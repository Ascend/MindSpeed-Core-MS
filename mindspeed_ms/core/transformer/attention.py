# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
""" self attention and cross attention """

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import mindspore as ms
from mindspore import Tensor, ops
from mindspore.communication import comm_func
import mindspore.common.dtype as mstype

from mindspeed_ms.core import parallel_state, tensor_parallel
from mindspeed_ms.core.transformer import Module
from mindspeed_ms.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from mindspeed_ms.core.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_world_size,
    get_context_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
)
from mindspeed_ms.core.context_parallel.ring_attention import RingAttention
from mindspeed_ms.core.context_parallel.flash_sp import FlashSP
from mindspeed_ms.core.transformer import TransformerConfig
from mindspeed_ms.core.utils import divide

from mindspeed_ms.core.transformer.enums import AttnMaskType
from mindspeed_ms.core.transformer import ModuleSpec, build_module
from mindspeed_ms.training import get_args as ms_args


@dataclass
class SelfAttentionSubmodules:
    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


@dataclass
class CrossAttentionSubmodules:
    linear_q: Union[ModuleSpec, type] = None
    linear_kv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None


def _merge_heads(x):
    """Merge attention heads."""
    # [B, N, S, D] -> [B, S, N, D]
    x = x.transpose(0, 2, 1, 3)
    bs, seq_len, num_heads, head_dim = x.shape
    # [B, S, N, D] -> [B, S ,H]
    merged_shape = (bs, seq_len, num_heads * head_dim)
    x_merged = x.reshape(merged_shape)
    return x_merged


class Attention(Module, ABC):
    """Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: Union[SelfAttentionSubmodules, CrossAttentionSubmodules],
            layer_number: int,
            attn_mask_type: AttnMaskType,
            attention_type: str,
    ):
        super().__init__(config=config)
        args = ms_args()
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, self.config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        # flash attn parameters
        self.use_flash_attention = args.use_flash_attn and attention_type == "self" \
            and self.attn_mask_type == AttnMaskType.causal
        if self.use_flash_attention:
            if self.attention_type != "self":
                raise NotImplementedError('FlashAttention code path only supports self-attention for now.')
            self.fa_config = self.config.fa_config
        self.enable_flash_sp = args.use_flash_sp

        self.sequence_parallel = self.config.sequence_parallel
        self.num_heads = self.config.num_attention_heads
        self.kv_channels = self.config.kv_channels
        self.norm_factor = math.sqrt(self.kv_channels)
        self.tp_group_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_group_size)
        self.use_lora = self.config.use_lora

        if get_context_parallel_world_size() > 1:
            if not self.enable_flash_sp:
                self.ring_attention = RingAttention(
                    self.num_heads,
                    input_layout="BNSD",
                    scale_value=1 / self.norm_factor,
                    sparse_mode=0
                )
            else:
                self.flash_sp = FlashSP(
                    self.num_heads,
                    input_layout="BSH",
                    scale_value=1 / self.norm_factor,
                    dp=get_data_parallel_world_size(),
                    mp=get_tensor_model_parallel_world_size(),
                    sp=get_context_parallel_world_size(),
                )

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
        )

        self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'

        # Output.
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
        )

    def _checkpointed_attention_forward(
            self,
            query,
            key,
            value,
            attention_mask,
            rotary_pos_emb=None,
            attn_mask_type=None,
            packed_seq_params=None,
    ):
        """Forward method with selective activation checkpointing."""

        def custom_forward(*inputs):
            query = inputs[0]
            key = inputs[1]
            value = inputs[2]
            attention_mask = inputs[3]
            attn_mask_type = inputs[5]
            attn_mask_type = AttnMaskType(attn_mask_type.item())
            output_ = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
            return output_

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        attn_mask_type = Tensor([attn_mask_type.value], dtype=ms.int32)
        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False,
            query,
            key,
            value,
            attention_mask,
            rotary_pos_emb,
            attn_mask_type,
        )

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_length: object, batch_size: object, dtype: object) -> object:
        """Allocate memory to store kv cache during inference."""

        return ms.mint.zeros(
            (inference_max_sequence_length,
             batch_size,
             self.num_query_groups_per_partition,
             self.hidden_size_per_attention_head),
            dtype=dtype,
        )

    def _adjust_key_value_for_inference(self, inference_params, key, value, rotary_pos_emb):
        """
        Saves the generated key and value tensors to the end of the buffers in inference_params.
        Returns the full size keys and values from the provided inference_params, as well as
        adjusted rotary_pos_emb.

        Returns a tuple: (key, value, rotary_pos_emb)
        """
        attn_mask_type = self.attn_mask_type
        if inference_params is None:
            return key, value, rotary_pos_emb, attn_mask_type

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        is_first_step = False
        if self.layer_number not in inference_params.key_value_memory_dict:
            inf_max_seq_length = inference_params.max_sequence_length
            inf_max_batch_size = inference_params.max_batch_size
            inference_key_memory = self._allocate_memory(
                inf_max_seq_length, inf_max_batch_size, key.dtype
            )
            inference_value_memory = self._allocate_memory(
                inf_max_seq_length, inf_max_batch_size, value.dtype
            )
            inference_params.key_value_memory_dict[self.layer_number] = (
                inference_key_memory,
                inference_value_memory,
            )
            is_first_step = True
        else:
            # Get the pre-allocated buffers for this layer
            inference_key_memory, inference_value_memory = inference_params.key_value_memory_dict[
                self.layer_number
            ]
            attn_mask_type = AttnMaskType.no_mask

        batch_start = inference_params.batch_size_offset
        batch_end = batch_start + key.size(1)
        assert batch_end <= inference_key_memory.size(1)
        sequence_start = inference_params.sequence_len_offset
        sequence_end = sequence_start + key.size(0)
        assert sequence_end <= inference_key_memory.size(0)
        # Copy key and values.
        inference_key_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = key
        inference_value_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = value
        key = inference_key_memory[:sequence_end, batch_start:batch_end, ...]
        value = inference_value_memory[:sequence_end, batch_start:batch_end, ...]

        # adjust the key rotary positional embedding
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # need to cross check this condition during inference
            # if not set_inference_key_value_memory:
            if not is_first_step:
                # In inference, we compute one token at a time.
                # Select the correct positional embedding
                # (only the last token in the sequence)
                q_pos_emb = q_pos_emb[sequence_end - 1: sequence_end]
            else:
                # In the first forward pass of inference,
                # we use the entire provided prefix.
                # q_pos_emb here has the rope embeddings of the entire
                # prefix + to-be-generated output so
                # we slice to just the prefix.
                q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
            k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
            rotary_pos_emb = (q_pos_emb, k_pos_emb)

        return key, value, rotary_pos_emb, attn_mask_type

    @abstractmethod
    def get_query_key_value_tensors(self, hidden_states, key_value_states):
        """
        This method needs to be implemented based on whether the derived class
        is "self-attn" or "cross-attn".
        """

    def construct(
            self,
            hidden_states,
            attention_mask,
            key_value_states=None,
            inference_params=None,
            rotary_pos_emb=None,
            packed_seq_params=None,
    ):
        """ construct """
        # hidden_states: [sq, b, h]

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2
        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params, key, value, rotary_pos_emb
        )

        # only SBH is supported in mindspeed.core.
        seq_len, bs, _ = hidden_states.shape

        # apply query, key, value projection
        if self.attention_type == "self":
            if self.sequence_parallel:
                seq_len = seq_len * self.tp_group_size

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None
            query = apply_rotary_pos_emb(query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q)
            key = apply_rotary_pos_emb(key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        # ==================================
        # core attention computation
        # ==================================

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        elif not self.use_flash_attention:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        elif get_context_parallel_world_size() <= 1:
            if attention_mask.ndim == 3:
                attention_mask = attention_mask.expand_dims(axis=1)
            if query.dtype == mstype.float32:
                query = query.astype(mstype.float16)
            if key.dtype == mstype.float32:
                key = key.astype(mstype.float16)
            if value.dtype == mstype.float32:
                value = value.astype(mstype.float16)
            attention_mask = attention_mask.astype(mstype.uint8)

            if self.fa_config and hasattr(self.fa_config, 'input_layout') and self.fa_config.input_layout == 'SBH':
                # SBND -> SBH
                fa_use_sbh = True
                query, key, value = [
                    x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]) for x in [query, key, value]
                ]
            else:
                # SBND -> BNSD
                fa_use_sbh = False
                query = query.transpose(1, 2, 0, 3)
                key = key.transpose(1, 2, 0, 3)
                value = value.transpose(1, 2, 0, 3)

            if self.fa_config:
                output = ops.flash_attention_score(
                    query,
                    key,
                    value,
                    self.num_heads_per_partition,
                    attn_mask=attention_mask,
                    scalar_value=1.0 / self.norm_factor,
                    **self.fa_config,
                )
            else:
                output = ops.flash_attention_score(
                    query,
                    key,
                    value,
                    self.num_heads_per_partition,
                    attn_mask=attention_mask,
                    scalar_value=1.0 / self.norm_factor,
                )
            if not fa_use_sbh:
                core_attn_out = _merge_heads(output)
                # BSH -> SBH
                core_attn_out = core_attn_out.swapaxes(0, 1)
            else:
                core_attn_out = output

        else:
            if query.dtype == mstype.float32:
                query = query.astype(mstype.float16)
            if key.dtype == mstype.float32:
                key = key.astype(mstype.float16)
            if value.dtype == mstype.float32:
                value = value.astype(mstype.float16)

            # SBND -> BNSD
            query = query.transpose(1, 2, 0, 3)
            key = key.transpose(1, 2, 0, 3)
            value = value.transpose(1, 2, 0, 3)

            if not self.enable_flash_sp:
                output = self.ring_attention(query, key, value)
            else:
                # BNSD to BSH
                query = query.transpose((0, 2, 1, 3)).reshape(bs, seq_len, -1)
                key = key.transpose((0, 2, 1, 3)).reshape(bs, seq_len, -1)
                value = value.transpose((0, 2, 1, 3)).reshape(bs, seq_len, -1)

                output = self.flash_sp(query, key, value)
                # BSH to BNSD
                output = output.reshape(bs, seq_len, -1, self.kv_channels).transpose(
                    (0, 2, 1, 3)
                )

            core_attn_out = _merge_heads(output)
            # BSH -> SBH
            core_attn_out = core_attn_out.swapaxes(0, 1)

        if packed_seq_params is not None:
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.linear_proj(core_attn_out)

        return output, bias


class SelfAttention(Attention):
    """Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: SelfAttentionSubmodules,
            layer_number: int,
            attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
        )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
        )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

    def run_realtime_tests(self):
        """Performs a consistency check.

        This function makes sure that tensors across devices are the same during an experiment.
        This is often not guaranteed to be so because of silent hardware failures (eg, memory
        corruption loading a checkpoint, network traffic corruption encountered during data transmission).

        (TODO) In the future, more tensors should be checked across the training run and
        checked every X iterations. This is left for future work. Equality of tensors is probably not
        required; transmitting hashes is sufficient."""

        if not self.config.qk_layernorm:
            return

        # check that all tensor parallel and data parallel ranks have the same
        # Q & K layernorm parameters.
        inputs = ms.mint.stack(
            [
                self.q_layernorm.weight.data,
                self.q_layernorm.bias.data,
                self.k_layernorm.weight.data,
                self.k_layernorm.bias.data,
            ]
        )
        dp_list = comm_func.all_gather_into_tensor(inputs, group=get_data_parallel_group())[0]

        def _compare(srcs, tgts, names, parallelism):
            if len(srcs) != len(tgts) or len(srcs) != len(names):
                raise ValueError(f"Length mismatch. Length of srcs: {len(srcs)}, tgts: {len(tgts)}, "
                                 f"names: {len(names)} should be equal")
            for src, tgt, name in zip(srcs, tgts, names):
                if not ms.mint.all(src == tgt):
                    diff = ops.norm(src - tgt)
                    raise ValueError(f"Discrepancy between {name} in {parallelism} ranks {i} and {rank}. Diff: {diff}")

        for i, dp in enumerate(dp_list):
            q_w, q_b, k_w, k_b = ops.unbind(dp)
            _compare(
                [q_w, q_b, k_w, k_b],
                [
                    self.q_layernorm.weight.data,
                    self.q_layernorm.bias.data,
                    self.k_layernorm.weight.data,
                    self.k_layernorm.bias.data,
                ],
                ["q_w", "q_b", "k_w", "k_b"],
                "DP",
            )

        tp_list = comm_func.all_gather_into_tensor(inputs, group=get_tensor_model_parallel_group())[0]
        for i, tp in enumerate(tp_list):
            q_w, q_b, k_w, k_b = ops.unbind(tp)
            _compare(
                [q_w, q_b, k_w, k_b],
                [
                    self.q_layernorm.weight.data,
                    self.q_layernorm.bias.data,
                    self.k_layernorm.weight.data,
                    self.k_layernorm.bias.data,
                ],
                ["q_w", "q_b", "k_w", "k_b"],
                "TP",
            )

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.shape[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        (query, key, value) = ms.mint.split(
            mixed_qkv,
            split_arg_list,
            dim=3,
        )

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.shape[0], query.shape[1], -1, self.hidden_size_per_attention_head)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value


class CrossAttention(Attention):
    """Cross-attention layer class

    Cross-attention layer takes input with size [s, b, h] and context with size
    [s, b, h] and returns output of the same size.
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: CrossAttentionSubmodules,
            layer_number: int,
            attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="cross",
        )

        if self.config.num_query_groups != self.config.num_attention_heads:
            raise ValueError(
                f"Group query attention is not currently supported in cross attention."
            )
        assert self.query_projection_size == self.kv_projection_size

        self.linear_q = build_module(
            submodules.linear_q,
            self.config.hidden_size,
            self.query_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )

        self.linear_kv = build_module(
            submodules.linear_kv,
            self.config.hidden_size,
            2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )

    def get_query_key_value_tensors(self, hidden_states, key_value_states):
        """
        Derives `query` tensor from `hidden_states`, and `key`/`value` tensors
        from `key_value_states`.
        """
        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        mixed_kv, _ = self.linear_kv(key_value_states)

        # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
        new_tensor_shape = mixed_kv.shape[:-1] + (
            self.num_attention_heads_per_partition,
            2 * self.hidden_size_per_attention_head,
        )
        mixed_kv = mixed_kv.view(*new_tensor_shape)

        # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
        (key, value) = tensor_parallel.split_tensor_along_last_dim(mixed_kv, 2)

        # Attention head [sq, b, h] --> [sq, b, hp]
        query, _ = self.linear_q(hidden_states)

        # [sq, b, hp] --> [sq, b, np, hn]
        new_tensor_shape = query.shape[:-1] + (
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        query = query.view(*new_tensor_shape)

        return query, key, value
