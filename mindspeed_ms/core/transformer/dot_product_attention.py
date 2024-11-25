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
""" Dot product attention """

import math

import mindspore as ms
from mindspore import Tensor, ops, mint
import mindspore.common.dtype as mstype

from mindspeed_ms.legacy.model.scale_mask_softmax import ScaleMaskSoftmax

from mindspeed_ms.core import parallel_state
from mindspeed_ms.core.utils import divide
from mindspeed_ms.core.tensor_parallel.random import get_rng_tracer
from mindspeed_ms.core.transformer import Module
from mindspeed_ms.core.transformer import TransformerConfig
from mindspeed_ms.core.transformer.enums import AttnMaskType
from mindspeed_ms.core.transformer.utils import attention_mask_func


class DotProductAttention(Module):
    """
    Region where selective activation recomputation is applied.
    This region is memory intensive but less compute intensive which
    makes activation checkpointing more efficient for LLMs (20B+).
    See Reducing Activation Recomputation in Large Transformer Models:
        https://arxiv.org/abs/2205.05198 for more details.

    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
            self,
            config: TransformerConfig,
            layer_number: int,
            attn_mask_type: AttnMaskType,
            attention_type: str,
            attention_dropout: float = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        if self.config.context_parallel_size != 1:
            raise ValueError("Context parallelism is not supported by DotProductAttention.")

        if self.config.window_size is not None:
            raise ValueError("Sliding Window Attention is not supported by DotProductAttention.")

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  # unused for now

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        compute_type = mstype.float32 if self.config.attention_softmax_in_fp32 else self.config.softmax_compute_dtype
        self.scale_mask_softmax = ScaleMaskSoftmax(
            mask_func=attention_mask_func,
            scale=coeff,
            softmax_compute_type=compute_type
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = ms.mint.nn.Dropout(
            p=self.config.attention_dropout if attention_dropout is None else attention_dropout
        )

    # pylint: disable=W0613
    def construct(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attention_mask: Tensor,
            attn_mask_type: AttnMaskType = None,
            packed_seq_params=None,
    ):
        """ construct """
        if packed_seq_params is not None:
            raise ValueError("Packed sequence is not supported by DotProductAttention.")

        # ===================================
        # Raw attention scores. [b, n/p, s, s]
        # ===================================

        # expand the key and value [sk, b, ng, hn] -> [sk, b, np, hn]
        # This is a noop for normal attention where ng == np. When using group query attention this
        # creates a view that has the keys and values virtually repeated along their dimension to
        # match the number of queries.

        # attn_mask_type is not used.
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            repeat_num = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
            key = mint.repeat_interleave(key, repeat_num, dim=2)
            value = mint.repeat_interleave(value, repeat_num, dim=2)

        # [b, np, sq, sk]
        output_size = (query.shape[1],
                       query.shape[2],
                       query.shape[0],
                       key.shape[0])
        # [sq, b, np, hn] -> [sq, b * np, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use simple strides
        # to extract the queries.
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)

        # pre-allocating input tensor: [b * np, sq, sk]
        matmul_input_buffer = ms.mint.zeros(
            (output_size[0] * output_size[1], output_size[2], output_size[3]), dtype=query.dtype,
        )
        matmul_result = ms.mint.baddbmm(
            matmul_input_buffer,
            query.swapaxes(0, 1),  # [b * np, sq, hn]
            key.swapaxes(0, 1).swapaxes(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.config.sequence_parallel:
            with get_rng_tracer().rng_fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value.shape[1],
            value.shape[2],
            query.shape[0],
            value.shape[3]
        )
        # change view [sk, b * np, hn]
        value = value.view(value.shape[0], output_size[0] * output_size[1], -1)
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)
        # matmul: [b * np, sq, hn]
        context = ops.bmm(attention_probs, value.swapaxes(0, 1))  # in ms.mint.bmm, only 3 dimension is supported
        # change view [b, np, sq, hn]
        context = context.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context.shape[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)

        return context
