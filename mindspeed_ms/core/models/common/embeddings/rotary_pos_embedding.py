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
# ============================================================================
"""Rotary position embedding."""

__all__ = ["RotaryEmbedding", "apply_rotary_pos_emb"]

import math
import mindspore as ms
from mindspore import Tensor, ops, mint
from mindspeed_ms.training.global_vars import get_args

from mindspeed_ms.legacy.model.module import Module
from mindspeed_ms.core.transformer.transformer_config import TransformerConfig
from mindspeed_ms.core.transformer.transformer_block import TransformerBlock
from mindspeed_ms.core import parallel_state


class RotaryEmbedding(Module):
    r"""
    Rotary positional embedding for language model.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained from transformer config.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings.
            Default: 1.0.
        rotary_interleaved (bool, optional): Determines the method of applying rotary embeddings to the input
            dimensions. Default: False.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Default: None.
        rotary_base (int, optional): Base period for rotary position embeddings. Default: 10000.

    Inputs:
        - **max_seq_len** (int) - Max sequence length of inputs.
        - **offset** (int) - The starting point for the position encoding.

    Outputs:
        - **emb** (Tensor) - Embeddings after applying RoPE.

    Raises:
        NotImplementedError: If `rotary_interleaved` is True.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import os
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.communication import init
        >>> from mindspore import ops
        >>> from mindspeed_ms.core.config import TransformerConfig
        >>> from mindspeed_ms.core.models.common.embeddings.rotary_pos_embedding import (
        ...     RotaryEmbedding,
        ...     apply_rotary_pos_emb
        ... )
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> rank_id = os.environ.get('RANK_ID')
        >>> if rank_id is not None:
        >>>     ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True)
        >>>     init()
        >>> seed_value = 42
        >>> ms.set_seed(seed_value)
        >>> np.random.seed(seed_value)
        >>> class MyAttention(nn.Cell):
        >>>     def __init__(self, config: TransformerConfig):
        >>>         super(MyAttention, self).__init__()
        >>>         self.config = config
        >>>     def construct(self, x, freqs):
        >>>         return apply_rotary_pos_emb(x, freqs, self.config)
        >>> class MyNet(nn.Cell):
        >>>     def __init__(self, config: TransformerConfig):
        >>>         super(MyNet, self).__init__()
        >>>         self.n_heads = config.num_attention_heads
        >>>         self.head_dim = dim // self.n_heads
        >>>         self.rotary_embedding = RotaryEmbedding(self.head_dim)
        >>>         self.attention = MyAttention(config)
        >>>         dp = config.data_parallel
        >>>         self.transpose = ops.Transpose().shard(((dp, 1, 1, 1),))
        >>>         self.transpose_back = ops.Transpose().shard(((dp, 1, 1, 1),))
        >>>         self.reshape = ops.Reshape()
        >>>     def construct(self, x: Tensor):
        >>>         bs_, seq_len_, dim_ = x.shape
        >>>         # [bs, seq_len, dim] -> [bs, seq_len, heads, head_dim]
        >>>         x = self.reshape(x, (bs_, seq_len_, self.n_heads, self.head_dim))
        >>>         # [bs, seq_len, heads, head_dim] -> [bs, heads, seq_len, head_dim]
        >>>         query = self.transpose(x, (0, 2, 1, 3))
        >>>         freqs = self.rotary_embedding(seq_len_)
        >>>         output = self.attention(query, freqs)
        >>>         # [bs, heads, seq_len, head_dim] -> [bs, seq_len, heads, head_dim]
        >>>         output = self.transpose_back(output, (0, 2, 1, 3))
        >>>         # [bs, seq_len, heads, head_dim] -> [bs, seq_len, dim]
        >>>         output = self.reshape(output, (bs_, seq_len_, dim_))
        >>>         return output
        >>> config_ = TransformerConfig()
        >>> config_.data_parallel = 1
        >>> config_.tensor_parallel = 1
        >>> config_.context_parallel = 1
        >>> config_.num_attention_heads = 8
        >>> bs = 2
        >>> seq_len = 4096
        >>> dim = 8192
        >>> input_shape = (bs, seq_len, dim)
        >>> net = MyNet(config_)
        >>> input_ = Tensor(np.random.standard_normal(input_shape).astype(np.float32))
        >>> output_ = net(input_)
        >>> print(output_.shape)
        (2, 4096, 8192)
    """

    def __init__(
            self,
            kv_channels,
            rotary_percent,
            rotary_interleaved=False,
            seq_len_interpolation_factor=None,
            rotary_base=10000,
    ):
        super().__init__()
        args = get_args()
        if args.rotary_base:
            self.rotary_base = args.rotary_base
        else:
            self.rotary_base = rotary_base

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.inv_freq = 1.0 / (self.rotary_base ** (mint.arange(0, dim, 2)[: (dim // 2)].astype(ms.float32) / dim))

        if hasattr(args, "rope_scaling_type") and args.rope_scaling_type == "llama3":
            self.inv_freq = apply_llama3_scaling(self.inv_freq)
        elif hasattr(args, "rope_scaling_type") and args.rope_scaling_type == "yarn":
            raise NotImplementedError('rope-scaling-type is not supported yarn for now.')

    def construct(self, max_seq_len, offset=0):
        """ Construct function of rotary embedding. """
        seq = (mint.arange(max_seq_len, dtype=self.inv_freq.dtype) + offset).astype(ms.float32)

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        freqs = ops.outer(seq, self.inv_freq)
        if not self.rotary_interleaved:
            emb = ops.concat((freqs, freqs), axis=-1)
        else:
            emb = ops.stack((freqs.view(-1, 1), freqs.view(-1, 1)), axis=-1).view(
                freqs.shape[0], -1
            )
        # emb [S, ..., D]
        emb = emb[:, None, None, :]
        if parallel_state.get_context_parallel_world_size() > 1:
            raise NotImplementedError('Rotary is not supported context parallel for now.')

        return Tensor(emb)

    def get_rotary_seq_len(
            self,
            inference_params,
            transformer: TransformerBlock,
            transformer_input: Tensor,
            transformer_config: TransformerConfig,
    ) -> float:
        """Function to get the rotary sequence length.

        Args:
            inference_params : Used during Inference time
            transformer (TransformerBlock): The transformer block (decoder/encoder) used by the model
            transformer_input (Tensor): _description_
            transformer_config (TransformerConfig): Transformer config used by the model

        Returns:
            float: The rotary sequence length
        """
        if inference_params is not None:
            rotary_seq_len = inference_params.max_sequence_length
        else:
            if transformer.set_hidden_states is not None:
                rotary_seq_len = transformer.set_hidden_states.value().shape[0]
            else:
                rotary_seq_len = transformer_input.shape[0]

            if transformer_config.sequence_parallel:
                rotary_seq_len *= transformer_config.tensor_model_parallel_size

        rotary_seq_len *= transformer_config.context_parallel_size

        return rotary_seq_len


def apply_llama3_scaling(freqs: ms.Tensor):
    """ apply llama3 scaling """
    args = get_args()
    original_length = args.original_max_position_embeddings

    low_freq_wavelen = original_length / args.low_freq_factor
    high_freq_wavelen = original_length / args.high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            freq = freq.tolist()
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            freq = freq / args.rope_scaling_factor
            freq = freq.tolist()
            new_freqs.append(freq)
        else:
            smooth = (original_length / wavelen - args.low_freq_factor) / (args.high_freq_factor - args.low_freq_factor)
            freq = (1 - smooth) * freq / args.rope_scaling_factor + smooth * freq
            freq = freq.tolist()
            new_freqs.append(freq)

    return ms.Tensor(new_freqs, dtype=freqs.dtype)


def _rotate_half(x, rotary_interleaved):
    if not rotary_interleaved:
        x1, x2 = mint.split(x, x.shape[-1] // 2, dim=-1)
        return ops.cat((-x2, x1), axis=-1)

    raise NotImplementedError('rotary_interleaved=True is not supported for now.')


def apply_rotary_pos_emb_bnsd(t, freqs, rotary_interleaved=False) -> Tensor:
    """
    Apply rotary positional embedding to input tensor.
    Please check https://kexue.fm/archives/8265 for detailed formulas

    Inputs:
        - **x** (Tensor) - Input tensor x with shape :math:`(B, N, S, D)`.
        - **freqs** (Tensor) - Rotary positional embedding tensor freq with shape :math:`(..., S, D)`.

    Outputs:
        - **output** (Tensor): The input tensor after applying RoPE.

    Supported Platforms:
        ``Ascend``
    """
    cos_ = mint.cos(freqs).to(t.dtype)
    sin_ = mint.sin(freqs).to(t.dtype)

    # rotate
    output = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    return output


# pylint: disable=missing-docstring
def _apply_fused_rotary_pos_emb(t: Tensor, freqs: Tensor, rotary_interleaved: bool = False) -> Tensor:
    rot_dim = freqs.shape[-1]
    t_shape_last_dim = t.shape[-1]
    if rot_dim != t_shape_last_dim:
        t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    cos_ = mint.cos(freqs).to(t.dtype)
    sin_ = mint.sin(freqs).to(t.dtype)
    mode = 1 if rotary_interleaved else 0
    t = ops.rotary_position_embedding(t, cos_, sin_, mode=mode)
    if rot_dim == t_shape_last_dim:
        return t
    return mint.cat((t, t_pass), dim=-1)


# pylint: disable=W0613
def apply_rotary_pos_emb(t, freqs, config, cu_seqlens=None) -> Tensor:
    if cu_seqlens is None:
        args = get_args()
        if args.use_fused_rotary_pos_emb:
            return _apply_fused_rotary_pos_emb(t, freqs)
        return apply_rotary_pos_emb_bnsd(t, freqs, rotary_interleaved=False)

    raise NotImplementedError('cu_seqlens input for apply_rotary_pos_emb() is not supported for now.')
