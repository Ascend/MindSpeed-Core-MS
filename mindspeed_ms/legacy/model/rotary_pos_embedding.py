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

import mindspore as ms
from mindspore import Tensor, ops, mint

from mindspeed_ms.core.config import get_args
from mindspeed_ms.core.parallel_state import get_context_parallel_rank, get_context_parallel_world_size
from .module import Module

def get_pos_emb_on_this_cp_rank(pos_emb, seq_dim):
    """
    Retrieve the positional embedding for the current compute node (CP rank).
    """
    args = get_args()
    if args.context_parallel_algo == 'megatron_cp_algo':
        if args.cp_attention_mask_type == 'general':
            pos_emb = _get_pos_emb_on_this_cp_rank_in_ulysses_cp(pos_emb, seq_dim)
        else:
            pos_emb = _get_pos_emb_on_this_cp_rank_in_megatron_cp(pos_emb, seq_dim)
    elif args.context_parallel_algo == 'ulysses_cp_algo':
        pos_emb = _get_pos_emb_on_this_cp_rank_in_ulysses_cp(pos_emb, seq_dim)
    elif args.context_parallel_algo == 'hybrid_cp_algo':
        raise ValueError(f"Only megatron_cp_algo and ulysses_cp_algo supported")
    return pos_emb

def _get_pos_emb_on_this_cp_rank_in_ulysses_cp(pos_emb, seq_dim):
    cp_size = get_context_parallel_world_size()
    cp_rank = get_context_parallel_rank()
    pos_emb = ops.chunk(pos_emb, cp_size, seq_dim)[cp_rank]

    return pos_emb

def _get_pos_emb_on_this_cp_rank_in_megatron_cp(pos_emb, seq_dim):
    cp_size = get_context_parallel_world_size()
    cp_rank = get_context_parallel_rank()
    cp_idx = Tensor([cp_rank, (2 * cp_size - cp_rank - 1)], ms.int32)
    pos_emb = pos_emb.view(
        *pos_emb.shape[0:seq_dim], 2 * cp_size, -1, *pos_emb.shape[(seq_dim + 1):]
    )
    pos_emb = ops.index_select(pos_emb, seq_dim, cp_idx)
    pos_emb = pos_emb.view(*pos_emb.shape[0:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :])
    return pos_emb

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

        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.communication import init
        >>> from mindspeed_ms.legacy.model.rotary_pos_embedding import RotaryEmbedding
        >>> from mindspeed_ms.core.config import ModelParallelConfig, TrainingConfig, TransformerConfig
        >>> from mindspeed_ms.core.parallel_state import initialize_model_parallel
        >>> class MyNet(nn.Cell):
        ...     def __init__(self, config: TransformerConfig):
        ...         super(MyNet, self).__init__()
        ...         self.rotary_embedding = RotaryEmbedding(config.seq_length)
        ...     def construct(self, x: Tensor):
        ...         emb = self.rotary_embedding(x.shape[1])
        ...         return emb
        >>> ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
        >>> init()
        >>> initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
        >>> parallel_config = ModelParallelConfig(tensor_model_parallel_size=1)
        >>> training_config = TrainingConfig(parallel_config=parallel_config)
        >>> config = TransformerConfig(seq_length=16,
        ...                            vocab_size=1,
        ...                            num_layers=1,
        ...                            num_attention_heads=16,
        ...                            hidden_size=256,
        ...                            ffn_hidden_size=256,
        ...                            parallel_config=parallel_config,
        ...                            training_config=training_config)
        >>> bs = 2
        >>> seq_len = 16
        >>> hidden_size = 256
        >>> input_shape = (bs, seq_len, hidden_size)
        >>> net = MyNet(config)
        >>> input = Tensor(np.random.standard_normal(input_shape).astype(np.float32))
        >>> output = net(input)
        >>> print(output.shape)
        (16, 1, 1, 16)
    """

    def __init__(
            self,
            kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=False,
            seq_len_interpolation_factor=None,
            rotary_base=10000,
        ):
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved
        if self.rotary_interleaved:
            raise NotImplementedError('Rotary interleaved is not supported for now.')

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.inv_freq = 1.0 / (
            rotary_base ** (mint.arange(0, dim, 2)[: (dim // 2)].astype(ms.float32) / dim)
        )

    def construct(self, max_seq_len, offset=0):
        """ Construct function of rotary embedding. """
        seq = (mint.arange(max_seq_len, dtype=self.inv_freq.dtype) + offset).astype(ms.float32)

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        freqs = ops.outer(seq, self.inv_freq)
        if not self.rotary_interleaved:
            emb = ops.concat((freqs, freqs), axis=-1)
        else:
            raise NotImplementedError('Rotary interleaved is not supported for now.')

        # emb [S, ..., D]
        emb = emb[:, None, None, :]
        if get_context_parallel_world_size() > 1:
            # slice rotary_pos_emb along sequence dimension and select the partition of the current CP rank
            emb = get_pos_emb_on_this_cp_rank(emb, 0)
        return Tensor(emb)


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
    t = ops.rotary_position_embedding(t.contiguous(), cos_, sin_, mode=mode)
    if rot_dim == t_shape_last_dim:
        return t
    return mint.cat((t, t_pass), dim=-1)


# pylint: disable=W0613
def apply_rotary_pos_emb(t, freqs, config, cu_seqlens=None) -> Tensor:
    if cu_seqlens is None:
        if config.apply_rope_fusion:
            return _apply_fused_rotary_pos_emb(t, freqs)
        return apply_rotary_pos_emb_bnsd(t, freqs, rotary_interleaved=False)

    raise NotImplementedError('cu_seqlens input for apply_rotary_pos_emb() is not supported for now.')
