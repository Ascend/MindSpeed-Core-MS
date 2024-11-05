# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

from typing import Any
import mindspore
from mindspore import Tensor, mint
from mindspore.nn import Cell
from mindspore.mint.distributed import get_world_size
from mindspore.communication.comm_func import all_to_all_single_with_output_shape


def single_all_to_all(input_, scatter_idx, gather_idx, group):
    seq_world_size = get_world_size(group)
    inp_shape = list(input_.shape)
    inp_shape[scatter_idx] = inp_shape[scatter_idx] // seq_world_size
    if scatter_idx < 2:
        input_t = input_.reshape(
            [seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).contiguous()
    else:
        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        input_t = input_.reshape(
            [-1, seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).swapaxes(0, 1).contiguous()
    output = mint.zeros_like(input_t)
    output = all_to_all_single_with_output_shape(output.shape, input_t, group=group)
    # if scattering the seq-dim, transpose the heads back to the original dimension
    # e.g., [cp, s/cp, b, n/cp, d] -> [s/cp, b, cp, n/cp, d]
    if scatter_idx < 2:
        output = output.swapaxes(0, 1).swapaxes(1, 2).contiguous()

    return output.reshape(
        inp_shape[: gather_idx] + [inp_shape[gather_idx] * seq_world_size, ] + inp_shape[gather_idx + 1:]).contiguous()


class _SeqAllToAll(mindspore.nn.Cell):
    def __init__(self, group, scatter_idx: int, gather_idx: int):
        super().__init__()
        self.group = group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def construct(self, input_: Tensor) -> Tensor:
        return single_all_to_all(input_, self.scatter_idx, self.gather_idx, self.group)

    def bprop(self, *args):
        return single_all_to_all(args[-1], self.gather_idx, self.scatter_idx, self.group)


class UlyssesContextAttention(mindspore.nn.Cell):
    """Initialization.
    Arguments:
        local_attention (Cell): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
            self,
            local_attention: Cell,
            sequence_process_group,
            scatter_idx: int = 2,
            gather_idx: int = 0,
    ) -> None:
        super(UlyssesContextAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.seq_all_to_all = _SeqAllToAll(self.spg, self.scatter_idx, self.gather_idx)
        self.seq_all_to_all2 = _SeqAllToAll(self.spg, self.gather_idx, self.scatter_idx)

    def construct(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # in shape : e.g., [s/p:h:]
        query_layer = self.seq_all_to_all(query)
        key_layer = self.seq_all_to_all(key)
        value_layer = self.seq_all_to_all(value)
        # out shape : e.g., [s:h/p:]
        context_layer = self.local_attn(query_layer, key_layer, value_layer, *args, **kwargs)
        output = self.seq_all_to_all2(context_layer)
        # out e.g., [s/p::h]
        return output