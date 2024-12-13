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
"""Comm utils."""

import mindspore.nn as nn
from mindspore import ops
from mindspore.communication.comm_func import all_gather_into_tensor
from mindspeed_ms.core.parallel_state import (
    CollectiveCommIntf,
    TPXCollectiveComm,
    TPYCollectiveComm
)


def split_along_last_dim(local_rank_input, comm_intf: CollectiveCommIntf = TPYCollectiveComm):
    """Split tensor along last dimension."""
    world_size = comm_intf.get_comm_group_world_size()
    if world_size == 1:
        return local_rank_input

    last_dim = local_rank_input.dim() - 1
    last_dim_size = local_rank_input.shape[last_dim] // world_size

    tensor_list = ops.split(local_rank_input, last_dim_size, axis=last_dim)

    rank = comm_intf.get_comm_rank()
    output = tensor_list[rank].contiguous()

    return output


def split_along_first_dim(local_rank_input, comm_intf: CollectiveCommIntf = TPXCollectiveComm):
    """Split tensor along first dimension."""
    world_size = comm_intf.get_comm_group_world_size()
    if world_size == 1:
        return local_rank_input

    dim_size = local_rank_input.shape[0]
    assert (
        dim_size % world_size == 0
    ), "First dimension of the tensor should be divisible by parallel size"
    local_dim_size = dim_size // world_size
    rank = comm_intf.get_comm_rank()
    dim_offset = rank * local_dim_size

    output = local_rank_input[dim_offset : dim_offset + local_dim_size].contiguous()

    return output


def gather_along_last_dim(local_rank_input, comm_intf: CollectiveCommIntf = TPYCollectiveComm):
    """Gather tensor along last dimension."""
    world_size = comm_intf.get_comm_group_world_size()

    if world_size == 1:
        return local_rank_input
    init_shape = local_rank_input.shape[0]

    local_rank_input, _ = all_gather_into_tensor(
        local_rank_input.contiguous(), group=comm_intf.get_comm_group(), async_op=False
    )
    return ops.concat(ops.split(local_rank_input, init_shape, axis=0), axis=-1)


def gather_along_first_dim(local_rank_input, comm_intf: CollectiveCommIntf = TPXCollectiveComm):
    """Gather tensor along first dimension."""
    world_size = comm_intf.get_comm_group_world_size()

    if world_size == 1:
        return local_rank_input
    local_rank_input, _ = all_gather_into_tensor(
        local_rank_input.contiguous(), group=comm_intf.get_comm_group(), async_op=False
    )
    return local_rank_input


class _ScatterAlongLastDim(nn.Cell):
    """Scatter tensor along last dimension class."""
    def __init__(self, comm_intf: CollectiveCommIntf):
        super().__init__()
        self.comm_intf = comm_intf

    def construct(self, input_):
        return split_along_last_dim(input_, self.comm_intf)

    def bprop(self, *args):
        return gather_along_last_dim(args[-1], self.comm_intf)  # , None


class _ScatterAlongFirstDim(nn.Cell):
    """Scatter tensor along first dimension class."""
    def __init__(self, comm_intf: CollectiveCommIntf):
        super().__init__()
        self.comm_intf = comm_intf

    def construct(self, input_):
        return split_along_first_dim(input_, self.comm_intf)

    def bprop(self, *args):
        return gather_along_first_dim(args[-1], self.comm_intf)  # , None


def auto_grad_scatter_along_last_dim(input_, comm_intf: CollectiveCommIntf):
    """Auto grad scatter along last dimension."""
    net = _ScatterAlongLastDim(comm_intf)
    return net(input_)


def auto_grad_scatter_along_first_dim(input_, comm_intf: CollectiveCommIntf):
    """Auto grad scatter along first dimension."""
    net = _ScatterAlongFirstDim(comm_intf)
    return net(input_)


class _GatherAlongLastDim(nn.Cell):
    """Gather tensor along last dimension class."""
    def __init__(self, comm_intf: CollectiveCommIntf):
        super().__init__()
        self.comm_intf = comm_intf

    def construct(self, input_):
        return gather_along_last_dim(input_, self.comm_intf)

    def bprop(self, *args):
        return split_along_last_dim(args[-1], self.comm_intf)


class _GatherAlongFirstDim(nn.Cell):
    """Gather tensor along first dimension class."""
    def __init__(self, comm_intf: CollectiveCommIntf):
        super().__init__()
        self.comm_intf = comm_intf

    def construct(self, input_):
        return gather_along_first_dim(input_, self.comm_intf)

    def bprop(self, *args):
        return split_along_first_dim(args[-1], self.comm_intf)


def auto_grad_gather_along_last_dim(input_, comm_intf: CollectiveCommIntf):
    """Auto grad gather along last dimension."""
    net = _GatherAlongLastDim(comm_intf)
    return net(input_)


def auto_grad_gather_along_first_dim(input_, comm_intf: CollectiveCommIntf):
    """Auto grad gather along first dimension."""
    net = _GatherAlongFirstDim(comm_intf)
    return net(input_)
