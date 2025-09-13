# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""
This module provides custom collective operations such as `_gather`,
which supports Ascend/NPU/CUDA devices and flexible tensor gathering across process groups.
"""

from typing import List
import torch.distributed as dist
import torch


def _gather(
        input_: torch.Tensor,
        pg: dist.ProcessGroup,
        dim: int = -1,
        gather_sizes: List = None
):
    """
    Gather tensors from all processes in the given process group along a specified dimension.
    Supports devices: 'cuda', 'npu', 'Ascend'.
    """
    input_ = input_.contiguous()
    world_size = dist.get_world_size(pg)

    # add Supports devices: 'Ascend'.
    if input_.device.type not in ["cuda", "npu", "Ascend"]:
        raise AssertionError("input tensor must in cuda, npu, or Ascend")

    if world_size == 1:
        return input_

    # all gather
    if gather_sizes is not None:
        tensor_list = []
        tensor_shape_base = input_.size()
        for i in range(world_size):
            tensor_shape = list(tensor_shape_base)
            tensor_shape[dim] = gather_sizes[i]
            tensor_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))
    else:
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, input_, group=pg)

    # concat
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output
