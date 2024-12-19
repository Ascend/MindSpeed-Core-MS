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
"""clip grad and scale grad"""
from mindspore import mint
from mindspore.communication import get_group_size, GlobalComm
from mindspore.communication.comm_func import all_reduce


def inplace_apply_to_tensor_list(func: callable):
    """Apply a function to a list of tensors in place.

    Args:
        func (callable): The function to apply to each tensor in the list.
    Returns:
        callable: The function that applies the input function to each tensor in the list in place.
    """

    def inplace_apply_func(tensor_list, *args, **kwargs):
        for idx, _ in enumerate(tensor_list):
            tensor_list[idx].copy_(func(tensor_list[idx], *args, **kwargs))

    return inplace_apply_func


def param_is_not_shared(param):
    """ get param not shaed """
    return not hasattr(param, 'shared') or not param.shared


def get_grad_norm_fp32(grads_for_norm, norm_type=2, parallel_group=None):
    """ get grad norm fp32 """
    total_norm = mint.norm(mint.stack([mint.norm(grad, norm_type) for grad in grads_for_norm],),
                           norm_type) ** norm_type

    if parallel_group is None:
        parallel_group = GlobalComm.WORLD_COMM_GROUP
    if get_group_size(parallel_group) > 1:
        total_norm = all_reduce(total_norm, "sum", parallel_group)[0]

    total_norm = total_norm.item() ** (1.0 / norm_type)
    return total_norm


def clip_grad_by_total_norm_fp32(parameters, max_norm, total_norm):
    """ clip gradients by global norm. """
    grads = []
    for param in parameters:
        if param.grad is not None:
            grads.append(param.grad)
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        grad_func = inplace_apply_to_tensor_list(mint.mul)
        grad_func(grads, clip_coeff)
