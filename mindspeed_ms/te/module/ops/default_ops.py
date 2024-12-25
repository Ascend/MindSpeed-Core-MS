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

"""default ops."""

from mindspore import mint
from mindspeed_ms.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size
)
from mindspeed_ms.te.module.ops.ascend_turbo_ops import AscendTurboOps
from mindspeed_ms.te.fp8 import fp8_matmul


class DefaultOps(AscendTurboOps):
    """DefaultOps"""

    @staticmethod
    def allgather_matmul(input_, weight, bias, fp8_meta=None, key=None):
        tp_world_size = get_tensor_model_parallel_world_size()
        tp_group = get_tensor_model_parallel_group()
        dim_size = list(input_.shape)
        dim_size[0] = dim_size[0] * tp_world_size

        total_input = input_.new_empty(dim_size)
        mint.distributed.all_gather_into_tensor(total_input, input_.contiguous(), group=tp_group, async_op=False)

        if fp8_meta is None or not fp8_meta.fp8_enable:
            output = mint.matmul(total_input, weight)
        else:
            output = fp8_matmul(total_input, weight, fp8_meta, key)

        return output, total_input

    @staticmethod
    def matmul_reduce_scatter(input_, weight, bias, fp8_meta=None, key=None):
        tp_world_size = get_tensor_model_parallel_world_size()
        tp_group = get_tensor_model_parallel_group()

        #TODO
        if fp8_meta is not None:
            input_, weight = fp8_meta.pre_communication(key, input_, weight)

        if fp8_meta is None or not fp8_meta.fp8_enable:
            output_ = mint.matmul(input_, weight)
        else:
            output_ = fp8_matmul(input_, weight, fp8_meta, key)

        dim_size = list(output_.shape)
        dim_size[0] = dim_size[0] // tp_world_size
        output = output_.new_empty(dim_size)
        mint.distributed.reduce_scatter_tensor(output, output_.contiguous(), group=tp_group)
        return output

    @staticmethod
    def matmul_all_reduce(input_, weight, bias, fp8_meta=None, key=None):
        tp_world_size = get_tensor_model_parallel_world_size()
        if fp8_meta is None or not fp8_meta.fp8_enable:
            output_ = mint.matmul(input_, weight)
        else:
            output_ = fp8_matmul(input_, weight, fp8_meta, key)

        if tp_world_size > 1:
            mint.distributed.all_reduce(output_, group=get_tensor_model_parallel_group())

        if bias is not None:
            output_ = output_ + bias
        return output_
