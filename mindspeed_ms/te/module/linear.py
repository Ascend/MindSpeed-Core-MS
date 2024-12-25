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

"""Parallel Linear Layer for TE."""

from typing import Any, Callable, Optional

import mindspore
from mindspore import nn, mint
from mindspore.common.api import _no_grad
from mindspore import Parameter

from mindspeed_ms.training import ModelParallelConfig
from mindspeed_ms.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
)
from mindspeed_ms.core.utils import divide

from mindspeed_ms.te.fp8 import fp8_matmul
from mindspeed_ms.te.fp8.metadata import FP8Metadata
from mindspeed_ms.te.module.ops import get_ops, DummyHandle
from mindspeed_ms.te.module.ops.ascend_turbo_ops import ASCEND_TURBO_CONFIG


class TEColumnParallelLinear(nn.Cell):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `ColumnParallelLinear` layer.
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            *,
            config: ModelParallelConfig,
            init_method: Callable,
            gather_output: bool,
            bias: bool,
            skip_bias_add: bool,
            is_expert: bool,
            skip_weight_param_allocation: bool = False,
    ):
        if gather_output:
            raise ValueError('Transformer Engine linear layers do not support gather_output = True')

        if is_expert:
            raise ValueError('Transformer Engine linear layers do not yet support MoE')

        super(TEColumnParallelLinear, self).__init__()
        self.fp8_meta = FP8Metadata(['fprop', 'dgrad', 'wgrad'])

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        self.config = config
        self.skip_weight_param_allocation = skip_weight_param_allocation

        world_size = get_tensor_model_parallel_world_size()

        self.output_size_per_partition = divide(output_size, world_size)

        # Initialize weight.
        if not skip_weight_param_allocation:
            self.weight = Parameter(mint.ones((self.output_size_per_partition, self.input_size),
                                              dtype=config.params_dtype))
            if config.perform_initialization:
                init_method(self.weight)
            setattr(self.weight, 'allreduce', True)
        else:
            self.weight = None

        if bias:
            self.bias = Parameter(mint.zeros(self.output_size_per_partition, dtype=config.params_dtype))
            if config.perform_initialization:
                # Always initialize bias to zero.
                with _no_grad():
                    self.bias.fill_(0)
            setattr(self.bias, 'allreduce', True)
        else:
            self.bias = None

        self.sequence_parallel = config.sequence_parallel and world_size > 1
        self.allreduce_dgrad = world_size > 1 and not self.sequence_parallel

    def construct(self, input_: mindspore.Tensor, weight: Optional[mindspore.Tensor] = None):
        """Forward function"""
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass"
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != expected_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)},"
                    f"not {expected_shape} as expected"
                )

        bias = self.bias if not self.skip_bias_add else None

        if self.sequence_parallel:
            cps = ColumnParallelSeq()
            output = cps(input_, weight, bias, self.fp8_meta)
        else:
            cpns = ColumnParallelNoSeq()
            output = cpns(input_, weight, bias, self.fp8_meta)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class ColumnParallelSeq(nn.Cell):
    """ColumnParallelSeq"""
    def construct(self, input_, weight, bias, fp8_meta):
        """forward function"""
        self.use_bias = bias is not None
        self.total_input = None

        output_parallel, total_input = get_ops().allgather_matmul(input_, weight.t(), None, fp8_meta, 'fprop')

        if ASCEND_TURBO_CONFIG.save_allgather_input:
            self.total_input = total_input

        return output_parallel

    def bprop(self, *args):
        """backward function"""
        input_, weight, bias, fp8_meta, grad_output = args[0], args[1], args[2], args[3], args[5]
        tp_group = get_tensor_model_parallel_group()
        tp_world_size = get_tensor_model_parallel_world_size()

        all_gather_handle, total_input = DummyHandle(), self.total_input
        grad_weight, grad_bias = None, None
        if weight.requires_grad and not ASCEND_TURBO_CONFIG.save_allgather_input:
            grad_output, input_ = fp8_meta.pre_communication('wgrad', grad_output, input_)
            all_gather_handle, total_input = async_gather_along_first_dim(input_, tp_group, tp_world_size)

        if not fp8_meta.fp8_enable:
            grad_input = grad_output.matmul(weight)
        else:
            grad_input = fp8_matmul(grad_output, weight, fp8_meta, 'dgrad')

        sub_grad_input = grad_input.new_empty(input_.shape)
        sub_grad_input.requires_grad = False

        reduce_scatter_handle = mint.distributed.reduce_scatter_tensor(sub_grad_input, grad_input, group=tp_group,
                                                                       async_op=True)

        if weight.requires_grad:
            grad_output, total_input = reshape_two_dim(grad_output), reshape_two_dim(total_input)
            all_gather_handle.wait()

            if not fp8_meta.fp8_enable:
                grad_weight = grad_output.t().matmul(total_input)
            else:
                grad_weight = fp8_matmul(grad_output, total_input, fp8_meta, 'wgrad', (True, False))
            grad_bias = grad_output.sum(dim=0) if self.use_bias and bias.requires_grad else None

        reduce_scatter_handle.wait()
        return sub_grad_input, grad_weight, grad_bias, None

class ColumnParallelNoSeq(nn.Cell):
    """ColumnParallelNoSeq"""
    def construct(self, input_, weight, bias, fp8_meta):
        """Forward function"""
        self.use_bias = bias is not None

        if fp8_meta is None or not fp8_meta.fp8_enable:
            output = mint.matmul(input_, weight.t())
        else:
            output = fp8_matmul(input_, weight.t(), fp8_meta, 'fprop')

        if bias is not None:
            output = output + bias
        return output

    def bprop(self, *args):
        """backward function"""
        input_, weight, bias, fp8_meta, grad_output = args[0], args[1], args[2], args[3], args[5]
        grad_input = grad_output.matmul(weight)
        tp_group = get_tensor_model_parallel_group()

        handle = mint.distributed.all_reduce(grad_input, group=tp_group, async_op=True)
        grad_weight, grad_bias = None, None

        if weight.requires_grad:
            grad_output = reshape_two_dim(grad_output)

            if fp8_meta is None or not fp8_meta.fp8_enable:
                grad_weight = grad_output.t().matmul(reshape_two_dim(input_))
            else:
                grad_weight = fp8_matmul(grad_output.t(), input_, fp8_meta, 'wgrad')

            handle.wait()
            grad_bias = grad_output.sum(dim=0) if bias.requires_grad and self.use_bias else None
        else:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None

class TERowParallelLinear(nn.Cell):
    """TERowParallelLinear"""
    def __init__(
            self,
            input_size: int,
            output_size: int,
            *,
            config: ModelParallelConfig,
            init_method: Callable,
            bias: bool,
            input_is_parallel: bool,
            skip_bias_add: bool,
            is_expert: bool,
    ):
        if not input_is_parallel:
            raise ValueError(
                "Transformer Engine linear layers do not support input_is_parallel = False"
            )

        if is_expert:
            raise ValueError('Transformer Engine linear layers do not yet support MoE')

        super(TERowParallelLinear, self).__init__()
        self.fp8_meta = FP8Metadata(['fprop', 'dgrad', 'wgrad'])

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        self.skip_bias_add = skip_bias_add
        self.sequence_parallel = config.sequence_parallel and config.tensor_model_parallel_size > 1

        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.weight = Parameter(mint.ones((self.output_size, self.input_size_per_partition), dtype=config.params_dtype))
        if config.perform_initialization:
            init_method(self.weight)
        setattr(self.weight, 'allreduce', True)

        if bias:
            self.bias = Parameter(mint.zeros(self.output_size, dtype=config.params_dtype))
            if config.perform_initialization:
                # Always initialize bias to zero.
                with _no_grad():
                    self.bias.fill_(0)
            setattr(self.bias, 'allreduce', True)
            setattr(self.bias, 'sequence_parallel', self.sequence_parallel)
        else:
            self.bias = None


    def construct(self, input_: mindspore.Tensor):
        """forward function"""
        if self.sequence_parallel:
            rps = RowParallelSeq()
            output = rps(input_, self.weight, None, self.fp8_meta)
        else:
            rpns = RowParallelNoSeq()
            output = rpns(input_, self.weight, None, self.fp8_meta)

        if not self.skip_bias_add:
            output = (output + self.bias) if self.bias is not None else output
            output_bias = None
        else:
            output_bias = self.bias

        return output, output_bias

    def set_extra_state(self, state: Any):
        """ Extra state is ignored """

    def get_extra_state(self) -> None:
        """ Keep compatibility with TE state dict. """
        return None


class RowParallelSeq(nn.Cell):
    """RowParallelSeq"""
    def construct(self, input_, weight, bias, fp8_meta):
        """forward function"""
        self.use_bias = bias is not None
        output_parallel = get_ops().matmul_reduce_scatter(input_, weight.t(), bias, fp8_meta, 'fprop')
        return output_parallel

    def bprop(self, *args):
        """backward function"""
        total_input, weight, bias, fp8_meta, grad_output = args[0], args[1], args[2], args[3], args[5]

        grad_input, grad_output = get_ops().allgather_matmul(grad_output, weight, None, fp8_meta, 'dgrad')
        grad_weight, grad_bias = None, None

        if weight.requires_grad:
            grad_output = reshape_two_dim(grad_output)
            total_input = reshape_two_dim(total_input)
            if not fp8_meta.fp8_enable:
                grad_weight = grad_output.t().matmul(total_input)
            else:
                grad_weight = fp8_matmul(grad_output, total_input, fp8_meta, 'wgrad', (True, False))
            grad_bias = grad_output.sum(dim=0) if self.use_bias and bias.requires_grad else None

        return grad_input, grad_weight, grad_bias, None

class RowParallelNoSeq(nn.Cell):
    """RowParallelNoSeq"""
    def construct(self, input_, weight, bias, fp8_meta):
        """forward function"""
        self.input_ = input_
        self.weight = weight
        self.use_bias = bias is not None
        self.fp8_meta = fp8_meta

        output_ = get_ops().matmul_all_reduce(input_, weight.t(), bias, fp8_meta, 'fprop')
        return output_

    def bprop(self, *args):
        """backward function"""
        total_input, weight, bias, fp8_meta, grad_output = args[0], args[1], args[2], args[3], args[5]

        if fp8_meta is None or not fp8_meta.fp8_enable:
            grad_input = grad_output.matmul(weight)
        else:
            grad_input = fp8_matmul(grad_output, weight, fp8_meta, 'dgrad')

        grad_weight, grad_bias = None, None

        if weight.requires_grad:
            grad_output = reshape_two_dim(grad_output)

            if fp8_meta is None or not fp8_meta.fp8_enable:
                grad_weight = grad_output.t().matmul(reshape_two_dim(total_input))
            else:
                grad_weight = fp8_matmul(grad_output.t(), reshape_two_dim(total_input), fp8_meta, 'wgrad')

            grad_bias = grad_output.sum(dim=0) if bias.requires_grad and self.use_bias else None

        return grad_input, grad_weight, grad_bias, None

def async_gather_along_first_dim(input_, group, world_size):
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size
    output_ = input_.new_empty(dim_size)
    output_.requires_grad = False
    work = mint.distributed.all_gather_into_tensor(output_, input_.contiguous(), group=group, async_op=True)
    return work, output_

def te_matmul(x, y, fp8_meta, key):
    if fp8_meta is None or not fp8_meta.fp8_enable:
        output = x.matmul(y)
    else:
        output = fp8_matmul(x, y, fp8_meta, key)
    return output

def reshape_two_dim(input_tensor):
    # Convert the tensor shapes to 2D for execution compatibility
    output = input_tensor.reshape(-1, input_tensor.shape[-1])
    if hasattr(input_tensor, 'fp8_amax'):
        setattr(output, 'fp8_amax', input_tensor.fp8_amax)
    return output
