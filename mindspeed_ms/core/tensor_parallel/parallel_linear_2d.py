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
"""ParallelLinear2D."""

from typing import Callable
import mindspore.ops.functional as F
from mindspore import Parameter, nn, ops
from mindspore.common.initializer import initializer
from mindspore.communication.comm_func import all_gather_into_tensor, reduce_scatter_tensor
from mindspeed_ms.core.parallel_state import CollectiveCommIntf


class ParallelLinear2D(nn.Cell):
    """Linear2D layer with row and column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: If True, do not add the bias term, instead
                       return it to be added by the caller. This
                       enables performance optimations where bias can
                       be fused with other elementwise operations.
        skip_weight_param_allocation: If True, weight parameter is not allocated and must be passed
                                      as a keyword argument `weight` during the forward pass. Note
                                      that this does not affect bias, which will be allocated if
                                      bias is True. Defaults to False.
        is_expert: If True, the layer is treated as an MoE expert layer.
        config: ModelParallelConfig object
        tp_comm_buffer_name: Communication buffer name is not used in
                             non-Transformer-Engine modules.
        partition_dim: divide with dim, column parallel set 0, row parallel set 1

    """

    def __init__(
            self,
            input_size,
            output_size,
            *,
            config,
            init_method: Callable,
            add_bias=True,
            gather_output=False,
            stride=1,
            keep_master_weight_for_test=False,
            skip_bias_add=False,
            skip_weight_param_allocation: bool = False,
            is_expert: bool = False,
            ag_comm_intf: CollectiveCommIntf = None,
            ag_sd_rcv_overlap_comm_intf=1,
            rs_comm_intf: CollectiveCommIntf = None,
            rs_sd_rcv_overlap_comm_intf=1,
            enable_overlap_ag_with_matmul=False,
            enable_overlap_matmul_with_rs=False,
            partition_dim: int = 1,
        ):
        super().__init__()
        self.config = config
        self.init_method = init_method
        self.stride = stride
        self.keep_master_weight_for_test = keep_master_weight_for_test
        self.add_bias = add_bias
        self.input_size = input_size
        self.output_size = output_size
        self.ag_comm_intf = ag_comm_intf
        self.rs_comm_intf = rs_comm_intf
        self.ag_comm_world_sz = ag_comm_intf.get_comm_group_world_size()
        self.rs_comm_world_sz = rs_comm_intf.get_comm_group_world_size()
        # when AG comm group is small, do overlap AG with matmul.
        self.enable_overlap_ag_with_matmul = enable_overlap_ag_with_matmul
        self.enable_overlap_matmul_with_rs = enable_overlap_matmul_with_rs
        self.ag_overlap_comm_intf = ag_sd_rcv_overlap_comm_intf
        self.rs_sd_rcv_overlap_comm_intf = rs_sd_rcv_overlap_comm_intf

        assert input_size % self.rs_comm_world_sz == 0, "input size should be divisible by tp-y"
        assert output_size % self.ag_comm_world_sz == 0, "output size should be divisible by tp-x"

        for param in [stride, ag_sd_rcv_overlap_comm_intf, rs_sd_rcv_overlap_comm_intf, partition_dim]:
            assert param == 1, F"param({param}) is not supported yet"
        for param in [gather_output, keep_master_weight_for_test, skip_weight_param_allocation,
                      is_expert, enable_overlap_ag_with_matmul, enable_overlap_matmul_with_rs]:
            assert param is False, F"param({param}) is not supported yet"

        self.input_size_per_partition = input_size // self.rs_comm_world_sz
        self.output_size_per_partition = output_size // self.ag_comm_world_sz
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert

        self.partition_dim = partition_dim
        self.param_init_dtype = self.config.params_dtype
        self.compute_dtype = self.config.compute_dtype

        self.weight = Parameter(initializer(init_method,
                                            [self.output_size_per_partition, self.input_size_per_partition],
                                            self.param_init_dtype,),
                                name="weight")

        if self.add_bias:
            self.bias = Parameter(initializer(init_method, [self.output_size_per_partition],
                                              self.param_init_dtype,), name="bias")
            self.internal_params = [self.weight, self.bias]
        else:
            self.bias = None
            self.internal_params = [self.weight]

    def construct(self, activation_input):
        """Construct"""
        origin_dtype = F.dtype(activation_input)
        self.weight = ops.cast(self.weight, self.compute_dtype)
        activation_input = ops.cast(activation_input, self.compute_dtype)

        total_input = all_gather_into_tensor(activation_input.contiguous(), self.ag_comm_intf.get_comm_group())[0]
        matmul_res = total_input @ self.weight.t()
        matmul_output = reduce_scatter_tensor(matmul_res, group=self.rs_comm_intf.get_comm_group())[0]
        output = (matmul_output + self.bias) if self.add_bias else matmul_output
        output = ops.cast(output, origin_dtype)

        return output

    def bprop(self, *args):
        """backpropagation."""
        origin_dtype = F.dtype(args[-1])
        self.weight = ops.cast(self.weight, self.compute_dtype)
        forward_input = ops.cast(args[0], self.compute_dtype)
        dout = ops.cast(args[2], self.compute_dtype)

        total_grad_output = all_gather_into_tensor(dout.contiguous(), self.rs_comm_intf.get_comm_group())[0]
        gathered_tensors, gathered_tensors_handle = all_gather_into_tensor(forward_input.contiguous(),
                                                                           group=self.ag_comm_intf.get_comm_group(),
                                                                           async_op=True)
        # [s/cp, b, E/x] @ [E/x, H/y]--> [s/cp, b, H/y] (partial sum)
        partial_grad_input = total_grad_output @ self.weight
        # [s/cp, b, H/y] (partial sum)---RS(X)--->[s/cp, b, H/(xy)] (full sum)
        grad_input, grad_input_handle = reduce_scatter_tensor(partial_grad_input,
                                                              group=self.ag_comm_intf.get_comm_group(), async_op=True)

        sb = total_grad_output.shape[0] * total_grad_output.shape[1]
        # [s/cp, b, E/x]--view--> [sb/cp, E/x]
        total_grad_output = total_grad_output.view(sb, total_grad_output.shape[2])
        if gathered_tensors_handle:
            gathered_tensors_handle.wait()

        # [s/(x*cp), b, h/y]---AG(X)--->[s/cp, b, h/y]
        total_activation_input = gathered_tensors
        # [s/cp, b, h/y]--view--> [sb/cp, h/y]
        total_activation_input = total_activation_input.view(sb, total_activation_input.shape[2])
        # [E/x, sb/cp] @ [sb/cp, h/y] ---> [E/x, h/y]
        grad_weight = total_grad_output.t() @ total_activation_input
        grad_bias = total_grad_output.sum(axis=0) if self.add_bias else None
        if grad_input_handle:
            grad_input_handle.wait()

        grad_input = ops.cast(grad_input, origin_dtype)
        grad_weight = ops.cast(grad_weight, origin_dtype)
        if self.add_bias:
            grad_bias = ops.cast(grad_bias, origin_dtype)
            return (grad_input,), {self.weight: grad_weight, self.bias: grad_bias}

        return (grad_input,), {self.weight: grad_weight}
