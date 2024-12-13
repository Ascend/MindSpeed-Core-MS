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
"""RMSNorm2D."""

import mindspore.ops.functional as F
import mindspore.common.dtype as mstype
from mindspore import Parameter, nn, ops
from mindspore.common.initializer import initializer
from mindspore.communication.comm_func import all_reduce
from mindspeed_ms.core.parallel_state import TPYCollectiveComm, CollectiveCommIntf, TPXCollectiveComm

class RMSNorm2D(nn.Cell):
    """RMS Normaliation 2d module
            Args:
                hidden_size (int): The width of input, i.e. hidden size
                eps (float): epsilon to use for the norm, default to 1e-6
                last_dim_split_comm_intf: All-reduce at last dim comm intf.
    """
    def __init__(self,
                 hidden_size: int,
                 eps: float = 1e-6,
                 last_dim_split_comm_intf: CollectiveCommIntf = TPYCollectiveComm(),
                 ):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.last_dim_split_comm_intf = last_dim_split_comm_intf
        self.last_dim_split_comm_world_size = self.last_dim_split_comm_intf.get_comm_group_world_size()
        # partitioning dimension
        assert self.hidden_size % self.last_dim_split_comm_world_size == 0, "{} is not divisible by {}".format(
            self.hidden_size, self.last_dim_split_comm_world_size)
        self.partitioned_dim = self.hidden_size // self.last_dim_split_comm_world_size
        self.weight = Parameter(initializer("ones", (self.partitioned_dim,), dtype=mstype.float32))
        self.internal_params = [self.weight]

    def construct(self, x):
        """construct"""
        origin_dtype = x.dtype
        x = ops.cast(x, mstype.float32)
        pow_mean_ = x.pow(2).mean(-1, keep_dims=True)
        pow_mean, _ = all_reduce(pow_mean_, group=self.last_dim_split_comm_intf.get_comm_group())
        var_x = ops.rsqrt(pow_mean / self.last_dim_split_comm_world_size + self.eps)
        norm_x = ops.mul(x, var_x)
        self.norm_x, self.var_x = norm_x, var_x
        norm_x = ops.cast(norm_x, origin_dtype)
        res = norm_x * self.weight
        return res

    def bprop(self, *args):
        """backpropagation"""
        origin_dtype = F.dtype(args[2])
        dout = ops.cast(args[2], mstype.float32)
        grad_norm_x = ops.mul(dout, self.weight)
        output_grad_mul_x_sum = (grad_norm_x * self.norm_x).sum(axis=-1, keepdims=True)
        output_grad_mul_x_sum, output_grad_mul_x_sum_handle = all_reduce(
            output_grad_mul_x_sum,
            group=self.last_dim_split_comm_intf.get_comm_group(),
            async_op=True
        )
        # calculate grad_weight
        grad_weight = ops.mul(dout, self.norm_x).sum(axis=list(range((len(dout.shape) - 1))))
        grad_weight = all_reduce(grad_weight, group=TPXCollectiveComm.get_comm_group())[0]
        if output_grad_mul_x_sum_handle:
            output_grad_mul_x_sum_handle.wait()
        grad_input = (grad_norm_x - self.norm_x * (output_grad_mul_x_sum / self.hidden_size)) * self.var_x
        grad_input = ops.cast(grad_input, args[0].dtype)
        grad_weight = ops.cast(grad_weight, origin_dtype)
        return (grad_input,), {self.weight: grad_weight}
