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
"""Normalization"""

__all__ = ["get_norm"]

import mindspore.common.dtype as mstype

from mindspore import nn, Parameter, Tensor, mint, ops
from mindspore.common.initializer import initializer


class LayerNorm(nn.Cell):
    r"""
    Layer norm operation.

    Args:
        normalized_shape (tuple): The shape of the input tensor
        eps (float): The epsilon value of the denominator. Default: ``1e-5``.
        params_dtype: The param init type. Default: ``mstype.float32``.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch, seq_length, hidden_size).

    Outputs:
        - Tensor with shape (batch, seq_length, hidden_size).
    """

    def __init__(self, normalized_shape, eps=1e-5, params_dtype=mstype.float32):
        super(LayerNorm, self).__init__()
        if params_dtype not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError("The type of parameter 'params_dtype' should in [float32, float16], "
                            "but got the type : {}.".format(type(params_dtype)))
        self.gamma = Parameter(initializer('ones', normalized_shape, params_dtype), name="gamma",
                               parallel_optimizer=False)
        self.beta = Parameter(initializer('zeros', normalized_shape, params_dtype), name="beta",
                              parallel_optimizer=False)
        self.eps = eps
        self.normalized_shape = normalized_shape

    def construct(self, x):
        """construct method"""
        output = mint.nn.functional.layer_norm(x, self.normalized_shape, self.gamma, self.beta, self.eps)
        return output


class RMSNorm(nn.Cell):
    r"""
    A self-defined RMSNorm operation using reduce mean.

    Args:
        dim (tuple): The shape of the input tensor
        eps (float): The epsilon value of the denominator. Default: ``1e-6``.
        sequence_parallel (bool): Set to true if sequence parallelism is being used,
          this marks the weights as needing to be allreduced. Default: ``False``.
        params_dtype (dtype.Number): The param init type. Default: ``mstype.float32``.
        scale (float): scale number for weight initialization. Default: ``1.0``.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch, seq_length, hidden_size).

    Outputs:
        - Tensor with shape (batch, seq_length, hidden_size).
    """

    def __init__(self, dim, eps=1e-6, sequence_parallel=False, params_dtype=mstype.float32,
                 scale=1.0):
        super(RMSNorm, self).__init__()
        self.eps = Tensor(float(eps), dtype=params_dtype)
        self.weight = Parameter(mint.ones((dim,), dtype=params_dtype) * scale)

        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def _norm(self, x):
        return mint.mul(x, mint.rsqrt(mint.add(mint.mean(mint.square(x), dim=-1, keepdim=True),
                                               self.eps)))

    def construct(self, x):
        """Forward of RMSNorm."""
        output = self._norm(x.float()).type_as(x)
        return mint.mul(output, self.weight)


class FusedRMSNorm(nn.Cell):
    r"""
    A RMSNorm fused kernel implementation.

    Args:
        dim (tuple): The shape of the input tensor
        eps (float): The epsilon value of the denominator. Default: ``1e-6``.
        params_dtype (dtype.Number): The param init type. Default: ``mstype.float32``.
        scale (float): scale number for weight initialization. Default: ``1.0``.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch, seq_length, hidden_size).

    Outputs:
        - Tensor with shape (batch, seq_length, hidden_size).
    """

    def __init__(self, dim, eps=1.e-6, params_dtype=mstype.float32, scale=1.0):
        super(FusedRMSNorm, self).__init__()
        self.eps = eps
        self.weight = Parameter(mint.ones((dim,), dtype=params_dtype) * scale, parallel_optimizer=False)

    def construct(self, x):
        """Forward of FusedRMSNorm."""
        output = ops.rms_norm(x, self.weight, self.eps)[0]
        return output


def get_norm(config, scale=1.0):
    r"""
    Get normalization layer.

    Args:
        config: The config of the model.

    Returns:
        callable, the normalization layer.
    """
    if config.normalization == "LayerNorm":
        return LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            params_dtype=config.params_dtype)
    if config.normalization == "RMSNorm":
        return RMSNorm(dim=config.hidden_size,
                       eps=config.layernorm_epsilon,
                       params_dtype=config.params_dtype,
                       scale=scale)
    if config.normalization == "FusedRMSNorm":
        return FusedRMSNorm(dim=config.hidden_size, eps=config.layernorm_epsilon, params_dtype=config.params_dtype,
                            scale=scale)

    raise Exception(f"unsupported norm type '{config.normalization}'.")
