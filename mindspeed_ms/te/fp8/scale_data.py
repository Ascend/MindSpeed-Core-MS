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

"""Scale Data for FP8."""

import math
from typing import List
import numpy as np

import mindspore
from mindspore import ops, nn, mint

from mindspeed_ms.te.fp8.constants import AMAX_COMPUTE_MAP
from mindspeed_ms.te.fp8.recipes.recipe import RecipeConfig


class ScaleData:
    """Scale Data for FP8."""
    def __init__(self, recipe_config: RecipeConfig, scale_shape: List[int] = None):
        if scale_shape is None:
            scale_shape = [1]
        self.config = recipe_config
        self.fp8_format = self.config.fp8_format.value
        self.fp8_max = self.config.fp8_format.value.max
        self.margin = self.config.margin
        self.amax_history_len = self.config.amax_history_len
        self.amax_history_current_len = 0
        self.scale_shape = scale_shape
        if self.config.amax_compute_algo not in AMAX_COMPUTE_MAP:
            raise AssertionError('Unsupported amax compute algo {}'.format(self.config.amax_compute_algo))
        self.amax_compute = AMAX_COMPUTE_MAP[self.config.amax_compute_algo]
        self.ori_dtype = None
        self.scale = mint.ones(self.scale_shape)
        self.scale_inv = 1 / self.scale

        self.amax_history = mint.zeros([self.amax_history_len] + self.scale_shape)
        self.amax = mint.zeros(self.scale_shape)

    def append_amax(self, amax):
        """append_amax"""
        if self.amax_history_current_len < self.amax_history_len:
            self.amax_history[self.amax_history_current_len, :].copy_(amax)
            self.amax_history_current_len += 1
        else:
            self.amax_history = self.amax_history.roll(-1, 1)
            self.amax_history[self.amax_history_len - 1, :].copy_(amax)

    def reduce_amax(self, group=None, async_op=False):
        """reduce_amax"""
        if group is not None and mindspore.mint.distributed.get_world_size(group) > 1:
            if self.amax_history_current_len < self.amax_history_len:
                amax = self.amax_history[self.amax_history_current_len - 1, :]
            else:
                amax = self.amax_history[self.amax_history_len - 1, :]
            handle = mint.distributed.all_reduce(amax, op="max", group=group, async_op=async_op)
            return handle
        return None

    def update_scale(self):
        """update_scale"""
        self.amax_compute(self.amax, self.amax_history)
        self.scale.copy_((self.fp8_max / self.amax) / (2 ** self.margin))
        self.scale_inv = 1 / self.scale

    def quantization(self, tensor: mindspore.Tensor):
        """quantization"""
        if self.scale.numel() == 1:
            quant_tensor = tensor * self.scale
        else:
            quant_tensor = self.apply_block_scale(tensor, self.scale)
        self.ori_dtype = quant_tensor.dtype
        return quant_tensor

    def dequantization(self, tensor: mindspore.Tensor):
        """dequantization"""
        if self.scale.numel() == 1:
            dequant_tensor = tensor * self.scale_inv
        else:
            dequant_tensor = self.apply_block_scale(tensor, self.scale_inv)
        return dequant_tensor.to(dtype=self.ori_dtype)

    def apply_block_scale(self, tensor, scale):
        """apply_block_scale"""
        output = tensor.clone()

        scale_len = self.scale.shape
        block_dim = self.config.block_dim

        assert math.ceil(output.shape[0] / block_dim[0]) == scale_len[0] and \
               math.ceil(output.shape[1] / block_dim[1]) == scale_len[1], \
            'shape not matched {}/{} != {}'.format(output.shape, block_dim, scale_len)

        for i in range(scale_len[0]):
            for j in range(scale_len[1]):
                i_start, i_end = i * block_dim[0], (i + 1) * block_dim[0]
                j_start, j_end = j * block_dim[1], (j + 1) * block_dim[1]
                output[i_start:i_end, j_start:j_end].copy_(output[i_start:i_end, j_start:j_end] * scale[i, j])

        return output


def finfo(dtype):
    """finfo"""
    return np.finfo(mindspore.dtype_to_nptype(dtype))


class LoraScaleData(ScaleData):
    """LoraScaleData"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ebits = self.config.fp8_format.value.ebits
        self.mbits = self.config.fp8_format.value.mbits

    def quantization(self, tensor: mindspore.Tensor):
        """quantization"""
        scaled_tensor = tensor * self.scale
        bias = int(2 ** (self.ebits - 1) - 1)
        num_clamp = mint.clamp(scaled_tensor, -self.fp8_max, self.fp8_max)
        # 计算non normalized的exp
        scales = mint.clamp(ops.stop_gradient(mint.floor(mint.log2(mint.abs(num_clamp)) + bias)), 1.0)
        scales = 2.0 ** (scales - self.mbits - bias)
        num = round_ste_func(num_clamp / scales) * scales

        return num

    def update_scale(self):
        """Default function to convert amax to scaling factor.
        Computing the scaling factor requires consideration of the following scenarios:
        1. amax == 0:
           No action is possible, set scale to  1.
        2. 0 < amax < tiny_amax
           The amax is too tiny that the scale becomes infinite in FP32.
           Set scale = FP32_max
        3. tiny_amax <= amax < FP32_max:
           Set scale = FP8_max (or scaled_max) / amax
        4. When amax == inf or amax == nan:
           No action is possible, set scale to 1.
        """
        sf = (self.fp8_max / self.amax) / (2 ** self.margin)
        sf = mint.where(self.amax > 0.0, sf, 1)
        sf = mint.where(mint.isfinite(self.amax), sf, 1)
        sf = mint.where(mint.isinf(sf), mint.full_like(sf, finfo(self.amax.dtype).max), sf)
        self.scale = sf
        self.scale_inv = 1 / self.scale


class RoundStraightThrough(nn.Cell):
    """RoundStraightThrough"""

    def construct(self, x):
        """forward"""
        return ops.round(x)

    def bprop(self, *args):
        """backward"""
        return args[-1]


round_ste_func = RoundStraightThrough()
