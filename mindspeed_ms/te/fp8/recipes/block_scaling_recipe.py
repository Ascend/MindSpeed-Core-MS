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

"""Block scaling recipe."""

import math

from mindspeed_ms.te.fp8 import set_amax
from mindspeed_ms.te.fp8.recipes.recipe import Recipe, RecipeConfig
from mindspeed_ms.te.fp8.scale_data import ScaleData


class BlockScalingRecipe(Recipe):
    """Block scaling recipe."""

    def __init__(self, key, recipe_config: RecipeConfig, input_shape, weight_shape) -> None:
        super().__init__(key, recipe_config, input_shape, weight_shape)
        if len(input_shape) != 2 or len(weight_shape) != 2:
            raise AssertionError('input shape or weight shape must be 2 dims.')
        if self.config.block_dim is None:
            self.config.block_dim = (128, 128)

        self.input_scale = ScaleData(recipe_config, self.get_scale_len(input_shape))
        self.weight_scale = ScaleData(recipe_config, self.get_scale_len(weight_shape))

    def get_scale_len(self, shape):
        """get scale len."""
        return [math.ceil(shape[0] / self.config.block_dim[0]),
                math.ceil(shape[1] / self.config.block_dim[1])]

    def pre_compute(self, inputs, weight):
        """pre_compute"""
        if not hasattr(inputs, 'fp8_amax'):
            set_amax(inputs, self.block_dim)
        if not hasattr(weight, 'fp8_amax'):
            set_amax(weight, self.block_dim)
        return super().pre_compute(inputs, weight)

    def quantization(self, tensor, scale_data: ScaleData):
        """quantization"""
        if tensor is None:
            return tensor

        scale_data.append_amax(tensor.fp8_amax)
        scale_data.update_scale()
        if not hasattr(tensor, 'is_fp8') or not tensor.is_fp8:  # if dtype is not fp8
            new_tensor = scale_data.quantization(tensor)  # cast to fp8
            setattr(new_tensor, 'fp8_amax', tensor.fp8_amax)
            setattr(new_tensor, 'is_fp8', True)
            tensor = new_tensor

        return tensor

    def dequantization(self, tensor):
        """dequantization"""
        raise RuntimeError('Block scaling has no dequantization method')

    def scale_matched(self, tensor, scale, block_dim):
        """scale_matched"""
        tensor_shape = tensor.shape
        scale_shape = scale.shape
        return math.ceil(tensor_shape[0] / block_dim[0]) == scale_shape[0] and \
               math.ceil(tensor_shape[1] / block_dim[1]) == scale_shape[1]

    def fp8_block_scaling_matmul(self, inputs, weight, transpose=(False, False)):
        """matmul for fp8 block scaling."""
        input_scale_inv = self.input_scale.scale_inv
        weight_scale_inv = self.weight_scale.scale_inv

        inputs = inputs.t() if transpose[0] else inputs
        weight = weight.t() if transpose[1] else weight
        input_scale_inv = input_scale_inv.t() if transpose[0] else input_scale_inv
        weight_scale_inv = weight_scale_inv.t() if transpose[1] else weight_scale_inv

        input_block_dim = (self.block_dim[1], self.block_dim[0]) if transpose[0] else self.block_dim
        weight_block_dim = (self.block_dim[1], self.block_dim[0]) if transpose[1] else self.block_dim

        if inputs.shape[1] is not weight.shape[0]:
            raise AssertionError('shape error.')

        if not self.scale_matched(inputs, input_scale_inv, input_block_dim) or \
           not self.scale_matched(weight, weight_scale_inv, weight_block_dim):
            raise AssertionError('inputs or weight shape not match scale. {}/{} != {} \t {}/{} != {}'
                                 .format(inputs.shape, input_block_dim, input_scale_inv.shape, weight.shape,
                                         weight_block_dim, weight_scale_inv.shape))

        # TODO torch.empty 对标
        output = inputs.new_empty((inputs.shape[0], weight.shape[1]))

        for i in range(inputs.shape[0]):
            for j in range(weight.shape[1]):
                line = inputs[i] * weight[:, j]

                input_single_scale_inv_i = i // input_block_dim[0]
                weight_single_scale_inv_j = j // weight_block_dim[1]

                for k in range(len(line)):
                    input_single_scale_inv_j = k // input_block_dim[1]
                    weight_single_scale_inv_i = k // weight_block_dim[0]

                    input_single_scale_inv = input_scale_inv[input_single_scale_inv_i, input_single_scale_inv_j]
                    weight_single_scale_inv = weight_scale_inv[weight_single_scale_inv_i, weight_single_scale_inv_j]
                    # TODO torch tensor.mul_
                    line[k] = line[k].mul(input_single_scale_inv * weight_single_scale_inv)

                output[i, j].copy_(line.sum())

        return output.to(dtype=self.input_scale.ori_dtype)
