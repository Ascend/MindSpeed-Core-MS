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

"""Current scaling recipe."""

from mindspeed_ms.te.fp8 import set_amax
from mindspeed_ms.te.fp8.recipes.recipe import Recipe, RecipeConfig
from mindspeed_ms.te.fp8.scale_data import ScaleData


class CurrentScalingRecipe(Recipe):
    """Current scaling recipe."""

    def __init__(self, key, recipe_config: RecipeConfig, input_shape, weight_shape) -> None:
        super().__init__(key, recipe_config, input_shape, weight_shape)
        self.block_dim = None
        self.input_scale = ScaleData(recipe_config)
        self.weight_scale = ScaleData(recipe_config)

    def pre_compute(self, inputs, weight):
        if not hasattr(inputs, 'fp8_amax'):
            set_amax(inputs, self.block_dim)
        if not hasattr(weight, 'fp8_amax'):
            set_amax(weight, self.block_dim)
        return super().pre_compute(inputs, weight)

    def quantization(self, tensor, scale_data: ScaleData):
        if tensor is None:
            return tensor

        scale_data.append_amax(tensor.fp8_amax)
        scale_data.reduce_amax(self.amax_reduce_group)
        scale_data.update_scale()
        if not hasattr(tensor, 'is_fp8') or not tensor.is_fp8:  # if dtype is not fp8
            new_tensor = scale_data.quantization(tensor)  # cast to fp8
            setattr(new_tensor, 'fp8_amax', tensor.fp8_amax)
            setattr(new_tensor, 'is_fp8', True)
            tensor = new_tensor

        return tensor

    def dequantization(self, tensor):
        dtype = self.input_scale.ori_dtype
        return (tensor * (self.input_scale.scale_inv * self.weight_scale.scale_inv)).to(dtype=dtype)
