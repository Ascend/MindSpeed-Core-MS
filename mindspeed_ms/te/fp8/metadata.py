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

"""Metadata for FP8"""

from dataclasses import dataclass
from typing import Type, Tuple

from mindspeed_ms.te.fp8.recipes.recipe import Recipe, RecipeConfig
from mindspeed_ms.te.fp8.state_manager import FP8GlobalStateManager


@dataclass
class FP8Config:
    default: Tuple[Type[Recipe], RecipeConfig] = None


class FP8Metadata:
    """Metadata for FP8"""
    def __init__(self, keys):
        for key in keys:
            setattr(self, key, None)

    @property
    def fp8_config(self):
        return FP8GlobalStateManager.FP8_CONFIG

    @property
    def fp8_enable(self):
        return FP8GlobalStateManager.FP8_ENABLED

    @property
    def fusion_matmul(self):
        return FP8GlobalStateManager.FUSION_MATMUL

    @staticmethod
    def create_recipe(key, config: Tuple[Type[Recipe], RecipeConfig], input_shape, weight_shape):
        recipe, recipe_config = config
        return recipe(key, recipe_config, input_shape, weight_shape)

    def init_recipes_if_necessarily(self, key, input_shape=None, weight_shape=None):
        if getattr(self, key) is None:
            fp8_config = self.get_fp8_config(key)
            recipe = self.create_recipe(key, fp8_config, input_shape, weight_shape)
            setattr(self, key, recipe)

    def get_fp8_config(self, key):
        fp8_config = self.fp8_config.default
        if hasattr(self.fp8_config, key) and getattr(self.fp8_config, key) is not None:
            fp8_config = getattr(self.fp8_config, key)
        return fp8_config

    def pre_communication(self, key, inputs, weight):
        if not self.fp8_enable or not self.get_fp8_config(key)[1].fp8_comm:
            return inputs, weight
        self.init_recipes_if_necessarily(key, inputs.shape, weight.shape)
        recipe = getattr(self, key)
        return (inputs, weight) if not recipe.fp8_comm else recipe.pre_communication(inputs, weight)

    def pre_compute(self, key, inputs, weight):
        if not self.fp8_enable or self.get_fp8_config(key)[1].fp8_comm:
            return inputs, weight
        self.init_recipes_if_necessarily(key, inputs.shape, weight.shape)
        recipe = getattr(self, key)
        return (inputs, weight) if recipe.fp8_comm else recipe.pre_compute(inputs, weight)
