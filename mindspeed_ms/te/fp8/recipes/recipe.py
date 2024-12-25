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

"""Recipe and recipe config."""

from abc import abstractmethod
from dataclasses import dataclass

from mindspeed_ms.te.fp8.constants import Format


@dataclass
class RecipeConfig:
    margin: int = 0
    fp8_format: Format = Format.E4M3
    amax_history_len: int = 10
    amax_compute_algo: str = 'max'
    amax_reduce_group: str = None
    block_dim: tuple = None
    fp8_comm: bool = False


class Recipe:
    """Recipe."""

    def __init__(self, key, recipe_config: RecipeConfig, input_shape, weight_shape):
        self.key = key
        self.config = recipe_config
        self.input_scale = None
        self.weight_scale = None
        self.input_shape = input_shape
        self.weight_scale = weight_shape

    def __getattr__(self, item):
        return getattr(self.config, str(item))

    def pre_communication(self, inputs, weight):
        inputs = self.quantization(inputs, self.input_scale)
        weight = self.quantization(weight, self.weight_scale)
        return inputs, weight

    def pre_compute(self, inputs, weight):
        inputs = self.quantization(inputs, self.input_scale)
        weight = self.quantization(weight, self.weight_scale)
        return inputs, weight

    @abstractmethod
    def quantization(self, tensor, scale_data):
        pass

    @abstractmethod
    def dequantization(self, tensor):
        pass
