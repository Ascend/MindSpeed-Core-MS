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

"""FP8 state manager."""

from mindspeed_ms.te.fp8.recipes.delayed_scaling_recipe import DelayedScalingRecipe


class FP8GlobalStateManager:
    """FP8 global state manager."""

    FP8_ENABLED = False
    FP8_CONFIG = None
    FP8_CALIBRATION = False
    FP8_DISTRIBUTED_GROUP = None
    IS_FIRST_FP8_MODULE = False
    FP8_GRAPH_CAPTURING = False
    FP8_AUTOCAST_DEPTH = 0
    FUSION_MATMUL = False

    @classmethod
    def fp8_autocast_enter(cls, enabled, fp8_config, calibrating, fp8_group, fp8_graph):
        """entry of FP8 autocast."""
        cls.FP8_ENABLED = enabled
        cls.FP8_CONFIG = fp8_config
        cls.FP8_CALIBRATION = calibrating
        cls.FP8_DISTRIBUTED_GROUP = fp8_group
        cls.FP8_GRAPH_CAPTURING = fp8_graph

        if cls.FP8_AUTOCAST_DEPTH == 0:
            cls.IS_FIRST_FP8_MODULE = True
        cls.FP8_AUTOCAST_DEPTH += 1

        if enabled and not cls.is_fp8_available():
            raise AssertionError('Device not support FP8.')

    @classmethod
    def fp8_autocast_exit(cls, enabled, fp8_graph):
        cls.FP8_AUTOCAST_DEPTH -= 1
        # Reduce only the non-FP8 weight modules here.
        # FP8 weight modules are reduced at the end of the optimizer
        # step after the weight amax is populated.
        # TODO torch.is_grad_enabled()
        if enabled and cls.FP8_AUTOCAST_DEPTH == 0 and not fp8_graph:
            for recipe in DelayedScalingRecipe.ALL_SCALING:
                recipe.finally_step()

    @classmethod
    def get_fp8_autocast_state(cls):
        """FP8 autocast state getter"""
        return (cls.FP8_ENABLED, cls.FP8_CONFIG, cls.FP8_CALIBRATION, cls.FP8_DISTRIBUTED_GROUP,
                cls.IS_FIRST_FP8_MODULE, cls.FP8_GRAPH_CAPTURING)

    @classmethod
    def set_fp8_autocast_state(cls, fp8_state):
        """FP8 autocast state setter"""
        (cls.FP8_ENABLED, cls.FP8_CONFIG, cls.FP8_CALIBRATION, cls.FP8_DISTRIBUTED_GROUP, cls.IS_FIRST_FP8_MODULE,
         cls.FP8_GRAPH_CAPTURING) = fp8_state

    @classmethod
    def is_fp8_available(cls) -> bool:
        return True
