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
"""Test FP8 Column Parallel Linear"""

import mindspore as ms
from mindspore import mint, ops, nn, grad
from mindspore.communication import init

from mindspeed_ms.training.arguments import parse_args
from mindspeed_ms.training.arguments import core_transformer_config_from_args
from mindspeed_ms.training.global_vars import set_args
from mindspeed_ms.training.initialize import _set_random_seed
from mindspeed_ms.core.parallel_state import initialize_model_parallel

from mindspeed_ms.te.fp8.fp8 import fp8_autocast
from mindspeed_ms.te.fp8.metadata import FP8Config
from mindspeed_ms.te.fp8.recipes.block_scaling_recipe import BlockScalingRecipe
from mindspeed_ms.te.fp8.recipes.recipe import RecipeConfig
from mindspeed_ms.te.fp8.recipes.current_scaling_recipe import CurrentScalingRecipe
from mindspeed_ms.te.fp8.recipes.delayed_scaling_recipe import DelayedScalingRecipe
from mindspeed_ms.te.module.linear import TEColumnParallelLinear

class ColumnModel(nn.Cell):
    """ColumnModel"""
    def __init__(self, config, input_size, output_size):
        super().__init__()

        word_size = mint.distributed.get_world_size()
        assert output_size % word_size == 0, 'output size can not div with world size.'
        self.linear = TEColumnParallelLinear(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False
        )

    def construct(self, x):
        """forward"""
        return self.linear(x)

class TestFP8Model:
    """TestFP8Model"""
    world_size = 2

    def test_fp8_column_model(self, recipe):
        """test column model"""
        iteration_num = 10
        input_size = 16
        output_size = 16

        args, _ = parse_args(None, True)
        args.params_dtype = ms.bfloat16
        args.num_attention_heads = 16
        args.hidden_size = 1024
        args.num_layers = 2
        args.tensor_model_parallel_size = 2
        args.sequence_parallel = True
        args.gradient_accumulation_fusion = False
        set_args(args)
        config = core_transformer_config_from_args(args)
        config.perform_initialization = False
        init()
        initialize_model_parallel(self.world_size, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        model_ori = ColumnModel(config, input_size, output_size)
        model_fp8 = ColumnModel(config, input_size, output_size)

        for r in recipe:
            fp8_config = FP8Config(default=(r, RecipeConfig(block_dim=(2, 2))))
            fp8_config.perform_initialization = False
            for i in range(iteration_num):
                print('start test {}'.format(i))
                inputs = ops.randn([4, input_size], dtype=ms.bfloat16)
                inputs.requires_grad = True
                # baseline
                output_ori = model_ori(inputs)[0]
                weights_ori = model_ori.trainable_params()
                grad_fn = grad(model_ori, grad_position=0, weights=weights_ori, has_aux=False)
                inputs_gradient_ori, params_gradient_ori = grad_fn(inputs)

                fp8_context = fp8_autocast(enabled=True, fp8_config=fp8_config)
                with fp8_context:
                    weights_fp8 = model_fp8.trainable_params()
                    output_fp8 = model_fp8(inputs)[0]
                    grad_fn = grad(model_fp8, grad_position=0, weights=weights_fp8, has_aux=False)
                    inputs_gradient_fp8, params_gradient_fp8 = grad_fn(inputs)
                ori_dtype = inputs_gradient_ori[0].dtype
                assert mint.allclose(output_ori, output_fp8.to(output_ori.dtype), atol=0.01, rtol=0.01)
                assert mint.allclose(inputs_gradient_ori[0], inputs_gradient_fp8[0].to(ori_dtype), atol=0.01, rtol=0.01)
                assert mint.allclose(params_gradient_ori[0], params_gradient_fp8[0].to(ori_dtype), atol=0.01, rtol=0.01)

recipes = [
    CurrentScalingRecipe,
    DelayedScalingRecipe,
    BlockScalingRecipe
]
ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, pynative_synchronize=True)
fp8_test = TestFP8Model()
fp8_test.test_fp8_column_model(recipes)
