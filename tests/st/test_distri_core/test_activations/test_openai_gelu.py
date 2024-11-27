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
"""Test Openai Gelu"""
import torch
import pytest
import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspeed_ms.legacy.model.utils import openai_gelu

np.random.seed(2024)

def megatron_gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))
def megatron_openai_gelu(x):
    return megatron_gelu_impl(x)

class TestOpenaiGelu:
    """A test class for openai gelu."""
    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=1)
    def test_openai_gelu(self):
        """
        Feature: test openai gelu.
        Description: compare relative error between megatron and mindspeed-ms
        Expectation: relative error smaller than 1e-4
        """
        value_shape = (8, 32, 256)
        np_value = np.random.randn(*value_shape).astype(np.float32)
        torch_tensor = torch.tensor(np_value, dtype=torch.float32)
        ms_tensor = ms.Tensor(np_value, dtype=mstype.float32)

        torch_output = megatron_openai_gelu(torch_tensor).numpy()
        ms_output = openai_gelu(ms_tensor).asnumpy()

        assert np.allclose(torch_output, ms_output, atol=1e-4), "openai gelu accuracy test fail !"
        print("openai gelu loss accuracy test passed.")
