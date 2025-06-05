# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Module for testing the transfer tool.
"""
import os
import shutil
import sys
import tempfile
import pytest

# 添加项目根目录到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from tools.transfer import (
    getfiles,
    convert_general_rules,
    convert_special_rules,
    convert_special_rules_by_line,
    convert_package,
)


def setup_and_teardown():
    """
    Feature: Setup and Teardown Environment
    Description: Prepare temporary directories and files for testing.
    Expectation: Successfully create and clean up temporary directories after each test.
    """
    # 创建临时目录用于测试
    origin_path = tempfile.mkdtemp()
    save_path = tempfile.mkdtemp()
    test_files = {
        "convert_ckpt.py": """
if __name__ == '__main__':
    import torch
    torch.cuda.current_device()
""",
        "mindspeed_llm/tasks/megatron_adaptor.py": """
optimizer_config_init_wrapper
from mindspeed.optimizer.distrib_optimizer import reuse_fp32_param_distrib_optimizer_init_wrapper
MegatronAdaptation.register(
    'megatron.core.parallel_state.get_nccl_options', 
    get_nccl_options_wrapper
)
""",
        "mindspeed_llm/core/datasets/blended_megatron_dataset_builder.py": """
from ..parallel_state import get_pipeline_model_parallel_node_info
logger = logging.getLogger(__name__)
gpus_per_node = torch.cuda.device_count()
current_rank = torch.cuda.current_device()
if args.tensor_model_parallel_size > gpus_per_node:
    return mpu.get_tensor_model_parallel_rank() == 0
""",
        "mindspeed_llm/core/models/common/embeddings/rotary_pos_embedding.py": """
for freq in freqs:
    wavelen = 2 * math.pi / freq
inv_freq_mask = 1.0 - YarnRotaryPositionEmbedding.yarn_linear_ramp_mask(
    low, high, dim // 2
).to(device=freqs.device, dtype=torch.float32)
if self.inv_freq.device.type == 'cpu':
    self.inv_freq = self.inv_freq.to(
        device=torch.cuda.current_device()
    )
t = npu_rotary_position_embedding(t.contiguous(), cos_, sin_, mode).to(t.dtype)
""",
        "mindspeed_llm/tasks/checkpoint/models.py": """
def _func_generator_set_weight(value):
    def func(self, **kwargs):
        return _get_dst_obj(self, value, **kwargs).weight.data.copy_(
            kwargs.get('data')
        )
    return func


def _func_generator_set_bias(value):
    def func(self, **kwargs):
        return _get_dst_obj(self, value, **kwargs).bias.data.copy_(
            kwargs.get('data')
        )
    return func


self.module = [AutoModelForCausalLM.from_pretrained(
    load_dir, device_map=device_map, trust_remote_code=trust_remote_code,
    local_files_only=True
)]
""",
        "mindspeed_llm/tasks/models/transformer/multi_head_latent_attention.py": """
output = torch.matmul(input_, self.weight.t())
"""
    }

    # 创建测试文件，并添加初始内容
    for file, content in test_files.items():
        file_path = os.path.join(origin_path, file)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(content)

    yield origin_path, save_path

    # 清理临时目录
    shutil.rmtree(origin_path)
    shutil.rmtree(save_path)


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.level0
@pytest.mark.run(order=1)
def test_getfiles(setup_and_teardown_fixture):
    """
    Feature: getfiles function
    Description: Test that all files in a directory are correctly retrieved.
    Expectation: All created files should be found and their paths should contain the origin path.
    """
    origin_path, _ = setup_and_teardown_fixture
    files = getfiles(origin_path)
    assert len(files) == len(setup_and_teardown_fixture[2])  # 确保所有文件都被获取
    for file in files:
        assert origin_path in file


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.level0
@pytest.mark.run(order=2)
def test_convert_general_rules(setup_and_teardown_fixture):
    """
    Feature: convert_general_rules function
    Description: Test that files are copied and renamed according to general rules.
    Expectation: Converted files should exist at the expected locations.
    """
    origin_path, save_path = setup_and_teardown_fixture
    convert_general_rules(origin_path, save_path)
    for file in setup_and_teardown_fixture[2].keys():
        converted_file_path = os.path.join(save_path, file)
        assert os.path.exists(converted_file_path)


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.level0
@pytest.mark.run(order=3)
def test_convert_special_rules(setup_and_teardown_fixture):
    """
    Feature: convert_special_rules function
    Description: Test that special regex replacements are applied on specific files.
    Expectation: Torch calls should be replaced by MindSpore-compatible ones.
    """
    origin_path, save_path = setup_and_teardown_fixture
    convert_special_rules(origin_path, save_path, package_name="megatron")
    for file, content in setup_and_teardown_fixture[2].items():
        converted_file_path = os.path.join(save_path, file)
        with open(converted_file_path, 'r') as f:
            new_content = f.read()
            if "torch.cuda.current_device()" in content:
                assert "get_local_rank()" in new_content
            if "from mindspeed.optimizer.distrib_optimizer import" in content:
                assert "from mindspeed.mindspore.optimizer.distrib_optimizer import" in new_content


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.level0
@pytest.mark.run(order=4)
def test_convert_special_rules_by_line(setup_and_teardown_fixture):
    """
    Feature: convert_special_rules_by_line function
    Description: Test that line-level code replacement is applied correctly.
    Expectation: Torch calls should be replaced by MindSpore-compatible ones.
    """
    origin_path, save_path = setup_and_teardown_fixture
    convert_special_rules_by_line(origin_path, save_path, package_name="megatron")
    for file, content in setup_and_teardown_fixture[2].items():
        converted_file_path = os.path.join(save_path, file)
        with open(converted_file_path, 'r') as f:
            new_content = f.read()
            if "torch.cuda.current_device()" in content:
                assert "get_local_rank()" in new_content
            if "from mindspeed.optimizer.distrib_optimizer import" in content:
                assert "from mindspeed.mindspore.optimizer.distrib_optimizer import" in new_content


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.level0
@pytest.mark.run(order=5)
def test_convert_package(setup_and_teardown_fixture):
    """
    Feature: convert_package function
    Description: Test full package conversion including both general and special rules.
    Expectation: Files should be created and transformed correctly.
    """
    origin_path, save_path = setup_and_teardown_fixture
    convert_package(origin_path, save_path, package_name="megatron")
    for file, content in setup_and_teardown_fixture[2].items():
        converted_file_path = os.path.join(save_path, file)
        assert os.path.exists(converted_file_path)
        with open(converted_file_path, 'r') as f:
            new_content = f.read()
            if "torch.cuda.current_device()" in content:
                assert "get_local_rank()" in new_content
            if "from mindspeed.optimizer.distrib_optimizer import" in content:
                assert "from mindspeed.mindspore.optimizer.distrib_optimizer import" in new_content

