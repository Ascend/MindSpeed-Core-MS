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
"""Test ParallelAttention with SelfAttention"""
import os
import pytest
from tests.st.test_distri_core.utils import compare_all_data


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestSelfAttention:
    """
    Compare ParallelAttention of legacy (Transformer V1)
        with SelfAttention of mcore (Transformer V2)
    """

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.run(order=1)
    def test_parallel_attention_legacy(self):
        """
        Feature: test case for transformer v1, i.g. legacy
        Description: run mindspore ParallelAttention to generate loss
        Expectation: test success
        """
        scripts_name = "run_self_attention.sh"
        devices = "0,1,2,3,4,5,6,7"
        log_dir = "msrun_legacy_log"

        test_path = os.path.split(os.path.realpath(__file__))[0]
        data_dir = os.path.join(test_path, "data/parallel/random_data/")
        ckpt_dir = os.path.join(test_path, "data/parallel/random_ckpt/")
        output_dir = os.path.join(test_path, "data/parallel/output/")

        cmd = f"bash {test_path}/{scripts_name} " + \
              f"{devices} " + \
              f"test_legacy " + \
              f"{data_dir} " + \
              f"{ckpt_dir} " + \
              f"{output_dir} " + \
              f"{test_path}/{log_dir}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        assert ret == 0, f"msrun failed, please check {test_path}/{log_dir}/worker_*.log"

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.run(order=2)
    def test_self_attention_mcore(self):
        """
        Feature: test case for transformer v2, i.g. core
        Description: run mindspore parallel_attention to generate pynative loss
        Expectation: test success
        """
        scripts_name = "run_self_attention.sh"
        devices = "0,1,2,3,4,5,6,7"
        log_dir = "msrun_mcore_log"

        test_path = os.path.split(os.path.realpath(__file__))[0]
        data_dir = os.path.join(test_path, "data/parallel/random_data/")
        ckpt_dir = os.path.join(test_path, "data/parallel/random_ckpt/")
        output_dir = os.path.join(test_path, "data/parallel/output/")

        cmd = f"bash {test_path}/{scripts_name} " + \
              f"{devices} " + \
              f"test_mcore " + \
              f"{data_dir} " + \
              f"{ckpt_dir} " + \
              f"{output_dir} " + \
              f"{test_path}/{log_dir}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        assert ret == 0, f"msrun failed, please check {test_path}/{log_dir}/worker_*.log"

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.run(order=3)
    def test_compare_loss(self):
        """
        Feature: test_compare_loss
        Description: compare relative error between pipeline loss and golden loss which with shared weight
        Expectation: relative error is 0
        """
        test_path = os.path.split(os.path.realpath(__file__))[0]
        data_dir = os.path.join(test_path, "data/parallel/output")

        compare_types = ["_forward", "_backward"]
        weight_dict = {"out_proj": "linear_proj",
                       "qkv_proj": "linear_qkv"}

        compare_all_data(data_dir, compare_types=compare_types, atol=0.0, rtol=0.0, print_error_point=False,
                         weight_dict=weight_dict)
