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
"""Test Pallel Mixtral"""
import os

import pytest


class TestMixtralWithContextParallel:
    """A test class for testing Linear."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_mixtral_pynative_cp2tp2pp2(self):
        """
        Feature: test mixtral with context parallel pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_mixtral_with_context_parallel.py"
        device_num = 8
        postfix = "_cp2tp2pp2"

        rm_file_patterns = ["npy_pynative*", f"msrun_log_pynative{postfix}*", "kernel_meta*"]
        print("")
        for rm_path in rm_file_patterns:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --yaml-cfg config_mixtral_small.yaml"
        cmd = f"msrun --worker_num={device_num} " + \
                    f"--local_worker_num={device_num} " + \
                    f"--master_port=8120 " + \
                    f"--log_dir=msrun_log_pynative{postfix} " + \
                    f"--join=True " + \
                    f"--cluster_time_out=300 " + \
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"
