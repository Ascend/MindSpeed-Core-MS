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
"""Test ParallelDP0GATHER"""
import os
import re
import pytest
import numpy as np


def grep_directory(directory_path, skip_num=0):
    """get loss from log file"""
    loss_dict = {}
    for file in os.listdir(directory_path):
        if "worker" in file:
            rank_id = file.split("_")[1].split(".")[0]
            log_file = os.path.join(directory_path, file)
            local_loss = []
            skipping = skip_num
            with open(log_file, 'r') as f:
                lines = f.readlines()
                re_pattern = r"lm loss: [0-9]*\.[0-9]*"
                for line in lines:
                    if "lm loss:" in line:
                        if skipping:
                            skipping = skipping - 1
                            continue
                        loss = re.findall(re_pattern, line)
                        local_loss.append(float(loss[0][9:]))
            loss_dict[rank_id] = local_loss
    return loss_dict


class TestParallelDP0GATHER:
    """A test class for testing DP0GATHER."""
    os.environ['HCCL_DETERMINISTIC'] = "true"
    os.environ['HCCL_IF_BASE_PORT'] = "30000"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=1)
    def test_dp0_gather_pynative_base(self):
        """
        Feature: test zero3 pynative
        Description: run pynative mode dp0 gather in bf16 mode
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_parallel_ckpt_dp0_gather.py --yaml-cfg test_dp0_gather.yaml --base True"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              f"--master_port=8199 " + \
              f"--log_dir=msrun_log_pynative_dp0_gather_base " + \
              f"--join=True " + \
              f"--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative_dp0_gather_ckpt/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative_dp0_gather_ckpt/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_dp0_gather_pynative(self):
        """
        Feature: test zero3 pynative
        Description: run pynative mode dp0 gather in bf16 mode
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_parallel_ckpt_dp0_gather.py --yaml-cfg test_dp0_gather.yaml"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              f"--master_port=8199 " + \
              f"--log_dir=msrun_log_pynative_dp0_gather_resume " + \
              f"--join=True " + \
              f"--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative_dp0_gather_golden/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative_dp0_gather_golden/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=4)
    def test_dp0_gather_check(self):
        """compare golden and actual"""
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        golden_loss = grep_directory(f"{sh_path}/msrun_log_pynative_dp0_gather_base", skip_num=3)
        actual_loss = grep_directory(f"{sh_path}/msrun_log_pynative_dp0_gather_resume")

        for key, value in actual_loss.items():
            for i in range(len(value)):
                assert np.allclose(value[i], golden_loss[key][i], atol=1e-3), f"rank {key} {i}th loss not equal"
