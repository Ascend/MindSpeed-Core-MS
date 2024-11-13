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
"""Test ParallelDDPZeRO3"""
import os
import re
import pytest
import numpy as np


def grep_directory(directory_path):
    """get loss from log file"""
    loss_dict = {}
    for file in os.listdir(directory_path):
        if "worker" in file:
            rank_id = file.split("_")[1].split(".")[0]
            log_file = os.path.join(directory_path, file)
            local_loss = []
            with open(log_file, 'r') as f:
                lines = f.readlines()
                re_pattern = r"Loss: *(\d+.?\d+e?\+?\-?\d*)"
                for line in lines:
                    if "Loss" in line:
                        loss = re.findall(re_pattern, line)
                        local_loss.append(float(loss[0]))
            loss_dict[rank_id] = local_loss
    return loss_dict


class TestParallelDDPZeRO3:
    """A test class for testing DDPZeRO3."""
    os.environ['HCCL_DETERMINISTIC'] = "true"
    os.environ['HCCL_IF_BASE_PORT'] = "30000"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=1)
    def test_ddp_zero3_pynative_make_ckpt(self):
        """
        Feature: test zero3 pynative
        Description: run pynative mode ddp zero3 in bf16 mode
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_parallel_ddp_zero3.py --yaml-cfg test_zero3.yaml --first True"
        device_num = 8

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              f"--master_port=8199 " + \
              f"--log_dir=msrun_log_pynative_ddp_zero3_ckpt " + \
              f"--join=True " + \
              f"--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative_ddp_zero3_ckpt/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative_ddp_zero3_ckpt/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_ddp_zero3_pynative_golden(self):
        """
        Feature: test zero3 pynative
        Description: run pynative mode ddp zero3 in bf16 mode
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_parallel_ddp_zero3.py --yaml-cfg test_zero3.yaml --golden True"
        device_num = 8

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              f"--master_port=8199 " + \
              f"--log_dir=msrun_log_pynative_ddp_zero3_golden " + \
              f"--join=True " + \
              f"--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative_ddp_zero3_golden/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative_ddp_zero3_golden/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=3)
    def test_ddp_zero3_pynative(self):
        """
        Feature: test zero3 pynative
        Description: run pynative mode ddp zero3 in bf16 mode
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_parallel_ddp_zero3.py --yaml-cfg test_zero3.yaml"
        device_num = 8

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              f"--master_port=8199 " + \
              f"--log_dir=msrun_log_pynative_ddp_zero3 " + \
              f"--join=True " + \
              f"--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative_ddp_zero3/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_pynative_ddp_zero3/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=4)
    def test_ddp_zero3_check(self):
        """compare golden and actual"""
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        golden_loss = grep_directory(f"{sh_path}/msrun_log_pynative_ddp_zero3_golden")
        actual_loss = grep_directory(f"{sh_path}/msrun_log_pynative_ddp_zero3")
        for key, value in actual_loss.items():
            for i in range(len(value)):
                assert np.allclose(value[i], golden_loss[key][i], atol=1e-3), f"rank {key} {i}th loss not equal"
