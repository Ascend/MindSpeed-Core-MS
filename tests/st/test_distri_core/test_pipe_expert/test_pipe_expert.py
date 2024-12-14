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
import numpy as np
from tests.st.test_distri_core.utils import read_loss_from_log


class TestPipeExpert:
    """A test class for testing Linear."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=1)
    def test_pipe_expert(self):
        """
        Feature: test pipe_expert under pynative
        Description: run pynative mode mixtral to test pipe_expert feature
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_pipe_expert.py"
        device_num = 8
        postfix = "_pipe_expert"

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --yaml-cfg config_pipe_expert.yaml"
        cmd = f"msrun --worker_num={device_num} " + \
                    f"--local_worker_num={device_num} " + \
                    f"--master_port=8119 " + \
                    f"--log_dir=msrun_log_pynative{postfix} " + \
                    f"--join=True " + \
                    f"--cluster_time_out=300 " + \
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

        # check loss
        pynative_log_path = f'msrun_log_pynative{postfix}/worker_0.log'
        pynative_loss = read_loss_from_log(pynative_log_path)
        print(f"pynative_loss are:\n{pynative_loss}")

        expect_loss = [4.193284, 4.192505, 4.1917257, 4.1909456, 4.1901655,
                       4.1893854, 4.1886053, 4.1878247, 4.1870437, 4.186263]

        expect_loss = np.array(expect_loss)
        print(f"expect_loss are:\n{expect_loss}")

        assert np.allclose(expect_loss, pynative_loss, atol=1.e-4, rtol=1e-4), \
               f"Expect relative error loss when use pipe_expert or not should be below 1e-4,\n" + \
               f"but got pynative loss:\n{pynative_loss},\n" + \
               f"and expect loss:\n{expect_loss},\n" + \
               "please check your code."


    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_pipe_expert_recompute(self):
        """
        Feature: test pipe_expert + recompute under pynative
        Description: run pynative mode mixtral to test pipe_expert + recompute feature
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_pipe_expert.py"
        device_num = 8
        postfix = "_pipe_expert_recompute"

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --yaml-cfg config_pipe_expert_recompute.yaml"
        cmd = f"msrun --worker_num={device_num} " + \
                    f"--local_worker_num={device_num} " + \
                    f"--master_port=8119 " + \
                    f"--log_dir=msrun_log_pynative{postfix} " + \
                    f"--join=True " + \
                    f"--cluster_time_out=300 " + \
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

        # check loss
        pynative_log_path = f'msrun_log_pynative{postfix}/worker_0.log'
        pynative_loss = read_loss_from_log(pynative_log_path)
        print(f"pynative_loss are:\n{pynative_loss}")

        expect_loss = [4.193284, 4.192505, 4.1917257, 4.1909456, 4.1901655,
                       4.1893854, 4.1886053, 4.1878247, 4.1870437, 4.186263]

        expect_loss = np.array(expect_loss)
        print(f"expect_loss are:\n{expect_loss}")

        assert np.allclose(expect_loss, pynative_loss, atol=1.e-4, rtol=1e-4), \
               f"Expect relative error loss when use pipe_expert + recompute or not should be below 1e-4,\n" + \
               f"but got pynative loss:\n{pynative_loss},\n" + \
               f"and expect loss:\n{expect_loss},\n" + \
               "please check your code."


    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=3)
    def test_pipe_expert_swap(self):
        """
        Feature: test pipe_expert + swap under pynative
        Description: run pynative mode mixtral to test pipe_expert + swap feature
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_pipe_expert.py"
        device_num = 8
        postfix = "_pipe_expert_swap"

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --yaml-cfg config_pipe_expert_swap.yaml"
        cmd = f"msrun --worker_num={device_num} " + \
                    f"--local_worker_num={device_num} " + \
                    f"--master_port=8119 " + \
                    f"--log_dir=msrun_log_pynative{postfix} " + \
                    f"--join=True " + \
                    f"--cluster_time_out=300 " + \
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

        # check loss
        pynative_log_path = f'msrun_log_pynative{postfix}/worker_0.log'
        pynative_loss = read_loss_from_log(pynative_log_path)
        print(f"pynative_loss are:\n{pynative_loss}")

        expect_loss = [4.193284, 4.192505, 4.1917257, 4.1909456, 4.1901655,
                       4.1893854, 4.1886053, 4.1878247, 4.1870437, 4.186263]

        expect_loss = np.array(expect_loss)
        print(f"expect_loss are:\n{expect_loss}")

        assert np.allclose(expect_loss, pynative_loss, atol=1.e-4, rtol=1e-4), \
               f"Expect relative error loss when use pipe_expert + swap or not should be below 1e-4,\n" + \
               f"but got pynative loss:\n{pynative_loss},\n" + \
               f"and expect loss:\n{expect_loss},\n" + \
               "please check your code."
