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


class TestMixtral:
    """A test class for testing Linear."""

    env_list = {
        # 'PYTHONPATH': f"/path/to/your/mindspore:{os.getenv('PYTHONPATH')}",
        }
    for k, v in env_list.items():
        os.environ[k] = v
    # os.system("ps -ef|grep pytest |grep -v grep|cut -c 9-16|xargs kill -9")

    def extract_loss_from_log(self, pynative_log_path: str):
        """extract loss from log_path"""

        assert os.path.exists(pynative_log_path), f"{pynative_log_path} did not exits"

        # check loss with golden loss
        pynative_loss = []
        with open(pynative_log_path, "r") as fp:
            for line in fp:
                if ", Loss: " in line:
                    line = line.strip().replace('[', '').replace(']', '').replace(',', '')
                    line = line.split(' ')
                    i = 0
                    for i, s in enumerate(line):
                        if "Loss:" in s:
                            print(f"{i}: {s} {line[i+1]}")
                            break
                    loss = float(line[i + 1])
                    pynative_loss.append(loss)
        pynative_loss = np.array(pynative_loss)
        return pynative_loss

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_mixtral_pynative_ep2tp2pp2(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_mixtral.py"
        device_num = 8
        postfix = "_ep2tp2pp2"

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
                    f"--master_port=8119 " + \
                    f"--log_dir=msrun_log_pynative{postfix} " + \
                    f"--join=True " + \
                    f"--cluster_time_out=300 " + \
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

        # check loss with golden loss
        pynative_log_path = f'msrun_log_pynative{postfix}/worker_4.log'
        pynative_loss = self.extract_loss_from_log(pynative_log_path)
        print(f"pynative_loss are:\n{pynative_loss}")

        golden_loss = [4.1485944, 4.1479816, 4.1473684, 4.146756, 4.146144,
                       4.145531, 4.144919, 4.144307, 4.143695, 4.1430836]

        golden_loss = np.array(golden_loss)
        print(f"golden_loss are:\n{golden_loss}")

        assert np.allclose(golden_loss, pynative_loss, atol=1.e-4, rtol=1e-4), \
               f"Expect relative error between pynative and golden loss below 1e-4,\n" + \
               f"but got pynative loss:\n{pynative_loss},\n" + \
               f"and golden loss:\n{golden_loss},\n" + \
               "please check your code."
