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
"""Test 2D TP language model"""

import os
import re
import numpy as np
import pytest


def read_loss_from_file(file_path):
    """ reading loss from log """
    losses = []
    with open(file_path, 'r') as file:
        for line in file:
            loss_str = re.search(r'Loss: (\d+\.\d+)', line)
            if loss_str:
                loss_value = float(loss_str.group(1))
                losses.append(loss_value)
    return losses

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestLanguageModel:
    """A test class for language model."""
    @pytest.mark.run(order=1)
    def test_2d_language_model_loss(self):
        """Test 2D TP language model loss."""
        os.environ['HCCL_BUFFSIZE'] = "200"
        # os.environ['ASCEND_RT_VISIBLE_DEVICES'] = "4,5,6,7"
        scripts_name = "run_language_model_2d.py"
        yaml_file = "test_language_model_2d.yaml"
        device_num = 4
        log_dir = "msrun_log_2d"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --yaml-cfg {yaml_file}"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              f"--master_port=8983 " + \
              f"--log_dir={log_dir} " + \
              f"--join=True " + \
              f"--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"

    @pytest.mark.run(order=2)
    def test_language_model_loss(self):
        """Test 1D TP language model loss."""
        os.environ['HCCL_BUFFSIZE'] = "200"
        # os.environ['ASCEND_RT_VISIBLE_DEVICES'] = "4,5,6,7"
        scripts_name = "run_language_model_2d.py"
        yaml_file = "test_language_model.yaml"
        device_num = 4
        log_dir = "msrun_log"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --yaml-cfg {yaml_file}"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              f"--master_port=8983 " + \
              f"--log_dir={log_dir} " + \
              f"--join=True " + \
              f"--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"

    @pytest.mark.run(order=3)
    def test_compare_loss(self):
        """Test loss"""
        log_path = './msrun_log/worker_0.log'
        log_path_2d = './msrun_log_2d/worker_0.log'
        language_model_loss = read_loss_from_file(log_path)
        language_model_loss_2d = read_loss_from_file(log_path_2d)
        language_model_loss = np.array(language_model_loss, np.float32)
        language_model_loss_2d = np.array(language_model_loss_2d, np.float32)
        print('tp_1d', language_model_loss, '\ntp2d', language_model_loss_2d)
        assert np.allclose(language_model_loss, language_model_loss_2d, rtol=1e-4),\
            "Language model loss accuracy test fail !"
        print("Language model loss accuracy test passed.")
