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
""" Test pretrian interface """
import os
import numpy as np
import pytest
from tests.st.test_distri_core.utils import read_loss_from_log


class TestPretrain:
    """A test class for language model."""
    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=1)
    def test_pretrain_single(self):
        """
        Feature: test pretrain interface with 1 devices.
        Description: train a language model using pretrain interface
        Expectation: test success
        """
        scripts_name = "run_pretrain_test.py"
        yaml_file = "pretrain_test_single.yaml"
        device_num = 1
        log_dir = "test_pretrain_single_log"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --yaml-cfg {yaml_file}"
        cmd = f"msrun --worker_num={device_num} " + \
                    f"--local_worker_num={device_num} " + \
                    f"--master_port=8133 " + \
                    f"--log_dir={log_dir} " + \
                    f"--join=True " + \
                    f"--cluster_time_out=300 " + \
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_compare_single_device_loss(self):
        """
        Feature: test_compare_single_device_loss
        Description: compare relative error between pretrain loss and golden loss
        Expectation: no error
        """
        log_path = './test_pretrain_single_log/worker_0.log'
        loss = read_loss_from_log(log_path)

        loss = np.array(loss, np.float32)
        golden_loss = np.array([4.851461410522461, 4.833165168762207, 4.796690940856934, \
                                4.8029866218566895, 4.732246398925781], np.float32)

        print(f"pretrain single device loss: {loss}", flush=True)
        print(f"golden loss: {golden_loss}", flush=True)
        assert np.allclose(loss, golden_loss, atol=0.0), "Pretrain process " \
                                                          "loss accuracy test fail !"
        print("============== Pretrain process loss accuracy test pass !!! ==============")

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=3)
    def test_pretrain_dp2tp2pp2vpp2sp_ddp_optimizer(self):
        """
        Feature: test pretrain interface with 8 devices.
        Description: train a language model using pretrain interface
        Expectation: test success
        """
        scripts_name = "run_pretrain_test.py"
        yaml_file = "pretrain_test_parallel.yaml"
        device_num = 8
        log_dir = "test_pretrain_parallel_log"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --yaml-cfg {yaml_file}"
        cmd = f"msrun --worker_num={device_num} " + \
                    f"--local_worker_num={device_num} " + \
                    f"--master_port=8132 " + \
                    f"--log_dir={log_dir} " + \
                    f"--join=True " + \
                    f"--cluster_time_out=300 " + \
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=4)
    def test_compare_multi_devices_loss(self):
        """
        Feature: test_compare_loss
        Description: compare relative error between pretrain loss and golden loss which with shared weight
        Expectation: no error
        """
        log_path = './test_pretrain_parallel_log/worker_7.log'
        loss = read_loss_from_log(log_path)

        loss = np.array(loss, np.float32)
        golden_loss = np.array([4.857857704162598, 4.852254867553711, 4.862484931945801, \
                                4.7300825119018555, 4.752025604248047], np.float32)

        print(f"pretrain multi devices loss: {loss}", flush=True)
        print(f"golden loss: {golden_loss}", flush=True)
        assert np.allclose(loss, golden_loss, atol=0.0), "Pretrain process " \
                                                          "loss accuracy test fail !"
        print("============== Pretrain process loss accuracy test pass !!! ==============")
