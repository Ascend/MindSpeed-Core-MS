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
"""Test Disable Grad Reduce"""
import os
import pytest
from tests.st.test_distri_core.utils import read_loss_from_log_list

class TestDisableGradReduce:
    """A test class for disable grad reduce."""
    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=1)
    def test_disable_grad_reduce_true(self):
        """
        Feature: test disable grad reduce.
        Description: set disable grad reduce true to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_disable_grad_reduce.py"
        yaml_file = "test_disable_grad_reduce.yaml"
        device_num = 2
        log_dir = "disable_grad_reduce_true_log"
        run_mode = "True"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --yaml-cfg {yaml_file} --run_mode {run_mode}"
        cmd = f"msrun --worker_num={device_num} " + \
                    f"--local_worker_num={device_num} " + \
                    f"--master_port=8142 " + \
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
    def test_disable_grad_reduce_false(self):
        """
        Feature: test disable grad reduce.
        Description: set disable grad reduce false to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_disable_grad_reduce.py"
        yaml_file = "test_disable_grad_reduce.yaml"
        device_num = 2
        log_dir = "disable_grad_reduce_false_log"
        run_mode = "False"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --yaml-cfg {yaml_file} --run_mode {run_mode}"
        cmd = f"msrun --worker_num={device_num} " + \
                    f"--local_worker_num={device_num} " + \
                    f"--master_port=8142 " + \
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
    @pytest.mark.env_onecard
    @pytest.mark.run(order=3)
    def test_compare_loss(self):
        """
        Feature: test_compare_loss
        Description: compare relative error between true loss and false loss
        Expectation: zero error
        """
        log_path_true = './disable_grad_reduce_true_log/worker_0.log'
        true_loss = read_loss_from_log_list(log_path_true)

        log_path_false = './disable_grad_reduce_false_log/worker_0.log'
        false_loss = read_loss_from_log_list(log_path_false)

        print(f"true loss: {true_loss}", flush=True)
        print(f"false loss: {false_loss}", flush=True)

        assert true_loss == false_loss, "disable grad reduce loss accuracy test fail!"
        print("disable grad reduce loss accuracy test passed.")
