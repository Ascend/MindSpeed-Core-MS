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
""" Test Resume_Training. """
import os
import pytest
import numpy as np
from tests.st.test_distri_core.utils import read_loss_from_log

MASTER_PORT = 8413


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestResumeTraining:
    """A test class for testing Linear."""
    env_list = {
        # 'ASCEND_RT_VISIBLE_DEVICES': '0,1,2,3',
        # 'ASCEND_RT_VISIBLE_DEVICES': '4,5,6,7',
        # 'ASCEND_GLOBAL_LOG_LEVEL': '3',
        # 'ASCEND_SLOG_PRINT_TO_STDOUT': '1',
        # 'ASCEND_GLOBAL_EVENT_ENABLE': '1',
        # 'GLOG_v': '0',
        # 'PYTHONPATH': f"/path/to/your/mindspore:{os.getenv('PYTHONPATH')}",
    }
    for k, v in env_list.items():
        os.environ[k] = v
    def extract_loss_from_log(self, pynative_log_path: str):
        '''extract loss from log_path'''
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
                    loss = float(line[i+1])
                    pynative_loss.append(loss)
        pynative_loss = np.array(pynative_loss)
        return pynative_loss

    @pytest.mark.level1
    @pytest.mark.run(order=0)
    def test_resume_training_pynative_ep1tp2pp2_step10(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        scripts_name = "run_resume_training.py"
        device_num = 4
        postfix = "_ep1tp2pp2_step10"
        rm_list = ["npy_pynative*", f"msrun_log_pynative{postfix}*", "kernel_meta*", f"output{postfix}"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --yaml-cfg=config_resume_training.yaml " + \
                      f"--crc_check " + \
                      f"--output_dir=output{postfix} " + \
                      f"--training_iters=10 " + \
                      f"--save_interval=5"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port={MASTER_PORT} "+\
                    f"--log_dir=msrun_log_pynative{postfix} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.run(order=1)
    def test_resume_training_pynative_ep1tp2pp2_step10_save_v1(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        scripts_name = "run_save_v1_ckpt.py"
        device_num = 4
        postfix = "_ep1tp2pp2_step10_v1"
        rm_list = ["npy_pynative*", f"msrun_log_pynative{postfix}*", "kernel_meta*", f"output{postfix}"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --yaml-cfg=config_resume_training.yaml " + \
                      f"--crc_check " + \
                      f"--output_dir=output{postfix} " + \
                      f"--training_iters=10 " + \
                      f"--resume_training " + \
                      f"--ckpt_step=5 " + \
                      f"--save_epoch_step=0_5 " + \
                      f"--load_checkpoint=output_ep1tp2pp2_step10 "
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port={MASTER_PORT+1} "+\
                    f"--log_dir=msrun_log_pynative{postfix} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.run(order=2)
    def test_resume_training_pynative_ep1tp2pp2_resume_from_step5(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "200"
        scripts_name = "run_resume_training.py"
        device_num = 4
        postfix = "_ep1tp2pp2_resume_from_step5"

        rm_list = ["npy_pynative*", f"msrun_log_pynative{postfix}*", "kernel_meta*", f"output{postfix}"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)
        scripts_cmd = f"{scripts_path} --yaml-cfg=config_resume_training.yaml " + \
                      f"--crc_check " + \
                      f"--output_dir=output{postfix} " + \
                      f"--training_iters=10 " + \
                      f"--resume_training " + \
                      f"--ckpt_step=5 " + \
                      f"--load_checkpoint=output_ep1tp2pp2_step10_v1 "
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port={MASTER_PORT+2} "+\
                    f"--log_dir=msrun_log_pynative{postfix} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

        # check loss with golden loss
        resume_log_path = f'msrun_log_pynative{postfix}/worker_2.log'
        resume_loss = read_loss_from_log(resume_log_path)

        golden_log_path = f'msrun_log_pynative_ep1tp2pp2_step10/worker_2.log'
        golden_loss = read_loss_from_log(golden_log_path)

        resume_loss = np.array(resume_loss)
        print(f"resume_loss are:\n{resume_loss}")
        golden_loss = np.array(golden_loss)[len(golden_loss)-len(resume_loss):]
        print(f"golden_loss are:\n{golden_loss}")

        assert np.allclose(golden_loss, resume_loss, atol=1.e-4, rtol=1e-4), \
               f"Expect relative error between resume and golden loss below 1e-4,\n" + \
               f"but got resume loss:\n{resume_loss},\n" + \
               f"and golden loss:\n{golden_loss},\n" + \
               "please check your code."

    @pytest.mark.level1
    @pytest.mark.run(order=3)
    def test_resume_training_pynative_dp2tp1pp2_step10(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        scripts_name = "run_resume_training.py"
        device_num = 4
        postfix = "_dp2tp1pp2_step10"
        rm_list = ["npy_pynative*", f"msrun_log_pynative{postfix}*", "kernel_meta*", f"output{postfix}"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} " +\
                      f"--yaml-cfg=./config_resume_training.yaml " +\
                      f"--crc_check " +\
                      f"--output_dir=output{postfix} " +\
                      f"--training_iters=10 " +\
                      f"--save_interval=6 " +\
                      f"--tp=1 " +\
                      f"--pp=2 " +\
                      f"--gbs=4 " +\
                      f"--epochs=2"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port={MASTER_PORT+3} "+\
                    f"--log_dir=msrun_log_pynative{postfix} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"run cmd is:\n{cmd}\n")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.run(order=4)
    def test_resume_training_pynative_dp2tp1pp2_step10_save_v1(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        scripts_name = "run_save_v1_ckpt.py"
        device_num = 4
        postfix = "_dp2tp1pp2_step10_v1"
        rm_list = ["npy_pynative*", f"msrun_log_pynative{postfix}*", "kernel_meta*", f"output{postfix}"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} " +\
                      f"--yaml-cfg=config_resume_training.yaml " +\
                      f"--crc_check " +\
                      f"--output_dir=output{postfix} " +\
                      f"--training_iters=10 " +\
                      f"--resume_training " +\
                      f"--ckpt_step=6 " + \
                      f"--save_epoch_step=1_1 " + \
                      f"--load_checkpoint=output_dp2tp1pp2_step10 " +\
                      f"--tp=1 " +\
                      f"--pp=2 " +\
                      f"--gbs=4 " +\
                      f"--epochs=2"
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port={MASTER_PORT+4} "+\
                    f"--log_dir=msrun_log_pynative{postfix} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"run cmd is:\n{cmd}\n")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.run(order=5)
    def test_resume_training_pynative_dp2tp1pp2_resume_from_step6(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        scripts_name = "run_resume_training.py"
        device_num = 4
        postfix = "_dp2tp1pp2_resume_from_step6"
        rm_list = ["npy_pynative*", f"msrun_log_pynative{postfix}*", "kernel_meta*", f"output{postfix}"]
        print("")
        for rm_path in rm_list:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} " +\
                      f"--yaml-cfg=config_resume_training.yaml " +\
                      f"--crc_check " +\
                      f"--output_dir=output{postfix} " +\
                      f"--resume_training " +\
                      f"--load_checkpoint=output_dp2tp1pp2_step10_v1 " +\
                      f"--training_iters=10 " +\
                      f"--save_interval=11 " +\
                      f"--tp=1 " +\
                      f"--pp=2 " +\
                      f"--gbs=4 " +\
                      f"--epochs=2 " +\
                      f"--ckpt_step=6 "
        cmd = f"msrun --worker_num={device_num} "+\
                    f"--local_worker_num={device_num} "+\
                    f"--master_port={MASTER_PORT+5} "+\
                    f"--log_dir=msrun_log_pynative{postfix} "+\
                    f"--join=True "+\
                    f"--cluster_time_out=300 "+\
                    f"{scripts_cmd}"
        print(f"run cmd is:\n{cmd}\n")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

        # check loss with golden loss
        resume_log_path = f'msrun_log_pynative{postfix}/worker_3.log'
        resume_loss = read_loss_from_log(resume_log_path)

        golden_log_path = f'msrun_log_pynative_dp2tp1pp2_step10/worker_3.log'
        golden_loss = read_loss_from_log(golden_log_path)

        resume_loss = np.array(resume_loss)
        print(f"resume_loss are:\n{resume_loss}")
        golden_loss = np.array(golden_loss)[len(golden_loss)-len(resume_loss):]
        print(f"golden_loss are:\n{golden_loss}")

        assert np.allclose(golden_loss, resume_loss, atol=1.e-4, rtol=1e-4), \
               f"Expect relative error between resume and golden loss below 1e-4,\n" + \
               f"but got resume loss:\n{resume_loss},\n" + \
               f"and golden loss:\n{golden_loss},\n" + \
               "please check your code."
