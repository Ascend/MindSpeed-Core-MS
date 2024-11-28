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
""" Test No Save/Load RNG """
import os
import yaml
import pytest
import numpy as np

MASTER_PORT = 8160
PRETRAIN_POSTFIX = "_ep1tp2pp2_save_no_rng"
FINETUNE_POSTFIX1 = "_ep1tp2pp2_load_no_rng"
FINETUNE_POSTFIX2 = "_ep1tp2pp2_load_rng"
BASE_YAML = "config_mixtral_small.yaml"


class TestNoRng:
    """A test class for testing --no-save-rng / --no-load-rng """

    env_list = {
        # 'PYTHONPATH': f"/path/to/your/mindspore:{os.getenv('PYTHONPATH')}",
        "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3",
        "ASCEND_LAUNCH_BLOCKING": "1",
        "HCCL_BUFFSIZE": "200",
        "HCCL_DETERMINISTIC": "true"
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
        return np.asarray(pynative_loss) if pynative_loss else []

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=0)
    def test_mixtral_pynative_no_save_rng(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        scripts_name = "save_checkpoint.py"
        device_num = 4
        postfix = PRETRAIN_POSTFIX

        rm_file_patterns = [
            "npy_pynative*",
            "kernel_meta*",
            f"msrun_log_pynative{postfix}*",
            f"output{postfix}"
        ]
        print("")
        for rm_path in rm_file_patterns:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]

        # Prepare YAML
        base_yaml = os.path.join(sh_path, BASE_YAML)
        test_yaml = os.path.join(sh_path, BASE_YAML.replace(
            ".yaml", f"{postfix}.yaml"))
        with open(base_yaml, "r") as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
            conf["save"] = f"output{postfix}"
            conf["no_save_rng"] = True
            conf["train_iters"] = 1
        with open(test_yaml, "w") as f:
            yaml.dump(conf, f)

        # Save Checkpoint
        scripts_path = os.path.join(sh_path, scripts_name)
        scripts_cmd = f"{scripts_path} --yaml-cfg {test_yaml}"
        cmd = f"msrun --worker_num={device_num} " + \
                    f"--local_worker_num={device_num} " + \
                    f"--master_port={MASTER_PORT} " + \
                    f"--log_dir=msrun_log_pynative{postfix} " + \
                    f"--join=True " + \
                    f"--cluster_time_out=300 " + \
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=1)
    def test_mixtral_pynative_no_load_rng(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        scripts_name = "pretrain_mixtral.py"
        device_num = 4
        postfix = FINETUNE_POSTFIX1

        rm_file_patterns = [
            "npy_pynative*",
            "kernel_meta*",
            f"msrun_log_pynative{postfix}*",
            f"output{postfix}"
        ]
        print("")
        for rm_path in rm_file_patterns:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]

        # Prepare YAML
        base_yaml = os.path.join(sh_path, BASE_YAML)
        test_yaml = os.path.join(sh_path, BASE_YAML.replace(
            ".yaml", f"{postfix}.yaml"))
        with open(base_yaml, "r") as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
            conf["save"] = f"output{postfix}"
            conf["load"] = f"output{PRETRAIN_POSTFIX}"
            conf["resume_training"] = True
            conf["no_load_rng"] = True
            conf["train_iters"] = 1
        with open(test_yaml, "w") as f:
            yaml.dump(conf, f)

        # Finetune
        scripts_path = os.path.join(sh_path, scripts_name)
        scripts_cmd = f"{scripts_path} --yaml-cfg {test_yaml}"
        cmd = f"msrun --worker_num={device_num} " + \
                    f"--local_worker_num={device_num} " + \
                    f"--master_port={MASTER_PORT+1} " + \
                    f"--log_dir=msrun_log_pynative{postfix} " + \
                    f"--join=True " + \
                    f"--cluster_time_out=300 " + \
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_mixtral_pynative_load_rng(self):
        """
        Feature: test mixtral pynative
        Description: run pynative mode mixtral to generate pynative loss
        Expectation: test success
        """
        scripts_name = "pretrain_mixtral.py"
        device_num = 4
        postfix = FINETUNE_POSTFIX2

        rm_file_patterns = [
            "npy_pynative*",
            "kernel_meta*",
            f"msrun_log_pynative{postfix}*",
            f"output{postfix}"
        ]
        print("")
        for rm_path in rm_file_patterns:
            rm_path = os.path.join(os.getcwd(), rm_path)
            print(f"removing {rm_path}")
            os.system(f"rm -rf {rm_path}")

        sh_path = os.path.split(os.path.realpath(__file__))[0]

        # Prepare YAML
        base_yaml = os.path.join(sh_path, BASE_YAML)
        test_yaml = os.path.join(sh_path, BASE_YAML.replace(
            ".yaml", f"{postfix}.yaml"))
        with open(base_yaml, "r") as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
            conf["save"] = f"output{postfix}"
            conf["load"] = f"output{PRETRAIN_POSTFIX}"
            conf["resume_training"] = True
            conf["no_load_rng"] = False
            conf["train_iters"] = 1
        with open(test_yaml, "w") as f:
            yaml.dump(conf, f)

        # Finetune
        scripts_path = os.path.join(sh_path, scripts_name)
        scripts_cmd = f"{scripts_path} --yaml-cfg {test_yaml}"
        cmd = f"msrun --worker_num={device_num} " + \
                    f"--local_worker_num={device_num} " + \
                    f"--master_port={MASTER_PORT+2} " + \
                    f"--log_dir=msrun_log_pynative{postfix} " + \
                    f"--join=True " + \
                    f"--cluster_time_out=300 " + \
                    f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_pynative{postfix}/worker_0.log -C 3")
        assert ret == 256, f"msrun failed, please check msrun_log_pynative{postfix}/worker_*.log"
