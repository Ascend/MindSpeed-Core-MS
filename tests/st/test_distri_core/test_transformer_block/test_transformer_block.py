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
"""Test TransformerBlock"""
import os
import shutil
import pytest
from tests.st.test_distri_core.utils import read_loss_from_log
import numpy as np


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestTransformerBlock:
    """A test class for TransformerBlock. """
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.run(order=1)
    def test_parallel_transformer_legacy(self):
        """
        Feature: test legacy parallel embedding.
        Description: run legacy parallel embedding to generate pynative loss
        Expectation: test success
        """
        scripts_name = "run_mindspore.sh"
        devices = "0,1,2,3"
        log_dir = "msrun_legacy_log"

        test_path = os.path.split(os.path.realpath(__file__))[0]
        data_dir = os.path.join(test_path, "data/parallel/random_data/")
        ckpt_dir = os.path.join(test_path, "data/parallel/random_ckpt/")
        folder_path = os.path.join(test_path, "data")
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print('Data removed')
        else:
            print('Data folder not exist')

        cmd = f"bash {test_path}/{scripts_name} "+\
                    f"{devices} "+\
                    f"test_legacy "+\
                    f"{data_dir} "+\
                    f"{ckpt_dir} "+\
                    f"{test_path}/{log_dir}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        assert ret == 0, f"msrun failed, please check {test_path}/{log_dir}/worker_*.log"

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.run(order=2)
    def test_transformer_block_mcore(self):
        """
        Feature: test mcore language model embedding.
        Description: run mcore language model embedding to generate pynative loss
        Expectation: test success
        """
        scripts_name = "run_mindspore.sh"
        devices = "0,1,2,3"
        log_dir = "msrun_mcore_log"

        test_path = os.path.split(os.path.realpath(__file__))[0]
        data_dir = os.path.join(test_path, "data/parallel/random_data/")
        ckpt_dir = os.path.join(test_path, "data/parallel/random_ckpt/")

        cmd = f"bash {test_path}/{scripts_name} "+\
                    f"{devices} "+\
                    f"test_mcore "+\
                    f"{data_dir} "+\
                    f"{ckpt_dir} "+\
                    f"{test_path}/{log_dir}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        assert ret == 0, f"msrun failed, please check {test_path}/{log_dir}/worker_*.log"

    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.run(order=3)
    def test_compare_loss(self):
        """
        Feature: test_compare_loss
        Description: compare relative error between pipeline loss and golden loss which with shared weight
        Expectation: relative error smaller than 1e-3
        """
        test_path = os.path.split(os.path.realpath(__file__))[0]

        legacy_log_path = os.path.join(test_path, "msrun_legacy_log/worker_3.log")
        legacy_loss = read_loss_from_log(legacy_log_path)
        assert legacy_loss, f"failed to read loss from {legacy_log_path}"

        mcore_log_path = os.path.join(test_path, "msrun_mcore_log/worker_3.log")
        mcore_loss = read_loss_from_log(mcore_log_path)
        assert mcore_loss, f"failed to read loss from {mcore_log_path}"

        print(f"legacy loss: {legacy_loss}", flush=True)
        print(f"mcore loss: {mcore_loss}", flush=True)

        assert np.allclose(legacy_loss, mcore_loss, atol=0.0, rtol=0.0), "loss accuracy test fail !"
        print("============== Custom interleaved staged pipeline net loss accuracy test pass !!! ==============")
