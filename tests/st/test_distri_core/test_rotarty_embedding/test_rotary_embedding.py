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
"""Test RotaryEmbedding"""
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


class TestRotaryEmbedding:
    """A test class for RotaryEmbedding"""
    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.run(order=1)
    def test_rotary_embedding_loss(self):
        """
        Feature: test rotary embedding.
        Description: run pynative mode language model to generate pynative loss
        Expectation: test success
        """
        log_dir = "rotary_embedding_log"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        cmd = "bash run_rotary_embedding.sh"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.run(order=2)
    def test_compare_loss(self):
        """
        Feature: test_compare_loss
        Description: compare relative error between test loss and golden loss
        Expectation: relative error smaller than 1e-3
        """
        log_path = './rotary_embedding_log/worker_0.log'
        loss = read_loss_from_file(log_path)

        loss = np.array(loss, np.float32)
        golden_loss = np.array([5.7579202], np.float32)

        print(f"rotary embedding loss: {loss}", flush=True)
        print(f"golden loss: {golden_loss}", flush=True)
        assert np.allclose(loss, golden_loss, atol=1e-3), "RotaryEmbedding loss accuracy test fail !"
        print("RotaryEmbedding loss accuracy test passed.")
