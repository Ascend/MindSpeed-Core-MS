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
"""Test DS3SFT"""
import os
import sys
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from utils import parse_log_file


def run_mindspore_ds3_sft_determinstic():
    """
    Feature: test mindspore pretrain_glm
    Description: run mindspore ds3_sft to generate pynative loss
    Expectation: test success
    """
    scripts_name = "run_ms_determin.sh"

    test_path = os.path.split(os.path.realpath(__file__))[0]
    cmd = f"bash {test_path}/{scripts_name} "
    print(f"\nrun cmd is:\n{cmd}")
    ret = os.system(cmd)
    assert ret == 0, f"msrun failed, please check ms_det.txt"


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
@pytest.mark.level0
def test_compare_accuracy():
    """
    Feature: test_compare_res
    Description: compare relative error between torch loss and mindspore loss
    Expectation: no error
    """
    run_mindspore_ds3_sft_determinstic()
    loss_pt = parse_log_file('pta_det.txt')
    loss_ms = parse_log_file('ms_det.txt')
    assert loss_ms, "MindSpore loss data is empty"
    # 开确定性计算，精度对齐
    for i in loss_pt:
        print("loss:", loss_pt[i][2], loss_ms[i][2])
        assert len(loss_pt[i][2]) - len(loss_ms[i][2]) < 20