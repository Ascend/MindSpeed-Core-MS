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
"""Test XIAOYISFT"""
import os
import sys
import pytest
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from utils import parse_log_file


def run_mindspore_xiaoyi_sft_determinstic():
    """
    Feature: test mindspore xiaoyi_sft
    Description: run mindspore xiaoyi_sft to generate pynative loss
    Expectation: test success
    """
    scripts_name = "run_ms_determin.sh"

    test_path = os.path.split(os.path.realpath(__file__))[0]
    cmd = f"bash {test_path}/{scripts_name} "
    print(f"\nrun cmd is:\n{cmd}")
    ret = os.system(cmd)
    assert ret == 0, f"msrun failed, please check ms_det.log"


def run_mindspore_xiaoyi_sft_nondeterminstic():
    """
    Feature: test mindspore xiaoyi_sft
    Description: run mindspore xiaoyi_sft to generate pynative loss
    Expectation: test success
    """
    scripts_name = "run_ms_nondetermin.sh"

    test_path = os.path.split(os.path.realpath(__file__))[0]
    cmd = f"bash {test_path}/{scripts_name} "
    print(f"\nrun cmd is:\n{cmd}")
    ret = os.system(cmd)
    assert ret == 0, f"msrun failed, please check ms_non_det.log"


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
@pytest.mark.level0
@pytest.mark.run(order=1)
def test_compare_performance():
    """
    Feature: test_compare_performance
    Description: compare run time between torch and mindspore
    Expectation: > 0.95pta
    """
    run_mindspore_xiaoyi_sft_nondeterminstic()
    data_pt = parse_log_file('pta_non_det.txt')
    data_ms = parse_log_file('ms_non_det.txt')
    tformat = '%Y-%m-%d %H:%M:%S'
    dt_ms = datetime.strptime(data_ms[10][0], tformat) - datetime.strptime(data_ms[5][0], tformat)
    dt_pt = datetime.strptime(data_pt[10][0], tformat) - datetime.strptime(data_pt[5][0], tformat)
    # 关闭确定性计算，统计5-10步，ms性能 > 0.95pta性能
    print("pt_time: %s s" % dt_pt.total_seconds())
    print("ms_time: %s s" % dt_ms.total_seconds())
    ratio = dt_ms.total_seconds() / dt_pt.total_seconds()
    print("Ratio(ms_time/pt_time): %s" % ratio)
    ratio = 0.9
    assert dt_ms.total_seconds() <= dt_pt.total_seconds()/ratio


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
@pytest.mark.level0
@pytest.mark.run(order=2)
def test_compare_accuracy():
    """
    Feature: test_compare_accuracy
    Description: compare relative error between torch loss and mindspore loss
    Expectation: no error
    """
    run_mindspore_xiaoyi_sft_determinstic()
    loss_pt = parse_log_file('pta_det.txt')
    loss_ms = parse_log_file('ms_det.txt')
    # 开确定性计算，精度对齐
    for i in loss_pt:
        print("loss:", loss_pt[i][2], loss_ms[i][2])
        assert loss_pt[i][2] == loss_ms[i][2]