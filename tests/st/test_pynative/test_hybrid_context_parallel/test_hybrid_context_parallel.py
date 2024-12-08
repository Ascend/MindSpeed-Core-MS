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
"""Test Hybrid Context Parallel"""
import os
import pytest

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestHybridAttnCP:
    """A test class for hybrid context parallel."""
    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_hybrid_cp(self):
        """
        Feature: test hybrid context parallel
        Description: run hybrid context parallel to generate logs
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_hybrid_context_parallel.py"
        device_num = 8
        log_dir = "msrun_log"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)
        scripts_cmd = f"{scripts_path} --yaml-cfg test_hybrid_context_parallel.yaml"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              f"--master_port=8118 " + \
              f"--join=True " + \
              f"--log_dir={log_dir} " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        # Check the success message in the logs
        os.system(f"grep -Ei 'ERROR' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"
