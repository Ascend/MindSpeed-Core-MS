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
"""Test RMSNorm2D"""

import os
import pytest

#
class TestRMSNorm2D:
    """A test class for language model."""
    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_rms_norm2d(self):
        """Test RMSNorm2D."""
        os.environ['HCCL_BUFFSIZE'] = "1"
        # os.environ['ASCEND_RT_VISIBLE_DEVICES'] = "4,5,6,7"
        scripts_name = "run_rms_norm2d.py"
        yaml_file = "test_tp_2d.yaml"
        device_num = 4
        log_dir = "rms_norm2d_log"
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
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"
