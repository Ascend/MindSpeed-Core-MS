# Copyright 2025 Huawei Technologies Co., Ltd
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
import os
import stat
import sys
import unittest
from pathlib import Path
import xmlrunner


# =============================
# ST test, run with shell
# =============================
def success_check(res):
    if res != 0:
        raise CommandFailedError(f"The command execution failed, return code is: {res}")


def success_check_ut(res):
    if len(res.failures) + len(res.errors) != 0:
        raise CommandFailedError(f"The command execution failed, return code is: {res}")


class STTest:
    def __init__(self):
        self.shell_file_list = []


    def run_shell(self):
        for shell_file in self.shell_file_list:
            success_check(os.system("sh {}".format(shell_file)))

# ===============================================
# UT test, run with pytest, waiting for more ...
# ===============================================


if __name__ == "__main__":
    st_test = STTest()
    st_test.run_shell()
    test_loader = unittest.TestLoader()
    discover = test_loader.discover(start_dir="./", pattern="test*.py")

    runner = unittest.TextTestRunner()
    success_check_ut(runner.run(discover))
