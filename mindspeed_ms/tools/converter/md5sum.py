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
'''check md5sum of the str serialized pt files '''
import argparse
import hashlib
from datetime import datetime
import torch


def compare_md5s(file_paths):
    md5_values = [calculate_md5(path) for path in file_paths]
    unique_md5s = set(md5_values)

    if len(unique_md5s) == 1:
        print(f"All files have the same md5sum value: {md5_values[0]}")
    else:
        print("Different md5sum")
        for i, md5 in enumerate(md5_values, 1):
            print(f"{file_paths[i-1]} md5sum: {md5}")


def calculate_md5(ckpt_file):
    state_dict = torch.load(ckpt_file, map_location='cpu')
    if 'args' in state_dict and 'consumed_train_samples' in state_dict['args']:
        state_dict['args'].consumed_train_samples = 0
    md5_value = hashlib.md5(str(state_dict).encode()).hexdigest()
    return md5_value


def log_with_time(log_str):
    now = datetime.now()
    print(f">{now.strftime('%H:%M:%S')}  : {log_str}", flush=True)


if __name__ == "__main__":
    log_with_time("-------------Comparing MD5Sum-------------")
    parser = argparse.ArgumentParser(description="Compare MD5Sum")
    parser.add_argument('files', metavar='file', type=str, nargs='+', help='Files to be compared')
    args = parser.parse_args()
    compare_md5s(args.files)
