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
import re
import logging
logging.basicConfig(level=logging.INFO)


def parse_memory_file(fname):
    p_memory = r'\| \d.*\| \d+.*\|.*\| (\d+).*\|'
    try:
        with open(fname, 'r') as f:
            context = f.read().split('\n')

        mems = []
        for line in context:
            m = re.match(p_memory, line)
            if m:
                mems.append(int(m.group(1)))

        if not mems:
            logging.warning("No memory data matched.")
            return None

        max_mem = max(mems)
    except FileNotFoundError:
        logging.warning(f"File not found: {fname}")
        return None
    except IOError as e:
        logging.error(f"Read file fail: {e}")
        return None
    except ValueError as e:
        logging.error(f"Value conversion error: {e}")
        return None

    return max_mem / 1024 if max_mem is not None else None


def parse_script(file):
    with open(file, 'r') as f:
        context = f.read().split('\n')
    p_gbs = r'.*global-batch-size (\d*).*'
    p_len = r'.*seq-length (\d*).*'
    gbs, length = None, None
    for line in context:
        match = re.match(p_gbs, line)
        if match:
            gbs = match.group(1)
        match = re.match(p_len, line)
        if match:
            length = match.group(1)
    return gbs, length


def parse_log_file(file):
    it_pattern = (r'.*\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] '
                  r'iteration\s*(\d*)\/.*lm loss: ([\d\.]*).*grad norm: ([\d\.]*).*')
    with open(file, 'r') as f:
        context = f.read().split('\n')
    data = {}
    for line in context:
        match = re.match(it_pattern, line)
        if match:
            data[int(match.group(2))] = match.groups()
    return data
