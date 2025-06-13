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
import time
import logging
import torch
logging.basicConfig(level=logging.INFO)

BREAKPOINT_COUNTER = 0
BREAKPOINT_TMP_FILE_PATH_DEFAULT = "/tmp/"
BREAKPOINT_TMP_FILE_PATH = os.getenv("BREAKPOINT_TMP_FILE_PATH",
                                     BREAKPOINT_TMP_FILE_PATH_DEFAULT)


def _get_counter_file():
    global BREAKPOINT_COUNTER
    return os.path.join(BREAKPOINT_TMP_FILE_PATH, f"debug_counter_{BREAKPOINT_COUNTER}.txt")


def breakpoint_(non_block=True):
    """
    distributed breakpoint for debug
    Args:
        block: whether to block other ranks. In case that
    """
    current_rank = torch.distributed.get_rank()
    debug_rank = os.environ.get("RANK_TO_DEBUG")
    if current_rank is None or debug_rank is None:
        raise Exception(f"RANK_TO_DEBUG/rank can't be None in debug mode, "
                        f"MS_NODE_ID: `{current_rank}`, RANK_TO_DEBUG: `{debug_rank}`")
    logging.info(f"current_rank: {current_rank}, debug_rank: {debug_rank}")

    counter_file = _get_counter_file()
    if str(current_rank) == str(debug_rank):
        if not os.path.exists(counter_file):
            with open(counter_file, "w") as f:
                f.write(f"DEBUGGING{counter_file}")
            logging.info(f"[{time.time()}]counter_file created: {counter_file}")
        import pdb
        pdb.set_trace() # press `n` and then `Enter` to reach your code
    elif not non_block:
        logging.info(f"[{time.time()}]waiting counter_file to be created...")
        while not os.path.exists(counter_file):
            time.sleep(1)
        logging.info(f"[{time.time()}]counter_file created, waiting to be cleaned...")
        while os.path.exists(counter_file):
            time.sleep(1)
            logging.info(f"[{time.time()}]blocking...")
        logging.info(f"[{time.time()}]counter_file cleaned, continue")
    else:
        # otherwise, other ranks don't need to wait
        logging.info(f"[{time.time()}]stepping in to debug field in background rank. (non blocking)")


def clear_():
    counter_file = _get_counter_file()
    if os.path.exists(counter_file):
        os.remove(counter_file)
        logging.info(f"[{time.time()}] counter_file cleared: {counter_file}")
    global BREAKPOINT_COUNTER
    BREAKPOINT_COUNTER += 1
