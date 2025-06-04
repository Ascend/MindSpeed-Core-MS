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
"""test pdb utils"""
import importlib
import os
import threading
import time

import pytest
import torch

from tools.debug_utils import pdb_utils


@pytest.fixture(autouse=True)
def reset_breakpoint_counter(monkeypatch, tmp_path):
    """
    Before each test case execution, reset BREAKPOINT_COUNTER to 0,
    and set the temporary directory to the tmp_path directory provided by pytest
    to prevent breakpoint files from polluting.
    """
    # Reset the global counter
    monkeypatch.setattr(pdb_utils, "BREAKPOINT_COUNTER", 0)
    # Override the temporary directory
    monkeypatch.setattr(pdb_utils, "BREAKPOINT_TMP_FILE_PATH", str(tmp_path))
    yield
    # No explicit cleanup needed after the test; tmp_path will clean up automatically.


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.level0
@pytest.mark.run(order=1)
def test_get_counter_file_path(tmp_path):
    """
    Feature: Get counter file path
    Description: Test whether the return value of _get_counter_file matches the expected format when the counter is 0
    and after updating BREAKPOINT_COUNTER to 5.
    Expectation: The returned path should be <tmp_path>/debug_counter_<counter>.txt for counter values 0 and 5.
    """
    # Initially BREAKPOINT_COUNTER == 0
    pdb_utils_mod = importlib.import_module("tools.debug_utils.pdb_utils")
    get_counter_file = getattr(pdb_utils_mod, "_get_counter_file")
    path_get0 = get_counter_file()
    expected0 = os.path.join(str(tmp_path), "debug_counter_0.txt")
    assert path_get0 == expected0

    # Modify the counter
    pdb_utils.BREAKPOINT_COUNTER = 5
    path_get5 = get_counter_file()
    expected5 = os.path.join(str(tmp_path), "debug_counter_5.txt")
    assert path_get5 == expected5


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.level0
@pytest.mark.run(order=2)
def test_breakpoint_at_debug_rank(monkeypatch, tmp_path):
    """
    Feature: Breakpoint creation and clearing when debugging rank
    Description: Simulate the situation where current process rank equals RANK_TO_DEBUG (both set to 2),
    intercept pdb.set_trace to avoid interactive debugging, call breakpoint_() to create the counter file,
    then call clear_() to delete it and increment BREAKPOINT_COUNTER.
    Expectation: breakpoint_() creates debug_counter_0.txt in tmp_path, and clear_() deletes that file and
    sets BREAKPOINT_COUNTER to 1.
    """
    # Set the current process rank to 2 and debug rank also to 2
    monkeypatch.setenv("RANK_TO_DEBUG", "2")
    # Monkeypatch torch.distributed.get_rank to return 2
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 2)

    # Intercept pdb.set_trace to avoid actually entering interactive debugging

    pdb = importlib.import_module("pdb")
    monkeypatch.setattr(pdb, "set_trace", lambda: None)

    # At this point BREAKPOINT_COUNTER == 0
    pdb_utils.breakpoint_()  # Should create file debug_counter_0.txt
    counter_file = tmp_path / "debug_counter_0.txt"
    assert counter_file.exists(), "When current_rank == debug_rank, a counter file should be created"

    # Call clear_ again: file is deleted, counter increments
    pdb_utils.clear_()
    assert not counter_file.exists(), "clear_() should delete the counter file"
    assert pdb_utils.BREAKPOINT_COUNTER == 1, "clear_() should increment BREAKPOINT_COUNTER to 1"


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.level0
@pytest.mark.run(order=3)
def test_breakpoint_non_debug_rank_non_block(monkeypatch, tmp_path):
    """
    Feature: Breakpoint skip for non-debug rank in non-block mode
    Description: Simulate the logic branch when current process rank (3) is not equal to RANK_TO_DEBUG (1) and
    non_block=True (default); call breakpoint_() and verify that it does nothing.
    Expectation: No breakpoint file is created in tmp_path and BREAKPOINT_COUNTER remains 0.
    """
    # Current rank is 3, debug rank is set to 1
    monkeypatch.setenv("RANK_TO_DEBUG", "1")
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)

    # Do not intercept pdb.set_trace; it should not enter that branch
    # As long as no exception is thrown, this branch is covered
    pdb_utils.breakpoint_()

    # Counter should not change, and there should be no counter file in directory
    assert pdb_utils.BREAKPOINT_COUNTER == 0
    files = list(tmp_path.iterdir())
    assert all("debug_counter" not in p.name for p in files)


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.level0
@pytest.mark.run(order=4)
def test_clear_without_existing_file(tmp_path):
    """
    Feature: Clear without existing counter file
    Description: Ensure no debug_counter_0.txt exists in tmp_path, then call clear_().
    Expectation: clear_() increments BREAKPOINT_COUNTER to 1 without raising an error, and no counter file is created.
    """
    # Ensure no debug_counter_0.txt exists
    counter_file = tmp_path / "debug_counter_0.txt"
    if counter_file.exists():
        counter_file.unlink()
    # Initial counter is 0
    assert pdb_utils.BREAKPOINT_COUNTER == 0
    pdb_utils.clear_()
    # clear_ will increment the counter
    assert pdb_utils.BREAKPOINT_COUNTER == 1
    # There should still be no file in directory
    assert not counter_file.exists()


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.level0
@pytest.mark.run(order=5)
def test_breakpoint_waiting_branch(monkeypatch, tmp_path, caplog):
    """
    Feature: Breakpoint waiting behavior for non-debug rank in blocking mode
    Description: Simulate current process rank (4) not equal to RANK_TO_DEBUG (2), create debug_counter_0.txt
    in tmp_path before calling breakpoint_() with non_block=False, and spawn a thread to delete the file after a
    short delay; capture log messages during the wait loop.
    Expectation: breakpoint_() will enter the waiting loop, logging at least one "blocking" message while waiting
    for debug_counter_0.txt to be removed.
    """
    # Set current_rank = 4, debug_rank = 2
    monkeypatch.setenv("RANK_TO_DEBUG", "2")
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 4)

    # Set non_block=False
    # First manually create a counter file named debug_counter_0.txt
    counter_path = tmp_path / "debug_counter_0.txt"
    counter_path.write_text("DEBUGGING")
    # Because the waiting branch will first detect that the file exists,
    # then loop waiting for the file to be deleted.
    # We delete it manually in another thread or after a delay; here we capture logs with caplog then delete it.
    # To avoid deadlock, delete it immediately after the next log check.

    # Intercept pdb.set_trace to prevent accidental entering
    pdb = importlib.import_module("pdb")
    monkeypatch.setattr(pdb, "set_trace", lambda: None)

    # Record logs
    caplog.set_level("INFO")

    def remove_file_later():
        # Wait 0.1s then delete the file to end the waiting loop
        time_to_wait = 0.1

        time.sleep(time_to_wait)
        if counter_path.exists():
            counter_path.unlink()

    remover = threading.Thread(target=remove_file_later)
    remover.start()

    # Call breakpoint_ to let it enter the waiting branch.
    # This call will see the counter_file already exists -> logging "waiting to be cleaned..."
    # Then we immediately delete the file to end the loop.
    # Because the loop sleeps 1s each time, there will be at least one "blocking..." log.
    # To cover this in testing, simulate this behavior in a small loop.
    pdb_utils.breakpoint_(non_block=False)
    remover.join(timeout=1.0)

    # Confirm there is at least one "blocking..." record in the logs
    assert any("blocking" in rec.getMessage() for rec in caplog.records)


if __name__ == "__main__":
    pytest.main()

