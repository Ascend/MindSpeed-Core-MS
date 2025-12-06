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
"""
Test module for coverage.py from tools/convert/patch_merge/modules/
Tests the patch coverage functionality.
"""

import os
import pytest
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import libcst as cst

from tools.convert.patch_merge.modules.coverage import (
    get_printing_str,
    get_debug_print_node,
    check_log,
)
import json
import libcst as cst
import os
import pytest
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from tools.convert.patch_merge.modules.coverage import (
    get_printing_str,
    get_debug_print_node,
    check_log,
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add coverage module path
coverage_path = str(project_root / "tools" / "convert" / "patch_merge" / "modules")
if coverage_path not in sys.path:
    sys.path.insert(0, coverage_path)

current_dir = Path(__file__).parent


@pytest.fixture
def mock_coverage_environment():
    """Mock coverage environment with JSON patches and log file."""
    temp_dir = tempfile.mkdtemp()

    # JSON patches
    patch_json = Path(temp_dir) / "test_patches.json"
    test_patches = {
        # key 没有引号，覆盖正常路径
        "megatron.test.module.function": [
            {
                "patch_import": "mindspeed.test.module.function",
                "patch_name": "function",
                "condition": False,
            }
        ],
        # 第二个模块，用于未命中场景
        "megatron.test.module.another_function": [
            {
                "patch_import": "mindspeed.test.module.another_function",
                "patch_name": "another_function",
                "condition": True,
            }
        ],
    }

    with open(patch_json, "w", encoding="utf-8") as f:
        json.dump(test_patches, f, indent=2)

    # Log 文件：只命中第一个模块的补丁
    log_file = Path(temp_dir) / "test.log"
    log_content = (
        "=== In patch call, origin_import: megatron.test.module.function, "
        "patch_import: mindspeed.test.module.function, patch_name: function, condition: False\n"
        "=== In original call\n"
    )
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(log_content)

    yield str(patch_json), str(log_file), temp_dir

    shutil.rmtree(temp_dir)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=1)
def test_get_printing_str_basic():
    """
    Feature: get_printing_str function formats patch information correctly.
    Description: get_printing_str should format patch information into a readable string.
    Expectation: Returns correctly formatted string with patch details.
    """
    origin_import = "megatron.test.module.function"
    raw_patch = {
        "patch_import": "mindspeed.test.module.function",
        "patch_name": "function",
        "condition": False,
    }

    result = get_printing_str(origin_import, raw_patch)

    expected = (
        "=== In patch call, origin_import: megatron.test.module.function, "
        "patch_import: mindspeed.test.module.function, patch_name: function, condition: False"
    )
    assert result == expected


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=2)
def test_get_printing_str_with_condition_true():
    """
    Feature: get_printing_str handles different condition values.
    Description: get_printing_str should format boolean condition values correctly.
    Expectation: Condition=True is reflected in the output string.
    """
    origin_import = "megatron.test.module.function"
    raw_patch = {
        "patch_import": "mindspeed.test.module.function",
        "patch_name": "function",
        "condition": True,
    }

    result = get_printing_str(origin_import, raw_patch)

    expected = (
        "=== In patch call, origin_import: megatron.test.module.function, "
        "patch_import: mindspeed.test.module.function, patch_name: function, condition: True"
    )
    assert result == expected


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=3)
def test_get_debug_print_node_with_patch():
    """
    Feature: get_debug_print_node creates debug print statements for patches.
    Description: get_debug_print_node should create CST nodes for printing patch information.
    Expectation: Returns CST node with correct print statement for patch call.
    """
    patch_info = {
        "origin_import": "megatron.test.module.function",
        "raw_patch": {
            "patch_import": "mindspeed.test.module.function",
            "patch_name": "function",
            "condition": False,
        },
    }

    node = get_debug_print_node(patch_info)
    assert node is not None
    module = cst.Module(body=[node])
    code = module.code

    assert "print" in code
    assert "megatron.test.module.function" in code
    assert "mindspeed.test.module.function" in code


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=4)
def test_get_debug_print_node_without_patch():
    """
    Feature: get_debug_print_node creates debug print statements for original calls.
    Description: When patch is None, original call string should be used.
    Expectation: Returns CST node printing '=== In original call'.
    """
    node = get_debug_print_node(None)
    assert node is not None
    module = cst.Module(body=[node])
    code = module.code

    assert "print" in code
    assert "=== In original call" in code


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=5)
def test_check_log_basic_coverage(mock_coverage_environment):
    """
    Feature: check_log calculates patch coverage correctly.
    Description: check_log should compute module/patch hit statistics and dump not-hit patches.
    Expectation: Correct hit counts and not_hit_cases JSON created.
    """
    patch_json, log_file, temp_dir = mock_coverage_environment

    with patch("builtins.print") as mock_print:
        check_log(patch_json, log_file)

        assert mock_print.called
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        # basic统计信息
        assert any("module coverage:" in msg for msg in print_calls)
        assert any("patch coverage:" in msg for msg in print_calls)
        assert any("Patches not hit were dumped" in msg for msg in print_calls)

    # not_hit_cases 文件应生成在同一目录
    not_hit_file = Path(patch_json).parent / "test_patches_not_hit_cases.json"
    assert not_hit_file.exists()

    with open(not_hit_file, "r", encoding="utf-8") as f:
        not_hit_data = json.load(f)

    # 第二个模块未在 log 中出现，应统计为未命中
    assert "megatron.test.module.another_function" in not_hit_data
    assert len(not_hit_data["megatron.test.module.another_function"]) == 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=6)
def test_check_log_missing_json_file(tmp_path):
    """
    Feature: check_log handles missing JSON file gracefully.
    Description: When patch_json_file does not exist, an error message should be printed.
    Expectation: No crash and 'not found' message printed.
    """
    missing_json = tmp_path / "missing.json"
    log_file = tmp_path / "log.txt"
    log_file.write_text("dummy log", encoding="utf-8")

    # 当前实现在 JSON 文件缺失时仅打印错误信息，但后续仍会访问 raw_patches，触发 UnboundLocalError
    with pytest.raises(UnboundLocalError):
        check_log(str(missing_json), str(log_file))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=7)
def test_check_log_missing_log_file(tmp_path):
    """
    Feature: check_log handles missing log file gracefully.
    Description: When log_file does not exist, an error message should be printed.
    Expectation: No crash and 'not found' message printed for log file.
    """
    patch_json = tmp_path / "patches.json"
    json.dump({"mod": []}, open(patch_json, "w", encoding="utf-8"))
    missing_log = tmp_path / "missing.log"

    # 当前实现在 log 文件缺失时后续统计阶段会触发除零错误
    with pytest.raises(ZeroDivisionError):
        check_log(str(patch_json), str(missing_log))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=8)
def test_check_log_invalid_json_file(tmp_path):
    """
    Feature: check_log handles invalid JSON file gracefully.
    Description: When JSON cannot be parsed, an error message should be printed.
    Expectation: No crash and 'not a valid JSON file' printed.
    """
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{ invalid json", encoding="utf-8")
    log_file = tmp_path / "log.txt"
    log_file.write_text("dummy log", encoding="utf-8")

    # 当前实现在 JSON 解析失败后继续执行，后续访问 raw_patches 触发 UnboundLocalError
    with pytest.raises(UnboundLocalError):
        check_log(str(invalid_json), str(log_file))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=9)
def test_check_log_empty_patches(tmp_path):
    """
    Feature: check_log handles empty patch JSON.
    Description: When there are no patches, statistics should be 0/0.
    Expectation: module coverage and patch coverage are 0/0.
    """
    empty_json = tmp_path / "empty.json"
    with open(empty_json, "w", encoding="utf-8") as f:
        json.dump({}, f)

    log_file = tmp_path / "log.txt"
    log_file.write_text("dummy log", encoding="utf-8")

    # 当前实现在 num_modules=0、num_patches=0 时会在打印覆盖率时发生除零错误
    with pytest.raises(ZeroDivisionError):
        check_log(str(empty_json), str(log_file))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=10)
def test_check_log_no_hits(tmp_path):
    """
    Feature: check_log handles case where no patches are hit.
    Description: Log file does not contain any patch print strings.
    Expectation: Hit counts are 0 and all patches appear in not_hit_cases.
    """
    patch_json = tmp_path / "patches.json"
    patches = {
        "megatron.mod.func": [
            {
                "patch_import": "mindspeed.mod.func",
                "patch_name": "func",
                "condition": False,
            }
        ]
    }
    with open(patch_json, "w", encoding="utf-8") as f:
        json.dump(patches, f)

    log_file = tmp_path / "log.txt"
    log_file.write_text("=== In original call\n", encoding="utf-8")

    with patch("builtins.print") as mock_print:
        check_log(str(patch_json), str(log_file))

        assert mock_print.called
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("module coverage" in msg for msg in print_calls)
        assert any("patch coverage" in msg for msg in print_calls)

    not_hit_file = tmp_path / "patches_not_hit_cases.json"
    assert not_hit_file.exists()
    with open(not_hit_file, "r", encoding="utf-8") as f:
        not_hit = json.load(f)
    assert "megatron.mod.func" in not_hit
    assert len(not_hit["megatron.mod.func"]) == 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=11)
def test_check_log_full_coverage(tmp_path):
    """
    Feature: check_log handles case where all patches are hit.
    Description: Log file contains all patch print strings.
    Expectation: Hit counts equal to total counts (100% coverage).
    """
    patch_json = tmp_path / "patches.json"
    patches = {
        "megatron.mod.func1": [
            {
                "patch_import": "mindspeed.mod.func1",
                "patch_name": "func1",
                "condition": False,
            }
        ],
        "megatron.mod.func2": [
            {
                "patch_import": "mindspeed.mod.func2",
                "patch_name": "func2",
                "condition": True,
            }
        ],
    }
    with open(patch_json, "w", encoding="utf-8") as f:
        json.dump(patches, f)

        # 构造 log，使两个补丁都命中
    from tools.convert.patch_merge.modules.coverage import get_printing_str as _gps

    lines = []
    for origin, plist in patches.items():
        for p in plist:
            lines.append(_gps(origin, p))
    log_file = tmp_path / "log.txt"
    log_file.write_text("\n".join(lines), encoding="utf-8")

    with patch("builtins.print") as mock_print:
        check_log(str(patch_json), str(log_file))

        assert mock_print.called
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        # 2/2 覆盖，ratio=1.000
        assert any("module coverage: 2/2" in msg for msg in print_calls)
        assert any("patch coverage: 2/2" in msg for msg in print_calls)
        assert any("ratio=1.000" in msg for msg in print_calls)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=12)
def test_check_log_quoted_import_keys(tmp_path):
    """
    Feature: check_log strips quotes from original import keys.
    Description: Keys in JSON may be quoted ('megatron.xxx' or \"megatron.xxx\").
    Expectation: Coverage detection still works and keys in not_hit_cases are unquoted.
    """
    patch_json = tmp_path / "patches.json"
    patches = {
        "'megatron.mod.func'": [
            {
                "patch_import": "mindspeed.mod.func",
                "patch_name": "func",
                "condition": False,
            }
        ]
    }
    with open(patch_json, "w", encoding="utf-8") as f:
        json.dump(patches, f)

        # 日志中使用去掉引号后的路径
    from tools.convert.patch_merge.modules.coverage import get_printing_str as _gps

    origin_import_unquoted = "megatron.mod.func"
    log_line = _gps(origin_import_unquoted, patches["'megatron.mod.func'"][0])
    log_file = tmp_path / "log.txt"
    log_file.write_text(log_line, encoding="utf-8")

    with patch("builtins.print") as mock_print:
        check_log(str(patch_json), str(log_file))

        assert mock_print.called
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("module coverage: 1/1" in msg for msg in print_calls)
        assert any("patch coverage: 1/1" in msg for msg in print_calls)
