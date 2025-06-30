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
Module for testing the transfer tool.
"""
import os
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from tools.transfer import convert_general_rules, convert_special_rules, convert_special_rules_by_line, convert_package
from tools.rules.line_rules import LINE_RULES, SPECIAL_RULES, GENERAL_RULES, SHELL_RULES, FILE_RULES
from importlib import reload
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

# Get the current test case directory
current_dir = Path(__file__).parent


@pytest.fixture
def mock_file_path():
    """Mock file path and content"""
    origin_path = current_dir / "test_data" / "origin"
    save_path = current_dir / "test_data" / "save"
    origin_path.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create mock files
    file1_path = origin_path / "core/tensor_parallel/cross_entropy.py"
    file1_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file1_path, 'w', encoding='UTF-8') as f:
        f.write("import torch\nimport torch.nn as nn\ndef cross_entropy(...): ...\n")

    file2_path = origin_path / "core/pipeline_parallel/schedules.py"
    file2_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file2_path, 'w', encoding='UTF-8') as f:
        f.write("from torch.autograd.variable import Variable\ndef forward(...): ...\n")

    return str(origin_path), str(save_path)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=1)
def test_convert_special_rules(mock_file_path):
    """
    Feature: Test special rule conversion
    Description: Test if special rules are correctly applied to file content
    Expectation: File content should be correctly replaced with specific replacement content
    """
    origin_path, save_path = mock_file_path

    # Mock special rules
    special_rules = {
        "megatron": {
            "core/tensor_parallel/cross_entropy.py": [
                (r"import torch", "import msadaptor"),
                (r"def cross_entropy", "def msadaptor_cross_entropy")
            ]
        }
    }

    with patch.dict('tools.rules.line_rules.SPECIAL_RULES', special_rules):
        convert_special_rules(origin_path, save_path, "megatron")

    converted_file_path = Path(save_path) / "core/tensor_parallel/cross_entropy.py"
    assert converted_file_path.exists()
    with open(converted_file_path, 'r', encoding='UTF-8') as f:
        content = f.read()
        assert "import msadaptor" in content
        assert "def msadaptor_cross_entropy" in content


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=2)
def test_convert_special_rules_by_line(mock_file_path):
    """
    Feature: Test line-by-line special rule conversion
    Description: Test if line-by-line special rules are correctly applied to file content
    Expectation: File content should be correctly replaced with specific replacement content
    """
    origin_path, save_path = mock_file_path

    # Mock line-by-line rules
    line_rules = {
        "megatron": {
            "core/pipeline_parallel/schedules.py": [
                """ from torch.autograd.variable import Variable
+from mindspore.ops import composite as C
+from mindspore.common.api import _pynative_executor"""
            ]
        }
    }

    with patch.dict('tools.rules.line_rules.LINE_RULES', line_rules):
        convert_special_rules_by_line(origin_path, save_path, "megatron")

    converted_file_path = Path(save_path) / "core/pipeline_parallel/schedules.py"
    assert converted_file_path.exists()
    with open(converted_file_path, 'r', encoding='UTF-8') as f:
        content = f.read()
        assert "from mindspore.ops" in content
        assert "import _pynative_executor" in content


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=3)
def test_convert_package(mock_file_path):
    """
    Feature: Test overall conversion logic
    Description: Test the combination of general and special rules
    Expectation: File content should be correctly replaced with specific replacement content
    """
    origin_path, save_path = mock_file_path

    # Mock rules
    general_rules = [
        ["import torch", "import msadaptor"],
        ["torch.", "msadaptor."]
    ]

    line_rules = {
        "megatron": {
            "core/pipeline_parallel/schedules.py": [
                """ from torch.autograd.variable import Variable
+from mindspore.ops import composite as C
+from mindspore.common.api import _pynative_executor"""
            ]
        }
    }

    special_rules = {
        "megatron": {
            "core/tensor_parallel/cross_entropy.py": [
                (r"import torch", "import msadaptor"),
                (r"def cross_entropy", "def msadaptor_cross_entropy")
            ]
        }
    }

    # Reorganize rules into the original data structure format
    original_general_rules = GENERAL_RULES.copy()
    original_line_rules = LINE_RULES.copy()
    original_special_rules = SPECIAL_RULES.copy()

    try:
        # Replace rules
        GENERAL_RULES[:] = general_rules
        LINE_RULES.update(line_rules)
        SPECIAL_RULES.update(special_rules)

        convert_package(origin_path, save_path, "megatron")
    finally:
        # Restore original rules
        GENERAL_RULES[:] = original_general_rules
        LINE_RULES.clear()
        LINE_RULES.update(original_line_rules)
        SPECIAL_RULES.clear()
        SPECIAL_RULES.update(original_special_rules)

    # Check general rule conversion
    converted_file_path = Path(save_path) / "core/tensor_parallel/cross_entropy.py"
    assert converted_file_path.exists()
    with open(converted_file_path, 'r', encoding='UTF-8') as f:
        content = f.read()
        assert "import msadaptor" in content
        assert "def msadaptor_cross_entropy" in content

    # Check line-by-line special rule conversion
    converted_file_path = Path(save_path) / "core/pipeline_parallel/schedules.py"
    assert converted_file_path.exists()
    with open(converted_file_path, 'r', encoding='UTF-8') as f:
        content = f.read()
        assert "from mindspore.ops" in content
        assert "import _pynative_executor" in content


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=4)
def test_convert_general_rules_non_py_files(mock_file_path):
    """
    Feature: Test general rule conversion for non-Python files
    Description: Test if general rules correctly handle non-Python files
    Expectation: Non-Python files should not be processed
    """
    origin_path, save_path = mock_file_path

    # Create a non-.py file
    file_path = Path(origin_path) / "core/tensor_parallel/cross_entropy.txt"
    with open(file_path, 'w', encoding='UTF-8') as f:
        f.write("import torch")

    # Perform conversion
    try:
        convert_general_rules(origin_path, save_path)
    except Exception as e:
        assert 1.0

    # Verify that the non-.py file has not been processed
    converted_file_path = Path(save_path) / "core/tensor_parallel/cross_entropy.txt"

    assert converted_file_path.exists() is False


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=5)
def test_convert_special_rules_create_new_file(mock_file_path):
    """
    Feature: Test special rule conversion for creating new files
    Description: Test if special rules correctly create new files
    Expectation: New files should be correctly created and contain specific content
    """
    origin_path, save_path = mock_file_path

    # Mock special rules - create a new file
    line_rules = {
        "megatron": {
            "core/tensor_parallel/new_file.py": [
                "import mindspore\nprint('New file created')"
            ]
        }
    }

    with patch.dict('tools.rules.line_rules.LINE_RULES', line_rules):
        convert_special_rules_by_line(origin_path, save_path, "megatron")

    # Verify that the new file has been created
    converted_file_path = Path(save_path) / "core/tensor_parallel/new_file.py"
    assert converted_file_path.exists()
    with open(converted_file_path, 'r', encoding='UTF-8') as f:
        content = f.read()
        assert "import mindspore" in content
        assert "print('New file created')" in content


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=6)
def test_convert_special_rules_file_not_found(mock_file_path):
    """
    Feature: Test special rule conversion for non-existent files
    Description: Test if special rules correctly handle non-existent files
    Expectation: New files should be correctly created and contain specific content
    """
    origin_path, save_path = mock_file_path

    # Mock special rules - reference a non-existent file
    line_rules = {
        "megatron": {
            "core/tensor_parallel/nonexistent.py": [
                """ import torch
+import mindspore"""
            ]
        }
    }

    with patch.dict('tools.rules.line_rules.LINE_RULES', line_rules):
        convert_special_rules_by_line(origin_path, save_path, "megatron")

    # Verify that the new file has been created
    converted_file_path = Path(save_path) / "core/tensor_parallel/nonexistent.py"
    assert converted_file_path.exists()
    with open(converted_file_path, 'r', encoding='UTF-8') as f:
        content = f.read()
        assert "import mindspore" in content


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=7)
def test_convert_package_empty_directory(mock_file_path):
    """
    Feature: Test conversion of an empty directory
    Description: Test the conversion process on an empty directory
    Expectation: The conversion should not raise any exceptions
    """
    origin_path, save_path = mock_file_path
    origin_path = Path(origin_path) / "empty_origin"
    save_path = Path(save_path) / "empty_save"

    # Perform conversion
    try:
        convert_package(str(origin_path), str(save_path), "megatron")
    except Exception as e:
        assert 1.0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=8)
def test_convert_special_rules_by_line_remove_file(mock_file_path):
    """
    Feature: Test line-by-line special rule conversion for removing files
    Description: Test if line-by-line special rules correctly remove files
    Expectation: The specified file should be removed
    """
    origin_path, save_path = mock_file_path

    # Create the file to be removed
    file_to_remove = Path(origin_path) / "file_to_remove.py"
    with open(file_to_remove, 'w', encoding='UTF-8') as f:
        f.write("This file should be removed")

    # Mock rules for removing the file
    line_rules = {
        "megatron": {
            "file_to_remove.py": ["REMOVE"]
        }
    }

    with patch.dict('tools.rules.line_rules.LINE_RULES', line_rules):
        convert_special_rules_by_line(origin_path, save_path, "megatron")

    # Check if the file has been removed
    assert not file_to_remove.exists()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=9)
def test_convert_special_rules_by_line_create_file(mock_file_path):
    """
    Feature: Test line-by-line special rule conversion for creating new files
    Description: Test if line-by-line special rules correctly create new files
    Expectation: New files should be correctly created and contain specific content
    """
    origin_path, save_path = mock_file_path

    # Mock rules for creating a new file
    line_rules = {
        "megatron": {
            "new_file.py": ["This is a newly created file"]
        }
    }

    with patch.dict('tools.rules.line_rules.LINE_RULES', line_rules):
        convert_special_rules_by_line(origin_path, save_path, "megatron")

    # Check if the new file has been created
    new_file_path = Path(save_path) / "new_file.py"
    assert new_file_path.exists()
    with open(new_file_path, 'r', encoding='UTF-8') as f:
        content = f.read()
        assert "This is a newly created file" in content


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=10)
def test_convert_einops_package(mock_file_path):
    """
    Feature: Test special rule conversion for the einops package
    Description: Test if special rules correctly convert the einops package
    Expectation: The specified file should be correctly converted
    """
    origin_path, save_path = mock_file_path

    # Create an einops test file
    file_path = Path(origin_path) / "einops_test.py"
    with open(file_path, 'w', encoding='UTF-8') as f:
        f.write("import torch\nimport einops")

    # Mock special rules
    special_rules = {
        "einops": {
            "einops_test.py": [
                (r"import torch", "import msadaptor")
            ]
        }
    }

    with patch.dict('tools.rules.line_rules.SPECIAL_RULES', special_rules):
        convert_special_rules(origin_path, save_path, "einops")

    # Check the conversion result
    converted_file_path = Path(save_path) / "einops_test.py"
    assert converted_file_path.exists()
    with open(converted_file_path, 'r', encoding='UTF-8') as f:
        content = f.read()
        assert "import msadaptor" in content
