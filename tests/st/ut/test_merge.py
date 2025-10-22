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
Fixed test module for merge.py functionality.
"""
import os
import pytest
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import types

from tools.convert.patch_merge.modules import merge

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add merge module path
merge_path = str(project_root / "tools" / "convert" / "patch_merge" / "modules")
if merge_path not in sys.path:
    sys.path.insert(0, merge_path)

current_dir = Path(__file__).parent


@pytest.fixture
def mock_merge_environment():
    """Mock merge environment with test data"""
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    root_dir = Path(temp_dir) / "MindSpeed-Core-MS"
    root_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test patch JSON file
    patch_json = root_dir / "test_patches.json"
    test_patches = {
        "'megatron.test.module.function'": [
            {
                "patch_import": "mindspeed.test.module.function",
                "patch_name": "function",
                "condition": False
            }
        ]
    }
    
    with open(patch_json, 'w', encoding='utf-8') as f:
        json.dump(test_patches, f, indent=2)
    
    # Create test source files
    megatron_dir = root_dir / "Megatron-LM" / "megatron" / "test" / "module"
    megatron_dir.mkdir(parents=True, exist_ok=True)
    
    mindspeed_dir = root_dir / "MindSpeed" / "mindspeed" / "test" / "module"
    mindspeed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create original function file
    original_file = megatron_dir / "module.py"
    original_content = '''def function(a, b):
    return a + b
'''
    with open(original_file, 'w', encoding='utf-8') as f:
        f.write(original_content)
    
    # Create patch function file
    patch_file = mindspeed_dir / "module.py"
    patch_content = '''def function(a, b):
    return a * b
'''
    with open(patch_file, 'w', encoding='utf-8') as f:
        f.write(patch_content)
    
    # Create adaptor file
    adaptor_file = root_dir / "MindSpeed-LLM" / "mindspeed_llm" / "tasks" / "megatron_adaptor.py"
    adaptor_file.parent.mkdir(parents=True, exist_ok=True)
    adaptor_content = '''from mindspeed_llm.tasks.megatron_adaptor import MegatronAdaptation

MegatronAdaptation.execute()
'''
    with open(adaptor_file, 'w', encoding='utf-8') as f:
        f.write(adaptor_content)
    
    yield str(root_dir), str(patch_json), str(original_file), str(patch_file), str(adaptor_file)
    
    # Cleanup
    shutil.rmtree(temp_dir)


def light_init(self, patches, root_dir):
    # mocked lightweight initializer for PatchMerger
    self.raw_patches = patches
    self.root = root_dir
    self.patch_replace_info = {}
    self.patch_func_infos = {}
    self.patch_wrapper_infos = {}
    self.patch_class_infos = {}
    self.all_patch_infos = [self.patch_replace_info, self.patch_func_infos, self.patch_wrapper_infos, self.patch_class_infos]
    self.cst_to_write = {}
    self.num_modules, self.num_patches = 0, 0
    from collections import defaultdict as _dd
    self.bad_parsed_cases = _dd(list)
    self.bad_handled_cases = _dd(list)
    self.adaptors = {}


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=1)
def test_merge_replacement_function_replacement(monkeypatch, tmp_path):
    """
    Feature: PatchMerger.merge_replacement handles function replacement correctly.
    Description: Test that function definitions are replaced from patch file to original file.
    Expectation: Function in original file is replaced with function from patch file.
    """
    # Create test files
    original_file = tmp_path / "megatron/func/original.py"
    patch_file = tmp_path / "mindspeed/func/patch.py"
    Path(original_file).parent.mkdir(parents=True, exist_ok=True)
    Path(patch_file).parent.mkdir(parents=True, exist_ok=True)
    
    original_content = '''def test_function(a, b):
    return a + b
'''
    patch_content = '''def test_function(a, b):
    return a * b
'''
    
    original_file.write_text(original_content, encoding="utf-8")
    patch_file.write_text(patch_content, encoding="utf-8")
    
    monkeypatch.setattr(merge.PatchMerger, "__init__", light_init, raising=True)
    
    # Mock get_cst to return parsed CST
    def mock_get_cst(self, file_path):
        import libcst as cst
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        return cst.parse_module(code)
    
    monkeypatch.setattr(merge.PatchMerger, "get_cst", mock_get_cst, raising=True)
    
    # Mock set_cst to track changes
    def mock_set_cst(self, file_path, cst_module):
        self.cst_to_write[file_path] = cst_module
    
    monkeypatch.setattr(merge.PatchMerger, "set_cst", mock_set_cst, raising=True)
    
    # Mock handle_annotate to avoid adaptor dependencies
    def mock_handle_annotate(self, patch_infos):
        pass
    
    monkeypatch.setattr(merge.PatchMerger, "handle_annotate", mock_handle_annotate, raising=True)
    
    # Mock handle_exc to track errors
    def mock_handle_exc(self, e, module_name, module_patch_infos):
        print(f"Error in {module_name}: {e}")
    
    monkeypatch.setattr(merge.PatchMerger, "handle_exc", mock_handle_exc, raising=True)
    
    # Create patch info
    patch_info = {
        "orign_file": str(original_file),
        "patch_file": str(patch_file),
        "module_orign_name": ("test_function", None, "test_function"),
        "module_patch_name": ("test_function", None, "test_function"),
        'orign_import': "megatron.func.original.test_function",
        'orign_import_root': "megatron",
        'patch_import': "mindspeed.func.patch.test_function",
        'patch_import_root': "mindspeed",
        "condition": False,
        "raw_patch": {"patch_import": "patch.test_function", "patch_name": "test_function", "condition": False}
    }
    
    pm = merge.PatchMerger({}, str(tmp_path))
    pm.patch_replace_info[str(original_file)] = {"test_function": [patch_info]}
    
    # Execute merge_replacement
    pm.merge_replacement()
    
    # Verify that original file was updated
    assert str(original_file) in pm.cst_to_write
    updated_cst = pm.cst_to_write[str(original_file)]
    updated_code = updated_cst.code
    
    # The function should be replaced (multiplication instead of addition)
    assert "return a * b" in updated_code
    assert "return a + b" not in updated_code


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=2)
def test_merge_replacement_class_replacement(monkeypatch, tmp_path):
    """
    Feature: PatchMerger.merge_replacement handles class replacement correctly.
    Description: Test that class definitions are replaced from patch file to original file.
    Expectation: Class in original file is replaced with class from patch file.
    """
    # Create test files
    original_file = tmp_path / "megatron/class/original.py"
    patch_file = tmp_path / "mindspeed/class/patch.py"
    Path(original_file).parent.mkdir(parents=True, exist_ok=True)
    Path(patch_file).parent.mkdir(parents=True, exist_ok=True)
    original_content = '''class TestClass:
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value
'''
    patch_content = '''class TestClass:
    def __init__(self, value):
        self.value = value * 2
    
    def get_value(self):
        return self.value * 2
'''
    
    original_file.write_text(original_content, encoding="utf-8")
    patch_file.write_text(patch_content, encoding="utf-8")
    
    monkeypatch.setattr(merge.PatchMerger, "__init__", light_init, raising=True)
    
    # Mock get_cst to return parsed CST
    def mock_get_cst(self, file_path):
        import libcst as cst
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        return cst.parse_module(code)
    
    monkeypatch.setattr(merge.PatchMerger, "get_cst", mock_get_cst, raising=True)
    
    # Mock set_cst to track changes
    def mock_set_cst(self, file_path, cst_module):
        self.cst_to_write[file_path] = cst_module
    
    monkeypatch.setattr(merge.PatchMerger, "set_cst", mock_set_cst, raising=True)
    
    # Mock handle_annotate to avoid adaptor dependencies
    def mock_handle_annotate(self, patch_infos):
        pass
    
    monkeypatch.setattr(merge.PatchMerger, "handle_annotate", mock_handle_annotate, raising=True)
    
    # Mock handle_exc to track errors
    def mock_handle_exc(self, e, module_name, module_patch_infos):
        print(f"Error in {module_name}: {e}")
    
    monkeypatch.setattr(merge.PatchMerger, "handle_exc", mock_handle_exc, raising=True)
    
    # Create patch info for class replacement
    patch_info = {
        "orign_file": str(original_file),
        "patch_file": str(patch_file),
        "module_orign_name": ("TestClass", "TestClass", None),
        "module_patch_name": ("TestClass", "TestClass", None),
        "condition": False,
        'orign_import': "megatron.class.original.test_function",
        'orign_import_root': "megatron",
        'patch_import': "mindspeed.class.patch.test_function",
        'patch_import_root': "mindspeed",
        "raw_patch": {"patch_import": "patch.TestClass", "patch_name": "TestClass", "condition": False}
    }
    
    pm = merge.PatchMerger({}, str(tmp_path))
    pm.patch_replace_info[str(original_file)] = {"TestClass": [patch_info]}
    
    # Execute merge_replacement
    pm.merge_replacement()
    
    # Verify that original file was updated
    assert str(original_file) in pm.cst_to_write
    updated_cst = pm.cst_to_write[str(original_file)]
    updated_code = updated_cst.code
    
    # The class should be replaced (value * 2 instead of value)
    assert "self.value = value * 2" in updated_code
    assert "return self.value * 2" in updated_code


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=3)
def test_merge_replacement_error_handling(monkeypatch, tmp_path):
    """
    Feature: PatchMerger.merge_replacement handles errors gracefully.
    Description: Test that errors during replacement are caught and handled properly.
    Expectation: Errors are logged and processing continues for other patches.
    """
    # Create test files
    original_file = tmp_path / "original.py"
    patch_file = tmp_path / "patch.py"
    
    original_content = '''def test_function(a, b):
    return a + b
'''
    patch_content = '''def test_function(a, b):
    return a * b
'''
    
    original_file.write_text(original_content, encoding="utf-8")
    patch_file.write_text(patch_content, encoding="utf-8")
    
    monkeypatch.setattr(merge.PatchMerger, "__init__", light_init, raising=True)
    
    # Mock get_cst to raise an error for patch file
    def mock_get_cst(self, file_path):
        if "patch" in str(file_path):
            raise Exception("Failed to parse patch file")
        import libcst as cst
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        return cst.parse_module(code)
    
    monkeypatch.setattr(merge.PatchMerger, "get_cst", mock_get_cst, raising=True)
    
    # Mock set_cst to track changes
    def mock_set_cst(self, file_path, cst_module):
        self.cst_to_write[file_path] = cst_module
    
    monkeypatch.setattr(merge.PatchMerger, "set_cst", mock_set_cst, raising=True)
    
    # Mock handle_annotate to avoid adaptor dependencies
    def mock_handle_annotate(self, patch_infos):
        pass
    
    monkeypatch.setattr(merge.PatchMerger, "handle_annotate", mock_handle_annotate, raising=True)
    
    # Track errors
    errors_caught = []
    def mock_handle_exc(self, e, module_name, module_patch_infos):
        errors_caught.append((module_name, str(e)))
    
    monkeypatch.setattr(merge.PatchMerger, "handle_exc", mock_handle_exc, raising=True)
    
    # Create patch info
    patch_info = {
        "orign_file": str(original_file),
        "patch_file": str(patch_file),
        "module_orign_name": ("test_function", None, "test_function"),
        "module_patch_name": ("test_function", None, "test_function"),
        "condition": False,
        "raw_patch": {"patch_import": "patch.test_function", "patch_name": "test_function", "condition": False}
    }
    
    pm = merge.PatchMerger({}, str(tmp_path))
    pm.patch_replace_info[str(original_file)] = {"test_function": [patch_info]}
    
    # Execute merge_replacement
    pm.merge_replacement()
    
    # Verify that error was caught and handled
    assert len(errors_caught) == 1
    assert errors_caught[0][0] == "test_function"
    assert "Failed to parse patch file" in errors_caught[0][1]


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=4)
def test_merge_replacement_multiple_patches_error(monkeypatch, tmp_path):
    """
    Feature: PatchMerger.merge_replacement handles multiple patches for same module error.
    Description: Test that having multiple replacement patches for the same module raises an error.
    Expectation: Exception is raised when multiple patches exist for the same module.
    """
    # Create test files
    original_file = tmp_path / "original.py"
    patch_file1 = tmp_path / "patch1.py"
    patch_file2 = tmp_path / "patch2.py"
    
    original_content = '''def test_function(a, b):
    return a + b
'''
    patch_content = '''def test_function(a, b):
    return a * b
'''
    
    original_file.write_text(original_content, encoding="utf-8")
    patch_file1.write_text(patch_content, encoding="utf-8")
    patch_file2.write_text(patch_content, encoding="utf-8")
    
    monkeypatch.setattr(merge.PatchMerger, "__init__", light_init, raising=True)
    
    # Mock get_cst to return parsed CST
    def mock_get_cst(self, file_path):
        import libcst as cst
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        return cst.parse_module(code)
    
    monkeypatch.setattr(merge.PatchMerger, "get_cst", mock_get_cst, raising=True)
    
    # Mock set_cst to track changes
    def mock_set_cst(self, file_path, cst_module):
        self.cst_to_write[file_path] = cst_module
    
    monkeypatch.setattr(merge.PatchMerger, "set_cst", mock_set_cst, raising=True)
    
    # Mock handle_annotate to avoid adaptor dependencies
    def mock_handle_annotate(self, patch_infos):
        pass
    
    monkeypatch.setattr(merge.PatchMerger, "handle_annotate", mock_handle_annotate, raising=True)
    
    # Track errors
    errors_caught = []
    def mock_handle_exc(self, e, module_name, module_patch_infos):
        errors_caught.append((module_name, str(e)))
    
    monkeypatch.setattr(merge.PatchMerger, "handle_exc", mock_handle_exc, raising=True)
    
    # Create multiple patch infos for the same module (should cause error)
    patch_info1 = {
        "orign_file": str(original_file),
        "patch_file": str(patch_file1),
        "module_orign_name": ("test_function", None, "test_function"),
        "module_patch_name": ("test_function", None, "test_function"),
        "condition": False,
        "raw_patch": {"patch_import": "patch1.test_function", "patch_name": "test_function", "condition": False}
    }
    
    patch_info2 = {
        "orign_file": str(original_file),
        "patch_file": str(patch_file2),
        "module_orign_name": ("test_function", None, "test_function"),
        "module_patch_name": ("test_function", None, "test_function"),
        "condition": False,
        "raw_patch": {"patch_import": "patch2.test_function", "patch_name": "test_function", "condition": False}
    }
    
    pm = merge.PatchMerger({}, str(tmp_path))
    pm.patch_replace_info[str(original_file)] = {"test_function": [patch_info1, patch_info2]}
    
    # Execute merge_replacement
    pm.merge_replacement()
    
    # Verify that error was caught for multiple patches
    assert len(errors_caught) == 1
    assert errors_caught[0][0] == "test_function"
    assert "Should only have 1 replacement for module" in errors_caught[0][1]


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=5)
def test_merge_get_module_name_function():
    """
    Feature: get_module_name function correctly formats module names.
    Description: get_module_name formats function names, class names, or class.method names.
    Expectation: Returns correctly formatted module names for different input combinations.
    """
    # Test function name only
    result = merge.get_module_name(None, 'function')
    assert result == 'function'
    
    # Test class name only
    result = merge.get_module_name('Class', None)
    assert result == 'Class'
    
    # Test class.method name
    result = merge.get_module_name('Class', 'method')
    assert result == 'Class.method'
    
    print("get_module_name function works correctly")


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=6)
def test_merge_time_tracker_decorator():
    """
    Feature: time_tracker decorator measures function execution time.
    Description: time_tracker decorator wraps functions to measure and log execution time.
    Expectation: Decorated function executes and timing information is logged.
    """
        
    # Test decorated function
    @merge.time_tracker
    def test_function():
        return "test_result"
    
    # Mock print to capture output
    with patch('builtins.print') as mock_print:
        result = test_function()
        
        # Check function result
        assert result == "test_result"
        
        # Check that timing information was printed
        assert mock_print.called
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("start test_function time:" in call for call in print_calls)
        assert any("finish test_function time:" in call for call in print_calls)
    
    print("time_tracker decorator works correctly")


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=7)
def test_merge_tik_tok_functions():
    """
    Feature: tik and tok functions manage timing stack.
    Description: tik starts timing and tok ends timing, managing a global timing stack.
    Expectation: Timing stack is correctly managed with start and end times.
    """ 
    # Reset global timing stack
    merge.START_TIMES = []
    
    # Mock print to capture output
    with patch('builtins.print') as mock_print:
        # Test tik function
        merge.tik("test_operation")
        assert len(merge.START_TIMES) == 1
        
        # Test tok function
        merge.tok("test_operation")
        assert len(merge.START_TIMES) == 0
        
        # Check that timing information was printed
        assert mock_print.called
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("start test_operation time:" in call for call in print_calls)
        assert any("finish test_operation time:" in call for call in print_calls)
    
    print("tik/tok functions work correctly")


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=8)
def test_merge_dump_json_function(mock_merge_environment):
    """
    Feature: dump_json_at_same_dir function saves data to JSON file.
    Description: dump_json_at_same_dir saves data to a JSON file in the same directory as the input file.
    Expectation: JSON file is created with correct data and naming convention.
    """
    root_dir, patch_json_path, original_file, patch_file, adaptor_file = mock_merge_environment

        
    # Test data
    test_data = {"test": "data", "number": 123}
    
    # Mock print to capture output
    with patch('builtins.print') as mock_print:
        # Test dump_json_at_same_dir function
        merge.dump_json_at_same_dir(patch_json_path, test_data, "test_suffix")
        
        # Check that file was created
        expected_file = Path(patch_json_path).parent / "test_patches_test_suffix.json"
        assert expected_file.exists()
        
        # Check file content
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            assert loaded_data == test_data
        
        # Check that success message was printed
        assert mock_print.called
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("test_suffix are dumped into" in call for call in print_calls)
    
    print("dump_json_at_same_dir function works correctly")



@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=9)
def test_merge_error_handling_invalid_json(mock_merge_environment):
    """
    Feature: merge function handles invalid JSON file gracefully.
    Description: merge function should handle cases where the patch JSON file is invalid or corrupted.
    Expectation: Appropriate error handling and logging for invalid JSON files.
    """
    root_dir, patch_json_path, original_file, patch_file, adaptor_file = mock_merge_environment
    
    # Create invalid JSON file
    invalid_json_path = Path(patch_json_path).parent / "invalid_patches.json"
    with open(invalid_json_path, 'w', encoding='utf-8') as f:
        f.write("{ invalid json content")
    
        
    # Test merge function with invalid JSON
    with pytest.raises(json.JSONDecodeError):
        merge.merge(root_dir, str(invalid_json_path), check=True)
    
    print("Error handling for invalid JSON works correctly")


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=10)
def test_merge_error_handling_missing_file(mock_merge_environment):
    """
    Feature: merge function handles missing patch file gracefully.
    Description: merge function should handle cases where the patch JSON file does not exist.
    Expectation: Appropriate error handling for missing files.
    """
    root_dir, patch_json_path, original_file, patch_file, adaptor_file = mock_merge_environment
    
    # Use non-existent file path
    missing_file_path = Path(patch_json_path).parent / "missing_patches.json"

        
    # Test merge function with missing file
    with pytest.raises(FileNotFoundError):
        merge.merge(root_dir, str(missing_file_path), check=True)
    
    print("Error handling for missing file works correctly")


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=11)
def test_merge_basic_functionality():
    """
    Feature: Basic merge functionality works without complex dependencies.
    Description: Test basic functions that don't require complex module imports.
    Expectation: Basic utility functions work correctly.
    """
        
    # Test basic utility functions
    assert merge.get_module_name(None, 'test_func') == 'test_func'
    assert merge.get_module_name('TestClass', None) == 'TestClass'
    assert merge.get_module_name('TestClass', 'test_method') == 'TestClass.test_method'
    
    # Test timing functions
    merge.START_TIMES = []
    merge.tik("test")
    assert len(merge.START_TIMES) == 1
    merge.tok("test")
    assert len(merge.START_TIMES) == 0
    
    print("Basic merge functionality works correctly")


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=12)
def test_parse_patch_infos_categorization(monkeypatch, tmp_path):
    """
    Feature: PatchMerger.parse_patch_infos categorizes patches correctly.
    Description: Ensure replacement, conditional func/class, and wrapper patches are routed to proper dicts.
    Expectation: patch_replace_info/patch_func_infos/patch_class_infos/patch_wrapper_infos populated accordingly.
    """
    # Build minimal fake file paths
    original_file = str(tmp_path / "Megatron-LM" / "megatron" / "pkg" / "mod.py")
    patch_file = str(tmp_path / "MindSpeed" / "mindspeed" / "pkg" / "mod.py")
    Path(original_file).parent.mkdir(parents=True, exist_ok=True)
    Path(patch_file).parent.mkdir(parents=True, exist_ok=True)
    Path(original_file).write_text("def function():\n    return 1\n", encoding="utf-8")
    Path(patch_file).write_text("def function():\n    return 2\n", encoding="utf-8")

    # Prepare raw patches covering four categories
    raw_patches = {
        "megatron.pkg.mod.function": [
            {"patch_import": "mindspeed.pkg.mod.function", "patch_name": "function", "condition": False}
        ],
        "megatron.pkg.ClassA": [
            {"patch_import": "mindspeed.pkg.ClassA", "patch_name": "ClassA", "condition": True}
        ],
        "megatron.pkg.mod.func2": [
            {"patch_import": "mindspeed.pkg.mod.func2", "patch_name": "func2", "condition": True}
        ],
        "megatron.pkg.mod.func3": [
            {"patch_import": "mindspeed.pkg.mod.func3_wrapper", "patch_name": "func3_wrapper", "condition": False}
        ],
    }

    monkeypatch.setattr(merge.PatchMerger, "__init__", light_init, raising=True)

    # Stub parse_path to map imports deterministically
    def fake_parse_path(source_packages, parent_module_path, module_name):
        # Determine if target is class or function by name capitalization or wrapper suffix
        full = f"{parent_module_path}.{module_name}" if module_name else parent_module_path
        if full.endswith("ClassA"):
            return ("mindspeed", patch_file, "ClassA", None)
        if full.endswith("func3_wrapper"):
            return ("mindspeed", patch_file, None, "func3_wrapper")
        # function cases (function, func2)
        if parent_module_path.startswith("megatron"):
            return ("megatron", original_file, None, module_name)
        return ("mindspeed", patch_file, None, module_name)

    monkeypatch.setattr(merge.PatchMerger, "parse_path", staticmethod(fake_parse_path), raising=True)

    pm = merge.PatchMerger(raw_patches, str(tmp_path))
    pm.parse_patch_infos()

    # Replacement
    assert len(pm.patch_replace_info) == 1
    assert original_file in pm.patch_replace_info
    assert "function" in pm.patch_replace_info[original_file]
    assert len(pm.patch_replace_info[original_file]["function"]) == 1

    # Conditional class
    assert len(pm.patch_class_infos) == 1

    # Conditional function
    assert len(pm.patch_func_infos) == 1

    # Wrapper
    assert len(pm.patch_wrapper_infos) == 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=13)
def test_parse_patch_infos_bad_parsed_case(monkeypatch, tmp_path):
    """
    Feature: PatchMerger.parse_patch_infos records bad parsed cases.
    Description: When parse_path raises, input patch is captured in bad_parsed_cases and processing continues.
    Expectation: bad_parsed_cases contains the problematic entry, other valid entries are categorized.
    """
    original_file = str(tmp_path / "Megatron-LM" / "megatron" / "pkg" / "mod.py")
    patch_file = str(tmp_path / "MindSpeed" / "mindspeed" / "pkg" / "mod.py")
    Path(original_file).parent.mkdir(parents=True, exist_ok=True)
    Path(patch_file).parent.mkdir(parents=True, exist_ok=True)
    Path(original_file).write_text("def ok():\n    return 1\n", encoding="utf-8")
    Path(patch_file).write_text("def ok():\n    return 2\n", encoding="utf-8")

    raw_patches = {
        "megatron.bad.missing.func": [
            {"patch_import": "mindspeed.bad.missing.func", "patch_name": "func", "condition": False}
        ],
        "megatron.pkg.mod.ok": [
            {"patch_import": "mindspeed.pkg.mod.ok", "patch_name": "ok", "condition": False}
        ],
    }

    monkeypatch.setattr(merge.PatchMerger, "__init__", light_init, raising=True)

    def fake_parse_path(source_packages, parent_module_path, module_name):
        full = f"{parent_module_path}.{module_name}" if module_name else parent_module_path
        if full.startswith("megatron.bad"):
            raise Exception("import failure")
        if parent_module_path.startswith("megatron"):
            return ("megatron", original_file, None, module_name)
        return ("mindspeed", patch_file, None, module_name)

    monkeypatch.setattr(merge.PatchMerger, "parse_path", staticmethod(fake_parse_path), raising=True)

    pm = merge.PatchMerger(raw_patches, str(tmp_path))
    pm.parse_patch_infos()

    # Bad parsed recorded
    assert "megatron.bad.missing.func" in pm.bad_parsed_cases
    assert len(pm.bad_parsed_cases["megatron.bad.missing.func"]) == 1

    # Valid entry categorized as replacement
    assert original_file in pm.patch_replace_info
    assert "ok" in pm.patch_replace_info[original_file]
    assert len(pm.patch_replace_info[original_file]["ok"]) == 1
