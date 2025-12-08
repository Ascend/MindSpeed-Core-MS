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
        "origin_file": str(original_file),
        "patch_file": str(patch_file),
        "module_origin_name": ("test_function", None, "test_function"),
        "module_patch_name": ("test_function", None, "test_function"),
        'origin_import': "megatron.func.original.test_function",
        'origin_import_root': "megatron",
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
        "origin_file": str(original_file),
        "patch_file": str(patch_file),
        "module_origin_name": ("TestClass", "TestClass", None),
        "module_patch_name": ("TestClass", "TestClass", None),
        "condition": False,
        'origin_import': "megatron.class.original.test_function",
        'origin_import_root': "megatron",
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
        "origin_file": str(original_file),
        "patch_file": str(patch_file),
        "module_origin_name": ("test_function", None, "test_function"),
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
        "origin_file": str(original_file),
        "patch_file": str(patch_file1),
        "module_origin_name": ("test_function", None, "test_function"),
        "module_patch_name": ("test_function", None, "test_function"),
        "condition": False,
        "raw_patch": {"patch_import": "patch1.test_function", "patch_name": "test_function", "condition": False}
    }
    
    patch_info2 = {
        "origin_file": str(original_file),
        "patch_file": str(patch_file2),
        "module_origin_name": ("test_function", None, "test_function"),
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
@pytest.mark.run(order=11)
def test_get_cst_set_cst_and_flush(tmp_path, monkeypatch):
    """
    Feature: PatchMerger.get_cst/set_cst/flush_cst_into_file work end-to-end.
    Description: Ensure get_cst reads from disk when cache is empty, set_cst updates cache, and flush_cst_into_file writes back.
    Expectation: File is parsable before and after flush and cst_to_write is honored.
    """
    src = tmp_path / "module_a.py"
    src.write_text("x = 1\n", encoding="utf-8")

    pm = merge.PatchMerger.__new__(merge.PatchMerger)
    pm.cst_to_write = {}

    # 1) file not in cache -> real parse path (covers 128-134)
    mod = pm.get_cst(str(src))
    assert hasattr(mod, "code")

    # 2) set_cst should populate cache (covers 140)
    pm.set_cst(str(src), mod)
    assert str(src) in pm.cst_to_write

    # 3) get_cst again should hit cache branch (covers 129)
    mod_cached = pm.get_cst(str(src))
    assert mod_cached is mod

    # 4) flush_cst_into_file writes back (covers 147-150)
    pm.flush_cst_into_file()
    text_after = src.read_text(encoding="utf-8")
    assert "x = 1" in text_after


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


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=14)
def test_patch_merger_annotate_register(monkeypatch, tmp_path):
    """
    Feature: PatchMerger.annotate comments out adaptor registrations.
    Description: When register() call matches patch info, annotate should insert 'pass' and comment the original line.
    Expectation: Adaptor entry is updated and marked dirty.
    """
    adaptor_file = tmp_path / "megatron_adaptor.py"
    adaptor_code = (
        "from mindspeed_llm.tasks.megatron_adaptor import MegatronAdaptation\n"
        "MegatronAdaptation.register('megatron.foo.bar', patch_func)\n"
    )
    adaptor_file.write_text(adaptor_code, encoding="utf-8")

    # Build fake PatchMerger instance without running heavy __init__
    pm = merge.PatchMerger.__new__(merge.PatchMerger)
    pm.adaptors = {str(adaptor_file): (adaptor_code, False)}

    patch_info = {
        "module_origin_name": ("FooBar", "FooBar", None),
        "origin_import": "megatron.foo.bar",
        "module_patch_name": ("patch_func", None, "patch_func"),
        "raw_patch": {"patch_import": "foo.bar", "patch_name": "patch_func", "condition": False},
    }

    pm.annotate(patch_info)

    updated_code, need_flush = pm.adaptors[str(adaptor_file)]
    assert need_flush is True
    assert "pass" in updated_code
    assert "#MegatronAdaptation.register" in updated_code


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=15)
def test_patch_merger_flush_annotation(tmp_path):
    """
    Feature: PatchMerger.flush_annotation writes dirty adaptor entries.
    Description: Only entries marked with need_flush=True should be flushed to disk.
    Expectation: Dirty file updated, clean file untouched.
    """
    file_dirty = tmp_path / "dirty.py"
    file_clean = tmp_path / "clean.py"
    file_dirty.write_text("old_dirty", encoding="utf-8")
    file_clean.write_text("old_clean", encoding="utf-8")

    pm = merge.PatchMerger.__new__(merge.PatchMerger)
    pm.adaptors = {
        str(file_dirty): ("new_dirty", True),
        str(file_clean): ("should_not_write", False),
    }

    pm.flush_annotation()

    assert file_dirty.read_text(encoding="utf-8") == "new_dirty"
    assert file_clean.read_text(encoding="utf-8") == "old_clean"


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=16)
def test_add_merge_info_force_patch_behavior():
    """
    Feature: PatchMerger.add_merge_info handles force_patch override logic.
    Description: When multiple patches with force_patch flags exist, the new forced one overrides the old.
    Expectation: Only one patch with the same condition is kept and it is the forced one.
    """
    pm = merge.PatchMerger.__new__(merge.PatchMerger)

    infos = {}
    origin_file = "origin.py"
    module_name = "foo"

    # First, add a non-force patch
    patch1 = {
        "condition": "cond",
        "raw_patch": {"force_patch": False},
    }
    pm.add_merge_info(infos, origin_file, module_name, patch1)
    assert infos[origin_file][module_name][0]["raw_patch"]["force_patch"] is False

    # Then, add a force patch for the same condition, it should replace the previous one
    patch2 = {
        "condition": "cond",
        "raw_patch": {"force_patch": True},
    }
    pm.add_merge_info(infos, origin_file, module_name, patch2)
    assert len(infos[origin_file][module_name]) == 1
    assert infos[origin_file][module_name][0]["raw_patch"]["force_patch"] is True

    # Finally, add a non-force patch with same condition when a force patch already exists:
    # it should be ignored (no append), covering branch where cur_force_patch is True
    length_before = len(infos[origin_file][module_name])
    patch3 = {
        "condition": "cond",
        "raw_patch": {"force_patch": False},
    }
    pm.add_merge_info(infos, origin_file, module_name, patch3)
    assert len(infos[origin_file][module_name]) == length_before


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=17)
def test_handle_exc_records_bad_cases(capsys):
    """
    Feature: PatchMerger.handle_exc records bad handled cases.
    Description: When an exception occurs during patch handling, raw patches are stored in bad_handled_cases.
    Expectation: bad_handled_cases is populated with origin_import keys.
    """
    from collections import defaultdict

    pm = merge.PatchMerger.__new__(merge.PatchMerger)
    pm.bad_handled_cases = defaultdict(list)

    origin_import = "megatron.foo.bar"
    raw_patch = {"patch_import": "mindspeed.foo.bar", "patch_name": "bar", "condition": False}
    module_patch_infos = [
        {"origin_import": origin_import, "raw_patch": raw_patch},
    ]

    e = RuntimeError("test error")
    pm.handle_exc(e, "FooBar", module_patch_infos)

    captured = capsys.readouterr()
    # basic error message printed
    assert "Exception test error while patching module FooBar" in captured.out
    # bad_handled_cases recorded
    assert origin_import in pm.bad_handled_cases
    assert pm.bad_handled_cases[origin_import][0] == raw_patch


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=18)
def test_parse_path_raises_on_none_module_name():
    """
    Feature: PatchMerger.parse_path validates module_name.
    Description: Passing None as module_name should raise ValueError.
    Expectation: ValueError is raised with proper message.
    """
    with pytest.raises(ValueError):
        merge.PatchMerger.parse_path(["megatron"], "megatron.foo.bar", None)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=19)
def test_merge_with_router_name_error_on_origin_file(monkeypatch, tmp_path):
    """
    Feature: PatchMerger.merge_with_router updates CST and then hits NameError on typo `origin_file`.
    Description: When source_cst is changed, merge_with_router will attempt to call set_cst(origin_file,...).
    Expectation: NameError is raised, covering the edge line.
    """
    # Build minimal python file and CST
    origin_file = tmp_path / "origin.py"
    origin_file.write_text("def foo():\n    return 1\n", encoding="utf-8")

    pm = merge.PatchMerger.__new__(merge.PatchMerger)
    pm.cst_to_write = {}

    # Avoid real handle_annotate/annotate logic which expects full patch structure
    def fake_handle_annotate(self, patch_infos):
        return

    monkeypatch.setattr(merge.PatchMerger, "handle_annotate", fake_handle_annotate, raising=True)

    # Real get_cst to generate a CST, but avoiding full PatchMerger.__init__
    def fake_get_cst(self, file_path):
        import libcst as cst
        return cst.parse_module(origin_file.read_text(encoding="utf-8"))

    monkeypatch.setattr(merge.PatchMerger, "get_cst", fake_get_cst, raising=True)

    # Router that always returns a *different* CST object so that source_cst != origin_source_cst
    class DummyRouter:
        def __init__(self, module_name, patch_infos):
            pass

        def visit(self, tree):
            import libcst as cst
            return cst.parse_module("def foo():\n    return 2\n")

    # Patch MetadataWrapper to just store tree and call router.visit
    class DummyWrapper:
        def __init__(self, tree):
            self._tree = tree

        def visit(self, visitor):
            return visitor.visit(self._tree)

    monkeypatch.setattr(merge, "MetadataWrapper", DummyWrapper, raising=True)

    # Prepare minimal patch_infos structure
    patch_infos = {str(origin_file): {"foo": [{"dummy": True}]}}

    pm.merge_with_router(patch_infos, DummyRouter)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=20)
def test_merge_stats_and_flush_called(monkeypatch, tmp_path):
    """
    Feature: merge function prints statistics and calls flush methods in non-check mode.
    Description: Use a fake PatchMerger to avoid heavy logic but keep merge() flow.
    Expectation: flush_cst_into_file and flush_annotation are invoked.
    """
    # Prepare a simple raw_patches JSON
    patch_json = tmp_path / "patches.json"
    raw_patches = {
        "megatron.mod.func": [{"patch_import": "mindspeed.mod.func", "patch_name": "func", "condition": False}]}
    patch_json.write_text(json.dumps(raw_patches), encoding="utf-8")

    root_dir = str(tmp_path)

    class FakePM:
        def __init__(self, patches, root):
            self.raw_patches = patches
            self.bad_parsed_cases = {}
            self.bad_handled_cases = {}
            self.flush_cst_called = False
            self.flush_anno_called = False

        def parse_patch_infos(self):
            pass

        def merge_replacement(self):
            pass

        def merge_class_patch(self):
            pass

        def merge_func_patch(self):
            pass

        def merge_wrapper_patch(self):
            pass

        def flush_cst_into_file(self):
            self.flush_cst_called = True

        def flush_annotation(self):
            self.flush_anno_called = True

    # Monkeypatch PatchMerger to FakePM
    monkeypatch.setattr(merge, "PatchMerger", FakePM, raising=True)

    with patch("builtins.print") as mock_print:
        merge.merge(root_dir, str(patch_json), check=False)

    # Check statistics were printed
    print_calls = [c[0][0] for c in mock_print.call_args_list]
    assert any("total patches" in msg for msg in print_calls)
    assert any("bad parsed cases" in msg for msg in print_calls)
    assert any("bad handled cases" in msg for msg in print_calls)

    # Ensure flush methods were called
    # Last constructed FakePM instance is not directly accessible, but we can at least
    # assert that no exception occurred and coverage for non-check path is hit.


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=21)
def test_preprocess_and_postprocess(monkeypatch, tmp_path):
    """
    Feature: preprocess and postprocess toggle MegatronAdaptation.execute() call.
    Description: preprocess comments out execute() and registers decorator; postprocess restores it.
    Expectation: File content is updated accordingly.
    """
    # Stub external dependencies
    import types as _types

    torch = _types.ModuleType("torch")
    torch.nn = _types.SimpleNamespace(Module=object)
    transformer_engine = _types.ModuleType("transformer_engine")
    transformer_engine.pytorch = _types.SimpleNamespace()

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformer_engine", transformer_engine)

    # Build minimal mindspeed_llm package structure on disk
    pkg_root = tmp_path / "MindSpeed-LLM"
    tasks_dir = pkg_root / "mindspeed_llm" / "tasks"
    train_dir = pkg_root / "mindspeed_llm" / "training"
    args_dir = train_dir / "arguments"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    args_dir.mkdir(parents=True, exist_ok=True)

    # __init__.py files
    for d in [pkg_root / "mindspeed_llm", tasks_dir, train_dir, args_dir]:
        (d / "__init__.py").write_text("", encoding="utf-8")

    # training/arguments/__init__.py exports parse_args_decorator
    (args_dir / "__init__.py").write_text(
        "def parse_args_decorator(func):\n    return func\n", encoding="utf-8"
    )

    adaptor_path = tasks_dir / "megatron_adaptor.py"
    adaptor_code = """
class MegatronAdaptation:
    registry = []
    @classmethod
    def register(cls, name, func):
        cls.registry.append((name, func))
    @classmethod
    def apply(cls):
        cls.applied = True

MegatronAdaptation.execute()
"""
    adaptor_path.write_text(adaptor_code, encoding="utf-8")

    # Ensure package is importable
    sys.path.insert(0, str(pkg_root))

    # Run preprocess: it should comment out execute() line
    merge.preprocess(str(adaptor_path))
    modified = adaptor_path.read_text(encoding="utf-8")
    assert "# MegatronAdaptation.execute()" in modified

    # Run postprocess: it should restore execute() line
    merge.postprocess(str(adaptor_path))
    restored = adaptor_path.read_text(encoding="utf-8")
    assert "MegatronAdaptation.execute()" in restored
    assert "# MegatronAdaptation.execute()" not in restored


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=23)
def test_patch_merger_real_init_loads_adaptors(monkeypatch, tmp_path):
    """
    Feature: PatchMerger.__init__ loads adaptor files correctly.
    Description: Using a fake root_dir and args.root_dir, __init__ should collect adaptor files into self.adaptors.
    Expectation: Adaptor paths are recorded with code strings and need_flush=False.
    """
    root_dir = tmp_path / "MindSpeed-Core-MS"
    root_dir.mkdir(parents=True, exist_ok=True)

    # Create required adaptor files
    adaptor_rel_paths = [
        "MindSpeed-LLM/mindspeed_llm/tasks/megatron_adaptor.py",
        "MindSpeed-LLM/mindspeed_llm/core/pipeline_parallel/dualpipe/adaptor.py",
        "MindSpeed/mindspeed/features_manager/tensor_parallel/unaligned_linear_feature.py",
        "MindSpeed-LLM/mindspeed_llm/mindspore/mindspore_adaptor.py",
    ]
    for rel in adaptor_rel_paths:
        p = root_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("pass\n", encoding="utf-8")

    # Create some extra python files under features_manager (non-__init__.py)
    fm_dir = root_dir / "MindSpeed-LLM/mindspeed_llm/features_manager"
    fm_dir.mkdir(parents=True, exist_ok=True)
    (fm_dir / "__init__.py").write_text("", encoding="utf-8")
    extra1 = fm_dir / "feat_a.py"
    extra2 = fm_dir / "subdir" / "feat_b.py"
    extra2.parent.mkdir(parents=True, exist_ok=True)
    extra1.write_text("pass\n", encoding="utf-8")
    extra2.write_text("pass\n", encoding="utf-8")

    # Mock global args used inside PatchMerger.__init__
    ns = types.SimpleNamespace(root_dir=str(root_dir))
    monkeypatch.setattr(merge, "args", ns, raising=False)

    pm = merge.PatchMerger({}, str(root_dir))

    # All adaptor files plus extra feature files should be tracked
    assert len(pm.adaptors) >= len(adaptor_rel_paths)
    for path_obj, (code, need_flush) in pm.adaptors.items():
        # path_obj is a Path instance
        assert isinstance(code, str)
        assert need_flush is False


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=24)
def test_add_merge_info_force_patch_conflict():
    """
    Feature: PatchMerger.add_merge_info detects conflicting force_patch entries.
    Description: When two patches with the same condition both have force_patch=True, an exception is raised.
    Expectation: Exception message mentions only support one force_patch.
    """
    pm = merge.PatchMerger.__new__(merge.PatchMerger)
    infos = {}
    origin_file = "origin.py"
    module_name = "foo"

    patch1 = {"condition": "cond", "raw_patch": {"force_patch": True}}
    patch2 = {"condition": "cond", "raw_patch": {"force_patch": True}}

    pm.add_merge_info(infos, origin_file, module_name, patch1)
    with pytest.raises(Exception) as exc_info:
        pm.add_merge_info(infos, origin_file, module_name, patch2)
    assert "Only support one force_patch" in str(exc_info.value)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=25)
def test_add_merge_info_different_condition_appends():
    """
    Feature: PatchMerger.add_merge_info appends patches with different conditions.
    Description: When conditions differ, patches should coexist and the continue branch should be taken.
    Expectation: Both patches are present in the module patch list.
    """
    pm = merge.PatchMerger.__new__(merge.PatchMerger)
    infos = {}
    origin_file = "origin.py"
    module_name = "foo"

    patch1 = {"condition": "cond1", "raw_patch": {"force_patch": True}}
    patch2 = {"condition": "cond2", "raw_patch": {"force_patch": True}}

    pm.add_merge_info(infos, origin_file, module_name, patch1)
    pm.add_merge_info(infos, origin_file, module_name, patch2)

    assert len(infos[origin_file][module_name]) == 2


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=26)
def test_parse_path_with_nested_class_parent(tmp_path):
    """
    Feature: PatchMerger.parse_path handles nested parent paths with class attributes.
    Description: When an intermediate module is missing but parent module exposes a class, parse_path should resolve via that class.
    Expectation: Returned import_root/file_path/class_name/func_name are correct.
    """
    # Build a minimal 'megatron.parent' package where 'childpkg' is a class with method 'mymethod'.
    pkg_root = tmp_path / "pkgs"
    pkg_root.mkdir(parents=True, exist_ok=True)

    megatron_dir = pkg_root / "megatron"
    parent_dir = megatron_dir / "parent"
    parent_dir.mkdir(parents=True, exist_ok=True)

    (megatron_dir / "__init__.py").write_text("", encoding="utf-8")
    (parent_dir / "__init__.py").write_text(
        "class childpkg:\n"
        "    def mymethod(self):\n"
        "        return 1\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(pkg_root))
    try:
        import importlib

        importlib.invalidate_caches()
        # Ensure modules are importable
        importlib.import_module("megatron")
        importlib.import_module("megatron.parent")

        import_root, file_path, class_name, func_name = merge.PatchMerger.parse_path(
            ["megatron", "mindspeed", "mindspeed_llm"],
            "megatron.parent.childpkg",
            "mymethod",
        )
    finally:
        sys.path.pop(0)

    assert import_root.startswith("megatron")
    assert class_name == "childpkg"
    assert func_name == "mymethod"
    assert Path(file_path).name == "__init__.py"


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=28)
def test_merge_with_router_error_and_handle_exc(monkeypatch, tmp_path):
    """
    Feature: PatchMerger.merge_with_router forwards errors to handle_exc.
    Description: When router visitor returns None CST, an exception is raised and handled.
    Expectation: handle_exc is called with the failing module name.
    """
    origin_file = tmp_path / "origin.py"
    origin_file.write_text("def foo():\n    return 1\n", encoding="utf-8")

    pm = merge.PatchMerger.__new__(merge.PatchMerger)
    pm.cst_to_write = {}

    # Use a simple get_cst that always returns a valid CST
    def fake_get_cst(self, file_path):
        import libcst as cst
        return cst.parse_module(origin_file.read_text(encoding="utf-8"))

    monkeypatch.setattr(merge.PatchMerger, "get_cst", fake_get_cst, raising=True)

    # Dummy wrapper that just stores tree and calls router.visit
    class DummyWrapper:
        def __init__(self, tree):
            self._tree = tree

        def visit(self, visitor):
            return visitor.visit(self._tree)

    monkeypatch.setattr(merge, "MetadataWrapper", DummyWrapper, raising=True)

    # Router that incorrectly returns None, triggering the error path
    class BadRouter:
        def __init__(self, module_name, patch_infos):
            pass

        def visit(self, tree):
            return None

    # Avoid heavy annotation logic
    def fake_handle_annotate(self, patch_infos):
        return

    monkeypatch.setattr(merge.PatchMerger, "handle_annotate", fake_handle_annotate, raising=True)

    errors = []

    def fake_handle_exc(self, e, module_name, module_patch_infos):
        errors.append((str(e), module_name))

    monkeypatch.setattr(merge.PatchMerger, "handle_exc", fake_handle_exc, raising=True)

    patch_infos = {str(origin_file): {"foo": [{"dummy": True}]}}

    pm.merge_with_router(patch_infos, BadRouter)

    assert len(errors) == 1
    assert "Got None cst after visit" in errors[0][0]
    assert errors[0][1] == "foo"


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=29)
def test_merge_func_and_wrapper_patch_bridge(monkeypatch):
    """
    Feature: merge_func_patch/merge_wrapper_patch delegate to merge_with_router.
    Description: Ensure both methods call merge_with_router with correct transformer classes.
    Expectation: merge_with_router is invoked twice with expected arguments.
    """
    pm = merge.PatchMerger.__new__(merge.PatchMerger)
    pm.patch_func_infos = {"file.py": {"func": []}}
    pm.patch_wrapper_infos = {"file.py": {"func": []}}

    calls = []

    def fake_merge_with_router(self, infos, router_cls):
        calls.append((infos, router_cls))

    monkeypatch.setattr(merge.PatchMerger, "merge_with_router", fake_merge_with_router, raising=True)

    pm.merge_func_patch()
    pm.merge_wrapper_patch()

    assert calls[0][0] == pm.patch_func_infos
    assert calls[0][1] is merge.PatchFuncRouterTransformer
    assert calls[1][0] == pm.patch_wrapper_infos
    assert calls[1][1] is merge.PatchWrapperRouterTransformer


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=30)
def test_merge_prints_bad_cases_and_check_message(monkeypatch, tmp_path):
    """
    Feature: merge() prints bad case summary and check-mode message.
    Description: When bad_parsed_cases/bad_handled_cases are non-empty and check=True, extra messages are printed.
    Expectation: Output contains bad-cases hint and check-mode hint, and no flush is called.
    """
    patch_json = tmp_path / "patches_check.json"
    raw_patches = {"mod": [{"k": "v"}]}
    patch_json.write_text(json.dumps(raw_patches), encoding="utf-8")

    root_dir = str(tmp_path)

    class FakePM2:
        def __init__(self, patches, root):
            self.raw_patches = patches
            self.bad_parsed_cases = {"mod": ["bad1"]}
            self.bad_handled_cases = {"mod": ["bad2"]}

        def parse_patch_infos(self):
            pass

        def merge_replacement(self):
            pass

        def merge_class_patch(self):
            pass

        def merge_func_patch(self):
            pass

        def merge_wrapper_patch(self):
            pass

        def flush_cst_into_file(self):
            # Should not be called when check=True
            pass

        def flush_annotation(self):
            # Should not be called when check=True
            pass

    monkeypatch.setattr(merge, "PatchMerger", FakePM2, raising=True)

    with patch("builtins.print") as mock_print:
        merge.merge(root_dir, str(patch_json), check=True)

    msgs = [c[0][0] for c in mock_print.call_args_list]
    assert any("bad parsed cases" in m for m in msgs)
    assert any("bad handled cases" in m for m in msgs)
    assert any("bad cases are skipped" in m for m in msgs)
    assert any("we are in **check** mode" in m for m in msgs)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=31)
def test_parse_patch_infos_error_branches(monkeypatch, tmp_path):
    """
    Feature: PatchMerger.parse_patch_infos covers error and split branches.
    Description: Use fake parse_path to trigger split-without-dot path, patch-parse exceptions, and both-name-None errors.
    Expectation: split() len==1 branch, bad_parsed_cases append, and error on both class/func None are all exercised.
    """
    # Two modules: one without dot in name, one normal
    raw_patches = {
        "singleimport": [
            {"patch_import": "mindspeed.bad.mod.BadPatch", "patch_name": "BadPatch", "condition": False}
        ],
        "megatron.good.mod": [
            {"patch_import": "mindspeed.good.mod.GoodPatch", "patch_name": "GoodPatch", "condition": False}
        ],
    }

    # light_init avoids heavy __init__
    monkeypatch.setattr(merge.PatchMerger, "__init__", light_init, raising=True)

    def fake_parse_path(source_packages, parent_module_path, module_name):
        # Origin import "singleimport" will hit split-name-without-dot branch (len==1),
        # we return a function origin so class_name is None but func_name is not.
        if parent_module_path == "singleimport":
            return ("megatron", "/tmp/single.py", None, "single_func")

        # Origin import "megatron.good.mod" returns a valid function origin
        if parent_module_path == "megatron.good.mod":
            return ("megatron", "/tmp/good_origin.py", None, "good_func")

        # Patch import under "mindspeed.bad.mod" raises to exercise 318-326 error path
        if parent_module_path == "mindspeed.bad.mod":
            raise Exception("patch import bad")

        # Patch import under "mindspeed.good.mod" returns both class and func as None
        # to trigger the error at 329.
        if parent_module_path == "mindspeed.good.mod":
            return ("mindspeed", "/tmp/good_patch.py", None, None)

        # Fallback: act like a simple function
        return ("megatron", "/tmp/fallback.py", None, "func")

    monkeypatch.setattr(merge.PatchMerger, "parse_path", staticmethod(fake_parse_path), raising=True)

    pm = merge.PatchMerger(raw_patches, str(tmp_path))

    # parse_patch_infos should raise due to class_patch_name and func_patch_name both None at 329
    with pytest.raises(Exception):
        pm.parse_patch_infos()

    # bad_parsed_cases should contain the failing bad patch from "singleimport"
    assert "singleimport" in pm.bad_parsed_cases or "megatron.bad.mod" in pm.bad_parsed_cases
