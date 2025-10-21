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
Test module for checkpointing.py from tools/load_ms_weights_to_pt/
Tests the load_wrapper decorator functionality
"""

import pytest
import sys
import os

# Add the tools directory to the path
tools_path = os.path.join(os.path.dirname(__file__), '../../../tools/load_ms_weights_to_pt')
sys.path.insert(0, tools_path)

from checkpointing import load_wrapper


class TestLoadWrapper:
    """Test cases for the load_wrapper decorator"""

    def test_load_wrapper_success(self):
        """Test load_wrapper when the original function succeeds"""
        # Create a mock function that succeeds
        @load_wrapper
        def mock_load_func(file_path):
            return {"data": "loaded successfully"}
        
        result = mock_load_func("test.pt")
        assert result == {"data": "loaded successfully"}

    def test_load_wrapper_with_kwargs(self):
        """Test load_wrapper with keyword arguments"""
        @load_wrapper
        def mock_load_func(file_path, map_location=None, weights_only=False):
            return {"file": file_path, "map_location": map_location, "weights_only": weights_only}
        
        result = mock_load_func("test.pt", map_location="cpu", weights_only=True)
        assert result["file"] == "test.pt"
        assert result["map_location"] == "cpu"
        assert result["weights_only"] is True

    def test_load_wrapper_preserves_function_metadata(self):
        """Test that load_wrapper preserves the original function's metadata"""
        @load_wrapper
        def documented_function(arg1, arg2):
            """This is a test function with documentation"""
            return arg1 + arg2
        
        # Check that the wrapper preserves the function name and docstring
        assert documented_function.__name__ == "documented_function"
        assert "This is a test function with documentation" in documented_function.__doc__

    def test_load_wrapper_with_multiple_args(self):
        """Test load_wrapper with multiple positional arguments"""
        @load_wrapper
        def mock_multi_arg_func(arg1, arg2, arg3):
            return arg1 + arg2 + arg3
        
        result = mock_multi_arg_func(1, 2, 3)
        assert result == 6

    def test_load_wrapper_with_mixed_args(self):
        """Test load_wrapper with mixed positional and keyword arguments"""
        @load_wrapper
        def mock_mixed_func(a, b, c=10, d=20):
            return a + b + c + d
        
        result = mock_mixed_func(1, 2, c=5)
        assert result == 28  # 1 + 2 + 5 + 20

    def test_load_wrapper_return_none(self):
        """Test load_wrapper when function returns None"""
        @load_wrapper
        def return_none_func():
            return None
        
        result = return_none_func()
        assert result is None

    def test_load_wrapper_with_complex_return_type(self):
        """Test load_wrapper with complex return types"""
        @load_wrapper
        def complex_return_func():
            return {
                "model": {"layer1": [1, 2, 3], "layer2": [4, 5, 6]},
                "optimizer": {"lr": 0.001, "momentum": 0.9},
                "epoch": 10
            }
        
        result = complex_return_func()
        assert isinstance(result, dict)
        assert "model" in result
        assert result["epoch"] == 10


class TestLoadWrapperIntegration:
    """Integration tests for load_wrapper"""

    def test_load_wrapper_with_file_operations(self, tmp_path):
        """Test load_wrapper with actual file path operations"""
        test_file = tmp_path / "test.pt"
        test_file.write_text("test data")
        
        @load_wrapper
        def read_file(path):
            with open(path, 'r') as f:
                return f.read()
        
        result = read_file(str(test_file))
        assert result == "test data"

    def test_load_wrapper_decorator_chain(self):
        """Test load_wrapper in a decorator chain"""
        def another_decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return f"decorated: {result}"
            return wrapper
        
        @another_decorator
        @load_wrapper
        def chained_function(value):
            return value
        
        result = chained_function("test")
        assert result == "decorated: test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

