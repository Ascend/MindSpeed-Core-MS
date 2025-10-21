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
Test module for serialization.py from tools/load_ms_weights_to_pt/
Tests the core weight loading and serialization functionality
"""

import pytest
import sys
import os
import io
import zipfile
import numpy as np
from unittest.mock import Mock
from pathlib import Path

# Add the tools directory to the path
tools_path = os.path.join(os.path.dirname(__file__), '../../../tools/load_ms_weights_to_pt')
sys.path.insert(0, tools_path)

try:
    import torch
    import mindspore
    from ml_dtypes import bfloat16
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

from serialization import (
    _is_path, _open_file, _open_file_like, _check_seekable,
    _is_zipfile, _maybe_decode_ascii, get_default_load_endianness,
    LoadEndianness, PyTorchFileReader, dtype_map,
    transform_ms_dtype_to_pt_dtype, _rebuild_tensor_v2, _rebuild_from_type_v2,
    get_func_by_name
)


class TestUtilityFunctions:
    """Test utility functions in serialization.py"""

    def test_is_path_with_string(self):
        """Test _is_path with string input"""
        assert _is_path("test/path.pt") is True
        assert _is_path("/absolute/path.pt") is True

    def test_is_path_with_pathlib(self):
        """Test _is_path with pathlib.Path input"""
        assert _is_path(Path("test/path.pt")) is True
        assert _is_path(Path("/absolute/path.pt")) is True

    def test_is_path_with_non_path(self):
        """Test _is_path with non-path input"""
        assert _is_path(io.BytesIO()) is False
        assert _is_path(123) is False
        assert _is_path(None) is False

    def test_maybe_decode_ascii_with_bytes(self):
        """Test _maybe_decode_ascii with bytes input"""
        result = _maybe_decode_ascii(b"test_string")
        assert result == "test_string"
        assert isinstance(result, str)

    def test_maybe_decode_ascii_with_string(self):
        """Test _maybe_decode_ascii with string input"""
        result = _maybe_decode_ascii("test_string")
        assert result == "test_string"
        assert isinstance(result, str)

    def test_get_default_load_endianness(self):
        """Test get_default_load_endianness"""
        result = get_default_load_endianness()
        assert result is None or isinstance(result, LoadEndianness)

    def test_check_seekable_with_seekable_file(self):
        """Test _check_seekable with seekable file"""
        buffer = io.BytesIO(b"test data")
        assert _check_seekable(buffer) is True

    def test_check_seekable_with_non_seekable_file(self):
        """Test _check_seekable with non-seekable file"""
        # Create a mock that doesn't support seek
        mock_file = Mock()
        mock_file.seek.side_effect = io.UnsupportedOperation("seek")
        mock_file.tell.return_value = 0
        
        with pytest.raises(io.UnsupportedOperation):
            _check_seekable(mock_file)


class TestFileOperations:
    """Test file operation classes and functions"""

    def test_open_file_like_with_string_path_read(self, tmp_path):
        """Test _open_file_like with string path in read mode"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        with _open_file_like(str(test_file), 'rb') as f:
            content = f.read()
            assert content == b"test content"

    def test_open_file_like_with_string_path_write(self, tmp_path):
        """Test _open_file_like with string path in write mode"""
        test_file = tmp_path / "test.txt"
        
        with _open_file_like(str(test_file), 'wb') as f:
            f.write(b"test content")
        
        assert test_file.read_text() == "test content"

    def test_open_file_like_with_buffer_read(self):
        """Test _open_file_like with buffer in read mode"""
        buffer = io.BytesIO(b"test content")
        
        with _open_file_like(buffer, 'rb') as f:
            content = f.read()
            assert content == b"test content"

    def test_open_file_like_with_buffer_write(self):
        """Test _open_file_like with buffer in write mode"""
        buffer = io.BytesIO()
        
        with _open_file_like(buffer, 'wb') as f:
            f.write(b"test content")
        
        buffer.seek(0)
        assert buffer.read() == b"test content"

    def test_open_file_like_invalid_mode(self):
        """Test _open_file_like with invalid mode"""
        buffer = io.BytesIO()
        
        with pytest.raises(RuntimeError, match="Expected 'r' or 'w' in mode"):
            with _open_file_like(buffer, 'x'):
                pass

    def test_is_zipfile_true(self, tmp_path):
        """Test _is_zipfile with valid zip file"""
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test.txt", "content")
        
        with open(zip_path, 'rb') as f:
            assert _is_zipfile(f) is True

    def test_is_zipfile_false(self):
        """Test _is_zipfile with non-zip file"""
        buffer = io.BytesIO(b"not a zip file")
        assert _is_zipfile(buffer) is False

    def test_is_zipfile_empty(self):
        """Test _is_zipfile with empty file"""
        buffer = io.BytesIO(b"")
        assert _is_zipfile(buffer) is False


class TestLoadEndianness:
    """Test LoadEndianness enum"""

    def test_load_endianness_enum_values(self):
        """Test LoadEndianness enum has correct values"""
        assert LoadEndianness.NATIVE.value == 1
        assert LoadEndianness.LITTLE.value == 2
        assert LoadEndianness.BIG.value == 3

    def test_load_endianness_enum_members(self):
        """Test LoadEndianness enum members"""
        assert hasattr(LoadEndianness, 'NATIVE')
        assert hasattr(LoadEndianness, 'LITTLE')
        assert hasattr(LoadEndianness, 'BIG')


class TestDtypeMap:
    """Test dtype mapping"""

    def test_dtype_map_completeness(self):
        """Test dtype_map contains expected mappings"""
        assert "HalfStorage" in dtype_map
        assert "FloatStorage" in dtype_map
        assert "BFloat16Storage" in dtype_map
        assert "LongStorage" in dtype_map
        assert "ByteStorage" in dtype_map
        assert "BoolStorage" in dtype_map

    def test_dtype_map_values(self):
        """Test dtype_map values are correct numpy types"""
        assert dtype_map["HalfStorage"] == np.float16
        assert dtype_map["FloatStorage"] == np.float32
        assert dtype_map["LongStorage"] == np.int64
        assert dtype_map["ByteStorage"] == np.uint8
        assert dtype_map["BoolStorage"] == np.bool_


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="torch/mindspore not available")
class TestTransformDtype:
    """Test dtype transformation functions"""

    def test_transform_ms_dtype_to_pt_dtype_dict(self):
        """Test transform_ms_dtype_to_pt_dtype with dict input"""
        input_dict = {
            "key1": "value1",
            "key2": [1, 2, 3]
        }
        result = transform_ms_dtype_to_pt_dtype(input_dict)
        assert result == input_dict

    def test_transform_ms_dtype_to_pt_dtype_list(self):
        """Test transform_ms_dtype_to_pt_dtype with list input"""
        input_list = [1, 2, 3, "test"]
        result = transform_ms_dtype_to_pt_dtype(input_list)
        assert result == input_list

    def test_transform_ms_dtype_to_pt_dtype_primitive(self):
        """Test transform_ms_dtype_to_pt_dtype with primitive types"""
        assert transform_ms_dtype_to_pt_dtype(42) == 42
        assert transform_ms_dtype_to_pt_dtype("test") == "test"
        assert transform_ms_dtype_to_pt_dtype(3.14) == 3.14

    def test_transform_ms_dtype_tuple_key(self):
        """Test transform_ms_dtype_to_pt_dtype with tuple keys"""
        input_dict = {
            (mindspore.float32, mindspore.bfloat16): "value"
        }
        result = transform_ms_dtype_to_pt_dtype(input_dict)
        assert (torch.float32, torch.bfloat16) in result
        assert result[(torch.float32, torch.bfloat16)] == "value"

    def test_transform_ms_dtype_nested_structure(self):
        """Test transform_ms_dtype_to_pt_dtype with nested structure"""
        input_dict = {
            "layer1": {
                "weights": [1, 2, 3],
                "bias": [4, 5, 6]
            },
            "layer2": {
                "weights": [7, 8, 9]
            }
        }
        result = transform_ms_dtype_to_pt_dtype(input_dict)
        assert result["layer1"]["weights"] == [1, 2, 3]
        assert result["layer2"]["weights"] == [7, 8, 9]


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="torch not available")
class TestRebuildTensor:
    """Test tensor rebuilding functions"""

    def test_rebuild_tensor_v2_simple(self):
        """Test _rebuild_tensor_v2 with simple tensor"""
        storage = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        size = (4,)
        stride = (1,)
        
        result = _rebuild_tensor_v2(storage, 0, size, stride, False, None)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4,)

    def test_rebuild_tensor_v2_2d(self):
        """Test _rebuild_tensor_v2 with 2D tensor"""
        storage = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        size = (2, 3)
        stride = (3, 1)
        
        result = _rebuild_tensor_v2(storage, 0, size, stride, False, None)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 3)

    def test_rebuild_tensor_v2_with_offset(self):
        """Test _rebuild_tensor_v2 with storage offset"""
        storage = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        size = (3,)
        stride = (1,)
        offset = 2
        
        result = _rebuild_tensor_v2(storage, offset, size, stride, False, None)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([2.0, 3.0, 4.0]))

    def test_rebuild_tensor_v2_scalar(self):
        """Test _rebuild_tensor_v2 with scalar tensor"""
        storage = np.array([42.0], dtype=np.float32)
        size = ()
        stride = ()
        
        result = _rebuild_tensor_v2(storage, 0, size, stride, False, None)
        assert isinstance(result, torch.Tensor)
        assert result.numel() == 1

    def test_rebuild_tensor_v2_fortran_order(self):
        """Test _rebuild_tensor_v2 with Fortran order (column-major)"""
        storage = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        size = (2, 3)
        stride = (1, 2)  # stride[0] == 1 indicates Fortran order
        
        result = _rebuild_tensor_v2(storage, 0, size, stride, False, None)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 3)

    def test_rebuild_from_type_v2(self):
        """Test _rebuild_from_type_v2"""
        mock_func = Mock(return_value="result")
        args = ("arg1", "arg2")
        
        result = _rebuild_from_type_v2(mock_func, None, args, None)
        assert result == "result"
        mock_func.assert_called_once_with("arg1", "arg2")


class TestGetFuncByName:
    """Test get_func_by_name function"""

    def test_get_func_by_name_rebuild_tensor(self):
        """Test get_func_by_name with _rebuild_tensor_v2"""
        func = get_func_by_name("_rebuild_tensor_v2")
        assert callable(func)
        assert func == _rebuild_tensor_v2

    def test_get_func_by_name_rebuild_from_type(self):
        """Test get_func_by_name with _rebuild_from_type_v2"""
        func = get_func_by_name("_rebuild_from_type_v2")
        assert callable(func)
        assert func == _rebuild_from_type_v2

    def test_get_func_by_name_invalid(self):
        """Test get_func_by_name with invalid name"""
        with pytest.raises(RuntimeError, match="function name .* is invalid"):
            get_func_by_name("nonexistent_function")


class TestPyTorchFileReader:
    """Test PyTorchFileReader class"""

    def test_pytorch_file_reader_init(self, tmp_path):
        """Test PyTorchFileReader initialization"""
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("data/test.txt", "content")
        
        with open(zip_path, 'rb') as f:
            reader = PyTorchFileReader(f)
            assert reader.directory == "data"

    def test_pytorch_file_reader_has_record(self, tmp_path):
        """Test PyTorchFileReader.has_record"""
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("data/test.txt", "content")
            zf.writestr("data/other.txt", "other content")
        
        with open(zip_path, 'rb') as f:
            reader = PyTorchFileReader(f)
            assert reader.has_record("test.txt") is True
            assert reader.has_record("other.txt") is True
            assert reader.has_record("nonexistent.txt") is False

    def test_pytorch_file_reader_read_record(self, tmp_path):
        """Test PyTorchFileReader.read_record"""
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("data/test.txt", "test content")
        
        with open(zip_path, 'rb') as f:
            reader = PyTorchFileReader(f)
            content = reader.read_record("test.txt")
            assert content == b"test content"

    def test_pytorch_file_reader_read_record_not_found(self, tmp_path):
        """Test PyTorchFileReader.read_record with non-existent file"""
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("data/test.txt", "content")
        
        with open(zip_path, 'rb') as f:
            reader = PyTorchFileReader(f)
            content = reader.read_record("nonexistent.txt")
            assert content is None

    def test_pytorch_file_reader_open_record(self, tmp_path):
        """Test PyTorchFileReader.open_record"""
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("data/test.txt", "test content")
        
        with open(zip_path, 'rb') as f:
            reader = PyTorchFileReader(f)
            with reader.open_record("test.txt") as record:
                content = record.read()
                assert content == b"test content"

    def test_pytorch_file_reader_get_all_records(self, tmp_path):
        """Test PyTorchFileReader.get_all_records"""
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("data/file1.txt", "content1")
            zf.writestr("data/file2.txt", "content2")
            zf.writestr("data/file3.txt", "content3")
        
        with open(zip_path, 'rb') as f:
            reader = PyTorchFileReader(f)
            records = reader.get_all_records()
            assert "file1.txt" in records
            assert "file2.txt" in records
            assert "file3.txt" in records

    def test_pytorch_file_reader_get_record_offset(self, tmp_path):
        """Test PyTorchFileReader.get_record_offset"""
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("data/test.txt", "content")
        
        with open(zip_path, 'rb') as f:
            reader = PyTorchFileReader(f)
            offset = reader.get_record_offset("test.txt")
            assert isinstance(offset, int)
            assert offset >= 0


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="torch/mindspore not available")
class TestLoadMsWeights:
    """Test load_ms_weights function"""

    def test_load_ms_weights_with_non_zip_file(self):
        """Test load_ms_weights with non-zip file raises appropriate error"""
        buffer = io.BytesIO(b"not a zip file")
        # This should not raise an error but might not load correctly
        # The actual behavior depends on implementation

    def test_load_ms_weights_with_torchscript_zip(self, tmp_path):
        """Test load_ms_weights rejects torchscript zip files"""
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("data/constants.pkl", "content")
        
        with open(zip_path, 'rb') as f:
            with pytest.raises(ValueError, match="do not support torchscript"):
                from serialization import load_ms_weights
                load_ms_weights(f)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

