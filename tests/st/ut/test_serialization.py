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
import pickle
from unittest.mock import Mock, patch, mock_open, MagicMock
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
    _is_path, _opener, _open_file, _open_buffer_writer, _open_buffer_reader,
    _check_seekable, _open_file_like, _is_zipfile, PyTorchFileReader,
    _open_zipfile_reader, _is_torchscript_zip, LoadEndianness,
    get_default_load_endianness, _maybe_decode_ascii, load_ms_weights,
    get_func_by_name, _load, transform_ms_dtype_to_pt_dtype,
    _rebuild_tensor_v2, _rebuild_from_type_v2,
    dtype_map, DTYPE_MAP, func_name_dict, main
)


class TestUtilityFunctions:
    """Test utility functions in serialization.py"""

    def test_is_path_with_string(self):
        """
        Feature: _is_path
        Description:  string path
        Expectation:  Success.
        """
        assert _is_path("test/path.pt") is True
        assert _is_path("/absolute/path.pt") is True


    def test_is_path_with_pathlib(self):
        """
        Feature: _is_path
        Description:  Path
        Expectation:  Success.
        """
        assert _is_path(Path("test/path.pt")) is True
        assert _is_path(Path("/absolute/path.pt")) is True


    def test_is_path_with_non_path(self):
        """
        Feature: _is_path
        Description:  Path
        Expectation:  Success.
        """
        assert _is_path(io.BytesIO()) is False
        assert _is_path(123) is False
        assert _is_path(None) is False


    def test_maybe_decode_ascii_with_bytes(self):
        """
        Feature: _maybe_decode_ascii
        Description:  bytes
        Expectation:  Success.
        """
        result = _maybe_decode_ascii(b"test_string")
        assert result == "test_string"
        assert isinstance(result, str)


    def test_maybe_decode_ascii_with_string(self):
        """
        Feature: _maybe_decode_ascii
        Description:  string
        Expectation:  Success.
        """
        result = _maybe_decode_ascii("test_string")
        assert result == "test_string"
        assert isinstance(result, str)


    def test_get_default_load_endianness(self):
        """
        Feature: get_default_load_endianness
        Description:  none
        Expectation:  Success.
        """
        result = get_default_load_endianness()
        assert result is None or isinstance(result, LoadEndianness)


    def test_check_seekable_with_seekable_file(self):
        """
        Feature: _check_seekable
        Description:  seekable file
        Expectation:  Success.
        """
        buffer = io.BytesIO(b"test data")
        assert _check_seekable(buffer) is True


    def test_check_seekable_with_non_seekable_file(self):
        """
        Feature: _check_seekable
        Description:  non-seekable file
        Expectation:  Success.
        """
        mock_file = Mock()
        mock_file.seek.side_effect = io.UnsupportedOperation("seek")
        mock_file.tell.return_value = 0

        with pytest.raises(io.UnsupportedOperation):
            _check_seekable(mock_file)


    def test_check_seekable_unsupported_operation_no_match(self):
        """
        Feature: _check_seekable
        Description:  seekable file
        Expectation:  Success.
        """
        class MockFile:
            def seek(self, pos):
                raise io.UnsupportedOperation("Operation not supported")

            def tell(self):
                return 0

        mock_file = MockFile()

        with pytest.raises(io.UnsupportedOperation) as excinfo:
            _check_seekable(mock_file)
        assert "Operation not supported" in str(excinfo.value)


class TestFileOperations:
    """Test file operation classes and functions"""

    def test_open_file_like_with_string_path_read(self, tmp_path):
        """
        Feature: _open_file_like
        Description:  string path in read mode
        Expectation:  Success.
        """
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        with _open_file_like(str(test_file), 'rb') as f:
            content = f.read()
            assert content == b"test content"


    def test_open_file_like_with_string_path_write(self, tmp_path):
        """
        Feature: _open_file_like
        Description:  string path in write mode
        Expectation:  Success.
        """
        test_file = tmp_path / "test.txt"

        with _open_file_like(str(test_file), 'wb') as f:
            f.write(b"test content")

        assert test_file.read_text() == "test content"


    def test_open_file_like_with_buffer_read(self):
        """
        Feature: _open_file_like
        Description:  buffer in read mode
        Expectation:  Success.
        """
        buffer = io.BytesIO(b"test content")

        with _open_file_like(buffer, 'rb') as f:
            content = f.read()
            assert content == b"test content"


    def test_open_file_like_with_buffer_write(self):
        """
        Feature: _open_file_like
        Description:  buffer in write mode
        Expectation:  Success.
        """
        buffer = io.BytesIO()

        with _open_file_like(buffer, 'wb') as f:
            f.write(b"test content")

        buffer.seek(0)
        assert buffer.read() == b"test content"


    def test_open_file_like_invalid_mode(self):
        """
        Feature: _open_file_like
        Description:  invalid mode
        Expectation:  Success.
        """
        buffer = io.BytesIO()

        with pytest.raises(RuntimeError, match="Expected 'r' or 'w' in mode"):
            with _open_file_like(buffer, 'x'):
                pass


    def test_is_zipfile_true(self, tmp_path):
        """
        Feature: _is_zipfile
        Description:  valid zip file
        Expectation:  Success.
        """
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test.txt", "content")

        with open(zip_path, 'rb') as f:
            assert _is_zipfile(f) is True


    def test_is_zipfile_false(self):
        """
        Feature: _is_zipfile
        Description:  non-zip file
        Expectation:  Success.
        """
        buffer = io.BytesIO(b"not a zip file")
        assert _is_zipfile(buffer) is False


    def test_is_zipfile_empty(self):
        """
        Feature: _is_zipfile
        Description:  empty file
        Expectation:  Success.
        """
        buffer = io.BytesIO(b"")
        assert _is_zipfile(buffer) is False


    class TestLoadEndianness:
        """Test LoadEndianness enum"""


    def test_load_endianness_enum_values(self):
        """
        Feature: LoadEndianness
        Description:  correct values
        Expectation:  Success.
        """
        assert LoadEndianness.NATIVE.value == 1
        assert LoadEndianness.LITTLE.value == 2
        assert LoadEndianness.BIG.value == 3


    def test_load_endianness_enum_members(self):
        """
        Feature: LoadEndianness
        Description:  enum members
        Expectation:  Success.
        """
        assert hasattr(LoadEndianness, 'NATIVE')
        assert hasattr(LoadEndianness, 'LITTLE')
        assert hasattr(LoadEndianness, 'BIG')


    class TestDictMap:
        """Test dtype, DTYPE_MAP, func_name_dict mapping"""
    def test_dtype_map_completeness(self):
        """
        Feature: dtype_map
        Description:  expected mappings
        Expectation:  Success.
        """
        assert "HalfStorage" in dtype_map
        assert "FloatStorage" in dtype_map
        assert "BFloat16Storage" in dtype_map
        assert "LongStorage" in dtype_map
        assert "ByteStorage" in dtype_map
        assert "BoolStorage" in dtype_map


    def test_dtype_map_values(self):
        """
        Feature: dtype_map
        Description:  expected values
        Expectation:  Success.
        """
        assert dtype_map["HalfStorage"] == np.float16
        assert dtype_map["FloatStorage"] == np.float32
        assert dtype_map["LongStorage"] == np.int64
        assert dtype_map["ByteStorage"] == np.uint8
        assert dtype_map["BoolStorage"] == np.bool_


    def test_DTYPE_MAP_completeness(self):
        """
        Feature: dtype_map
        Description:  expected mappings
        Expectation:  Success.
        """
        assert mindspore.float32 in DTYPE_MAP
        assert mindspore.bfloat16 in DTYPE_MAP


    def test_DTYPE_MAP_values(self):
        """
        Feature: dtype_map
        Description:  expected values
        Expectation:  Success.
        """
        assert DTYPE_MAP[mindspore.float32] == torch.float32
        assert DTYPE_MAP[mindspore.bfloat16] == torch.bfloat16


    def test_func_name_dict_completeness(self):
        """
        Feature: func_name_dict
        Description:  expected mappings
        Expectation:  Success.
        """
        assert '_rebuild_from_type_v2' in func_name_dict
        assert '_rebuild_tensor_v2' in func_name_dict


    def test_func_name_dict_values(self):
        """
        Feature: func_name_dict
        Description:  expected values
        Expectation:  Success.
        """
        assert func_name_dict['_rebuild_from_type_v2'] == _rebuild_from_type_v2
        assert func_name_dict['_rebuild_tensor_v2'] == _rebuild_tensor_v2


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="torch/mindspore not available")
class TestTransformDtype:
    """Test dtype transformation functions"""

    def test_transform_ms_dtype_to_pt_dtype_dict(self):
        """
        Feature: transform_ms_dtype_to_pt_dtype
        Description:  dict input
        Expectation:  Success.
        """
        input_dict = {
            "key1": "value1",
            "key2": [1, 2, 3]
        }
        result = transform_ms_dtype_to_pt_dtype(input_dict)
        assert result == input_dict


    def test_transform_ms_dtype_to_pt_dtype_dict_Value_Error(self):
        """
        Feature: transform_ms_dtype_to_pt_dtype
        Description:  tuple key dict raise Value Error
        Expectation:  Success.
        """
        state = {
            (mindspore.float32, mindspore.int32): "value1"
        }
        with pytest.raises(ValueError) as exc_info:
            transform_ms_dtype_to_pt_dtype(state)
        assert str(exc_info.value) == "convert error, unsupported dtype Int32"


    def test_transform_ms_dtype_to_pt_dtype_list(self):
        """
        Feature: transform_ms_dtype_to_pt_dtype
        Description:  list input
        Expectation:  Success.
        """
        input_list = [1, 2, 3, "test"]
        result = transform_ms_dtype_to_pt_dtype(input_list)
        assert result == input_list


    def test_transform_ms_dtype_to_pt_dtype_primitive(self):
        """
        Feature: transform_ms_dtype_to_pt_dtype
        Description:  primitive types
        Expectation:  Success.
        """
        assert transform_ms_dtype_to_pt_dtype(42) == 42
        assert transform_ms_dtype_to_pt_dtype("test") == "test"
        assert transform_ms_dtype_to_pt_dtype(3.14) == 3.14


    def test_transform_ms_dtype_tuple_key(self):
        """
        Feature: transform_ms_dtype_to_pt_dtype
        Description:  tuple keys
        Expectation:  Success.
        """
        input_dict = {
            (mindspore.float32, mindspore.bfloat16): "value"
        }
        result = transform_ms_dtype_to_pt_dtype(input_dict)
        assert (torch.float32, torch.bfloat16) in result
        assert result[(torch.float32, torch.bfloat16)] == "value"


    def test_transform_ms_dtype_nested_structure(self):
        """
        Feature: transform_ms_dtype_to_pt_dtype
        Description:  nested structure
        Expectation:  Success.
        """
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
        """
        Feature: _rebuild_tensor_v2
        Description:  simple tensor
        Expectation:  Success.
        """
        storage = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        size = (4,)
        stride = (1,)

        result = _rebuild_tensor_v2(storage, 0, size, stride, False, None)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4,)


    def test_rebuild_tensor_v2_2d(self):
        """
        Feature: _rebuild_tensor_v2
        Description:  2D tensor
        Expectation:  Success.
        """
        storage = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        size = (2, 3)
        stride = (3, 1)

        result = _rebuild_tensor_v2(storage, 0, size, stride, False, None)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 3)


    def test_rebuild_tensor_v2_with_offset(self):
        """
        Feature: _rebuild_tensor_v2
        Description:  storage offset
        Expectation:  Success.
        """
        storage = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        size = (3,)
        stride = (1,)
        offset = 2

        result = _rebuild_tensor_v2(storage, offset, size, stride, False, None)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([2.0, 3.0, 4.0]))


    def test_rebuild_tensor_v2_scalar(self):
        """
        Feature: _rebuild_tensor_v2
        Description:  scalar tensor
        Expectation:  Success.
        """
        storage = np.array([42.0], dtype=np.float32)
        size = ()
        stride = ()

        result = _rebuild_tensor_v2(storage, 0, size, stride, False, None)
        assert isinstance(result, torch.Tensor)
        assert result.numel() == 1


    def test_rebuild_tensor_v2_fortran_order(self):
        """
        Feature: _rebuild_tensor_v2
        Description:  Fortran order (column-major)
        Expectation:  Success.
        """
        storage = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        size = (2, 3)
        stride = (1, 2)  # stride[0] == 1 indicates Fortran order

        result = _rebuild_tensor_v2(storage, 0, size, stride, False, None)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 3)


    def test_rebuild_from_type_v2(self):
        """
        Feature: _rebuild_from_type_v2
        Description:  simple input
        Expectation:  Success.
        """
        mock_func = Mock(return_value="result")
        args = ("arg1", "arg2")

        result = _rebuild_from_type_v2(mock_func, None, args, None)
        assert result == "result"
        mock_func.assert_called_once_with("arg1", "arg2")


class TestGetFuncByName:
    """Test get_func_by_name function"""


    def test_get_func_by_name_rebuild_tensor(self):
        """
        Feature: get_func_by_name
        Description:  func name _rebuild_tensor_v2
        Expectation:  Success.
        """
        func = get_func_by_name("_rebuild_tensor_v2")
        assert callable(func)
        assert func == _rebuild_tensor_v2


    def test_get_func_by_name_rebuild_from_type(self):
        """
        Feature: get_func_by_name
        Description:  func name _rebuild_from_type_v2
        Expectation:  Success.
        """
        func = get_func_by_name("_rebuild_from_type_v2")
        assert callable(func)
        assert func == _rebuild_from_type_v2


    def test_get_func_by_name_invalid(self):
        """
        Feature: get_func_by_name
        Description:  invalid name
        Expectation:  Success.
        """
        with pytest.raises(RuntimeError, match="function name .* is invalid"):
            get_func_by_name("nonexistent_function")


class TestLoad:
    """Test _load class"""

    @patch('serialization._load')
    def test_load_has_byteorder_little(self, mock_load_func):
        """
        Feature: _load
        Description:  byteorder mark, little endian
        Expectation:  Error.
        """
        mock_zip = MagicMock()
        mock_zip.has_record.return_value = True
        mock_zip.read_record.return_value = b'little'

        with pytest.raises(AttributeError, match="__call__"):
            _load(mock_zip, None, pickle)


class TestPyTorchFileReader:
    """Test PyTorchFileReader class"""

    def test_pytorch_file_reader_init(self, tmp_path):
        """
        Feature: PyTorchFileReader initialization
        Description:  zipfile
        Expectation:  Success.
        """
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("data/test.txt", "content")

        with open(zip_path, 'rb') as f:
            reader = PyTorchFileReader(f)
            assert reader.directory == "data"


    def test_pytorch_file_reader_has_record(self, tmp_path):
        """
        Feature: PyTorchFileReader has_record
        Description:  zipfile content
        Expectation:  Success.
        """
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
        """
        Feature: PyTorchFileReader read_record
        Description:  zipfile content output
        Expectation:  Success.
        """
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("data/test.txt", "test content")

        with open(zip_path, 'rb') as f:
            reader = PyTorchFileReader(f)
            content = reader.read_record("test.txt")
            assert content == b"test content"


    def test_pytorch_file_reader_read_record_not_found(self, tmp_path):
        """
        Feature: PyTorchFileReader read_record
        Description:  nonexistent zipfile content output
        Expectation:  Success.
        """
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("data/test.txt", "content")

        with open(zip_path, 'rb') as f:
            reader = PyTorchFileReader(f)
            content = reader.read_record("nonexistent.txt")
            assert content is None


    def test_pytorch_file_reader_open_record(self, tmp_path):
        """
        Feature: PyTorchFileReader open_record
        Description:  nonexistent record and exist record
        Expectation:  Success.
        """
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("data/test.txt", "test content")

        with open(zip_path, 'rb') as f:
            reader = PyTorchFileReader(f)
            with reader.open_record("test.txt") as record:
                content = record.read()
                assert content == b"test content"

            result = reader.open_record("nonexistent.txt")
            assert result is None


    def test_pytorch_file_reader_get_all_records(self, tmp_path):
        """
        Feature: PyTorchFileReader get_all_records
        Description:  output all records
        Expectation:  Success.
        """
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
        """
        Feature: PyTorchFileReader get_record_offset
        Description:   nonexistent record and exist record
        Expectation:  Success.
        """
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("data/test.txt", "content")

        with open(zip_path, 'rb') as f:
            reader = PyTorchFileReader(f)
            offset = reader.get_record_offset("test.txt")
            assert isinstance(offset, int)
            assert offset >= 0

            result = reader.get_record_offset("nonexistent.txt")
            assert result is None


    @patch('zipfile.ZipFile')
    def test_pytorch_file_reader_init_offset(self, mock_zipfile, tmp_path):
        """
        Feature: PyTorchFileReader initialization
        Description:   init offset
        Expectation:  Success.
        """

        mock_zipfile.return_value.namelist.return_value = ['model/weights.pth']

        class MockZipFile:
            def __init__(self, records, offset, len):
                self.records = records
                self.offset = offset
                self.len = len
                self.position = 0

            def seek(self, position):
                self.position = position

            def read(self, size=None):
                if size is None:
                    size = self.length - self.position
                start = self.offset + self.position
                end = start + size
                self.position += size
                return self.records[start:end]

        mock_file = MockZipFile(b'dummy data' * 10, 5, 50)
        reader = PyTorchFileReader(mock_file)
        out = reader.file


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="torch/mindspore not available")
class TestLoadMsWeights:
    """Test load_ms_weights function"""

    def test_load_ms_weights_with_torchscript_zip(self, tmp_path):
        """
        Feature: load_ms_weights
        Description:  zipfile
        Expectation:  Success.
        """
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("data/constants.pkl", "content")

        with open(zip_path, 'rb') as f:
            with pytest.raises(ValueError, match="do not support torchscript"):
                from serialization import load_ms_weights
                load_ms_weights(f, None, None)


class TestMain:
    """Test recursive_print in main function"""

    @patch('serialization.load_ms_weights')
    def test_main_output_when_recursive_print_state_is_tensor_list(self,mock_load_ms_weights,capfd):
        """
        Feature: print main
        Description:  tensor list
        Expectation:  Success.
        """

        def load_ms_weights_tensor_list(file):
            state_list = [
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[3.0, 4.0], [3.0, 4.0]])
            ]
            return state_list

        mock_load_ms_weights.side_effect = load_ms_weights_tensor_list

        expected_output = (
            '0 torch.float32 torch.Size([2, 2]) 10.0\n'
            '0.1 torch.float32 torch.Size([2, 2]) 14.0\n'
        )

        main()
        output = capfd.readouterr().out
        assert output == expected_output

    @patch('serialization.load_ms_weights')
    def test_main_output_when_recursive_print_state_is_string(self,mock_load_ms_weights,capfd):
        """
        Feature: print main
        Description:  string list
        Expectation:  Success.
        """

        def load_ms_weights_tensor_list(file):
            state_list = 'output_weights'
            return state_list

        mock_load_ms_weights.side_effect = load_ms_weights_tensor_list

        expected_output = (
            " <class 'str'> output_weights\n"
        )

        main()
        output = capfd.readouterr().out
        assert output == expected_output

    @patch('serialization.load_ms_weights')
    def test_main_output_when_recursive_print_state_is_tensor_dict(self,mock_load_ms_weights,capfd):
        """
        Feature: print main
        Description:  tensor dict
        Expectation:  Success.
        """

        def load_ms_weights_tensor_list(file):
            state_dict = {
                    'layer1': {
                        'weight': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                        'bias': torch.tensor([0.1, 0.2])
                    },
                    'layer2': {
                        'weight': torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                        'bias': torch.tensor([0.3, 0.4])
                    }
                }
            return state_dict

        mock_load_ms_weights.side_effect = load_ms_weights_tensor_list

        expected_output = (
                "layer1.weight torch.float32 torch.Size([2, 2]) 10.0\n"
                "layer1.weight.bias torch.float32 torch.Size([2]) 0.30000001192092896\n"
                "layer1.weight.bias.layer2.weight torch.float32 torch.Size([2, 2]) 26.0\n"
                "layer1.weight.bias.layer2.weight.bias torch.float32 torch.Size([2]) 0.7000000476837158\n"
        )

        main()
        output = capfd.readouterr().out
        assert output == expected_output

if __name__ == "main":
    pytest.main([__file__, "-v"])