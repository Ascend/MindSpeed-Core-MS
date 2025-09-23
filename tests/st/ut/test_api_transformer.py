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
Module for testing APITransformer.
"""
import os
import pytest
import sys
import json
from pathlib import Path
from unittest.mock import patch
import libcst as cst
from libcst.metadata import MetadataWrapper
from tools.convert.modules.api_transformer import APITransformer
from tools.convert.modules.string_transformer import StringTransformer, PairTransformer
from tools.convert.modules.utils import source_file_iterator, case_insensitive_replace, FileConverter
from multiprocessing import Pool
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
current_dir = Path(__file__).parent


@pytest.fixture
def mock_file_path():
    """Mock file path and content"""
    origin_path = current_dir / "test_data" / "origin"
    save_path = current_dir / "test_data" / "save"
    origin_path.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)

    file1 = origin_path / "test_api_transformer.py"
    file_FileConverter = origin_path / "test_api_transformer_FileConverter.py"
    context = '''# torch comment
import torch
import torch_npu
import torch.distributed as dist
import torch.nn as nn
import safetensors.torch
from torch.nn import Linear, ReLU
from safetensors.torch import load_model
from .utils import helper
from some.module import *
"replace search_str with replacement_str in original_str, ignore case"
"replace Search_str with replacement_str in original_str, ignore case"
"replace Search_Str with replacement_str in original_str, ignore case"
"replace SEARCH_STR with replacement_str in original_str, ignore case"


def func(a: torch.tensor, b: int):
    return a * b
str1 = 'torch.nn'
str2 = 'torch'

num1 = nn.functional.relu()
x = torch.matmul(a, b)
x_device = f"torch.device: {torch.device}"
world_size = dist.get_world_size(pg)
'''
    with open(file1, 'w', encoding='UTF-8') as f:
        f.write(context)
    with open(file_FileConverter, 'w', encoding='UTF-8') as f:
        f.write(context)

    file2 = origin_path / "serialization.py"
    with open(file2, 'w', encoding='UTF-8') as f:
        f.write("if mod_name == 'msadapter':\n    return str(name)\n")

    file3 = origin_path / "proxy.py"
    with open(file3, 'w', encoding='UTF-8') as f:
        f.write("{'msadapter': 'msadapter'}")

    return str(origin_path), str(save_path)


def apply_ApiTransformer_to_file(file_path, save_file_path, current_name="torch", new_name="msadapter", string_mapping=None):
    """Apply APITransformer to a file and save output."""
    if string_mapping is None:
        string_mapping = [("torch", "msadapter")]

    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()

    tree = MetadataWrapper(cst.parse_module(source))
    transformer = APITransformer(current_name, new_name, string_mapping)
    new_tree = tree.visit(transformer)

    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    with open(save_file_path, 'w', encoding='utf-8') as f:
        f.write(new_tree.code)


def apply_StringTransformer_to_file(file_path, save_file_path):
    """Apply StringTransformer to a file and save output."""
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()

    tree = MetadataWrapper(cst.parse_module(source))
    transformer = StringTransformer([('msadapter', 'torch'),])
    new_tree = tree.visit(transformer)

    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    with open(save_file_path, 'w', encoding='utf-8') as f:
        f.write(new_tree.code)


def apply_PairTransformer_to_file(file_path, save_file_path):
    """Apply PairTransformer to a file and save output."""
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()

    tree = MetadataWrapper(cst.parse_module(source))
    transformer = PairTransformer([(('msadapter', 'msadapter'), ('torch', 'msadapter')),])
    new_tree = tree.visit(transformer)

    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    with open(save_file_path, 'w', encoding='utf-8') as f:
        f.write(new_tree.code)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=1)
def test_StringTransformer_leave_SimpleString(mock_file_path):
    """
    Feature: StringTransformer replaces strings within string literals.
    Description: Given a Python file containing the string 'msadapter', replaces it with 'torch' inside string contexts.
    Expectation: Output file contains 'torch' in place of 'msadapter' within strings.
    """
    origin_path, save_path = mock_file_path
    origin_file = Path(origin_path) / "serialization.py"
    save_file = Path(save_path) / "serialization.py"

    apply_StringTransformer_to_file(str(origin_file), str(save_file))
    assert save_file.exists()
    with open(save_file, 'r', encoding='UTF-8') as f:
        content = f.read()
        assert "torch" in content


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=2)
def test_PairTransformer_leave_DictElement(mock_file_path):
    """
    Feature: PairTransformer replaces key-value string pairs in dict literals.
    Description: Given a dict element string "'msadapter': 'msadapter'", replaces it with "'torch': 'msadapter'".
    Expectation: Output file contains the transformed dict element string "'torch': 'msadapter'".
    """
    origin_path, save_path = mock_file_path
    origin_file = Path(origin_path) / "proxy.py"
    save_file = Path(save_path) / "proxy.py"

    apply_PairTransformer_to_file(str(origin_file), str(save_file))
    assert save_file.exists()
    with open(save_file, 'r', encoding='UTF-8') as f:
        content = f.read()
        assert "'torch': 'msadapter'" in content


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=3)
def test_api_transformer_comment_docstring_string_replace(mock_file_path):
    """
    Feature: APITransformer replaces strings in comments and f-strings.
    Description: Input file contains 'torch' in comment and f-string; transformer replaces with 'msadapter'.
    Expectation: Output contains '# msadapter comment' and f"msadapter.device" in transformed content.
    """
    origin_path, save_path = mock_file_path
    origin_file = Path(origin_path) / "test_api_transformer.py"
    save_file = Path(save_path) / "test_api_transformer.py"

    apply_ApiTransformer_to_file(str(origin_file), str(save_file))
    assert save_file.exists()
    with open(save_file, 'r', encoding='UTF-8') as f:
        content = f.read()
        assert "# msadapter comment" in content
        assert 'f"msadapter.device' in content


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=4)
def test_api_transformer_name_replace(mock_file_path):
    """
    Feature: APITransformer renames imported module names.
    Description: Input contains 'import torch' and 'import torch_npu'; these are mapped to 'msadapter' equivalents.
    Expectation: Output contains 'import msadapter' and 'import msadapter_npu'.
    """
    origin_path, save_path = mock_file_path
    origin_file = Path(origin_path) / "test_api_transformer.py"
    save_file = Path(save_path) / "test_api_transformer.py"

    apply_ApiTransformer_to_file(str(origin_file), str(save_file))

    assert save_file.exists()
    with open(save_file, 'r', encoding='UTF-8') as f:
        content = f.read()
        assert "import msadapter" in content
        assert "import msadapter_npu" in content


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=5)
def test_api_transformer_call_mapping_api(mock_file_path):
    """
    Feature: APITransformer maps specific API calls to target framework equivalents.
    Description: Input contains 'torch.matmul(a, b)'; this is mapped to 'mindspore.mint.matmul(a, b)'.
    Expectation: Output contains 'mindspore.mint.matmul(a, b)' and imports 'import mindspore'.
    """
    origin_path, save_path = mock_file_path
    origin_file = Path(origin_path) / "test_api_transformer.py"
    save_file = Path(save_path) / "test_api_transformer.py"

    apply_ApiTransformer_to_file(str(origin_file), str(save_file))

    assert save_file.exists()
    with open(save_file, 'r', encoding='UTF-8') as f:
        content = f.read()
        assert "import mindspore" in content
        assert "mindspore.mint.matmul(a, b)" in content


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=6)
def test_api_transformer_safetensors_ignored(mock_file_path):
    """
    Feature: APITransformer skips replacement in safetensors.torch namespace.
    Description: Input contains 'import safetensors.torch'.
    Expectation: These remain unchanged in output; 'safetensors.torch' is preserved.
    """
    origin_path, save_path = mock_file_path
    origin_file = Path(origin_path) / "test_api_transformer.py"
    save_file = Path(save_path) / "test_api_transformer.py"

    apply_ApiTransformer_to_file(str(origin_file), str(save_file))

    assert save_file.exists()
    with open(save_file, 'r', encoding='UTF-8') as f:
        content = f.read()
        assert "import safetensors.torch" in content


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=7)
def test_case_insensitive_replace(mock_file_path):
    """
    Feature: Utility function replaces substrings ignoring case.
    Description: Input string contains 'search_str' in multiple case variants.
    Expectation: All variants are replaced with 'replacement_str' in output.
    """
    origin_path, save_path = mock_file_path
    origin_file = Path(origin_path) / "test_api_transformer.py"

    with open(origin_file, 'r', encoding='UTF-8') as f:
        content = f.read()
        out = case_insensitive_replace(content, "search_str", "replacement_str")
        assert "search_str" not in out
        assert "Search_str" not in out
        assert "Search_Str" not in out
        assert "SEARCH_STR" not in out


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=8)
def test_FileConverter(mock_file_path):
    """
    Feature: FileConverter applies APITransformer and writes back to file.
    Description: Input file contains torch APIs and references; FileConverter maps them using APITransformer.
    Expectation: File is modified in-place with all expected replacements: msadapter imports, mindspore APIs, etc.
    """
    origin_path, save_path = mock_file_path
    origin_file = Path(origin_path) / "test_api_transformer_FileConverter.py"
    string_mapping = [('torch', 'msadapter'), ('torch_npu', 'msadapter_npu')]
    file_converter = FileConverter(APITransformer, ('torch', 'msadapter', string_mapping))
    file_converter.convert(origin_file)
    with open(origin_file, 'r', encoding='UTF-8') as f:
        content = f.read()
        assert "import safetensors.torch" in content
        assert "mindspore.mint.matmul(a, b)" in content
        assert "import msadapter_npu" in content
        assert "import msadapter" in content
        assert 'f"msadapter.device' in content
        assert "import mindspore" in content


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=9)
def test_Convert(mock_file_path):
    """
    Feature: Convert processes multiple files via multiprocessing using FileConverter.
    Description: Source directory contains Python files; Convert traverses and transforms all using pool.imap.
    Expectation: Each file is correctly transformed — imports, APIs, and strings replaced as expected.
    """
    origin_path, save_path = mock_file_path
    origin_file = Path(origin_path) / "test_api_transformer.py"
    file_iterator = source_file_iterator(origin_path)
    string_mapping = [('torch', 'msadapter'), ('torch_npu', 'msadapter_npu')]
    file_converter = FileConverter(APITransformer, ('torch', 'msadapter', string_mapping))
    with Pool(processes=32) as pool:
        results = list(tqdm(pool.imap(file_converter.convert, file_iterator), desc="Processing"))

    with open(origin_file, 'r', encoding='UTF-8') as f:
        content = f.read()
        assert "import safetensors.torch" in content
        assert "mindspore.mint.matmul(a, b)" in content
        assert "import msadapter_npu" in content
        assert "import msadapter" in content
        assert 'f"msadapter.device' in content
        assert "import mindspore" in content
