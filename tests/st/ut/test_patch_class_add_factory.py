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
Test module for patch_class_add_factory.py functionality.
"""
import os
import pytest
import sys
from pathlib import Path

import libcst as cst
from libcst.metadata import MetadataWrapper, ParentNodeProvider

from tools.convert.patch_merge.modules import patch_class_add_factory

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add module path
module_path = str(project_root / "tools" / "convert" / "patch_merge" / "modules")
if module_path not in sys.path:
    sys.path.insert(0, module_path)

# Import the real patch_import_collector and register it under the top-level module name
from tools.convert.patch_merge.modules import patch_import_collector as _patch_import_collector  # noqa: E402
sys.modules.setdefault("patch_import_collector", _patch_import_collector)

current_dir = Path(__file__).parent


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=1)
def test_grep_in_files_basic_match(tmp_path):
    """
    Feature: patch_class_add_factory.grep_in_files
    Description: grep_in_files should return only .py files containing the pattern.
    Expectation: Only the file containing the target class name is reported.
    """
    target_dir = tmp_path / "src"
    target_dir.mkdir()

    match_file = target_dir / "match.py"
    nomatch_file = target_dir / "no_match.py"
    other_ext = target_dir / "readme.txt"

    match_file.write_text("class MyTargetClass:\n    pass\n", encoding="utf-8")
    nomatch_file.write_text("class OtherClass:\n    pass\n", encoding="utf-8")
    other_ext.write_text("MyTargetClass", encoding="utf-8")

    result = patch_class_add_factory.grep_in_files(str(target_dir), r"MyTargetClass")

    assert str(match_file) in result
    assert str(nomatch_file) not in result
    assert str(other_ext) not in result


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=2)
def test_grep_in_files_non_recursive(tmp_path):
    """
    Feature: patch_class_add_factory.grep_in_files
    Description: Non-recursive search scans only the top-level directory.
    Expectation: Files in subdirectories are ignored when recursive=False.
    """
    root = tmp_path / "root"
    root.mkdir()
    sub = root / "sub"
    sub.mkdir()

    top_file = root / "top.py"
    sub_file = sub / "sub.py"

    top_file.write_text("TARGET = 1\n", encoding="utf-8")
    sub_file.write_text("TARGET = 2\n", encoding="utf-8")

    result_non_recursive = patch_class_add_factory.grep_in_files(str(root), r"TARGET", recursive=False)
    result_recursive = patch_class_add_factory.grep_in_files(str(root), r"TARGET", recursive=True)

    assert str(top_file) in result_non_recursive
    assert str(sub_file) not in result_non_recursive
    assert str(top_file) in result_recursive
    assert str(sub_file) in result_recursive


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=3)
def test_patch_class_factory_build_and_insert_before_class():
    """
    Feature: PatchClassFactoryTransformer._build_factory_def & leave_ClassDef
    Description: Factory class is generated and placed before the original class definition.
    Expectation: The resulting module includes MyClassFactory before MyClass and contains expected methods/imports.
    """
    code = """
class MyClass:
    def method(self):
        return "orig"
"""
    module = cst.parse_module(code)

    patches = [
        {
            "patch_import": "pkg.mod.PatchMyClass",
            "condition": [],
            "orign_import": "megatron.pkg.mod.MyClass",
            "raw_patch": {
                "patch_import": "pkg.mod.PatchMyClass",
                "patch_name": "MyClass",
                "condition": [],
            },
        },
        {
            "patch_import": "pkg.mod.PatchMyClass2",
            "condition": ["args.rank == 0"],
            "orign_import": "megatron.pkg.mod.MyClass",
            "raw_patch": {
                "patch_import": "pkg.mod.PatchMyClass2",
                "patch_name": "MyClass",
                "condition": ["args.rank == 0"],
            },
        },
    ]

    wrapper = MetadataWrapper(module)
    transformer = patch_class_add_factory.PatchClassFactoryTransformer("MyClass", patches)
    new_module = wrapper.visit(transformer)
    new_code = new_module.code

    assert "class MyClassFactory" in new_code
    factory_pos = new_code.index("class MyClassFactory")
    orig_pos = new_code.index("class MyClass:\n")
    assert factory_pos < orig_pos

    assert "def create_instance(" in new_code
    assert "def get_class(" in new_code

    assert "from megatron.training import get_args" in new_code
    assert "global_args = get_args()" in new_code
    assert "global_args.rank == 0" in new_code


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=4)
def test_patch_class_factory_merged_name_builder_unique():
    """
    Feature: PatchClassFactoryTransformer._merged_name_builder
    Description: Generates deterministic patch class names for each unique patch_import.
    Expectation: Identical patch_imports share the same alias; different ones do not.
    """
    patches = [
        {
            "patch_import": "pkg.mod.PatchA",
            "condition": [],
            "orign_import": "megatron.pkg.mod.MyClass",
            "raw_patch": {
                "patch_import": "pkg.mod.PatchA",
                "patch_name": "MyClass",
                "condition": [],
            },
        },
        {
            "patch_import": "pkg.mod.PatchB",
            "condition": [],
            "orign_import": "megatron.pkg.mod.MyClass",
            "raw_patch": {
                "patch_import": "pkg.mod.PatchB",
                "patch_name": "MyClass",
                "condition": [],
            },
        },
    ]
    transformer = patch_class_add_factory.PatchClassFactoryTransformer("MyClass", patches)

    name_a1 = transformer._merged_name_builder("MyClass", patches[0])
    name_a2 = transformer._merged_name_builder("MyClass", patches[0])
    name_b = transformer._merged_name_builder("MyClass", patches[1])

    assert name_a1 == name_a2
    assert name_a1 != name_b
    assert name_a1.startswith("MyClassPatch")
    assert name_b.startswith("MyClassPatch")


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=5)
def test_patch_class_call_transformer_leave_name_rewrites_and_add_import():
    """
    Feature: PatchClassCallTransformer.leave_Name & leave_Module
    Description: Name-form references should be replaced with ClassFactory.get_class(), and a factory import is added.
    Expectation: Rewritten call sites plus `from pkg.mod import MyClassFactory`.
    """
    code = """
from pkg.mod import MyClass

def fn():
    x = MyClass()
    return MyClass
"""
    module = cst.parse_module(code)
    wrapper = MetadataWrapper(module)
    parent_provider = wrapper.resolve(ParentNodeProvider)

    transformer = patch_class_add_factory.PatchClassCallTransformer(
        cls_name="MyClass",
        original_import_name="pkg.mod.MyClass",
        parent_provider=parent_provider,
    )
    new_module = wrapper.visit(transformer)
    new_code = new_module.code

    assert "MyClassFactory.get_class()" in new_code
    assert "from pkg.mod import MyClassFactory" in new_code


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=6)
def test_patch_class_call_transformer_leave_attribute_rewrites_ref():
    """
    Feature: PatchClassCallTransformer.leave_Attribute
    Description: Attribute-form references (e.g., ns.MyClass) should also be routed through the factory.
    Expectation: Attribute usage is rewritten, original imports remain, and factory import is appended.
    """
    code = """
import pkg.mod as mod

def fn():
    x = mod.MyClass
    return x
"""
    module = cst.parse_module(code)
    wrapper = MetadataWrapper(module)
    parent_provider = wrapper.resolve(ParentNodeProvider)

    transformer = patch_class_add_factory.PatchClassCallTransformer(
        cls_name="MyClass",
        original_import_name="pkg.mod.MyClass",
        parent_provider=parent_provider,
    )
    new_module = wrapper.visit(transformer)
    new_code = new_module.code

    assert "MyClassFactory.get_class()" in new_code
    assert "import pkg.mod as mod" in new_code
    assert "from pkg.mod import MyClassFactory" in new_code
