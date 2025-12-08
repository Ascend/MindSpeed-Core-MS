# Copyright 2025 Huawei Technologies Co., Ltd
# ============================================================================
"""
Test module for patch_func_router.py from tools/convert/patch_merge/modules/
Tests the function routing transformer behavior.
"""

import os
import sys
from pathlib import Path

import pytest
import libcst as cst
from libcst.metadata import MetadataWrapper, ParentNodeProvider

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.convert.patch_merge.modules.patch_func_router import (  # noqa: E402
    PatchFuncRouterTransformer,
)


def _make_patch(
    patch_import: str,
    condition,
    origin_import: str = "megatron.pkg.mod.func",
    cond_imports=None,
):
    """Build a patch dictionary aligned with real parser expectations for reuse."""
    if cond_imports is None:
        cond_imports = []
    return {
        "patch_import": patch_import,
        "condition": condition,
        "origin_import": origin_import,
        "raw_patch": {
            "patch_import": patch_import,
            "patch_name": patch_import.split(".")[-1],
            "condition": condition,
            "condition_import": cond_imports,
        },
    }


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=1)
def test_merged_name_builder_generates_stable_unique_names():
    """
    Feature: PatchFuncRouterTransformer._merged_name_builder
    Description: Generates unique aliases per patch_import while keeping identical imports stable.
    Expectation: Same patch_import yields identical aliases, different patch_imports yield different aliases.
    """
    patches = [
        _make_patch("pkg.mod.func_a", []),
        _make_patch("pkg.mod.func_b", []),
    ]
    router = PatchFuncRouterTransformer("target_func", patches)

    name_a1 = router._merged_name_builder("target_func", patches[0])
    name_a2 = router._merged_name_builder("target_func", patches[0])
    name_b = router._merged_name_builder("target_func", patches[1])

    assert name_a1 == name_a2
    assert name_a1 != name_b
    assert name_a1.startswith("target_func_patchfunc_")
    assert name_b.startswith("target_func_patchfunc_")


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=2)
def test_create_default_call_for_plain_function():
    """
    Feature: PatchFuncRouterTransformer.create_default_call
    Description: Validates default call creation for standalone functions.
    Expectation: Returns statements such as `return func_name(args...)`.
    """
    router = PatchFuncRouterTransformer("foo", [])

    func_args = [
        cst.Arg(value=cst.Name("x")),
        cst.Arg(value=cst.Name("y")),
    ]

    default_stmt = router.create_default_call(
        default_name="foo_default",
        is_class_method=False,
        is_self_method=False,
        func_args=func_args,
    )

    # Wrap into a module to render the generated code for assertion
    code = cst.Module(body=[default_stmt]).code
    assert "return foo_default(x, y)" in code


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=3)
def test_create_default_call_for_self_method_and_classmethod():
    """
    Feature: PatchFuncRouterTransformer.create_default_call
    Description: Ensures instance and class/static methods receive appropriate default-call shapes.
    Expectation: Instance methods call `self.func`, class/static methods call `ClassName.func`.
    """
    # Instance method scenario
    router_self = PatchFuncRouterTransformer("MyClass.method", [])
    func_args_self = [
        cst.Arg(value=cst.Name("self")),
        cst.Arg(value=cst.Name("x")),
    ]
    default_self = router_self.create_default_call(
        default_name="method_default",
        is_class_method=True,
        is_self_method=True,
        func_args=func_args_self,
    )
    code_self = cst.Module(body=[default_self]).code
    # Instance methods should drop `self` from forwarded arguments
    assert "return self.method_default(x)" in code_self

    # Class/static method scenario
    router_cls = PatchFuncRouterTransformer("MyClass.method", [])
    router_cls.cls_name = "MyClass"  # Explicitly set class name for class/static flow
    func_args_cls = [
        cst.Arg(value=cst.Name("x")),
        cst.Arg(value=cst.Name("y")),
    ]
    default_cls = router_cls.create_default_call(
        default_name="method_default",
        is_class_method=True,
        is_self_method=False,
        func_args=func_args_cls,
    )
    code_cls = cst.Module(body=[default_cls]).code
    assert "return MyClass.method_default(x, y)" in code_cls


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=4)
def test_build_calls_injects_condition_imports_and_router_func():
    """
    Feature: PatchFuncRouterTransformer._build_calls
    Description: Builds router wrappers for target functions including conditional imports and global_args.
    Expectation: Default implementation is retained, router version is added, and imports/branches are correct.
    """
    code = """
def target(x, y, *, z=1):
    return x + y + z
"""
    module = cst.parse_module(code)

    # One unconditional patch and one conditional patch requiring condition_import
    patches = [
        _make_patch("pkg.mod.patch_func1", []),
        _make_patch(
            "pkg.mod.patch_func2",
            ["args.rank == 0"],
            cond_imports=["cond.module.cond_func"],
        ),
    ]

    router = PatchFuncRouterTransformer("target", patches)
    wrapper = MetadataWrapper(module)
    # Run the transformer so leave_FunctionDef triggers _build_calls and LibCST expands FlattenSentinel automatically
    new_module = wrapper.visit(router)
    new_code = new_module.code

    # 1) Original function renamed to target_default
    assert "def target_default(" in new_code

    # 2) Router wrapper `target` is added
    assert "def target(" in new_code

    # 3) get_args import and global_args initialization injected
    assert "from megatron.training import get_args" in new_code
    assert "global_args = get_args()" in new_code

    # 4) condition_import entries materialize as real imports
    assert "from cond.module import cond_func" in new_code

    # 5) Condition expressions replace args with global_args
    assert "global_args.rank == 0" in new_code


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=5)
def test_leave_functiondef_only_targets_matching_function():
    """
    Feature: PatchFuncRouterTransformer.leave_FunctionDef
    Description: Ensures only matching module_name functions get router replacements.
    Expectation: Only function `target` is rewritten into default + router pair.
    """
    code = """
def untouched(a):
    return a

def target(a, b):
    return a + b
"""
    module = cst.parse_module(code)
    wrapper = MetadataWrapper(module)

    patches = [
        _make_patch("pkg.mod.patch_target", []),
    ]
    router = PatchFuncRouterTransformer("target", patches)

    new_module = wrapper.visit(router)
    new_code = new_module.code

    # Function `untouched` keeps its original definition
    assert "def untouched(a):" in new_code
    assert "def target_default(" in new_code
    assert "def target(" in new_code


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.run(order=6)
def test_leave_functiondef_handles_class_method_scope():
    """
    Feature: PatchFuncRouterTransformer.leave_FunctionDef for class methods
    Description: Routers should be generated only when the class name matches the module_name target.
    Expectation: Target class methods are rewritten, off-target class methods stay untouched.
    """
    code = """
class OtherClass:
    def target(self, x):
        return x

class MyClass:
    def target(self, x):
        return x
"""
    module = cst.parse_module(code)
    wrapper = MetadataWrapper(module)

    patches = [
        _make_patch("pkg.mod.patch_method", []),
    ]
    router = PatchFuncRouterTransformer("MyClass.target", patches)

    new_module = wrapper.visit(router)
    new_code = new_module.code

    # `OtherClass.target` remains unchanged
    assert "class OtherClass" in new_code
    assert "def target(self, x):" in new_code  # Still shows once

    # `MyClass` should now contain both `target_default` and the router `target`
    assert "class MyClass" in new_code
    assert "def target_default(self, x):" in new_code
    assert "def target(self, x):" in new_code
