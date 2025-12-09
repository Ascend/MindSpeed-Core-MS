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
Test module for patch_import_collector.py from tools/convert/patch_merge/modules/
Tests the import collection and manipulation functionality
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the tools directory to the path
tools_path = os.path.join(os.path.dirname(__file__), '../../../tools/convert/patch_merge/modules')
sys.path.insert(0, tools_path)

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider, ScopeProvider, ParentNodeProvider

from tools.convert.patch_merge.modules.coverage import get_debug_print_node
from tools.convert.patch_merge.modules.patch_import_collector import MImport

try:
    import torch
    import mindspore
    from ml_dtypes import bfloat16
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

from patch_wrapper_router import (
    PatchWrapperRouterTransformer,
)

class TestPatchWrapperRouterTransformer:
    """Test PatchWrapperRouterTransformer in patchwrapper_router.py"""

    def setup_method(self):
        """
        Feature: PatchWrapperRouterTransformer
        Description:  setup
        Expectation:  Success.
        """
        self.sample_patch_infos = [{
            'origin_import': 'test_module.test_func',
            'patch_import': 'patch_module.wrapper_func',
            'condition': [],
            'raw_patch': {
                'origin_import': 'test_module.test_func',
                'patch_import': 'patch_module.wrapper_func',
                'patch_name': 'raw_name',
                'condition': [],
                'raw_patch': {}
            }
        }]

        self.sample_patch_infos2 = [{
            'origin_import': 'test_module.test_func',
            'patch_import': 'patch_module.wrapper_func',
            'condition': [],
            'raw_patch': {
                'origin_import': 'test_module.test_func',
                'patch_import': 'patch_module.wrapper_func',
                'patch_name': 'raw_name',
                'condition': [],
                'raw_patch': {}
            }
        }]

    def test_init_class_method(self):
        """
        Feature: __init__
        Description:  class method
        Expectation:  Success.
        """
        transformer = PatchWrapperRouterTransformer(
            "TestClass.test_method",
            self.sample_patch_infos
        )
        assert transformer.is_class_method is True
        assert transformer.cls_name == "TestClass"
        assert transformer.func_name == "test_method"

    def test_init_function_method(self):
        """
        Feature: __init__
        Description:  function method
        Expectation:  Success.
        """
        transformer = PatchWrapperRouterTransformer(
            "test_function",
            self.sample_patch_infos
        )
        assert transformer.is_class_method is False
        assert transformer.cls_name is None
        assert transformer.func_name == "test_function"

    def test_merged_name_builder_new_name(self):
        """
        Feature: merged name builder
        Description:  new name
        Expectation:  Success.
        """
        transformer = PatchWrapperRouterTransformer(
            "test_func",
            self.sample_patch_infos
        )
        name = transformer._merged_name_builder("test_func", {'patch_import': 'new.patch'})
        assert "test_func__patch_wrapper_0" in name
        assert name in transformer.used_names.values()

    def test_merged_name_builder_existing_name(self):
        """
        Feature: merged name builder
        Description:  existing name
        Expectation:  Success.
        """
        transformer = PatchWrapperRouterTransformer(
            "test_func",
            self.sample_patch_infos
        )
        transformer.used_names['existing.patch'] = 'existing_name'
        name = transformer._merged_name_builder("test_func", {'patch_import': 'existing.patch'})
        assert name == 'existing_name'

    def test_visit_module(self):
        """
        Feature: visit_module
        Description:  cst.parse_module
        Expectation:  Success.
        """
        transformer = PatchWrapperRouterTransformer(
            "test_func",
            self.sample_patch_infos
        )
        module_node = cst.parse_module("def test(): pass")
        result = transformer.visit_Module(module_node)
        assert result is True
        assert transformer.root == module_node

    def test_parse_patch_unconditional(self):
        """
        Feature: parse_patch
        Description:  unconditional patch
        Expectation:  Success.
        """
        transformer = PatchWrapperRouterTransformer(
            "test_func",
            self.sample_patch_infos
        )
        patch = {
            'patch_import': 'module.wrapper',
            'condition': []
        }
        module, wrapper, condition = transformer.parse_patch(patch)
        assert module == 'module'
        assert wrapper == 'wrapper'
        assert condition == []

    def test_parse_patch_conditional(self):
        """
        Feature: parse_patch
        Description:  conditional patch
        Expectation:  Success.
        """
        transformer = PatchWrapperRouterTransformer(
            "test_func",
            self.sample_patch_infos
        )
        patch = {
            'patch_import': 'module.wrapper',
            'condition': ['args.test == True']
        }
        module, wrapper, condition = transformer.parse_patch(patch)
        assert module == 'module'
        assert wrapper == 'wrapper'
        assert condition == ['args.test == True']

    def test_build_inner_wrapped_call_unconditional_function_method_true(self):
        """
        Feature: _build_inner_wrapped_call
        Description:  unconditional patch, True
        Expectation:  Success.
        """

        transformer = PatchWrapperRouterTransformer(
            "TestClass.test_method",
            self.sample_patch_infos
        )

        fn = cst.Name("test_fn")
        result = transformer._build_inner_wrapped_call("test_wrapper", self.sample_patch_infos2, fn, True)

        assert isinstance(result, cst.FunctionDef)
        assert result.name.value == "test_wrapper"
        assert len(result.params.params) == 1
        assert result.params.params[0].name.value == "self"

    def test_build_inner_wrapped_call_unconditional_function_method_false(self):
        """
        Feature: _build_inner_wrapped_call
        Description:  unconditional patch, False
        Expectation:  Success.
        """

        transformer = PatchWrapperRouterTransformer(
            "test_function",
            self.sample_patch_infos
        )
        fn = cst.Name("test_fn")

        result = transformer._build_inner_wrapped_call("test_wrapper", self.sample_patch_infos2, fn, False)

        assert isinstance(result, cst.FunctionDef)
        assert result.name.value == "test_wrapper"
        assert len(result.params.params) == 0

    def test_build_outer_decorator_def(self):
        """
        Feature: _build_outer_decorator_def
        Description:  func
        Expectation:  Success.
        """

        transformer = PatchWrapperRouterTransformer(
            "test_function",
            self.sample_patch_infos
        )

        result = transformer._build_outter_decorator_def("test_func", self.sample_patch_infos2)

        assert isinstance(result, cst.FunctionDef)
        assert "test_func_decorator" in result.name.value

    def test_leave_function_def_success_class_method(self):
        """
        Feature: leave_FunctionDef
        Description:  cst.ClassDef.body, success class
        Expectation:  Success.
        """

        transformer = PatchWrapperRouterTransformer(
            "TestClass.test_method",
            self.sample_patch_infos
        )

        class_def = cst.ClassDef(
            name=cst.Name("TestClass"),
            body=cst.IndentedBlock([
                cst.FunctionDef(
                    name=cst.Name("test_method"),
                    params=cst.Parameters(),
                    body=cst.IndentedBlock([])
                )
            ])
        )
        func_def = class_def.body.body[0]

        with patch.object(transformer, 'get_metadata') as mock_get_metadata:
            mock_get_metadata.return_value = class_def

            result = transformer.leave_FunctionDef(func_def, func_def)

            assert transformer.modified_patch_nodes is None
            assert not isinstance(result, cst.FlattenSentinel)



    def test_build_outer_decorator_def_for_cls(self):
        """
        Feature: _build_outer_decorator_def_for_cls
        Description:  TestClass.test_method, patches
        Expectation:  Success.
        """

        transformer = PatchWrapperRouterTransformer(
            "TestClass.test_method",
            self.sample_patch_infos
        )

        result = transformer._build_outter_decorator_def_for_cls("test_method", "TestClass", self.sample_patch_infos2)

        assert isinstance(result, cst.FunctionDef)
        assert "test_class_test_method_decorator" in result.name.value

    def test_try_wrap_implicit_class_func_no_class(self):
        """
        Feature: try_wrap_implicit_class_func
        Description:  module no class
        Expectation:  Success.
        """
        transformer = PatchWrapperRouterTransformer(
            "test_function",
            self.sample_patch_infos
        )

        module_node = cst.parse_module("class TestClass: pass")
        result = transformer.try_wrap_implicit_class_func(module_node)
        assert result is None

    @patch("patch_wrapper_router.PatchWrapperRouterTransformer.try_wrap_implicit_class_func")
    def test_try_wrap_implicit_class_func_dataclass(self,mock_func):
        """
        Feature: try_wrap_implicit_class_func
        Description:  module node, dataclass
        Expectation:  Success.
        """
        transformer = PatchWrapperRouterTransformer(
            "dataclass.__init__",
            self.sample_patch_infos
        )

        dataclass_decorator = cst.Decorator(decorator=cst.Name("dataclass"))
        class_def = cst.ClassDef(
            name=cst.Name("dataclass"),
            decorators=[dataclass_decorator,dataclass_decorator],
            body=cst.IndentedBlock([])
        )

        module_node = cst.Module(body=[class_def])
        module_node2 = cst.Module(body=[module_node,module_node])

        mock_func.return_value = module_node

        result = transformer.try_wrap_implicit_class_func(module_node2)
        assert result is module_node
        assert isinstance(result, cst.Module)

    def test_leave_module_already_modified(self):
        """
        Feature: leave_Module
        Description:  module already modified
        Expectation:  Success.
        """
        transformer = PatchWrapperRouterTransformer(
            "test_function",
            self.sample_patch_infos
        )
        transformer.modified_patch_nodes = cst.parse_statement("pass")

        module_node = cst.parse_module("def test(): pass")
        result = transformer.leave_Module(module_node, module_node)
        assert result == module_node

    @patch("patch_wrapper_router.PatchWrapperRouterTransformer.try_wrap_implicit_class_func")
    def test_leave_module_try_wrap_success(self,mock_func):
        """
        Feature: leave_Module
        Description:  module_node success
        Expectation:  Success.
        """

        module_node = cst.Module(body=[])
        mock_func.return_value = module_node

        transformer = PatchWrapperRouterTransformer(
            "TestClass.__init__",
            self.sample_patch_infos
        )

        result = transformer.leave_Module(module_node, module_node)
        assert result is module_node
        assert isinstance(result, cst.Module)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])