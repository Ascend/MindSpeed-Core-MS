# Copyright (c) Huawei Technologies Co., Ltd 2025.  All rights reserved.
import re
import os
from pathlib import Path
from collections import defaultdict
import libcst as cst
from libcst.metadata import PositionProvider, ScopeProvider, MetadataWrapper, ParentNodeProvider
from libcst import matchers

from .coverage import get_debug_print_node

def grep_in_files(directory: str, pattern: str, recursive: bool = True, file_pattern: str = ".py"):
    """
    Search for the content that matches the pattern in the specified directory 
    and return a list of file paths containing that content.
    """
    matched_files = []
    regex = re.compile(pattern)

    for root, _, files in os.walk(directory):
        for file_rela_path in files:
            if not file_rela_path.endswith(file_pattern):
                continue
            file_path = Path(root) / file_rela_path
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if regex.search(content):
                        matched_files.append(str(file_path))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        if not recursive:
            break

    return matched_files

class PatchClassFactoryTransformer(cst.CSTTransformer):
    """
    Merge the conditional class patch into a factory class
    """
    METADATA_DEPENDENCIES = (PositionProvider, ScopeProvider, ParentNodeProvider)

    def __init__(self, module_name, patch_infos):
        self.patch_infos = patch_infos
        self.cls_name = module_name
        if '.' in module_name:
            raise Exception(f"Generating Factory with function {module_name}")

        self.root = None
        self.used_names = {}
        self.to_import = []
        self.modified_patch_nodes = []

    def _merged_name_builder(self, name, patch):
        """
        Obtain the category name of the patch
        """
        patch_import = patch['patch_import']
        if patch_import in self.used_names:
            return self.used_names[patch_import]

        post_fix = f'Patch{len(self.used_names)}'
        self.used_names[patch_import] = f'{name}{post_fix}'
        return self.used_names[patch_import]


    def _merged_branch_builder(self, calls, default_call):
        """
        Build branch nodes
        """
        debug_node = get_debug_print_node(patch=None)
        current_node = cst.Else(
            body=cst.IndentedBlock([debug_node, default_call])  #_indented_block_builder(debug_node, default_call)
        )
        for call, patch in calls.items():
            patch_call_name = self._merged_name_builder(self.cls_name, patch)
            condition, patch_import = patch["condition"], patch["patch_import"]

            if len(condition) == 0:
                condition = 'True'
            elif len(condition) == 1:
                condition = condition[0]
            else:
                condition = ' and '.join(condition)
            condition = condition.replace("args", "global_args")  # fix duplicate name "*args"
            patch_import_module = '.'.join(patch_import.split(".")[:-1])
            patch_import_func = patch_import.split(".")[-1]

            debug_node = get_debug_print_node(patch)
            new_import = cst.parse_statement(f"from {patch_import_module} import {patch_import_func} as {patch_call_name}")
            current_node = cst.If(
                test=cst.parse_expression(condition),
                body=cst.IndentedBlock([debug_node, new_import, call]),
                orelse=current_node   # Bottom-up construction of the branch nodes
            )
        return current_node


    def _build_create_instance_func(self, cls_name, patches):
        """
        The create_instance method of the factory class is used for instantiation
        """
        create_instance_func = cst.FunctionDef(
            name=cst.Name("create_instance"),
            params=cst.Parameters(
                params=[],
                star_kwarg=cst.Param(  # **kwargs
                    name=cst.Name("kwargs"),
                    star="**"
                ),
                star_arg=cst.Param(  # *args
                    name=cst.Name("args"),
                    star="*"
                ),
            ),
            body=[],
            decorators=[cst.Decorator(decorator=cst.Name("staticmethod"))]
        )
        func_args = []
        if create_instance_func.params.star_arg:
            func_args.append(cst.Arg(star="*", value=cst.Name("args")))
        if create_instance_func.params.star_kwarg:
            func_args.append(cst.Arg(star="**", value=cst.Name("kwargs")))

        # create default call
        default_call = cst.SimpleStatementLine([
            cst.Return(
                cst.Call(
                    func=cst.Name(cls_name),
                    args=func_args
                )
        )])

        # create patch calls
        patched_calls = {}
        for patch in patches:
            patch_class_name = self._merged_name_builder(cls_name, patch)

            patch_call = cst.SimpleStatementLine([
                cst.Return(
                    cst.Call(
                        func=cst.Name(patch_class_name),
                        args=func_args
                    )
            )])
            patched_calls[patch_call] = patch

        # assemble nodes
        args_import_node = cst.parse_statement(f"from megatron.training import get_args")
        get_args_node = cst.parse_statement(f"global_args = get_args()")
        branch_node = self._merged_branch_builder(patched_calls, default_call)

        create_instance_func = create_instance_func.with_changes(
            body=cst.IndentedBlock(body=[args_import_node, get_args_node, branch_node])
        )

        return create_instance_func


    def _build_get_class_func(self, cls_name, patches):
        """
        The get_class method of the factory class is used for static method calls
        """
        get_class_func = cst.FunctionDef(
            name=cst.Name("get_class"),
            params=cst.Parameters(params=[]),
            body=[],
            decorators=[cst.Decorator(decorator=cst.Name("staticmethod"))]
        )

        # create default call
        default_call = cst.parse_statement(f"return {cls_name}")

        # create patch calls
        patched_calls = {}
        for patch in patches:
            patch_class_name = self._merged_name_builder(cls_name, patch)
            patch_call = cst.parse_statement(f"return {patch_class_name}")
            patched_calls[patch_call] = patch

        # assemble nodes
        args_import_node = cst.parse_statement(f"from megatron.training import get_args")
        get_args_node = cst.parse_statement(f"global_args = get_args()")
        branch_node = self._merged_branch_builder(patched_calls, default_call)

        get_class_func = get_class_func.with_changes(
            body=cst.IndentedBlock(body=[args_import_node, get_args_node, branch_node])
        )

        return get_class_func


    def _build_factory_def(self, patches):
        """
        Create a factory class definition node
        """
        cls_name = self.cls_name
        factory_name = cls_name + "Factory"
        # build factory class def
        class_factory_class = cst.ClassDef(
            name=cst.Name(factory_name),
            body=[]
        )

        # build create_instance func and get_class_func
        create_instance_func = self._build_create_instance_func(cls_name, patches)
        get_class_func = self._build_get_class_func(cls_name, patches)

        class_factory_class = class_factory_class.with_changes(
            body=cst.IndentedBlock(body=[create_instance_func, cst.EmptyLine(), get_class_func])
        )

        return class_factory_class


    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.CSTNode:
        """
        Add the factory class before the original class definition node
        """
        cls_name = original_node.name.value
        if cls_name != self.cls_name:
            return updated_node
        
        print(f"[DEBUG] Building factory for class {cls_name}")

        class_factory = self._build_factory_def(self.patch_infos)

        return cst.FlattenSentinel([
            cst.EmptyLine(),
            cst.EmptyLine(),
            class_factory,
            original_node,
        ])


class PatchClassCallTransformer(cst.CSTTransformer):
    """
    Replace the original class call statement with a factory class call
    """
    METADATA_DEPENDENCIES = (PositionProvider, ScopeProvider, ParentNodeProvider)

    def __init__(self, cls_name, original_import_name, parent_provider):
        self.cls_name = cls_name
        self.factory_name = cls_name + "Factory"
        self.original_import_name = original_import_name

        self.root = None
        self.used_names = defaultdict(list)
        self.to_import = set()

        self.has_change = False

        self.parent_provider = parent_provider


    def leave_Name(self, original_node, updated_node) -> bool:
        """
        Replace all classes with ClassFactory.get_class()
        """
        parent = self.parent_provider.get(original_node)
        if isinstance(parent, cst.Import) or isinstance(parent, cst.ImportFrom) or isinstance(parent, cst.ImportAlias) or isinstance(parent, cst.Attribute):
            return updated_node

        if updated_node.value == self.cls_name:
            self.has_change = True

            return cst.Call(
                func=cst.Attribute(
                    value=cst.Name(
                        value=self.factory_name,
                    ),
                    attr=cst.Name("get_class")
                )
            )

        return updated_node
    
    def leave_Attribute(self, original_node, updated_node) -> bool:
        """
        Replace all A.b.Class with ClassFactory.get_class()
        """
        parent = self.parent_provider.get(original_node)
        if isinstance(parent, cst.Import) or isinstance(parent, cst.ImportFrom) or isinstance(parent, cst.ImportAlias):
            return updated_node

        if updated_node.attr.value == self.cls_name:
            self.has_change = True

            return cst.Call(
                func=cst.Attribute(
                    value=cst.Name(
                        value=self.factory_name,
                    ),
                    attr=cst.Name("get_class")
                )
            )

        return updated_node
    
    def leave_Module(self, original_node, updated_node):
        if not self.has_change:
            return updated_node

        # Add the import of the factory class
        idx = self.original_import_name.rfind('.' + self.cls_name)
        original_import_from = self.original_import_name[: idx]
        from patch_import_collector import insert_top_level_imports, MImport
        extra_import = [MImport(is_from=True, module=original_import_from, name=self.factory_name)]

        return insert_top_level_imports(updated_node, extra_import)