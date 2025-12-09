# Copyright (c) Huawei Technologies Co., Ltd 2025.  All rights reserved.
import re
from collections import defaultdict
import importlib
import sys
import libcst as cst
from libcst.metadata import PositionProvider, ScopeProvider, ParentNodeProvider
import inflection
from tools.convert.patch_merge.modules.coverage import get_debug_print_node

from tools.convert.patch_merge.modules.patch_import_collector import MImport


class PatchWrapperRouterTransformer(cst.CSTTransformer):
    """
    Merge the wrapper patch into a function decorator
    """
    METADATA_DEPENDENCIES = (PositionProvider, ScopeProvider, ParentNodeProvider)

    def __init__(self, module_name, patch_infos):
        self.origin_import = patch_infos[0]['origin_import']
        self.module_name = module_name
        if '.' in module_name:
            self.is_class_method = True
            self.cls_name, self.func_name = module_name.split('.')
        else:
            self.is_class_method = False
            self.cls_name, self.func_name = None, module_name
        self.patch_infos = patch_infos

        self.root = None

        self.used_names = {}
        self.extra_imports = set()

        self.debug_printing_str = []

        self.modified_patch_nodes = None
        self.metadata = None

    def _merged_name_builder(self, name, patch):
        """
        Get the wrapper alias
        """
        patch_import = patch['patch_import']
        if patch_import in self.used_names:
            return self.used_names[patch_import]

        post_fix = f'_patch_wrapper_{len(self.used_names)}'

        self.used_names[patch_import] = f"{name.strip('_')}_{post_fix}"
        return self.used_names[patch_import]


    def visit_Module(self, node):
        self.root = node
        return True


    def parse_patch(self, patch):
        """
        Get the import path, wrapper name and condition information in the patch
        """
        patch_import, condition = patch['patch_import'], patch['condition']
        patch_import_module = '.'.join(patch_import.split(".")[:-1])
        wrapper_name = patch_import.split(".")[-1]
        return patch_import_module, wrapper_name, condition


    def _build_inner_wrapped_call(self, name, patches, fn, call_with_self):
        """
        Build the wrapper node inside the decorator
        """
        inner_wrapper_body = []

        # Create default statements "wrapped_fn = fn"
        wrapped_fn = cst.AssignTarget(target=cst.Name("wrapped_fn"))
        assign_wrap_default = cst.SimpleStatementLine([
            cst.Assign(
                targets=[wrapped_fn],
                value=fn,
            )
        ])
        inner_wrapper_body.append(assign_wrap_default)

        # Handle unconditional wrapper
        patches_with_condition = []
        for patch in patches:
            patch_import_module, wrapper_origin_name, condition = self.parse_patch(patch)

            if len(condition) > 0:
                patches_with_condition.append(patch)
                continue

            debug_node = get_debug_print_node(patch)
            wrapper_name = self._merged_name_builder(self.func_name, patch)
            new_import = cst.parse_statement(f"from {patch_import_module} import {wrapper_origin_name} as {wrapper_name}")

            assign_wrap = cst.parse_statement(f"{wrapped_fn.target.value} = {wrapper_name}({wrapped_fn.target.value})")
            inner_wrapper_body.extend([debug_node, new_import, assign_wrap])

        # Handle condition import
        if len(patches_with_condition) > 0:
            condition_imports = {MImport(is_from=True, module="megatron.training", name="get_args")}  # from megatron.training import get_args
            for patch in patches:
                if 'condition_import' not in patch['raw_patch']:
                    continue
                for cond_imp in patch['raw_patch']['condition_import']:
                    module, imp_name = cond_imp.rsplit('.', 1)
                    condition_imports.add(MImport(is_from=True, module=module, name=imp_name))
            
            condition_imports = [MImport.mimport_to_cstimport(imp) for imp in condition_imports]

            get_args_node = cst.parse_statement(f"global_args = get_args()")
            inner_wrapper_body.extend([
                cst.EmptyLine(comment=cst.Comment("### condition import start ###")),
                *condition_imports, 
                get_args_node,
                cst.EmptyLine(comment=cst.Comment("### condition import end ###"))])

        # Handle conditional wrappers
        branch_node = []
        for patch in patches_with_condition:
            patch_import_module, wrapper_origin_name, condition = self.parse_patch(patch)

            if len(condition) <= 0:
                raise Exception(f"Patch {patch} has no condition, but is in the conditional wrapper list")

            debug_node = get_debug_print_node(patch)
            wrapper_name = self._merged_name_builder(self.func_name, patch)
            new_import = cst.parse_statement(f"from {patch_import_module} import {wrapper_origin_name} as {wrapper_name}")

            assign_wrap = cst.parse_statement(f"{wrapped_fn.target.value} = {wrapper_name}({wrapped_fn.target.value})")
            assign_wrap_block = cst.IndentedBlock([debug_node, new_import, assign_wrap])

            if len(condition) == 1:
                condition = condition[0]
            else:
                condition = ' and '.join(condition)
            condition = condition.replace("args", "global_args")
            branch_node.append(cst.If(
                test=cst.parse_expression(condition),
                body=assign_wrap_block,
            ))
        if len(branch_node) > 0:
            inner_wrapper_body.extend(branch_node)

        # Create the call statement return wrapped_fn(*args, **kwargs)
        if call_with_self:
            wrapped_fn_call = cst.parse_statement(f"return {wrapped_fn.target.value}(self, *args, **kwargs)")
        else:
            wrapped_fn_call = cst.parse_statement(f"return {wrapped_fn.target.value}(*args, **kwargs)")
        inner_wrapper_body.append(wrapped_fn_call)

        # Assembly node
        inner_wrapper_params=cst.Parameters(
            params=[
                cst.Param(name=cst.Name("self")),
            ] if self.is_class_method else [],
            star_kwarg=cst.Param(  # **kwargs
                name=cst.Name("kwargs"),
                star="**"
            ),
            star_arg=cst.Param(  # *args
                name=cst.Name("args"),
                star="*"
            ),
        )
        inner_wrapper_node = cst.FunctionDef(  # def inner wrapper
            name=cst.Name(value=name),
            params=inner_wrapper_params,
            body=cst.IndentedBlock(body=inner_wrapper_body),
        )

        return inner_wrapper_node


    def _build_outter_decorator_def(self, func_name, patches):
        """
        Build the decorator definition
        """
        fn = cst.Name("fn")

        inner_wrapper_node = self._build_inner_wrapped_call("wrapped_call", patches, fn, self.is_class_method)
        inner_wrapper_node = inner_wrapper_node.with_changes(decorators=[  # add @wraps(fn)
            cst.Decorator(decorator=cst.Call(
                func=cst.Name("wraps"),
                args=[cst.Arg(value=fn)],
        ))])
        
        # Each individual statement (such as return or import) needs to be wrapped in a SimpleStatementLine
        # Otherwise, indentation issues will occur
        wraps_import_node = cst.parse_statement(f"from functools import wraps")
        return_wrapper_node = cst.SimpleStatementLine([cst.Return(value=cst.Expr(value=inner_wrapper_node.name))])
        deco_name = f"{func_name.strip('_')}_decorator"
        print(f"[DEBUG] Create decorator {deco_name} for function {func_name}")
        deco_def_node = cst.FunctionDef(  # def inner wrapper
            name=cst.Name(value=deco_name),
            params=fn,
            body=cst.IndentedBlock(body=[wraps_import_node, inner_wrapper_node, return_wrapper_node]),
        )

        return deco_def_node


    def leave_FunctionDef(self, original_node, updated_node):
        """
        Add decorators to the function
        """
        func_name = original_node.name.value

        if func_name != self.func_name:
            return updated_node
        
        # get class name for class method
        if self.is_class_method:
            class_name = None
            node = original_node
            while True:
                try:
                    parent = self.get_metadata(ParentNodeProvider, node)
                except KeyError:  # reach root node
                    break

                if parent is None:
                    break
                if isinstance(parent, cst.ClassDef):
                    break
                node = parent
                class_name = parent.name.value
            if class_name != self.cls_name:  # not this class's method
                return updated_node

        # build decorator definition
        outter_deco_def_node = self._build_outter_decorator_def(func_name, self.patch_infos)

        # add func decorator
        func_decorator_node = cst.Decorator(decorator=outter_deco_def_node.name)
        new_decorators = (func_decorator_node,) + original_node.decorators
        update_with_wrapper_node = cst.FlattenSentinel([
            outter_deco_def_node,
            cst.EmptyLine(),
            original_node.with_changes(decorators=new_decorators),
        ])

        self.modified_patch_nodes = update_with_wrapper_node
        return update_with_wrapper_node


    def _build_outter_decorator_def_for_cls(self, func_name, cls_name, patches):
        """
        For default class methods (such as dataclass.__init__), wrapper needs to be added through class decorators
        """
        fn = cst.Name("fn")
        cls = cst.Name("cls")

        fn_assign_node = cst.parse_statement(f"{fn.value} = {cls.value}.{func_name}")
        wraps_import_node = cst.parse_statement(f"from functools import wraps")

        inner_wrapper_node = self._build_inner_wrapped_call("wrapped_call", patches, fn, self.is_class_method)
        inner_wrapper_node = inner_wrapper_node.with_changes(decorators=[  # add @wraps(fn)
            cst.Decorator(decorator=cst.Call(
                func=cst.Name("wraps"),
                args=[cst.Arg(value=fn)],
        ))])

        # Update the class function to the wrapped function
        wrapped_assign_node = cst.parse_statement(f"{cls.value}.{func_name} = {inner_wrapper_node.name.value}")
        return_wrapper_node = cst.parse_statement(f"return {cls.value}")

        deco_name = f"{inflection.underscore(cls_name)}_{func_name.strip('_')}_decorator"
        print(f"[DEBUG] Create decorator {deco_name} for class {cls_name} function {func_name}")
        deco_def_node = cst.FunctionDef(  # def inner wrapper
            name=cst.Name(value=deco_name),
            params=cls,
            body=cst.IndentedBlock(body=[fn_assign_node, wraps_import_node, inner_wrapper_node, wrapped_assign_node, return_wrapper_node]),
        )

        return deco_def_node


    def try_wrap_implicit_class_func(self, origin_node):
        """
        Handle default class method (such as the __init__ generated by dataclass) or inherited method
        """
        if self.cls_name is None:
            return None

        update_idx = -1
        body = list(origin_node.body)
        is_data_class = lambda node: len(node.decorators) > 0 and node.decorators[-1].decorator.value == "dataclass"

        for i, node in enumerate(body):
            if not isinstance(node, cst.ClassDef) or node.name.value != self.cls_name:
                continue
            origin_class_node = node

            if is_data_class(node):   # wrap dataclass
                # Build decorator
                outter_deco_def_node = self._build_outter_decorator_def_for_cls(self.func_name, self.cls_name, self.patch_infos)

                # Add decorator to the class
                func_decorator_node = cst.Decorator(decorator=outter_deco_def_node.name)
                new_decorators = (func_decorator_node,) + origin_class_node.decorators
                updated_class_node = [
                    cst.EmptyLine(),
                    outter_deco_def_node,
                    origin_class_node.with_changes(decorators=new_decorators),
                ]
            else:  # Override the inherited func, and wrap the origin func inside
                fn = cst.parse_statement(f"super().{self.func_name}.__func__").body[0]
                wrapped_def = self._build_inner_wrapped_call(self.func_name, self.patch_infos, fn, self.is_class_method)

                class_body = list(origin_class_node.body.body)
                class_body.extend([cst.EmptyLine(), wrapped_def])

                updated_class_node = [origin_class_node.with_changes(body=cst.IndentedBlock(class_body))]

            # Record update index
            update_idx = i
            break

        if update_idx == -1:
            return None
        
        # Insert the updated node
        body = body[: update_idx] + updated_class_node + body[update_idx + 1:]
        new_node = origin_node.with_changes(body=body)

        return new_node

    
    def leave_Module(self, original_node, updated_node):
        if self.modified_patch_nodes is not None:  # patch has already merged
            return updated_node

        print(f"[WARNING] Function to be wrapped not found. module_origin_name: {self.module_name}")

        new_node = self.try_wrap_implicit_class_func(original_node)
        if new_node: return new_node

        raise Exception(f"No module found to be wrapped, original import: {self.origin_import}")

