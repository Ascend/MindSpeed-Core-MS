# Copyright (c) Huawei Technologies Co., Ltd 2025.  All rights reserved.
import re
from collections import defaultdict
import libcst as cst
from libcst.metadata import PositionProvider, ScopeProvider, MetadataWrapper, ParentNodeProvider
from .coverage import get_debug_print_node
from .patch_import_collector import MImport

class PatchFuncRouterTransformer(cst.CSTTransformer):
    """
    Merge the conditional function patch into a routing function
    """
    METADATA_DEPENDENCIES = (PositionProvider, ScopeProvider, ParentNodeProvider)

    def __init__(self, module_name, patch_infos):
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


    def _merged_name_builder(self, name, patch):
        """
        Get the alias of the patch function
        """
        patch_import = patch['patch_import']
        if patch_import in self.used_names:
            return self.used_names[patch_import]

        post_fix = f'patchfunc_{len(self.used_names)}'

        self.used_names[patch_import] = f'{name}_{post_fix}'
        return self.used_names[patch_import]


    def _merged_branch_builder(self, calls, default_call):
        """
        Build function routing branch nodes
        """
        debug_node = get_debug_print_node(patch=None)
        current_node = cst.Else(
            body=cst.IndentedBlock([debug_node, default_call])
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
            condition = condition.replace("args", "global_args")  # bugggggggggggggg, duplicate name *args
            patch_import_module = '.'.join(patch_import.split(".")[:-1])
            patch_import_func = patch_import.split(".")[-1]

            debug_node = get_debug_print_node(patch)
            new_import = cst.parse_statement(f"from {patch_import_module} import {patch_import_func} as {patch_call_name}")
            current_node = cst.If(
                test=cst.parse_expression(condition),
                body=cst.IndentedBlock([debug_node, new_import, call]),
                orelse=current_node   # Bottom-up, use the current node as the "else" branch of the next condition
            )
        return current_node


    def visit_Module(self, node):
        self.root = node
        return True


    def create_default_call(self, default_name, is_class_method, is_self_method, func_args):
        if not is_class_method:
            return cst.SimpleStatementLine([cst.Return(
                cst.Call(
                    func=cst.Name(default_name),
                    args=func_args
                )
            )])

        if is_self_method:
            return cst.SimpleStatementLine([cst.Return(
                cst.Call(
                    func=cst.Attribute(
                        value=cst.Name("self"),
                        attr=cst.Name(default_name)
                    ),
                    args=func_args[1:] # remove param "self"
                )
            )])

        return cst.SimpleStatementLine([cst.Return(
            cst.Call(
                func=cst.Attribute(
                    value=cst.Name(self.cls_name),  # static/cls method call
                    attr=cst.Name(default_name)
                ),
                args=func_args
            )
        )])


    def _build_calls(self, patches, original_node, updated_node, is_class_method):
        """
        Build router function calls based on the patches
        """
        func_name = original_node.name.value

        # get function params
        func_params = original_node.params
        params = []
        params.extend(func_params.params)
        if len(func_params.kwonly_params) > 0 and not isinstance(func_params.star_arg, cst.ParamStar):
            raise Exception("Param list should start with * for keyword-only arguments")
        params.extend(func_params.kwonly_params)
        params.extend(func_params.posonly_params)

        star_params = []
        if func_params.star_arg and func_params.star_arg != cst.MaybeSentinel.DEFAULT and \
                not isinstance(func_params.star_arg, cst.ParamStar):
            star_params.append(func_params.star_arg)
        if func_params.star_kwarg and func_params.star_kwarg != cst.MaybeSentinel.DEFAULT:
            star_params.append(func_params.star_kwarg)


        func_args = []
        func_args.extend([cst.Arg(value=cst.Name(p.name.value)) for p in params])
        func_args.extend([cst.Arg(star=sp.star, value=sp.name) for sp in star_params])

        # is self method or static/class method
        decos = [deco.decorator.value if isinstance(deco, cst.Name) else "" for deco in original_node.decorators]
        is_self_method = "staticmethod" not in decos and "classmethod" not in decos

        # create default call
        default_name = f"{func_name}_default"
        default_call = self.create_default_call(default_name, is_class_method, is_self_method, func_args)

        # create patch calls
        patched_calls = {}
        for patch in patches:
            patch_call_name = self._merged_name_builder(func_name, patch)

            patch_call = cst.SimpleStatementLine([cst.Return(
                cst.Call(
                    func=cst.Name(patch_call_name),
                    args=func_args
                )
            )])
            patched_calls[patch_call] = patch

        # build branch node
        branch_node = self._merged_branch_builder(patched_calls, default_call)

        # collect condition imports
        condition_imports = {MImport(is_from=True, module="megatron.training", name="get_args")}  # from megatron.training import get_args
        for patch in patches:
            if 'condition_import' not in patch['raw_patch']:
                continue
            for cond_imp in patch['raw_patch']['condition_import']:
                module, imp_name = cond_imp.rsplit('.', 1)
                condition_imports.add(MImport(is_from=True, module=module, name=imp_name))
        condition_imports = [MImport.mimport_to_cstimport(imp) for imp in condition_imports]

        # assemble nodes
        get_args_node = cst.parse_statement(f"global_args = get_args()")
        new_node = cst.FunctionDef(
            name=cst.Name(value=func_name),
            params=original_node.params.deep_clone(),
            body=cst.IndentedBlock(body=[
                cst.EmptyLine(comment=cst.Comment("### condition import start ###")),
                *condition_imports, 
                cst.EmptyLine(comment=cst.Comment("### condition import end ###")),
                get_args_node, 
                branch_node]
            )
        )
        return cst.FlattenSentinel([
            original_node.with_changes(name=cst.Name(default_name)),
            cst.EmptyLine(),
            new_node
        ])


    def leave_FunctionDef(self, original_node, updated_node):
        """
        Find the function definition and add the function routing node
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
            if parent is None:
                return updated_node

            class_name = parent.name.value

            if class_name != self.cls_name:  # not this class's method
                return updated_node

        print(f"[DEBUG] building router func for {func_name}" \
            + f" in {class_name}" if self.cls_name is not None else "")

        return self._build_calls(self.patch_infos, original_node, updated_node, self.is_class_method)
