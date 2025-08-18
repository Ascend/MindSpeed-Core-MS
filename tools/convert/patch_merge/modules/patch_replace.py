# Copyright (c) Huawei Technologies Co., Ltd 2025.  All rights reserved.
import re
from collections import defaultdict

import libcst as cst
from libcst.metadata import PositionProvider, ScopeProvider, MetadataWrapper, ParentNodeProvider
from libcst import matchers
from patch_import_collector import (PatchImportCollector, MImport, find_import_for_call, 
                            insert_top_level_imports, get_top_level_imports, get_imports_from_def, get_last_import_index)
from coverage import get_debug_print_node

from typing import Set


class PatchReplaceTransformer(cst.CSTTransformer):
    """
    Unconditional function/class replacement: Move the patch module definition to the original file
    Entry points: leave_ClassDef, leave_FunctionDef, leave_Module
    """
    METADATA_DEPENDENCIES = (PositionProvider, ScopeProvider, ParentNodeProvider)

    def __init__(self, patch, patch_cst):
        self.patch = patch

        self.module_orign_name, self.class_orign_name, self.func_orign_name = patch['module_orign_name']
        self.module_patch_name, self.class_patch_name, self.func_patch_name = patch["module_patch_name"]

        # Collect the imports in the patch node
        self.patch_import_root = patch['patch_import_root']
        self.top_level_imports_in_patch, _ = get_top_level_imports(patch_cst, self.patch_import_root)
        self.import_sources_in_patch = self.top_level_imports_in_patch + get_imports_from_def(patch_cst, self.patch_import_root)

        self.import_source_in_this = None
        self.extra_imports: Set[MImport] = set()

        self.root = None
        self.cur_class = None
        self.patch_cst = patch_cst
        self.do_replace = False


    def visit_Module(self, node):
        """
        Collect the imports in the current node for deduplication
        """
        self.orign_import_root = self.patch['orign_import_root']
        self.top_level_imports_in_this, _ = get_top_level_imports(node, self.orign_import_root)
        self.import_source_in_this = set(self.top_level_imports_in_this + get_imports_from_def(node, self.orign_import_root))


    def visit_ClassDef(self, node):
        cls_name = node.name.value
        self.cur_class = cls_name

        if cls_name != self.class_orign_name:
            return node


    def collect_imports_into_func_node(self, patch_func_node, exclude_names):
        """
        Collect the imports needed in the patch function and add them at the beginning of the function definition
        (Since the original definition will be replaced with a patch definition, the classes/methods used in the patch function 
        need to be imported)
        """
        # Collect the imports involved in the function body
        print(f"[DEBUG] Collecting imports from definition of function {patch_func_node.name.value} in patch file...")
        collector = PatchImportCollector(self.import_sources_in_patch, exclude_names=exclude_names)
        patch_func_node.body.visit(collector)
        local_extra_imports = collector.extra_imports
        local_extra_imports -= self.import_source_in_this   # remove imports/def/class if exists in original file
        print(f"[DEBUG] extra_imports: {local_extra_imports}")

        # Collect the imports involved in the function parameters
        collector = PatchImportCollector(self.import_sources_in_patch, exclude_names=exclude_names)
        patch_func_node.params.visit(collector)
        self.extra_imports |= collector.extra_imports

        updated_body = patch_func_node.body.body
        updated_body = list(updated_body)

        imports = [MImport.mimport_to_cstimport(imp) for imp in local_extra_imports]
        debug_node = [get_debug_print_node(self.patch)]

        updated_body =  [cst.EmptyLine(comment=cst.Comment("### patch import start ###"))] \
                + imports \
                + debug_node \
                + [cst.EmptyLine(comment=cst.Comment("### patch import end ###"))] \
                + [cst.EmptyLine()] \
                + updated_body
        
        patch_func_node = patch_func_node.with_changes(body=cst.IndentedBlock(body=updated_body))
        return patch_func_node


    def get_class_node_from_patch_cst(self):
        """
        Obtain the new class definition node from the patch file
        """
        patch_class_node = None
        idx = -1
        for i, node in enumerate(self.patch_cst.body):
            if isinstance(node, cst.ClassDef) and node.name.value == self.class_patch_name:
                patch_class_node = node.deep_clone()
                idx = i
                break
        
        if patch_class_node is None:
            raise Exception(f"Class {self.class_patch_name} not found in patch file")

        # walk all func def in class, append import from head
        class_body = list(patch_class_node.body.body)
        updated_body = []
        for node in class_body:
            updated_body.append(node)
            if isinstance(node, cst.FunctionDef):
                updated_body[-1] = self.collect_imports_into_func_node(
                    node, exclude_names=[self.func_patch_name, self.class_patch_name])
        patch_class_node = patch_class_node.with_changes(body=cst.IndentedBlock(updated_body))

        return patch_class_node


    def get_inherit_alias(self, patch_class_node):
        """
        If the new class inherits from the original class, find the alias of the original class in the patch file
        """
        if len(patch_class_node.bases) <= 0:
            return False, None

        # Find an alias
        base_names = [ba.value for ba in patch_class_node.bases]
        orign_import = self.patch['orign_import']
        orign_import_parent, class_name = orign_import.rsplit('.', 1)
        orign_imp = MImport(is_from=True, module=orign_import_parent, name=class_name)

        for base_name in base_names:
            base_imp = find_import_for_call(base_name, self.top_level_imports_in_patch)
            if base_imp.equal(orign_imp, compare_asname=False):
                if not base_imp.asname:
                    raise Exception(f"Class {self.class_orign_name} is not imported with alias in patch file")
                return True, base_imp.asname

        return False, None


    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.CSTNode:
        """
        Class definition replacement
        """
        cls_name = original_node.name.value
        self.cur_class = None

        # is func patch
        if self.func_orign_name is not None:
            return updated_node

        # is class patch but didn't hit name
        if cls_name != self.class_orign_name:
            return updated_node
        
        # get new class def from patch_module
        patch_class_node = self.get_class_node_from_patch_cst()

        is_inherit_with_alias, alias_name = self.get_inherit_alias(patch_class_node)

        # import base classes from patch file
        for base in patch_class_node.bases:
            if base.value.value == alias_name:
                continue
            self.extra_imports.add(find_import_for_call(base.value, self.import_sources_in_patch))
        self.extra_imports -= self.import_source_in_this

        self.do_replace = True
        if is_inherit_with_alias:  # If the alias is inherited, perform the addition
            if is_inherit_with_alias:
                updated_node = updated_node.with_changes(name=cst.Name(alias_name))
            return cst.FlattenSentinel([  # New patch class added
                updated_node, 
                cst.EmptyLine(),
                patch_class_node,
            ])
        else: # With the same name, perform the replacement
            print(f"[DEBUG] Replacing definition of {self.class_orign_name} from patch class {self.class_patch_name}")
            if self.class_orign_name != self.class_patch_name:  # Replace with the original class name
                return patch_class_node.with_changes(name=cst.Name(self.class_orign_name))
            return patch_class_node


    def get_func_node_from_patch_cst(self):
        """
        Obtain the new function definition node from the patch file
        """
        patch_func_node = None
        idx = -1
        for i, node in enumerate(self.patch_cst.body):
            if isinstance(node, cst.FunctionDef):
                if node.name.value == self.func_patch_name:
                    patch_func_node = node.deep_clone()
                    idx = i
                    break
        
        if patch_func_node is None:
            raise Exception(f"Function {self.func_patch_name} not found in patch file")

        patch_func_node = self.collect_imports_into_func_node(
            patch_func_node, exclude_names=[self.func_patch_name, self.class_patch_name])

        return patch_func_node


    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.CSTNode:
        """
        Function definition replacement
        """
        func_name = original_node.name.value

        # is classpatch
        if self.class_orign_name is not None and self.func_orign_name is None:
            return updated_node

        # is func patch but didn't hit name
        if self.func_orign_name is not None and func_name != self.func_orign_name:
            return updated_node

        # is class func patch but didn't hit class
        if self.class_orign_name is not None and self.cur_class != self.class_orign_name:
            return updated_node
        
        patch_func_node = self.get_func_node_from_patch_cst()

        self.do_replace = True
        print(f"[DEBUG] Replacing definition of {self.module_orign_name} from patch function {self.module_patch_name}")

        return updated_node.with_changes(params=patch_func_node.params, body=patch_func_node.body)


    def try_patch_variable(self, updated_node):
        """
        Special case: patch the class to the variable
        """
        print(f"[DEBUG] Trying find module {self.func_orign_name} in variable assignment")

        is_orign_module_var = self.class_orign_name is None and self.func_orign_name is not None
        is_patch_module_class = self.class_patch_name is not None and self.func_patch_name is None

        if not is_orign_module_var or not is_patch_module_class:
            return updated_node

        update_idx = -1
        body = list(updated_node.body)
        for i, node in enumerate(body):
            if matchers.matches(node, matchers.SimpleStatementLine(body=[matchers.Assign()])):
                assign_node = node.body[0]
                if len(assign_node.targets) != 1:
                    raise Exception(f"Unexpected assignment node: {assign_node.targets} in {self.module_orign_name}")
                if assign_node.targets[0].target.value != self.func_orign_name:
                    continue
                
                update_idx = i
                new_assign_node = assign_node.with_changes(value=cst.Name(self.class_patch_name))
                body[i] = new_assign_node
        
        if update_idx == -1:
            print("[DEBUG] No variable found to be patched")
            return updated_node

        class_def_node = self.get_class_node_from_patch_cst()
        body = body[:update_idx] + [class_def_node] + body[update_idx:]
        self.do_replace = True
        return updated_node.with_changes(body=body)


    def try_patch_implict_class_func(self, updated_node):
        """
        Special case: Class function rewriting or addition
        """
        print(f"[DEBUG] Trying find module {self.func_orign_name} in implict class func")

        if self.class_orign_name is None or self.func_orign_name is None:
            return updated_node

        root_body = list(updated_node.body)
        orign_class_node, patch_func_node = None, None
        for i, node in enumerate(root_body):
            if not isinstance(node, cst.ClassDef) or node.name.value != self.class_orign_name:
                continue
            orign_class_node = node

            patch_func_node = self.get_func_node_from_patch_cst()
            if self.func_orign_name != self.func_patch_name:
                patch_func_node = patch_func_node.with_changes(name=cst.Name(self.func_orign_name))
            
            class_body = list(orign_class_node.body.body)
            class_body.append(patch_func_node)
            orign_class_node = orign_class_node.with_changes(body=cst.IndentedBlock(class_body))
            root_body[i] = orign_class_node
            break
        
        if orign_class_node and patch_func_node:
            self.do_replace = True
            return updated_node.with_changes(body=root_body)
        
        return updated_node


    def try_patch_to_end(self, updated_node):
        """
        Special case: patch a non-exist module definition to the end of the file
        """
        if self.func_patch_name is not None:
            patch_node = self.get_func_node_from_patch_cst()
        else:
            patch_node = self.get_class_node_from_patch_cst()

        print(f"[DEBUG] Trying append module {patch_node.name.value} at the end of file")

        body = list(updated_node.body)
        body.append(patch_node)

        return updated_node.with_changes(body=body)


    def leave_Module(self, original_node, updated_node):
        if not self.do_replace:
            updated_node = self.try_patch_variable(updated_node)

        if not self.do_replace:
            updated_node = self.try_patch_implict_class_func(updated_node)

        if not self.do_replace:
            print(f"[WARNING] Class or function to be replaced not found." \
                    f"module_orign_name: {self.module_orign_name}, module_patch_name: {self.module_patch_name}")

            updated_node = self.try_patch_to_end(updated_node)

        if len(self.extra_imports) > 0:
            self.extra_imports -= self.import_source_in_this
            return insert_top_level_imports(updated_node, self.extra_imports)
        
        return updated_node


class PatchClassNodeRemover(cst.CSTTransformer):
    """
    Remove the class/function definitions in the patch file and only retain the definitions in the original file
    It is used to fix the issue of multiple definitions in gpt_layer_spec.py
    """
    def __init__(self, patch_infos) -> None:
        self.class_name_map = {}
        for patch in patch_infos:
            class_patch_name = patch['module_patch_name'][1]
            orign_import_root = patch['orign_import_root']
            class_orign_name = patch['module_orign_name'][1]
            if class_patch_name not in self.class_name_map:
                self.class_name_map[class_patch_name] = (orign_import_root, class_orign_name)
        if None in self.class_name_map:
            raise Exception("Got None class name to be removed.")

        self.orign_import_roots = [patch['orign_import_root'] for patch in patch_infos]

        self.has_removed = False

    def leave_ClassDef(self, original_node, updated_node):
        if updated_node.name.value in self.class_name_map:
            return cst.RemoveFromParent()

        return updated_node
    
    def leave_Module(self, original_node, updated_node):
        extra_imports = []
        for class_patch_name, (orign_import_root, class_orign_name) in self.class_name_map.items():
            extra_imports.append(MImport(is_from=True, module=orign_import_root, name=class_orign_name))
            if class_patch_name != class_orign_name:
                extra_imports[-1].asname = class_patch_name

        return insert_top_level_imports(updated_node, extra_imports)
