# Copyright (c) Huawei Technologies Co., Ltd 2025.  All rights reserved.
import re
import libcst as cst
from libcst.metadata import PositionProvider, ScopeProvider
from .utils import get_docstring, case_insensitive_replace, create_nested_attribute_or_name
from pathlib import Path
import json

def load_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

class APITransformer(cst.CSTTransformer):
    """
    cst transformer module to map torch api to msadapter api
    """
    METADATA_DEPENDENCIES = (PositionProvider, ScopeProvider)

    def __init__(self, current_name, new_name, string_mapping):
        self.string_mapping = string_mapping
        self.current_name = current_name
        self.new_name = new_name
        self.root = None
        self.alias_map = {}
        self.support_api = load_json_file(f"{Path(__file__).parent}/../mapping_resources/api_mapping.json")
        self.matched = False

    def _update_docstring(self, original_node, updated_node):
        docstring = get_docstring(original_node)
        if docstring:
            new_value = case_insensitive_replace(docstring.value, self.current_name, self.new_name)
            new_statementline = cst.SimpleStatementLine(body=[cst.Expr(cst.SimpleString(new_value))])
            return updated_node.with_changes(
                body=updated_node.body.with_changes(
                    body=(new_statementline,) + updated_node.body.body[1:]
                )
            )
        return updated_node
    
    def leave_FormattedStringText(self, original_node, updated_node):
        new_value = original_node.value
        for (current_name, new_name) in self.string_mapping:
            pattern = fr'\b{current_name}\.'
            new_value = re.sub(pattern, f'{new_name}.', new_value)
        return updated_node.with_changes(value=new_value)

    def leave_SimpleString(self, original_node, updated_node):
        new_value = original_node.value
        for (current_name, new_name) in self.string_mapping:
            if new_value.strip('"\'').startswith(current_name):
                new_value = re.sub(current_name, new_name, new_value)
        return updated_node.with_changes(value=new_value)

    def _flat_alias_if_possible(self, call_split):
        alias = call_split[0]
        module = self.alias_map.get(alias)
        if module is not None:
            call_split[0] = module
        return '.'.join(call_split)
    
    def visit_Module(self, node):
        self.root = node

    def _parse_alias(self, code):
        pattern = r'(.*) +as +(.*)'
        m = re.match(pattern, code)
        if m:
            full_path, alias = m.groups()
            return full_path, alias
        return None, None

    def visit_Import(self, node):
        guard_this_import = False
        for import_alias in node.names:
            code = self.root.code_for_node(import_alias).strip(' ,')
            full_path, alias = self._parse_alias(code)
            if alias:
                self.alias_map[alias] = full_path
            if 'safetensors' in code:
                guard_this_import = True
        return not guard_this_import
    
    def visit_ImportFrom(self, node):
        if node.module is None: # no need to consider relative import when doing api mapping
            return True
        if isinstance(node.names, cst.ImportStar): # we don't handle from xx import * right now
            return True
        module = self.root.code_for_node(node.module)
        guard_this_import = 'safetensors' in module
        for import_alias in node.names:
            code = self.root.code_for_node(import_alias).strip(' ,')
            sub_path, alias = self._parse_alias(code)
            if alias is None:
                sub_path, alias = code, code
            self.alias_map[alias] = f'{module}.{sub_path}'
        return not guard_this_import
    
    def leave_Call(self, original_node, updated_node):
        if isinstance(original_node.func, cst.Call): # func()()
            return updated_node
        if isinstance(original_node.func.value, cst.Call): # func().f()
            return updated_node
        code = self.root.code_for_node(original_node.func)
        call_split = code.split('.')
        de_aliased_call = self._flat_alias_if_possible(call_split)
        mapped_call = self.support_api.get(de_aliased_call)
        if mapped_call is not None:
            self.matched = True
            return updated_node.with_changes(func=create_nested_attribute_or_name(mapped_call))
        return updated_node

    def leave_Name(self, original_node, updated_node):
        if original_node.value == f'{self.current_name}_npu':
            return updated_node.with_changes(value=f'{self.new_name}_npu')
        if original_node.value == self.current_name:
            return updated_node.with_changes(value=self.new_name)
        return updated_node
    
    def leave_Comment(self, original_node, updated_node):
        new_value = case_insensitive_replace(original_node.value, self.current_name, self.new_name)
        return updated_node.with_changes(value=new_value)
    
    def leave_ClassDef(self, original_node, updated_node):
        return self._update_docstring(original_node, updated_node)
    
    def leave_FunctionDef(self, original_node, updated_node):
        return self._update_docstring(original_node, updated_node)

    def leave_Module(self, original_node, updated_node):
        if not self.matched:
            return updated_node
        import_stmt = "import mindspore"
        new_import = cst.parse_statement(import_stmt)
        nodes = list(updated_node.body)
        insert_index = 0
        for i, node in enumerate(nodes):
            if '__future__' not in self.root.code_for_node(node):
                insert_index = i
                break
        new_body = nodes[:insert_index] + [new_import] + nodes[insert_index:]
        return updated_node.with_changes(body=new_body)
