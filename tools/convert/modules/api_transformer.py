# Copyright (c) Huawei Technologies Co., Ltd 2012-2020.  All rights reserved.
import re
import libcst as cst
from libcst.metadata import PositionProvider, ScopeProvider
from .utils import get_docstring, case_insensitive_replace


class APITransformer(cst.CSTTransformer):
    """
    cst transformer module to map torch api to msadapter api
    """
    METADATA_DEPENDENCIES = (PositionProvider, ScopeProvider)

    def __init__(self, current_name, new_name):
        self.current_name = current_name
        self.new_name = new_name
        self.root = None
        self.imported_modules = {}
        self.alias_map = {} # to be merged into imported_modules
    
    def _import_chain_to_name(self, chain):
        for i, c in enumerate(chain):
            if c != '.':
                break
        return '.'.join(chain[:i]) + '.'.join(chain[i:])

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

    def _get_call_chain_if_normal_call(self, call):
        chain = call.split('.')
        if len(chain) < 1:
            return []
        for c in chain:
            if len(c) == 0 or re.match(r'^[a-zA-Z\_]*$', c) is None:
                return []
        return chain
    
    def _flat_alias_if_possible(self, call_chain):
        alias = call_chain[0]
        if alias in self.alias_map:
            return self.alias_map[map] + call_chain[1:]
        return call_chain
    
    def _name_or_attribute_decoder(self, tmp_node):
        attribute_chain = []
        while isinstance(tmp_node, cst.Attribute):
            attribute_chain.append(tmp_node.attr.value)
            tmp_node = tmp_node.value
        attribute_chain.append(tmp_node.value)
        return attribute_chain[::-1]
    
    def _parse_import_alias(self, import_alias):
        import_chain = self._name_or_attribute_decoder(import_alias.name)
        asname = None
        if import_alias.asname is not None:
            asname = import_alias.asname.name.value
        return asname, import_chain
    
    def visit_Module(self, node):
        self.root = node

    def visit_Import(self, node):
        for import_alias in node.names:
            alias, chain = self._parse_import_alias(import_alias)
            if alias:
                self.alias_map[alias] = chain
                self.imported_modules[alias] = chain
            else:
                fake_alias = self._import_chain_to_name(chain)
                self.imported_modules[fake_alias] = chain
        return True
    
    def visit_ImportFrom(self, node):
        if node.module is None: # relative import
            from_chain = ["."] * len(node.relative)
        else:
            from_chain = self._name_or_attribute_decoder(node.module)
        if isinstance(node.names, cst.ImportStar):
            return True
        for import_alias in node.names:
            alias, chain = self._parse_import_alias(import_alias)
            if len(chain) == 1 and alias is None:
                alias = chain[0]
            self.alias_map[alias] = from_chain + chain
            self.imported_modules[alias] = chain
        return True

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
