# Copyright (c) Huawei Technologies Co., Ltd 2025.  All rights reserved.
import libcst as cst
from libcst import matchers
from libcst.metadata import PositionProvider, ScopeProvider
from collections import defaultdict
from libcst.helpers import get_full_name_for_node
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MImport:
    """
    Record the import node
        - from <module> import <name> as <asname>
        - import <name> as <asname>
    """
    name: str
    is_from: bool = False
    module: str = None
    asname: str = None

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise Exception(f"Expect string name, got {type(self.name)}")
        if self.is_from and self.module is None:
            raise Exception("Module is required for 'from' import")

        if not self.name:
            raise Exception(f"Expect non-empty name, got {self.name}")

    def __eq__(self, other: object) -> bool:
        return self.equal(other, compare_asname=True)
    
    def __hash__(self) -> int:
        return hash((self.name, self.is_from, self.asname))

    def equal(self, other, compare_asname):
        """
        equal imports: equal names and inclusion in the import path
        """
        if self.is_from != other.is_from:
            return False
        
        is_equal = self.name == other.name
        if self.is_from:
            is_equal &= \
                (self.module in other.module or other.module in self.module)

        if compare_asname:
            is_equal &= \
                (self.asname == other.asname)

        return is_equal

    def __str__(self):
        imp_str = ""
        if self.is_from:
            imp_str = f"from {self.module} import {self.name}"
        else:
            imp_str = f"import {self.name}"
        
        if self.asname:
            imp_str += f" as {self.asname}"
        
        return imp_str

    def is_source(self, name):
        self_name = self.asname if self.asname else self.name
        return self_name == name

    @staticmethod
    def mimport_to_cstimport(mimport, ret_with_simpline=True):
        node = cst.parse_statement(str(mimport))
        if ret_with_simpline:
            return node
        return node.body[0]
    
    @staticmethod
    def cstimport_to_mimport(cstimport, relative_import_root=""):
        if isinstance(cstimport, cst.SimpleStatementLine):
            imp = cstimport.body[0]
        else:
            imp = cstimport
        
        imports = []
        names = imp.names
        if isinstance(imp, cst.Import):
            for alias in names:  # Split multiple imports in one statement into multiple ones
                if isinstance(alias.name, cst.Attribute):
                    name = cst.Module([]).code_for_node(alias.name)
                else:
                    name = alias.name.value
                asname = alias.asname.name.value if alias.asname else None
                imports.append(MImport(name=name, asname=asname))
        elif isinstance(imp, cst.ImportFrom):
            for alias in names:
                module_str = cst.Module([]).code_for_node(imp.module) if imp.module is not None else ""

                if len(imp.relative) > 0:   # from .a.b import c --> from path.to.a.b import c
                    if not relative_import_root:
                        raise Exception("Relative import requires import root, but got empty")

                    # Take the import path forward based on the number of points
                    module_path = relative_import_root.split('.')
                    module_path = '.'.join(module_path[:-len(imp.relative)])   

                    if module_str:
                        module_str = f"{module_path}.{module_str}"
                    else:
                        module_str = module_path
                name = alias.name.value
                asname = alias.asname.name.value if alias.asname else None
                imports.append(MImport(is_from=True, module=module_str, name=name, asname=asname))

        return imports


def get_top_level_imports(module, import_root=""):
    """
    Obtain all top-level imports. For relative imports, add import_root to change them to absolute path imports
    """
    def is_import(node):
        return matchers.matches(node, matchers.SimpleStatementLine(body=[matchers.Import() | matchers.ImportFrom()]))

    imports = []
    last_import_index = -1
    for i, node in enumerate(module.body):
        if is_import(node):
            imports.extend(MImport.cstimport_to_mimport(node, import_root))
            last_import_index = i
        elif matchers.matches(node, matchers.Try()):
            imports.extend(get_top_level_imports(node.body, import_root)[0])
            last_import_index = i

    return imports, last_import_index


def get_last_import_index(root):
    """
    Obtain the index of the last top-level import for inserting the import node
    """
    last_import_index = 0
    for i, node in enumerate(root.body):
        if matchers.matches(node, matchers.SimpleStatementLine(body=[matchers.Import() | matchers.ImportFrom()])):
            last_import_index = i
        elif matchers.matches(node, matchers.Try()):
            if get_last_import_index(node.body) != -1:  # recursively get imports in try-except
                last_import_index = i
    return last_import_index

def insert_top_level_imports(root, mimports: List[MImport]):
    """
    Insert the input import into the cst node root
    """
    nodes = list(root.body)
    last_import_index = get_last_import_index(root)

    imports = [cst.parse_statement(str(imp)) for imp in mimports]

    if len(imports) > 0:
        new_body = (nodes[:last_import_index + 1] 
                + [cst.EmptyLine(), cst.EmptyLine(comment=cst.Comment("### patch import start ###"))]
                + imports 
                + [cst.EmptyLine(comment=cst.Comment("### patch import end ###")), cst.EmptyLine()]
                + nodes[last_import_index + 1:])
        
        return root.with_changes(body=new_body)

    return root


def get_imports_from_def(module, import_root):
    """
    Collect the function /class definitions in the form of "from <import_root> import <func/class def>"
    """
    imports = []
    for _, node in enumerate(module.body):
        if matchers.matches(node, matchers.FunctionDef()) or matchers.matches(node, matchers.ClassDef()):
            imports.append(MImport(is_from=True, module=import_root, name=node.name.value))
    return imports

def find_import_for_call(
    name: cst.Name,
    all_imports: List[MImport],
) -> Optional[MImport]:
    """
    Given a Call node, try to find out the import statement it uses.
    """
    # Extract the root name of the call recursively (foo.bar() → 'foo')
    def get_root_name(node: cst.CSTNode) -> Optional[str]:
        if isinstance(node, cst.Name):
            return node.value
        elif isinstance(node, cst.Attribute):
            return get_root_name(node.value)
        return None
    root_name = get_root_name(name)

    if not root_name:
        return None

    for imp in all_imports:
        if imp.is_source(root_name):
            return imp

    # Reserved for builtin functions 
    if is_builtin_function(root_name):
        return None

    return None


def is_builtin_function(name: str) -> bool:
    return name in dir(__builtins__)


class PatchImportCollector(cst.CSTVisitor):
    '''
    Given 'import_sources', collect the import statements involved in a node
    '''
    METADATA_DEPENDENCIES = (PositionProvider, ScopeProvider)

    def __init__(self, import_sources, exclude_names=None):
        self.builtins = set(dir(__builtins__))
        self.local_defs = set()

        if exclude_names is None:
            self.exclude_names = []
        else:
            self.exclude_names = exclude_names

        self.calls = defaultdict(list)
        self.import_strs_from_calls = None
        self.import_sources = import_sources
        self.extra_imports = set()

    def visit_Name(self, node: cst.Name) -> bool:
        if node.value in self.exclude_names:
            return True
        mimport = find_import_for_call(node, self.import_sources)
        if mimport is not None:
            self.extra_imports.add(mimport)

        return False
