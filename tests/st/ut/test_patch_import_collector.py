# Copyright 2025 Huawei Technologies Co., Ltd
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

from patch_import_collector import (
    MImport,
    get_top_level_imports,
    get_last_import_index,
    insert_top_level_imports,
    get_imports_from_def,
    find_import_for_call,
    is_builtin_function,
    PatchImportCollector
)



class TestMImport:
    """Test cases for MImport dataclass"""

    def test_mimport_simple_import(self):
        """Test MImport for simple import statement"""
        imp = MImport(name="os")
        assert imp.name == "os"
        assert imp.is_from is False
        assert imp.module is None
        assert imp.asname is None

    def test_mimport_import_with_alias(self):
        """Test MImport for import with alias"""
        imp = MImport(name="numpy", asname="np")
        assert imp.name == "numpy"
        assert imp.asname == "np"
        assert imp.is_from is False

    def test_mimport_from_import(self):
        """Test MImport for 'from' import statement"""
        imp = MImport(name="load", is_from=True, module="torch")
        assert imp.name == "load"
        assert imp.is_from is True
        assert imp.module == "torch"

    def test_mimport_from_import_with_alias(self):
        """Test MImport for 'from' import with alias"""
        imp = MImport(name="Function", is_from=True, module="torch.autograd", asname="F")
        assert imp.name == "Function"
        assert imp.module == "torch.autograd"
        assert imp.asname == "F"

    def test_mimport_invalid_name_type(self):
        """Test MImport raises exception for non-string name"""
        with pytest.raises(Exception, match="Expect string name"):
            MImport(name=123)

    def test_mimport_empty_name(self):
        """Test MImport raises exception for empty name"""
        with pytest.raises(Exception, match="Expect non-empty name"):
            MImport(name="")

    def test_mimport_from_without_module(self):
        """Test MImport raises exception for 'from' import without module"""
        with pytest.raises(Exception, match="Module is required"):
            MImport(name="func", is_from=True)

    def test_mimport_equality_simple(self):
        """Test MImport equality for simple imports"""
        imp1 = MImport(name="os")
        imp2 = MImport(name="os")
        assert imp1 == imp2

    def test_mimport_equality_with_alias(self):
        """Test MImport equality with aliases"""
        imp1 = MImport(name="numpy", asname="np")
        imp2 = MImport(name="numpy", asname="np")
        imp3 = MImport(name="numpy", asname="numpy_alias")
        assert imp1 == imp2
        assert imp1 != imp3

    def test_mimport_equality_from_import(self):
        """Test MImport equality for 'from' imports"""
        imp1 = MImport(name="load", is_from=True, module="torch")
        imp2 = MImport(name="load", is_from=True, module="torch")
        assert imp1 == imp2

    def test_mimport_equal_method_with_module_inclusion(self):
        """Test MImport equal method with module path inclusion"""
        imp1 = MImport(name="func", is_from=True, module="a.b.c")
        imp2 = MImport(name="func", is_from=True, module="a.b")
        assert imp1.equal(imp2, compare_asname=False)

    def test_mimport_hash(self):
        """Test MImport hash function"""
        imp1 = MImport(name="os")
        imp2 = MImport(name="os")
        assert hash(imp1) == hash(imp2)
        
        imp_set = {imp1, imp2}
        assert len(imp_set) == 1  # Same imports should hash to same value

    def test_mimport_str_simple_import(self):
        """Test MImport string representation for simple import"""
        imp = MImport(name="os")
        assert str(imp) == "import os"

    def test_mimport_str_import_with_alias(self):
        """Test MImport string representation for import with alias"""
        imp = MImport(name="numpy", asname="np")
        assert str(imp) == "import numpy as np"

    def test_mimport_str_from_import(self):
        """Test MImport string representation for 'from' import"""
        imp = MImport(name="load", is_from=True, module="torch")
        assert str(imp) == "from torch import load"

    def test_mimport_str_from_import_with_alias(self):
        """Test MImport string representation for 'from' import with alias"""
        imp = MImport(name="nn", is_from=True, module="torch", asname="neural_net")
        assert str(imp) == "from torch import nn as neural_net"

    def test_mimport_is_source_without_alias(self):
        """Test MImport is_source method without alias"""
        imp = MImport(name="torch")
        assert imp.is_source("torch") is True
        assert imp.is_source("numpy") is False

    def test_mimport_is_source_with_alias(self):
        """Test MImport is_source method with alias"""
        imp = MImport(name="numpy", asname="np")
        assert imp.is_source("np") is True
        assert imp.is_source("numpy") is False

    def test_mimport_to_cstimport_simple(self):
        """Test converting MImport to CST import node"""
        imp = MImport(name="os")
        cst_node = MImport.mimport_to_cstimport(imp)
        assert isinstance(cst_node, cst.SimpleStatementLine)
        assert isinstance(cst_node.body[0], cst.Import)

    def test_mimport_to_cstimport_from_import(self):
        """Test converting MImport to CST from import node"""
        imp = MImport(name="load", is_from=True, module="torch")
        cst_node = MImport.mimport_to_cstimport(imp)
        assert isinstance(cst_node, cst.SimpleStatementLine)
        assert isinstance(cst_node.body[0], cst.ImportFrom)

    def test_mimport_to_cstimport_without_simpline(self):
        """Test converting MImport to CST without SimpleStatementLine wrapper"""
        imp = MImport(name="os")
        cst_node = MImport.mimport_to_cstimport(imp, ret_with_simpline=False)
        assert isinstance(cst_node, cst.Import)

    def test_cstimport_to_mimport_simple_import(self):
        """Test converting CST import to MImport"""
        code = "import os"
        module = cst.parse_module(code)
        cst_import = module.body[0]
        imports = MImport.cstimport_to_mimport(cst_import)
        assert len(imports) == 1
        assert imports[0].name == "os"
        assert imports[0].is_from is False

    def test_cstimport_to_mimport_import_with_alias(self):
        """Test converting CST import with alias to MImport"""
        code = "import numpy as np"
        module = cst.parse_module(code)
        cst_import = module.body[0]
        imports = MImport.cstimport_to_mimport(cst_import)
        assert len(imports) == 1
        assert imports[0].name == "numpy"
        assert imports[0].asname == "np"

    def test_cstimport_to_mimport_from_import(self):
        """Test converting CST from import to MImport"""
        code = "from torch import nn"
        module = cst.parse_module(code)
        cst_import = module.body[0]
        imports = MImport.cstimport_to_mimport(cst_import)
        assert len(imports) == 1
        assert imports[0].name == "nn"
        assert imports[0].is_from is True
        assert imports[0].module == "torch"

    def test_cstimport_to_mimport_multiple_imports(self):
        """Test converting CST with multiple imports in one statement"""
        code = "import os, sys, json"
        module = cst.parse_module(code)
        cst_import = module.body[0]
        imports = MImport.cstimport_to_mimport(cst_import)
        assert len(imports) == 3
        assert imports[0].name == "os"
        assert imports[1].name == "sys"
        assert imports[2].name == "json"

    def test_cstimport_to_mimport_relative_import(self):
        """Test converting CST relative import to MImport"""
        code = "from .module import func"
        module = cst.parse_module(code)
        cst_import = module.body[0]
        imports = MImport.cstimport_to_mimport(cst_import, relative_import_root="package.subpackage")
        assert len(imports) == 1
        assert imports[0].name == "func"
        assert imports[0].module == "package.module"

    def test_cstimport_to_mimport_relative_import_no_root(self):
        """Test converting CST relative import without import root raises exception"""
        code = "from .module import func"
        module = cst.parse_module(code)
        cst_import = module.body[0]
        with pytest.raises(Exception, match="Relative import requires import root"):
            MImport.cstimport_to_mimport(cst_import)

    def test_cstimport_to_mimport_double_relative_import(self):
        """Test converting CST double-dot relative import"""
        code = "from ..module import func"
        module = cst.parse_module(code)
        cst_import = module.body[0]
        imports = MImport.cstimport_to_mimport(cst_import, relative_import_root="package.sub1.sub2")
        assert len(imports) == 1
        assert imports[0].module == "package.module"

    def test_cstimport_to_mimport_dotted_import(self):
        """Test converting CST dotted import to MImport"""
        code = "import torch.nn"
        module = cst.parse_module(code)
        cst_import = module.body[0]
        imports = MImport.cstimport_to_mimport(cst_import)
        assert len(imports) == 1
        assert imports[0].name == "torch.nn"



class TestGetTopLevelImports:
    """Test cases for get_top_level_imports function"""

    def test_get_top_level_imports_simple(self):
        """Test getting top-level imports from simple module"""
        code = """import os
import sys
"""
        module = cst.parse_module(code)
        imports, last_index = get_top_level_imports(module)
        assert len(imports) == 2
        assert imports[0].name == "os"
        assert imports[1].name == "sys"
        assert last_index == 1

    def test_get_top_level_imports_mixed(self):
        """Test getting mixed import styles"""
        code = """import os
from torch import nn
import sys as system
"""
        module = cst.parse_module(code)
        imports, last_index = get_top_level_imports(module)
        assert len(imports) == 3
        assert imports[0].name == "os"
        assert imports[1].name == "nn"
        assert imports[1].is_from is True
        assert imports[2].name == "sys"
        assert imports[2].asname == "system"

    def test_get_top_level_imports_with_code_after(self):
        """Test getting imports when code follows"""
        code = """import os
import sys

def func():
    pass
"""
        module = cst.parse_module(code)
        imports, last_index = get_top_level_imports(module)
        assert len(imports) == 2
        assert last_index == 1

    def test_get_top_level_imports_in_try_block(self):
        """Test getting imports from try-except blocks"""
        code = """import os
try:
    import special_module
except ImportError:
    pass
"""
        module = cst.parse_module(code)
        imports, last_index = get_top_level_imports(module)
        assert len(imports) == 2
        assert imports[0].name == "os"
        assert imports[1].name == "special_module"

    def test_get_top_level_imports_with_relative_import(self):
        """Test getting imports with relative paths"""
        code = """from .module import func"""
        module = cst.parse_module(code)
        imports, last_index = get_top_level_imports(module, import_root="package.subpackage")
        assert len(imports) == 1
        assert imports[0].module == "package.module"

    def test_get_top_level_imports_empty_module(self):
        """Test getting imports from empty module"""
        code = ""
        module = cst.parse_module(code)
        imports, last_index = get_top_level_imports(module)
        assert len(imports) == 0
        assert last_index == -1

    def test_get_top_level_imports_no_imports(self):
        """Test getting imports from module with no imports"""
        code = """def func():
    pass

class MyClass:
    pass
"""
        module = cst.parse_module(code)
        imports, last_index = get_top_level_imports(module)
        assert len(imports) == 0
        assert last_index == -1



class TestGetLastImportIndex:
    """Test cases for get_last_import_index function"""

    def test_get_last_import_index_simple(self):
        """Test getting last import index from simple module"""
        code = """import os
import sys
import json
"""
        module = cst.parse_module(code)
        last_index = get_last_import_index(module)
        assert last_index == 2

    def test_get_last_import_index_with_code_after(self):
        """Test getting last import index when code follows"""
        code = """import os
import sys

def func():
    pass
"""
        module = cst.parse_module(code)
        last_index = get_last_import_index(module)
        assert last_index == 1

    def test_get_last_import_index_with_try_block(self):
        """Test getting last import index with try-except"""
        code = """import os
try:
    import special
except:
    pass

def func():
    pass
"""
        module = cst.parse_module(code)
        last_index = get_last_import_index(module)
        assert last_index == 1



class TestInsertTopLevelImports:
    """Test cases for insert_top_level_imports function"""

    def test_insert_top_level_imports_simple(self):
        """Test inserting imports into module"""
        code = """import os
import sys

def func():
    pass
"""
        module = cst.parse_module(code)
        new_imports = [MImport(name="torch"), MImport(name="numpy", asname="np")]
        new_module = insert_top_level_imports(module, new_imports)
        new_code = new_module.code
        assert "import torch" in new_code
        assert "import numpy as np" in new_code
        assert "### patch import start ###" in new_code
        assert "### patch import end ###" in new_code

    def test_insert_top_level_imports_empty_list(self):
        """Test inserting empty import list doesn't change module"""
        code = """import os

def func():
    pass
"""
        module = cst.parse_module(code)
        new_module = insert_top_level_imports(module, [])
        assert new_module.code == code

    def test_insert_top_level_imports_from_import(self):
        """Test inserting 'from' imports"""
        code = """import os

def func():
    pass
"""
        module = cst.parse_module(code)
        new_imports = [MImport(name="nn", is_from=True, module="torch")]
        new_module = insert_top_level_imports(module, new_imports)
        new_code = new_module.code
        assert "from torch import nn" in new_code



class TestGetImportsFromDef:
    """Test cases for get_imports_from_def function"""

    def test_get_imports_from_def_functions(self):
        """Test collecting function definitions as imports"""
        code = """def func1():
    pass

def func2():
    pass
"""
        module = cst.parse_module(code)
        imports = get_imports_from_def(module, "mymodule")
        assert len(imports) == 2
        assert imports[0].name == "func1"
        assert imports[0].module == "mymodule"
        assert imports[0].is_from is True
        assert imports[1].name == "func2"

    def test_get_imports_from_def_classes(self):
        """Test collecting class definitions as imports"""
        code = """class MyClass:
    pass

class AnotherClass:
    pass
"""
        module = cst.parse_module(code)
        imports = get_imports_from_def(module, "mymodule")
        assert len(imports) == 2
        assert imports[0].name == "MyClass"
        assert imports[1].name == "AnotherClass"

    def test_get_imports_from_def_mixed(self):
        """Test collecting mixed function and class definitions"""
        code = """def my_func():
    pass

class MyClass:
    pass

x = 10
"""
        module = cst.parse_module(code)
        imports = get_imports_from_def(module, "mymodule")
        assert len(imports) == 2
        assert imports[0].name == "my_func"
        assert imports[1].name == "MyClass"

    def test_get_imports_from_def_empty(self):
        """Test collecting from module with no definitions"""
        code = """x = 10
y = 20
"""
        module = cst.parse_module(code)
        imports = get_imports_from_def(module, "mymodule")
        assert len(imports) == 0



class TestFindImportForCall:
    """Test cases for find_import_for_call function"""

    def test_find_import_for_call_simple(self):
        """Test finding import for simple call"""
        name = cst.Name("torch")
        imports = [MImport(name="torch")]
        result = find_import_for_call(name, imports)
        assert result is not None
        assert result.name == "torch"

    def test_find_import_for_call_with_alias(self):
        """Test finding import for call using alias"""
        name = cst.Name("np")
        imports = [MImport(name="numpy", asname="np")]
        result = find_import_for_call(name, imports)
        assert result is not None
        assert result.name == "numpy"

    def test_find_import_for_call_attribute(self):
        """Test finding import for attribute access"""
        code = "torch.nn"
        module = cst.parse_module(code)
        expr = module.body[0].body[0].value
        imports = [MImport(name="torch")]
        result = find_import_for_call(expr, imports)
        assert result is not None
        assert result.name == "torch"

    def test_find_import_for_call_not_found(self):
        """Test finding import for call that doesn't exist"""
        name = cst.Name("unknown")
        imports = [MImport(name="torch")]
        result = find_import_for_call(name, imports)
        assert result is None

    def test_find_import_for_call_builtin(self):
        """Test finding import for builtin function returns None"""
        name = cst.Name("print")
        imports = []
        result = find_import_for_call(name, imports)
        assert result is None



class TestIsBuiltinFunction:
    """Test cases for is_builtin_function"""

    def test_is_builtin_function_custom(self):
        """Test that custom function is not recognized as builtin"""
        assert is_builtin_function("my_custom_func") is False

    def test_is_builtin_function_module(self):
        """Test that module names are not recognized as builtin"""
        assert is_builtin_function("torch") is False
        assert is_builtin_function("numpy") is False



class TestPatchImportCollector:
    """Test cases for PatchImportCollector class"""

    def test_patch_import_collector_init(self):
        """Test PatchImportCollector initialization"""
        imports = [MImport(name="torch")]
        collector = PatchImportCollector(imports)
        assert collector.import_sources == imports
        assert collector.exclude_names == []
        assert len(collector.extra_imports) == 0

    def test_patch_import_collector_init_with_exclude(self):
        """Test PatchImportCollector initialization with exclude names"""
        imports = [MImport(name="torch")]
        exclude = ["self", "cls"]
        collector = PatchImportCollector(imports, exclude_names=exclude)
        assert collector.exclude_names == exclude

    def test_patch_import_collector_visit_name(self):
        """Test PatchImportCollector visiting Name nodes"""
        code = """torch.tensor([1, 2, 3])"""
        module = cst.parse_module(code)
        imports = [MImport(name="torch")]
        collector = PatchImportCollector(imports)
        
        wrapper = cst.MetadataWrapper(module)
        wrapper.visit(collector)
        
        assert len(collector.extra_imports) == 1
        assert list(collector.extra_imports)[0].name == "torch"

    def test_patch_import_collector_visit_excluded_name(self):
        """Test PatchImportCollector skips excluded names"""
        code = """self.value"""
        module = cst.parse_module(code)
        imports = [MImport(name="self")]
        collector = PatchImportCollector(imports, exclude_names=["self"])
        
        wrapper = cst.MetadataWrapper(module)
        wrapper.visit(collector)
        
        assert len(collector.extra_imports) == 0

    def test_patch_import_collector_builtin_ignored(self):
        """Test PatchImportCollector ignores builtin functions"""
        code = """print("hello")"""
        module = cst.parse_module(code)
        imports = []
        collector = PatchImportCollector(imports)
        
        wrapper = cst.MetadataWrapper(module)
        wrapper.visit(collector)
        
        assert len(collector.extra_imports) == 0



class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_full_import_workflow(self):
        """Test complete workflow of extracting and inserting imports"""
        # Original code
        original_code = """import os
import sys

def my_func():
    pass
"""
        module = cst.parse_module(original_code)
        
        # Get existing imports
        imports, _ = get_top_level_imports(module)
        assert len(imports) == 2
        
        # Add new imports
        new_imports = [MImport(name="torch"), MImport(name="numpy", asname="np")]
        modified_module = insert_top_level_imports(module, new_imports)
        
        # Verify the result
        modified_code = modified_module.code
        assert "import torch" in modified_code
        assert "import numpy as np" in modified_code
        assert "def my_func():" in modified_code

    def test_roundtrip_mimport_cst_conversion(self):
        """Test converting MImport to CST and back"""
        # Create MImport
        original = MImport(name="torch", asname="th")
        
        # Convert to CST
        cst_node = MImport.mimport_to_cstimport(original)
        
        # Convert back to MImport
        converted = MImport.cstimport_to_mimport(cst_node)
        
        assert len(converted) == 1
        assert converted[0].name == original.name
        assert converted[0].asname == original.asname


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

