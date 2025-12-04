# Copyright (c) Huawei Technologies Co., Ltd 2025.  All rights reserved.
import pytest
import libcst as cst
from libcst.metadata import MetadataWrapper
from tools.convert.patch_merge.modules.patch_replace import PatchReplaceTransformer, PatchClassNodeRemover
from tools.convert.patch_merge.modules.patch_import_collector import MImport


class TestPatchReplaceTransformer:
    """Test the functionality of PatchReplaceTransformer class"""
    
    def setUp(self):
        """Set up test environment"""
        pass
    
    def test_function_replacement(self):
        """
        Feature: Test function replacement functionality
        Description: Create original and patch code, then use PatchReplaceTransformer to replace the original function with the patched version
        Expectation: The original function should be replaced with the patched function while maintaining the original function name
        """
        # Original code
        original_code = """
def test_func(x, y):
    return x + y
"""
        
        # Patch code
        patch_code = """
def patched_test_func(a, b):
    return a * b
"""
        
        # Create patch information
        patch = {
            'module_orign_name': (None, None, 'test_func'),
            'module_patch_name': (None, None, 'patched_test_func'),
            'patch_import_root': '.',
            'orign_import_root': '.',
            'orign_import': 'module.test_func',
            'raw_patch': {
                'patch_import': 'module.patched_test_func',
                'patch_name': 'patched_test_func',
                'condition': None
            }
        }
        
        # Parse code
        original_cst = cst.parse_module(original_code)
        patch_cst = cst.parse_module(patch_code)
        
        # Create transformer and apply
        transformer = PatchReplaceTransformer(patch, patch_cst)
        wrapper = MetadataWrapper(original_cst)
        modified_cst = wrapper.visit(transformer)
        
        # Verify results
        modified_code = modified_cst.code
        assert 'def test_func(a, b):' in modified_code
        assert 'return a * b' in modified_code
    
    def test_class_replacement(self):
        """
        Feature: Test class replacement functionality
        Description: Create original and patch classes, then use PatchReplaceTransformer to replace the original class with the patched version
        Expectation: The original class should be replaced with the patched class while maintaining the original class name and including all new methods from the patch
        """
        # Original code
        original_code = """
class TestClass:
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value
"""
        
        # Patch code
        patch_code = """
class PatchedTestClass:
    def __init__(self, value):
        self.value = value * 2
    
    def get_value(self):
        return self.value
    
    def set_value(self, new_value):
        self.value = new_value
"""
        
        # Create patch information
        patch = {
            'module_orign_name': (None, 'TestClass', None),
            'module_patch_name': (None, 'PatchedTestClass', None),
            'patch_import_root': '.',
            'orign_import_root': '.',
            'orign_import': 'module.TestClass',
            'raw_patch': {
                'patch_import': 'module.PatchedTestClass',
                'patch_name': 'PatchedTestClass',
                'condition': None
            }
        }
        
        # Parse code
        original_cst = cst.parse_module(original_code)
        patch_cst = cst.parse_module(patch_code)
        
        # Create transformer and apply
        transformer = PatchReplaceTransformer(patch, patch_cst)
        wrapper = MetadataWrapper(original_cst)
        modified_cst = wrapper.visit(transformer)
        
        # Verify results
        modified_code = modified_cst.code
        assert 'class TestClass:' in modified_code
        assert 'self.value = value * 2' in modified_code
        assert 'def set_value(self, new_value):' in modified_code
    
    def test_class_method_replacement(self):
        """
        Feature: Test class method replacement functionality
        Description: Create an original class with a method, then use PatchReplaceTransformer to replace just that specific method with a patched version
        Expectation: Only the targeted method should be replaced while keeping the rest of the class unchanged
        """
        # Original code
        original_code = """
class TestClass:
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value
"""
        
        # Patch code
        patch_code = """
def patched_get_value(self):
    return self.value * 2
"""
        
        # Create patch information
        patch = {
            'module_orign_name': (None, 'TestClass', 'get_value'),
            'module_patch_name': (None, None, 'patched_get_value'),
            'patch_import_root': '.',
            'orign_import_root': '.',
            'orign_import': 'module.TestClass.get_value',
            'raw_patch': {
                'patch_import': 'module.patched_get_value',
                'patch_name': 'patched_get_value',
                'condition': None
            }
        }
        
        # Parse code
        original_cst = cst.parse_module(original_code)
        patch_cst = cst.parse_module(patch_code)
        
        # Create transformer and apply
        transformer = PatchReplaceTransformer(patch, patch_cst)
        wrapper = MetadataWrapper(original_cst)
        modified_cst = wrapper.visit(transformer)
        
        # Verify results
        modified_code = modified_cst.code
        assert 'def get_value(self):' in modified_code
        assert 'return self.value * 2' in modified_code
    
    def test_variable_replacement(self):
        """
        Feature: Test variable replacement with class functionality
        Description: Create original code with a variable and patch code with a class, then use PatchReplaceTransformer to replace the variable with the class
        Expectation: The variable should be replaced with an instance of the patched class, and the class definition should be added to the file
        """
        # Original code
        original_code = """
TEST_VAR = None
"""
        
        # Patch code
        patch_code = """
class TestVarClass:
    def __init__(self):
        self.value = "patched"
    
    def get_value(self):
        return self.value
"""
        
        # Create patch information
        patch = {
            'module_orign_name': (None, None, 'TEST_VAR'),
            'module_patch_name': (None, 'TestVarClass', None),
            'patch_import_root': '.',
            'orign_import_root': '.',
            'orign_import': 'module.TEST_VAR',
            'raw_patch': {
                'patch_import': 'module.TestVarClass',
                'patch_name': 'TestVarClass',
                'condition': None
            }
        }
        
        # Parse code
        original_cst = cst.parse_module(original_code)
        patch_cst = cst.parse_module(patch_code)
        
        # Create transformer and apply
        transformer = PatchReplaceTransformer(patch, patch_cst)
        wrapper = MetadataWrapper(original_cst)
        modified_cst = wrapper.visit(transformer)
        
        # Verify results
        modified_code = modified_cst.code
        assert 'class TestVarClass:' in modified_code
        assert 'TEST_VAR = TestVarClass' in modified_code
    
    def test_append_new_definition(self):
        """
        Feature: 测试在原始文件中不存在目标函数时，将新函数添加到文件末尾的功能
        Description: 创建一个空的原始文件和一个包含新函数的补丁文件，然后使用PatchReplaceTransformer将新函数添加到原始文件末尾
        Expectation: 新函数定义应该被正确添加到原始文件的末尾
        """
        # Original code
        original_code = """
# Empty file
"""
        
        # Patch code
        patch_code = """
def new_function():
    return "This is a new function"
"""
        
        # Create patch information
        patch = {
            'module_orign_name': (None, None, 'non_existent_function'),
            'module_patch_name': (None, None, 'new_function'),
            'patch_import_root': '.',
            'orign_import_root': '.',
            'orign_import': 'module.non_existent_function',
            'raw_patch': {
                'patch_import': 'module.new_function',
                'patch_name': 'new_function',
                'condition': None
            }
        }
        
        # Parse code
        original_cst = cst.parse_module(original_code)
        patch_cst = cst.parse_module(patch_code)
        
        # Create transformer and apply
        transformer = PatchReplaceTransformer(patch, patch_cst)
        wrapper = MetadataWrapper(original_cst)
        modified_cst = wrapper.visit(transformer)
        
        # Verify results
        modified_code = modified_cst.code
        assert 'def new_function():' in modified_code
        assert 'return "This is a new function"' in modified_code
    
    def test_import_collection(self):
        """
        Feature: Test import collection functionality
        Description: Create original code with a simple function, then use PatchReplaceTransformer to replace it with a patched function that includes additional imports
        Expectation: The original function should be replaced with the patched version, and any imports from the patch should be properly collected and added to the file
        """
        # 原始代码
        original_code = """
def test_func():
    return None
"""
        
        # Patch code - Using additional imports
        patch_code = """
import math

def patched_test_func():
    return math.sqrt(16)
"""
        
        # Create patch information
        patch = {
            'module_orign_name': (None, None, 'test_func'),
            'module_patch_name': (None, None, 'patched_test_func'),
            'patch_import_root': '.',
            'orign_import_root': '.',
            'orign_import': 'module.test_func',
            'raw_patch': {
                'patch_import': 'module.patched_test_func',
                'patch_name': 'patched_test_func',
                'condition': None
            }
        }
        
        # Parse code
        original_cst = cst.parse_module(original_code)
        patch_cst = cst.parse_module(patch_code)
        
        # Create transformer and apply
        transformer = PatchReplaceTransformer(patch, patch_cst)
        wrapper = MetadataWrapper(original_cst)
        modified_cst = wrapper.visit(transformer)
        
        # Verify results
        modified_code = modified_cst.code
        assert 'import math' in modified_code
        assert 'return math.sqrt(16)' in modified_code
    
    def test_class_inheritance_with_alias(self):
        """
        Feature: 测试带别名的类继承替换功能
        Description: 创建一个原始类，然后使用带有别名继承的补丁类替换它，验证PatchReplaceTransformer能否正确处理这种情况
        Expectation: 原始类应该被补丁类替换，并且补丁类应该正确继承原始类，同时添加新的功能
        """
        # Original code
        original_code = """
class OriginalClass:
    def __init__(self, value):
        self.value = value
"""
        
        # Patch code - Inherits from original class with alias
        patch_code = """
from module import OriginalClass as BaseClass

class PatchedClass(BaseClass):
    def __init__(self, value):
        super().__init__(value)
        self.extra = "extra"
"""
        
        # Create patch information
        patch = {
            'module_orign_name': (None, 'OriginalClass', None),
            'module_patch_name': (None, 'PatchedClass', None),
            'patch_import_root': '.',
            'orign_import_root': '.',
            'orign_import': 'module.OriginalClass',
            'raw_patch': {
                'patch_import': 'module.PatchedClass',
                'patch_name': 'PatchedClass',
                'condition': None
            }
        }
        
        # Parse code
        original_cst = cst.parse_module(original_code)
        patch_cst = cst.parse_module(patch_code)
        
        # Create transformer and apply
        transformer = PatchReplaceTransformer(patch, patch_cst)
        wrapper = MetadataWrapper(original_cst)
        modified_cst = wrapper.visit(transformer)
        
        # Verify results
        modified_code = modified_cst.code
        assert 'class BaseClass:' in modified_code
        assert 'class PatchedClass(BaseClass):' in modified_code
    
    def test_try_patch_implict_class_func(self):
        """
        Feature: 测试向类中添加新方法的功能
        Description: 创建一个原始类（只有__init__方法），然后使用PatchReplaceTransformer尝试添加一个原始类中不存在的新方法
        Expectation: 新方法应该被成功添加到原始类中，而不是替换现有方法
        """
        # Original code
        original_code = """
class TestClass:
    def __init__(self, value):
        self.value = value
"""
        
        # Patch code
        patch_code = """
def patched_new_method(self):
    return self.value * 2
"""
        
        # Create patch information
        patch = {
            'module_orign_name': (None, 'TestClass', 'new_method'),
            'module_patch_name': (None, None, 'patched_new_method'),
            'patch_import_root': '.',
            'orign_import_root': '.',
            'orign_import': 'module.TestClass.new_method',
            'raw_patch': {
                'patch_import': 'module.patched_new_method',
                'patch_name': 'patched_new_method',
                'condition': None
            }
        }
        
        # Parse code
        original_cst = cst.parse_module(original_code)
        patch_cst = cst.parse_module(patch_code)
        
        # Create transformer and apply
        transformer = PatchReplaceTransformer(patch, patch_cst)
        wrapper = MetadataWrapper(original_cst)
        modified_cst = wrapper.visit(transformer)
        
        # Verify results
        modified_code = modified_cst.code
        assert 'def new_method(self):' in modified_code
        assert 'return self.value * 2' in modified_code
    
    def test_try_patch_to_end_class(self):
        """
        Feature: 测试将新类添加到文件末尾的功能
        Description: 创建一个空的原始文件，然后使用PatchReplaceTransformer尝试将一个新类添加到不存在的目标类位置
        Expectation: 当目标类不存在时，新类应该被添加到文件的末尾
        """
        # Original code
        original_code = """
# Empty file
"""
        
        # Patch code
        patch_code = """
class NewClass:
    def __init__(self):
        self.value = "new"
"""
        
        # Create patch information
        patch = {
            'module_orign_name': (None, 'NonExistentClass', None),
            'module_patch_name': (None, 'NewClass', None),
            'patch_import_root': '.',
            'orign_import_root': '.',
            'orign_import': 'module.NonExistentClass',
            'raw_patch': {
                'patch_import': 'module.NewClass',
                'patch_name': 'NewClass',
                'condition': None
            }
        }
        
        # Parse code
        original_cst = cst.parse_module(original_code)
        patch_cst = cst.parse_module(patch_code)
        
        # Create transformer and apply
        transformer = PatchReplaceTransformer(patch, patch_cst)
        wrapper = MetadataWrapper(original_cst)
        modified_cst = wrapper.visit(transformer)
        
        # Verify results
        modified_code = modified_cst.code
        assert 'class NewClass:' in modified_code
    
    def test_try_patch_variable_with_class(self):
        """
        Feature: 测试将变量替换为类实例的功能
        Description: 创建一个包含变量的原始代码，然后使用PatchReplaceTransformer将该变量替换为一个新类的实例
        Expectation: 变量应该被替换为类的实例，同时类定义应该被添加到文件中
        """
        # Original code
        original_code = """
TEST_VARIABLE = None
"""
        
        # Patch code
        patch_code = """
class TestClass:
    def __init__(self):
        self.value = "patched"
"""
        
        # Create patch information
        patch = {
            'module_orign_name': (None, None, 'TEST_VARIABLE'),
            'module_patch_name': (None, 'TestClass', None),
            'patch_import_root': '.',
            'orign_import_root': '.',
            'orign_import': 'module.TEST_VARIABLE',
            'raw_patch': {
                'patch_import': 'module.TestClass',
                'patch_name': 'TestClass',
                'condition': None
            }
        }
        
        # Parse code
        original_cst = cst.parse_module(original_code)
        patch_cst = cst.parse_module(patch_code)
        
        # Create transformer and apply
        transformer = PatchReplaceTransformer(patch, patch_cst)
        wrapper = MetadataWrapper(original_cst)
        modified_cst = wrapper.visit(transformer)
        
        # Verify results
        modified_code = modified_cst.code
        assert 'class TestClass:' in modified_code
        assert 'TEST_VARIABLE = TestClass' in modified_code


class TestPatchClassNodeRemover:
    """Test the functionality of PatchClassNodeRemover class"""
    
    def setUp(self):
        """Set up test environment"""
        pass
    
    def test_remove_class_definitions(self):
        """
        Feature: 测试类定义移除功能
        Description: 创建包含两个类定义（RemovedClass和KeptClass）的原始代码，然后使用PatchClassNodeRemover移除指定的类（RemovedClass）
        Expectation: RemovedClass应该从代码中被移除，而KeptClass应该保留在代码中
        """
        # Original code - Contains class to be removed
        original_code = """
class RemovedClass:
    def __init__(self):
        self.value = "to be removed"

class KeptClass:
    def __init__(self):
        self.value = "to be kept"
"""
        
        # Create patch information
        patch_infos = [
            {
                'module_patch_name': (None, 'RemovedClass', None),
                'orign_import_root': 'original.module',
                'module_orign_name': (None, 'OriginalClass', None)
            }
        ]
        
        # Parse code
        original_cst = cst.parse_module(original_code)
        
        # Create remover and apply
        remover = PatchClassNodeRemover(patch_infos)
        modified_cst = original_cst.visit(remover)
        
        # Verify results
        modified_code = modified_cst.code
        assert 'class RemovedClass:' not in modified_code
        assert 'class KeptClass:' in modified_code
    
    def test_add_imports_for_removed_classes(self):
        """
        Feature: 测试为被移除的类添加导入的功能
        Description: 创建包含RemovedClass的原始代码，然后使用PatchClassNodeRemover移除该类，并验证是否为被移除的类添加了相应的导入语句
        Expectation: RemovedClass应该从代码中被移除，并且应该添加导入语句 'from original.module import OriginalClass as RemovedClass'
        """
        # Original code - Contains class to be removed
        original_code = """
class RemovedClass:
    def __init__(self):
        self.value = "to be removed"
"""
        
        # 创建补丁信息
        patch_infos = [
            {
                'module_patch_name': (None, 'RemovedClass', None),
                'orign_import_root': 'original.module',
                'module_orign_name': (None, 'OriginalClass', None)
            }
        ]
        
        # 解析代码
        original_cst = cst.parse_module(original_code)
        
        # Create remover and apply
        remover = PatchClassNodeRemover(patch_infos)
        modified_cst = original_cst.visit(remover)
        
        # 验证结果
        modified_code = modified_cst.code
        assert 'from original.module import OriginalClass as RemovedClass' in modified_code



