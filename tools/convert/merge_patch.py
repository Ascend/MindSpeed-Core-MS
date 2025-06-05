# Copyright (c) Huawei Technologies Co., Ltd 2025.  All rights reserved.
from os.path import abspath, dirname, exists
import argparse
import libcst as cst


def check(path):
    if not isinstance(path, str):
        print("plz input string")
        return None
    if path.startswith('/'):
        if not exists(path):
            print(f"{path} dont exist")
    else:
        path = dirname(abspath(__file__)) + '/' + path
        if not exists(path):
            print(f"{path} dont exist")
    return path


class PatchDeal(cst.CSTTransformer):
    """ Process MindSpeed-LLM mindspore patch file

        Args: 
            patch_path: mindspore patch file abs path
            patch_class_name: mindspore patch class
    
    """
    def __init__(self, patch_path, patch_class_name):
        self.root = None
        self.patch_path = patch_path.split("/")[:-1]
        self.mindsporeadaptation_name = patch_class_name
        self.mindsporeadaptation_tree = None
        self.mindsporeadaptation_patch_func = set() # mindspore调用的非装饰器patch名集合
        self.mindsporeadaptation_in_flag = 0
        self.mindspore_patch_func = {"MegatronAdaptation.register", "MindSporeAdaptation.register"}

    def leave_If(self, original_node, updated_node):
        temp = self.root.code_for_node(original_node)
        if 'ai_framework' in temp:
            return cst.RemoveFromParent()
        return updated_node

    def leave_ImportFrom(self, original_node, updated_node):
        module_name = self.root.code_for_node(original_node.module)
        if len(original_node.relative) == 2:
            module_name = self.patch_path[-2] + '.' + module_name
            module = cst.parse_expression(module_name)
            return updated_node.with_changes(relative=[], module=module)
        return updated_node

    def visit_Call(self, node):
        if (self.mindsporeadaptation_in_flag == 1) and (isinstance(node.func, cst.Attribute)) and \
            ("".join(self._get_call_chain(node.func)) in self.mindspore_patch_func):
            args = node.args
            if (args[0].value.value.startswith("\'megatron") or args[0].value.value.startswith("\'mindspeed")) and \
                not args[1].value.value.endswith('wrapper') and not args[1].value.value.endswith('decorator'):
                self.mindsporeadaptation_patch_func.add(args[0].value.value)
        return True

    def visit_ClassDef(self, node):
        if node.name.value == self.mindsporeadaptation_name:
            self.mindsporeadaptation_in_flag = 1

    def leave_ClassDef(self, original_node, updated_node):
        if original_node.name.value == self.mindsporeadaptation_name:
            self.mindsporeadaptation_in_flag = 0
            self.mindsporeadaptation_tree = updated_node
        return updated_node

    def visit_Module(self, node):
        self.root = node
        return True

    def _get_call_chain(self, attribute_node: cst.Attribute) -> list:
        chain = []
        current = attribute_node
        while isinstance(current, cst.Attribute):
            chain.insert(0, current.attr.value)
            current = current.value
        if isinstance(current, cst.Name):
            chain.insert(0, current.value)
        res = chain[0]
        for i in chain[1:]:
            res += '.'
            res += i
        return res


def mindspeed_llm_patch_process(patch_path, patch_class_name):
    with open(patch_path, "r") as f:
        code = f.read()
        module = cst.parse_module(code)
        transformer = PatchDeal(patch_path, patch_class_name)
        modified_module = module.visit(transformer)
        return transformer.mindsporeadaptation_tree, transformer.mindsporeadaptation_patch_func


class MindSpeedLLMOriginDeal(cst.CSTTransformer):
    """ Process MindSpeed-LLM mindspore origin file

        Args: 
            target_class_name: execute patch method name
            add_class: mindspore patch cst tree
            add_classname: mindspore patch class name
            delete_set: patch in origin file need to delete
    
    """
    def __init__(self, target_class_name, add_class, add_classname, delete_set):
        self.root = None
        self.target_class_name = target_class_name
        self.class_to_add = add_class
        self.class_to_add_flag = 0
        self.orgin_class_name = add_classname
        self.megatron_patch_delete_set = delete_set
        self.delete_simplestatementline_flag = 0
        self.delete_simplestatementline_switch = 0
        self.delete_import_set = set()

    def visit_Call(self, node):
        return True

    def visit_ClassDef(self, node):
        if node.name.value == self.orgin_class_name:
            self.delete_simplestatementline_switch = 1

    def leave_ClassDef(self, original_node, updated_node):
        if updated_node.name.value == self.orgin_class_name:
            self.delete_simplestatementline_switch = 0
        return updated_node

    def leave_Call(self, original_node, updated_node):
        if isinstance(original_node.func, cst.Attribute):
            args = original_node.args
            if len(args) > 0 and isinstance(args[0].value, cst._nodes.expression.SimpleString) and \
                args[0].value.value in self.megatron_patch_delete_set and \
                not args[1].value.value.endswith('wrapper') and \
                not args[1].value.value.endswith('decorator'):
                if self.delete_simplestatementline_switch == 0:
                    self.delete_simplestatementline_flag = 1
                    self.delete_import_set.add(args[1].value.value)
                    return updated_node
            called_function = self._get_call_chain(original_node.func)
            if called_function == self.target_class_name:
                self.class_to_add_flag = 1
        return updated_node

    def leave_SimpleStatementLine(self, original_node, updated_node):
        if self.delete_simplestatementline_flag == 1:
            self.delete_simplestatementline_flag = 0
            return cst.RemoveFromParent()
        if self.class_to_add_flag == 1:
            self.class_to_add_flag = 0
            return cst.FlattenSentinel([self.class_to_add, updated_node])
        return updated_node

    def _get_call_chain(self, attribute_node: cst.Attribute) -> list:
        chain = []
        current = attribute_node
        while isinstance(current, cst.Attribute):
            chain.insert(0, current.attr.value)
            current = current.value
        if isinstance(current, cst.Name):
            chain.insert(0, current.value)
        res = chain[0]
        for i in chain[1:]:
            res += '.'
            res += i
        return res

    def leave_ImportFrom(self, original_node, updated_node):
        new_names = []
        for i in original_node.names:
            temp = i.name.value
            if temp != self.orgin_class_name: # 删除msspore adaptation的import
                new_names.append(i)
        if not new_names:
            return cst.RemoveFromParent()
        if len(new_names) == len(original_node.names):
            return updated_node
        return updated_node.with_changes(names=new_names)

    def visit_Module(self, node):
        self.root = node
        return True


class DeleteImport(cst.CSTTransformer):
    def __init__(self, delete_import_set, orgin_class_name):
        self.delete_import_set = delete_import_set
        self.orgin_class_name = orgin_class_name
        self.dont_delete_import_switch = 0
        self.root = None

    def visit_Module(self, node):
        self.root = node
        return True

    def visit_ClassDef(self, node):
        if node.name.value == self.orgin_class_name:
            self.dont_delete_import_switch = 1

    def leave_ClassDef(self, original_node, updated_node):
        if updated_node.name.value == self.orgin_class_name:
            self.dont_delete_import_switch = 0
        return updated_node

    def leave_ImportFrom(self, original_node, updated_node):
        if self.dont_delete_import_switch == 1:
            return updated_node
        new_names = []
        for i in original_node.names:
            temp = i.name.value
            if temp not in self.delete_import_set:
                new_names.append(i)
            else:
                new_names.append(i)
        if not new_names:
            return cst.RemoveFromParent()
        if len(new_names) == len(original_node.names):
            return updated_node
        return updated_node


def mindspeed_llm_origin_process(origin_path, target_class_name, patch_code, add_classname, delete_set):
    with open(origin_path, "r+") as f:
        code = f.read()
        modified_module = cst.parse_module(code)
        transformer = MindSpeedLLMOriginDeal(target_class_name, patch_code, add_classname, delete_set)
        modified_module = modified_module.visit(transformer)
        transformer = DeleteImport(transformer.delete_import_set, add_classname)
        modified_module = modified_module.visit(transformer)
        f.seek(0)
        f.write(modified_module.code)
        return modified_module


class MindSpeedPatchDeal(cst.CSTTransformer):
    """ Process MindSpeed mindspore patch file

        Args: 
            patch_path: mindspore patch file abs path
            patch_func_name: mindspore patch function
    
    """
    def __init__(self, patch_path, patch_func_name):
        self.root = None
        self.patch_path = patch_path.split("/")[:-1]
        self.mindsporeadaptation_name = patch_func_name
        self.mindsporeadaptation_tree = None # 需要迁移的代码块
        self.mindsporeadaptation_patch_func = set() # mindspore调用的非装饰器patch名集合
        self.mindsporeadaptation_in_flag = 0
        self.mindspore_patch_func = {"aspm.register_patch"}

    def leave_If(self, original_node, updated_node):
        cond_prev = self.root.code_for_node(original_node)
        if cond_prev.find("ai_framework"):
            condition = original_node.test
            or_chain = []
            current = condition
            while isinstance(current, cst.BooleanOperation) and isinstance(current.operator, cst.Or):
                or_chain.append(current.right)
                current = current.left
            or_chain.append(current)
            if len(or_chain) > 0:
                last_condition = or_chain[0]
            new_if = updated_node.with_changes(test=last_condition)
            return new_if
        return updated_node

    def leave_ImportFrom(self, original_node, updated_node):
        module_name = self.root.code_for_node(original_node.module)
        if len(original_node.relative) == 1:
            module_name = self.patch_path[-2] + '.' + self.patch_path[-1] + '.' + module_name
            module = cst.parse_expression(module_name)
            return updated_node.with_changes(relative=[], module=module)
        return updated_node

    def visit_Call(self, node):
        if (self.mindsporeadaptation_in_flag == 1) and (isinstance(node.func, cst.Attribute)) and \
                ("".join(self._get_call_chain(node.func)) in self.mindspore_patch_func): # 支持类方法调用
            args = node.args
            if (args[0].value.value.startswith("\'megatron") or args[0].value.value.startswith("\'mindspeed")) and \
                not args[1].value.value.endswith('wrapper') and not args[1].value.value.endswith('decorator'):
                self.mindsporeadaptation_patch_func.add(args[0].value.value)
        return True

    def visit_FunctionDef(self, node):
        if node.name.value == self.mindsporeadaptation_name:
            self.mindsporeadaptation_in_flag = 1

    def leave_FunctionDef(self, original_node, updated_node):
        if original_node.name.value == self.mindsporeadaptation_name:
            self.mindsporeadaptation_in_flag = 0
            self.mindsporeadaptation_tree = updated_node
        return updated_node

    def visit_Module(self, node):
        self.root = node
        return True

    def _get_call_chain(self, attribute_node: cst.Attribute):
        chain = []
        current = attribute_node
        while isinstance(current, cst.Attribute):
            chain.insert(0, current.attr.value)
            current = current.value
        if isinstance(current, cst.Name):
            chain.insert(0, current.value)
        res = chain[0]
        for i in chain[1:]:
            res += '.'
            res += i
        return res


class MindSpeedOriginDeal(cst.CSTTransformer):
    """ Process MindSpeed mindspore origin file

        Args: 
            target_func_name: execute patch function name
            add_func: mindspore patch cst tree
            add_funcname: mindspore patch function name
            delete_set: patch in origin file need to delete
    
    """

    def __init__(self, target_func_name, add_func, add_funcname, delete_set):
        self.root = None
        self.target_func_name = target_func_name
        self.func_to_add = add_func
        self.orgin_func_name = add_funcname
        self.megatron_patch_delete_set = delete_set
        self.delete_simplestatementline_flag = 0
        self.delete_simplestatementline_switch = 0
        self.delete_import_set = set()

    def visit_Call(self, node):
        return True

    def visit_FunctionDef(self, node):
        if node.name.value == self.orgin_func_name:
            self.delete_simplestatementline_switch = 1

    def leave_FunctionDef(self, original_node, updated_node):
        if updated_node.name.value == self.orgin_func_name:
            self.delete_simplestatementline_switch = 0
        elif updated_node.name.value == self.target_func_name:
            return cst.FlattenSentinel([self.func_to_add, updated_node])
        return updated_node

    def leave_Call(self, original_node, updated_node):
        if isinstance(original_node.func, cst.Attribute):
            args = original_node.args
            if len(args) > 0 and isinstance(args[0].value, cst._nodes.expression.SimpleString) and \
                args[0].value.value in self.megatron_patch_delete_set and \
                not args[1].value.value.endswith('wrapper') and \
                not args[1].value.value.endswith('decorator'):
                if self.delete_simplestatementline_switch == 0:
                    self.delete_simplestatementline_flag = 1
                    self.delete_import_set.add(args[1].value.value)
        return updated_node

    def leave_SimpleStatementLine(self, original_node, updated_node):
        if self.delete_simplestatementline_flag == 1:
            self.delete_simplestatementline_flag = 0
            return cst.RemoveFromParent()
        return updated_node

    def leave_ImportFrom(self, original_node, updated_node):
        new_names = []
        for i in original_node.names:
            temp = i.name.value
            if temp != self.orgin_func_name:
                new_names.append(i)
        if not new_names:
            return cst.RemoveFromParent()
        if len(new_names) == len(original_node.names):
            return updated_node
        return updated_node.with_changes(names=new_names)

    def visit_Module(self, node):
        self.root = node
        return True


def mindspeed_origin_process(origin_path, target_func_name, patch_code, add_funcname, delete_set):
    with open(origin_path, "r+") as f:
        code = f.read()
        modified_module = cst.parse_module(code)
        transformer = MindSpeedOriginDeal(target_func_name, patch_code, add_funcname, delete_set)
        modified_module = modified_module.visit(transformer)
        transformer = DeleteImport(transformer.delete_import_set, add_funcname)
        modified_module = modified_module.visit(transformer)
        f.seek(0)
        f.write(modified_module.code)
        return modified_module


def mindspeed_patch_process(patch_path, patch_func_name):
    with open(patch_path, "r") as f:
        code = f.read()
        modified_module = cst.parse_module(code)
        transformer = MindSpeedPatchDeal(patch_path, patch_func_name)
        modified_module = modified_module.visit(transformer)
        return transformer.mindsporeadaptation_tree, transformer.mindsporeadaptation_patch_func


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mindspeed_path", type=str, default=None,
                        help="origin mindspeed package path")
    parser.add_argument("--mindspeed_llm_path", type=str, default=None,
                        help="origin mindspeed package path")
    args = parser.parse_args()

    if args.mindspeed_llm_path:
        mindspeed_llm_patch_path = args.mindspeed_llm_path + "mindspeed_llm/mindspore/mindspore_adaptor.py"
        mindspeed_llm_origin_path = args.mindspeed_llm_path + "mindspeed_llm/tasks/megatron_adaptor.py"
    if args.mindspeed_path:
        mindspeed_patch_path = args.mindspeed_path + "mindspeed/mindspore/mindspore_adaptor.py"
        mindspeed_origin_path = args.mindspeed_path + "mindspeed/megatron_adaptor.py"

    mindspeed_llm_target_class_name = "MegatronAdaptation.execute"
    mindspeed_llm_patch_class_name = "MindSporeAdaptation"
    mindspeed_patch_func_name = "mindspore_adaptation"
    mindspeed_target_func_name = "exe_adaptation"

    mindspeed_llm_patch_abs_path = check(mindspeed_llm_patch_path)
    mindspeed_llm_patch_code, mindspeed_llm_delete_set = \
        mindspeed_llm_patch_process(mindspeed_llm_patch_abs_path, mindspeed_llm_patch_class_name)

    mindspeed_llm_origin_abs_path = check(mindspeed_llm_origin_path)
    mindspeed_llm_origin_modified_module = \
        mindspeed_llm_origin_process(mindspeed_llm_origin_abs_path, mindspeed_llm_target_class_name, 
        mindspeed_llm_patch_code, mindspeed_llm_patch_class_name, mindspeed_llm_delete_set)

    mindspeed_patch_abs_path = check(mindspeed_patch_path)
    mindspeed_patch_code, mindspeed_delete_set = \
        mindspeed_patch_process(mindspeed_patch_abs_path, mindspeed_patch_func_name)

    mindspeed_origin_abs_path = check(mindspeed_origin_path)
    mindspeed_origin_modified_module = \
        mindspeed_origin_process(mindspeed_origin_abs_path, mindspeed_target_func_name, mindspeed_patch_code, 
        mindspeed_patch_func_name, mindspeed_delete_set)