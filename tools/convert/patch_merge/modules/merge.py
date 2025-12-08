# Copyright (c) Huawei Technologies Co., Ltd 2025.  All rights reserved.
import functools
import json
import os
import sys
import re
from pathlib import Path
from collections import defaultdict
import traceback
from pprint import pprint, pformat
import argparse
import copy
import importlib
import inspect

import libcst as cst
from libcst import matchers
from libcst.metadata import MetadataWrapper, ParentNodeProvider
from .patch_replace import PatchReplaceTransformer, PatchClassNodeRemover
from .patch_class_add_factory import grep_in_files, PatchClassFactoryTransformer, PatchClassCallTransformer
from .patch_func_router import PatchFuncRouterTransformer
from .patch_wrapper_router import PatchWrapperRouterTransformer

from datetime import datetime
START_TIMES=[]


def tik(info=""):
    """
    Start timer
    """
    global START_TIMES
    start_time = datetime.now()
    START_TIMES.append(start_time)
    print(f"[INFO] start {info} time: {start_time}")


def tok(info=""):
    """
    End timer
    """
    global START_TIMES
    end_time = datetime.now()
    start_time = START_TIMES.pop()
    delta = end_time - start_time
    print(f"[INFO] finish {info} time: {end_time}, elapsed time: {delta.seconds}s")


def time_tracker(func):
    """
    Decorator to track the execution time of a function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tik(func.__name__)
        ret = func(*args, **kwargs)
        tok(func.__name__)
        return ret
    return wrapper


def get_module_name(class_name, func_name):
    """
    Get the function name, class name or class method name
    """
    if class_name is None and func_name is not None:  # func
        return func_name
    
    if class_name is not None and func_name is None:  # class
        return class_name

    return f"{class_name}.{func_name}"  # class fun: Foo.bar


class PatchMerger:
    """
    The patch merging entry class
    """
    def __init__(self, raw_patches, root):
        self.raw_patches = raw_patches
        self.root = root

        # parsed patches
        self.patch_replace_info = {}
        self.patch_func_infos = {}
        self.patch_wrapper_infos = {}
        self.patch_class_infos = {}
        self.all_patch_infos = [self.patch_replace_info, self.patch_func_infos, self.patch_wrapper_infos,  self.patch_class_infos]

        # cst of files for lazy flushing
        self.cst_to_write = {}

        # logging infos
        self.num_modules, self.num_patches = 0, 0
        self.bad_parsed_cases = defaultdict(list)
        self.bad_handled_cases = defaultdict(list)

        # megatron_adaptor files
        adaptor_paths = [
            Path(args.root_dir) / Path("MindSpeed-LLM/mindspeed_llm/tasks/megatron_adaptor.py"),
            Path(args.root_dir) / Path("MindSpeed-LLM/mindspeed_llm/core/pipeline_parallel/dualpipe/adaptor.py"),
            Path(args.root_dir) / Path("MindSpeed/mindspeed/features_manager/tensor_parallel/unaligned_linear_feature.py"),
            Path(args.root_dir) / Path("MindSpeed-LLM/mindspeed_llm/mindspore/mindspore_adaptor.py")
        ]

        def find_python_files(path):
            directory = Path(path)
            python_files = directory.glob('**/*.py')
            python_files = [str(f) for f in python_files if f.name != '__init__.py']
            return python_files

        # Get the adaptor path
        target_directory = Path(args.root_dir) / Path("MindSpeed-LLM/mindspeed_llm/features_manager/")
        py_files = find_python_files(target_directory)
        adaptor_paths += py_files

        self.adaptors = {}
        for path in adaptor_paths:
            abs_path = Path(path)
            with open(abs_path, 'r') as f:
                adaptor_code = f.read()
            self.adaptors[abs_path] = (adaptor_code, False)  # (code, need_flush)

    def get_cst(self, file_path):
        """
        Obtain the cst node corresponding to the file
        """
        if file_path in self.cst_to_write:
            source_cst = self.cst_to_write[file_path]
        else:
            with open(file_path, 'r') as f:
                code = f.read()
            source_cst = cst.parse_module(code)
        return source_cst

    def set_cst(self, file_path, cst_module):
        """
        Record the cst node corresponding to the file
        """
        self.cst_to_write[file_path] = cst_module

    @time_tracker
    def flush_cst_into_file(self):
        """
        Flush the updated cst to the file
        """
        for file_path, cst_module in self.cst_to_write.items():
            print(f"[INFO] flushing cst into {file_path}")
            with open(file_path, 'w') as f:
                f.write(cst_module.code)

    def handle_exc(self, e, module_name, module_patch_infos):
        """
        Handling error patch and printing
        """
        print(f"[ERROR] Exception {str(e)} while patching module {module_name}, raw patches: ")
        raw_patches = [(patch_info['origin_import'], patch_info['raw_patch']) for patch_info in module_patch_infos]
        pprint(raw_patches)
        print(f"**********************")
        print(traceback.format_exc())
        print(f"**********************")

        for origin_import, raw_patch in raw_patches:
            self.bad_handled_cases[origin_import].append(raw_patch)

    @staticmethod
    def parse_path(source_packages, parent_module_path, module_name):
        """
        Parse the file and the corresponding class/function name based on the import path and module name
        
        Input:
            - source_packages: restricted package name: ['megatron', 'mindspeed', 'mindspeed_llm']
            - parent_module_path: The import path retrieved
            - module_name: Module name (generally written as from <parent_module_path> import <module_name> in megatron_adaptor)

        Output:
            - import_root: The actual import path corresponding to the module definition 
                (the module may be imported from __init__.py, and the actual module location will be found here).
            - file: The file path where the module definition is located
            - class_name: Class name. If the module is not a class method or a class, return None
            - func_name: Function name. If the module is a class, return None
        """
        if module_name is None:
            raise ValueError("module_name cannot be None")

        # try import
        modules = parent_module_path.split('.')
        parent, module = None, None
        for i in range(1, len(modules) + 1):
            parent_path = '.'.join(modules[:i - 1])
            path = '.'.join(modules[:i])
            try:
                importlib.import_module(path)
            except ModuleNotFoundError as e:
                if not parent_path or not hasattr(importlib.import_module(parent_path), modules[i - 1]):
                    raise ModuleNotFoundError(e) from e
                else:
                    parent = getattr(importlib.import_module(parent_path), modules[i - 1])
                    if not hasattr(parent, module_name):
                        raise RuntimeError('no exist {} of {}'.format(module_name, parent))
                    break

        if not parent:
            parent = sys.modules[parent_module_path]

        # get module
        if not hasattr(parent, module_name):
            print(f"[WARNING] No exist {module_name} of {parent}")
            module = None
        else:
            module = getattr(parent, module_name)

        # get import root & file by "inspect.getmodule(module)"
        def get_source_module():
            if module is None:
                return inspect.getmodule(parent)
            # prefer class than function to prevent getting the module of a function defined in another class
            if inspect.isclass(module):
                return inspect.getmodule(module)
            if inspect.isclass(parent):
                return inspect.getmodule(parent)
            return inspect.getmodule(module)

        source_module = get_source_module()
        if source_module.__name__.split('.')[0] not in source_packages:
            raise Exception(f"Source package need to be in {source_packages}, got {source_module.__name__}")

        import_root, file_path = source_module.__name__, source_module.__file__

        if inspect.isclass(parent):
            return import_root, file_path, parent.__name__, module_name
        elif inspect.isclass(module):
            return import_root, file_path, module_name, None
        else:
            return import_root, file_path, None, module_name

    def add_merge_info(self, infos_dict, original_file, module_name, patch_info):
        if original_file not in infos_dict:
            infos_dict[original_file] = {}
        
        if module_name not in infos_dict[original_file]:
            infos_dict[original_file][module_name] = []

        need_append = True
        if 'force_patch' in patch_info['raw_patch']:
            infos = infos_dict[original_file][module_name]
            new_force_patch = patch_info['raw_patch']['force_patch']
            
            for i, info in enumerate(infos):
                if patch_info['condition'] != info['condition']:
                    continue
                
                cur_force_patch = info['raw_patch']['force_patch']
                if cur_force_patch:
                    if new_force_patch:
                        raise Exception(f"Only support one force_patch in {original_file}, {module_name}")
                    need_append = False
                elif new_force_patch:
                    infos[i] = patch_info
                    need_append = False

            infos_dict[original_file][module_name] = infos
        
        if need_append:
            infos_dict[original_file][module_name].append(patch_info)

    @time_tracker
    def parse_patch_infos(self):
        """
        Parse the patch json file and categorize and save it as "Dict[Dict[List]]"
            - Unconditional replacement: patch_replace_info
            - Conditional class substitution: patch_class_infos
            - Conditional function substitution: patch_func_infos
            - Conditional function decoration:patch_wrapper_infos
        """
        handled_packages = ['megatron', 'mindspeed', 'mindspeed_llm']

        # split import_path into parent_path and module_name
        def split(import_path):
            split_name = import_path.rsplit('.', 1)
            if len(split_name) == 1:
                parent_module_path, module_name = split_name, None
            else:
                parent_module_path, module_name = split_name
            
            return parent_module_path, module_name

        for origin_import, module_raw_patches in self.raw_patches.items():
            origin_import = origin_import[1:-1] if origin_import.startswith("'") else origin_import # fix "'megatron.xxx.xxx'"
            split_name = origin_import.rsplit('.', 1)
            parent_module_path, module_name = split(origin_import)

            # parse original import module
            try:
                origin_import_root, original_file, class_name, func_name = PatchMerger.parse_path(handled_packages, parent_module_path, module_name)
            except Exception as e:
                print(f"[ERROR] While parsing original import: {origin_import}, raising exception {e}")
                print(origin_import)
                pprint(module_raw_patches)
                print(f"**********************")
                print(traceback.format_exc())
                print(f"**********************")
                self.bad_parsed_cases[origin_import].extend(module_raw_patches)
                continue

            if class_name is None and func_name is None:
                raise Exception(f"[ERROR] While parsing original import{origin_import}, both class_name and func_name are None")
            module_origin_name = get_module_name(class_name, func_name)

            self.num_patches += len(module_raw_patches)

            # parse patch import modules of the original module
            for module_raw_patch in module_raw_patches:
                patch_import, _, condition = module_raw_patch['patch_import'], module_raw_patch['patch_name'], module_raw_patch['condition']
                parent_module_path, module_name = split(patch_import)
                try:
                    patch_import_root, patch_file, class_patch_name, func_patch_name = PatchMerger.parse_path(handled_packages, parent_module_path, module_name)
                except Exception as e:
                    print(f"[ERROR] While parsing patch import {patch_import}, raising exception: {e}")
                    print(origin_import)
                    pprint(module_raw_patch)
                    print(f"**********************")
                    print(traceback.format_exc())
                    print(f"**********************")
                    self.bad_parsed_cases[origin_import].append(module_raw_patch)
                    continue

                if class_patch_name is None and func_patch_name is None:
                    raise Exception(f"[ERROR] While parsing patch import {patch_import}, both class_name and func_name are None")

                module_patch_name = get_module_name(class_patch_name, func_patch_name)

                is_class_patch = (class_patch_name is not None) and (func_patch_name is None)
                is_wrapper = module_patch_name.endswith("_wrapper") or module_patch_name.endswith("_decorator")

                patch_info = {
                    "origin_file": original_file,
                    'origin_import': origin_import,
                    'origin_import_root': origin_import_root,
                    "patch_file": patch_file,
                    'patch_import': patch_import,
                    'patch_import_root': patch_import_root,
                    "module_origin_name": (module_origin_name, class_name, func_name),
                    "module_patch_name": (module_patch_name, class_patch_name, func_patch_name),
                    "condition": condition,
                    "raw_patch": module_raw_patch
                }

                # separate patch infos
                if not is_wrapper and not condition:
                    self.add_merge_info(self.patch_replace_info, original_file, module_origin_name, patch_info)
                    continue
                if is_class_patch:
                    self.add_merge_info(self.patch_class_infos, original_file, module_origin_name, patch_info)
                elif is_wrapper:
                    self.add_merge_info(self.patch_wrapper_infos, original_file, module_origin_name, patch_info)
                else:
                    self.add_merge_info(self.patch_func_infos, original_file, module_origin_name, patch_info)

        print(f"[INFO] =======================total {len(self.raw_patches)}====================")
        print(f"[INFO] =======================parsed failed {len(self.bad_parsed_cases)}====================")

    def annotate(self, patch):
        """
        Annotate the register/register_patch statement in megatron_adaptor based on the input patch.
        A dict is used to record the modified code and then flash it into the file all at once after processing is completed.
        """
        origin_import = patch['origin_import']
        parent, module = origin_import.rsplit('.', 1)
        parent, module = re.escape(parent), re.escape('.' + module)
        origin_import = re.escape(origin_import)
        _, class_patch_name, func_patch_name = patch['module_patch_name']
        patch_name = func_patch_name if func_patch_name is not None else class_patch_name
        patch_name = re.escape(patch_name)

        module_origin_name, _, _ = patch['module_origin_name']
        patterns = [
            rf'^((?:[^\S\n])*)(?:MegatronAdaptation|patch_manages?r|pm|aspm|MindSporeAdaptation)\.(?:register|register_patch)\(\s*[\'\"]{origin_import}[\'\"]\s*,\s*{patch_name}(?:,\s*force_patch=.*?)?\)',
            rf'^((?:[^\S\n])*)(?:MegatronAdaptation|patch_manages?r|pm|aspm|MindSporeAdaptation)\.(?:register|register_patch)\(\s*[\'\"]{parent}[\'\"]\s*[\'\"]{module}[\'\"]\s*,\s*{patch_name}\)',
            rf'^((?:[^\S\n])*){module_origin_name}\s*=\s*{patch_name}\s*$'
        ]

        def replacer(match):
            matched_text = match.group(0)
            blanks = match.group(1)
            # replaced with "pass"
            commented_text = re.sub(r'^', f'{blanks}#', matched_text, flags=re.MULTILINE)
            commented_text = f"{blanks}pass\n{commented_text}"
            return commented_text

        for file_path, (code, _) in self.adaptors.items():
            for pattern in patterns:
                match = re.search(pattern, code, flags=re.MULTILINE | re.DOTALL)
                if not match:
                    continue
                print(f"[DEBUG] Comment {match.group(0)} in {file_path}")
                code = re.sub(
                    pattern,
                    replacer,
                    code,
                    count=1,
                    flags=re.MULTILINE | re.DOTALL # | re.VERBOSE | re.UNICODE
                )
                self.adaptors[file_path] = (code, True)  # update

                # Comment only one patch at a time to prevent patches with the same name from not being found later
                return

        raise Exception(f"Register patch not found in adaptor, comment failed. patch {patch['origin_import']}: {patch['raw_patch']}")

    def handle_annotate(self, patch_infos):
        """
        Annotation patches
        """
        for patch_info in patch_infos:
            self.annotate(patch_info)

    @time_tracker
    def flush_annotation(self):
        """
        Flush the comments into the adaptor file
        """
        for file_path, (code, need_flush) in self.adaptors.items():
            if not need_flush:
                continue
            print(f"[INFO] flushing annotation into {file_path}")
            with open(file_path, 'w') as f:
                f.write(code)

    @time_tracker
    def merge_replacement(self):
        """
        Unconditional replacement: Replace the definitions in the patch file to the original file
        """
        patch_class_node_to_remove = defaultdict(list)

        for origin_file, all_module_patch_infos in self.patch_replace_info.items():  # All the patched modules in the file
            print(f"[INFO] Merging file in merge_replacement: {origin_file}")
            # 1. read original cst
            source_cst = self.get_cst(origin_file)
            origin_source_cst, updated_source_cst = source_cst, None
            source_wrapper = MetadataWrapper(source_cst)

            # 2. collect extra imports in patch and do replacement
            for module_name, module_patch_infos in all_module_patch_infos.items():  # All patches of the module
                try:
                    if len(module_patch_infos) != 1:
                        raise Exception(f"Should only have 1 replacement for module {module_name}, got {len(module_patch_infos)}")
                    patch_info = module_patch_infos[0]

                    # 2. read patch cst
                    patch_file = patch_info['patch_file']
                    patch_cst = self.get_cst(patch_file)

                    # 3. do replace
                    replacer = PatchReplaceTransformer(patch_info, patch_cst)
                    updated_source_cst = source_wrapper.visit(replacer)
                    if updated_source_cst is None:
                        raise Exception("Got None cst after visit")

                    # 4. record class node to be removed in patch_file, instead we import from original file
                    # Only handle replacing class by class
                    if replacer.func_origin_name is None and replacer.func_patch_name is None:
                        patch_class_node_to_remove[(patch_file, module_name)].append(patch_info)

                    # 5. annotate megatron_adaptor
                    self.handle_annotate(module_patch_infos)

                    # 6. update source_cst
                    source_cst = updated_source_cst
                    source_wrapper = MetadataWrapper(source_cst)
                except Exception as e:
                    self.handle_exc(e, module_name, module_patch_infos)

            # 7. update original cst in dict
            if source_cst != origin_source_cst:
                self.set_cst(origin_file, source_cst)

        # 8. do remove nodes in patch cst
        for (patch_file, module_name), patch_infos in patch_class_node_to_remove.items():
            try:
                patch_cst = self.get_cst(patch_file)
                remover = PatchClassNodeRemover(patch_infos)
                patch_cst = patch_cst.visit(remover)
            except Exception as e:
                self.handle_exc(e, module_name, patch_infos)

            if patch_cst is not None:
                self.set_cst(patch_file, patch_cst)

    @time_tracker
    def merge_class_patch(self):
        '''
        Conditional class substitution: Add a factory class and change class instantiation/static method calls to factory class calls
        '''
        for origin_file, all_module_patch_infos in self.patch_class_infos.items():
            print(f"[INFO] Merging file in merge_class_patch: {origin_file}")
            # 1. read original cst
            source_cst = self.get_cst(origin_file)
            origin_source_cst, updated_source_cst = source_cst, None
            source_wrapper = MetadataWrapper(source_cst)

            for module_name, module_patch_infos in all_module_patch_infos.items():
                try:
                    # 2. walk through all Callings and modify callings to factory call
                    cls_name = module_name
                    origin_import = module_patch_infos[0]['origin_import']
                    files = grep_in_files(os.path.join(self.root, "Megatron-LM", "megatron"), cls_name)
                    print(f"[INFO] walking {len(files)} files in megatron where {cls_name} is found...")
                    walked_cst = {}
                    for file_path in files:
                        if file_path == origin_file:
                            continue
                        call_cst = self.get_cst(file_path)
                        wrapper = MetadataWrapper(call_cst)
                        parent_provider = wrapper.resolve(ParentNodeProvider)
                        walker = PatchClassCallTransformer(cls_name, origin_import, parent_provider)

                        new_code = wrapper.visit(walker)
                        print(f"[DEBUG] After walking file {file_path}, has_change={walker.has_change}")
                        if walker.has_change:
                            walked_cst[file_path] = new_code

                    # 3. add factory class define
                    fac = PatchClassFactoryTransformer(module_name, module_patch_infos)
                    updated_source_cst = source_wrapper.visit(fac)
                    if updated_source_cst is None:
                        raise Exception("Got None cst after visit")

                    # 4. annotate megatron_adaptor
                    self.handle_annotate(module_patch_infos)

                    # 5. update source cst
                    source_cst = updated_source_cst
                    source_wrapper = MetadataWrapper(source_cst)
                    for file_path, cst in walked_cst.items():
                        self.set_cst(file_path, cst)
                except Exception as e:
                    self.handle_exc(e, module_name, module_patch_infos)

            # 6. update original cst in dict
            if source_cst != origin_source_cst:
                self.set_cst(origin_file, source_cst)

    def merge_with_router(self, patch_infos, router_trans_cls):
        """
        Conditional function substitution/wrapper: Create a routing function and route to the corresponding patch function based on the conditions
        """
        for origin_file, all_module_patch_infos in patch_infos.items():
            print(f"[INFO] Merging file {origin_file} in merge_with_router with {router_trans_cls.__name__}")
            # 1. read original cst
            source_cst = self.get_cst(origin_file)
            origin_source_cst, updated_source_cst = source_cst, None
            source_wrapper = MetadataWrapper(source_cst)

            for module_name, module_patch_infos in all_module_patch_infos.items():
                try:
                    # 2. add function router
                    merger = router_trans_cls(module_name, module_patch_infos)
                    updated_source_cst = source_wrapper.visit(merger)
                    if updated_source_cst is None:
                        raise Exception("Got None cst after visit")

                    # 3. annotate megatron_adaptor
                    self.handle_annotate(module_patch_infos)

                    # 4. update source cst
                    source_cst = updated_source_cst
                    source_wrapper = MetadataWrapper(source_cst)
                except Exception as e:
                    self.handle_exc(e, module_name, module_patch_infos)

            # 5. update original cst in dict
            if source_cst != origin_source_cst:
                self.set_cst(origin_file, source_cst)

    @time_tracker
    def merge_func_patch(self):
        """
        Conditional function substitution
        """
        self.merge_with_router(self.patch_func_infos, PatchFuncRouterTransformer)

    @time_tracker
    def merge_wrapper_patch(self):
        """
        Conditional/Unconditional function decorators
        """
        self.merge_with_router(self.patch_wrapper_infos, PatchWrapperRouterTransformer)


@time_tracker
def dump_json_at_same_dir(patch_json_file, data, name_suffix):
    """
    Save the erroneous patch to the same-level directory where the parsed json file is located
    """
    dirname, filename = os.path.split(patch_json_file)
    name, suffix = filename.rsplit('.', 1)
    dumped_file = os.path.join(dirname, f"{name}_{name_suffix}.{suffix}")

    json_indent = 4
    with open(dumped_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=json_indent)
    
    print(f"[INFO] {name_suffix} are dumped into {dumped_file}")


@time_tracker
def merge(root_dir, json_file, check):
    """
    The entry function of patch merging
    """
    # 1. Read json, extract patches, parse and classify
    patch_json = Path(json_file)
    print(f"[INFO] raw patches: {patch_json}")
    with open(patch_json, 'r', encoding='utf-8') as f:
        raw_patches = json.load(f)
    pm = PatchMerger(raw_patches, root_dir)
    pm.parse_patch_infos()

    # 2. Handle unconditional function substitution and unconditional class substitution
    pm.merge_replacement()

    # 3. Handle conditional class substitutions
    pm.merge_class_patch()

    # 4. Handle conditional function patches
    pm.merge_func_patch()

    # 5. Handle function wrapper
    pm.merge_wrapper_patch()
    
    # 6. Printing of statistical information
    get_num_patches = lambda patches: sum([len(p) for p in patches.values()])
    num_patches = get_num_patches(pm.raw_patches)
    num_bad_parsed_patches = get_num_patches(pm.bad_parsed_cases)
    num_bad_handled_patches = get_num_patches(pm.bad_handled_cases)
    print("===============================================")
    print(f"total patches: {num_patches} in {len(pm.raw_patches)} modules")
    print(f"bad parsed cases: {num_bad_parsed_patches} in {len(pm.bad_parsed_cases)} modules")
    print(f"bad handled cases: {num_bad_handled_patches} in {len(pm.bad_handled_cases)} modules")

    if num_bad_parsed_patches > 0 or num_bad_handled_patches > 0:
        print(f"(bad cases are skipped. grep '[ERROR]' to find more detail...)")
    if check:
        print(f"(Changes are not flushed into files since we are in **check** mode)")
    print("===============================================")

    # 7. Save the error patch
    dump_json_at_same_dir(json_file, pm.bad_parsed_cases, "bad_parsed_cases")
    dump_json_at_same_dir(json_file, pm.bad_handled_cases, "bad_handled_cases")

    # 8. Flush cst into the file
    if not check:
        pm.flush_cst_into_file()
        pm.flush_annotation()


@time_tracker
def preprocess(mindspeed_llm_adaptor):
    """
    Preprocess to resolve the import error in megatron and ensure that importlib can obtain modules from megatron
    """
    import torch
    import transformer_engine
    import types

    # Add the dummy module, otherwise importlib will report an error
    sys.modules["transformer_engine.common"] = types.ModuleType("transformer_engine.common")
    setattr(transformer_engine, "common", sys.modules["transformer_engine.common"])
    sys.modules["transformer_engine.common.recipe"] = types.ModuleType("transformer_engine.common.recipe")
    setattr(transformer_engine.common, "recipe", sys.modules["transformer_engine.common.recipe"])
    setattr(sys.modules["transformer_engine.common.recipe"], "DelayedScaling", torch.nn.Module)
    sys.modules["transformer_engine.pytorch.distributed"] = types.ModuleType("transformer_engine.pytorch.distributed")
    setattr(transformer_engine.pytorch, "distributed", sys.modules["transformer_engine.pytorch.distributed"])
    setattr(transformer_engine.pytorch.distributed, "CudaRNGStatesTracker", torch.nn.Module)

    sys.modules["amp_C"] = types.ModuleType("amp_C")
    amp_C = sys.modules["amp_C"]
    setattr(amp_C, "multi_tensor_l2norm", None)
    setattr(amp_C, "multi_tensor_scale", None)

    # Comment out "execute" in mindspeed_llm/task/megatron_adaptor.py
    # Otherwise the incorrect module will be obtained
    with open(mindspeed_llm_adaptor, 'r') as f:
        code = f.read()
    code = code.replace("MegatronAdaptation.execute()", "# MegatronAdaptation.execute()")
    with open(mindspeed_llm_adaptor, 'w') as f:
        f.write(code)

    # Fix circular import issue
    from mindspeed_llm.tasks.megatron_adaptor import MegatronAdaptation
    from mindspeed_llm.training.arguments import parse_args_decorator
    MegatronAdaptation.register('megatron.training.arguments.parse_args', parse_args_decorator)
    MegatronAdaptation.apply()


@time_tracker
def postprocess(mindspeed_llm_adaptor):
    """
    Uncomment "execute" in mindspeed_llm/task/megatron_adaptor.py
    """
    with open(mindspeed_llm_adaptor, 'r') as f:
        code = f.read()
    code = code.replace("# MegatronAdaptation.execute()", "MegatronAdaptation.execute()")
    with open(mindspeed_llm_adaptor, 'w') as f:
        f.write(code)


if __name__ == '__main__':
    tik("total")

    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', help='MindSpeed-Core-MS directory')
    parser.add_argument('--json-file', help='Path of the JSON file parsed by patches')
    parser.add_argument('--check', action='store_true', default=False, help='Check and do not write to the file')
    args = parser.parse_args()

    llm_adaptor_path = Path(args.root_dir) / Path("MindSpeed-LLM/mindspeed_llm/tasks/megatron_adaptor.py")
    preprocess(llm_adaptor_path)

    merge(args.root_dir, args.json_file, args.check)

    postprocess(llm_adaptor_path)

    tok("total")
