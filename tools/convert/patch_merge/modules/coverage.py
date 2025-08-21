# Copyright (c) Huawei Technologies Co., Ltd 2025.  All rights reserved.
import json
import libcst as cst
import re
from collections import defaultdict
import os
import json
import argparse


def get_printing_str(orign_import, raw_patch):
    patch_import = raw_patch["patch_import"]
    patch_name = raw_patch["patch_name"]
    condition = raw_patch["condition"]

    pstr = f"=== In patch call, orign_import: {orign_import}, patch_import: {patch_import}, patch_name: {patch_name}, condition: {condition}"
    # print(str)
    return pstr


def get_debug_print_node(patch):
    """
    Add printing node for coverage statistics
    """
    if patch is None:
        pstr = f"=== In original call"
    else:
        pstr = get_printing_str(patch["orign_import"], patch["raw_patch"])

    debug_node = cst.parse_statement(f"print(\"\"\"{pstr}\"\"\", flush=True)")

    return debug_node


def check_log(patch_json_file, log_file):
    """
    Calculate the coverage rate based on log file and json file
    """
    try:
        with open(patch_json_file, 'r', encoding='utf-8') as f:
            raw_patches = json.load(f)
    except FileNotFoundError:
        print(f"File '{patch_json_file}' not found")
    except json.JSONDecodeError:
        print(f"File '{patch_json_file}' is not a valid JSON file")
    
    try:
        with open(log_file, 'r') as f:
            log = f.read()
    except FileNotFoundError:
        print(f"File '{log_file}' not found")

    num_modules = len(raw_patches)
    num_patches = sum([len(pat) for pat in raw_patches.values()])
    
    hit_patch_cnt = 0
    hit_module_cnt = 0
    not_hit_patches = defaultdict(list)
    for orign_import, patches in raw_patches.items():
        # Remove the quotation marks at the beginning and end
        orign_import = orign_import[1:-1] if orign_import[0] == '\"' or orign_import[0] == '\'' else orign_import
        hit = False
        for patch in patches:
            pstr = get_printing_str(orign_import, patch)
            if pstr in log:
                hit_patch_cnt += 1
                hit = True
            else:
                not_hit_patches[orign_import].append(patch)
        
        if hit:
            hit_module_cnt += 1
    
    dirname, filename = os.path.split(patch_json_file)
    name, suffix = filename.rsplit('.', 1)
    not_hit_json = os.path.join(dirname, f"{name}_not_hit_cases.{suffix}")
    with open(not_hit_json, 'w', encoding='utf-8') as f:
        json.dump(not_hit_patches, f, ensure_ascii=False, indent=4)
    
    print("===============================================")
    print(f"module coverage: {hit_module_cnt}/{num_modules}, ratio={hit_module_cnt / num_modules :.3f}")
    print(f"patch coverage: {hit_patch_cnt}/{num_patches}, ratio={hit_patch_cnt / num_patches :.3f}")
    print(f"(Patches not hit were dumped into {not_hit_json})")
    print("===============================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check the patch coverage')
    parser.add_argument('--json-file', help='The path of the input JSON file')
    parser.add_argument('--log-file', default=None, help='The path of the input log file')
    args = parser.parse_args()

    check_log(args.json_file, args.log_file)