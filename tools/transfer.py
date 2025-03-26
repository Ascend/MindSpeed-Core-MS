# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""TRANSFER TOOL"""


import argparse
import os
import re
from rules.general_rules import GENERAL_RULES, SHELL_RULES, FILE_RULES
from rules.special_rules import SPECIAL_RULES
from rules.line_rules import LINE_RULES


def getfiles(folder_path):
    """getfiles"""
    filenames = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            filenames.append(file_path)
    return filenames


def convert_general_rules(origin_path, save_path):
    """convert_general_rules"""
    filenames = getfiles(origin_path)
    for file_path in filenames:
        # open
        print(file_path)
        if ".pyc" in file_path or (".py" not in file_path and ".sh" not in file_path):
            continue

        with open(file_path, 'r', encoding='UTF-8') as file:
            data = file.read()

        # save
        relative_path = file_path.split(origin_path)[-1]
        for rule in FILE_RULES:
            search_text = rule[0]
            replace_text = rule[1]
            relative_path = relative_path.replace(search_text, replace_text)
        convert_file_path = os.path.join(save_path, relative_path)
        os.makedirs(os.path.dirname(convert_file_path), exist_ok=True)
        with open(convert_file_path, 'w', encoding='UTF-8') as file:
            file.write(data)


def convert_special_rules(origin_path, save_path, package_name="megatron"):
    """convert_special_rules"""
    cur_special_rules = SPECIAL_RULES[package_name]
    for file_path, rules in cur_special_rules.items():
        # open
        oeign_file_path = os.path.join(origin_path, file_path)
        with open(oeign_file_path, 'r', encoding='UTF-8', newline='') as file:
            data = file.read()

        # replace
        for rule in rules:
            patter = rule[0]
            replace_text = rule[1]
            if patter == "":
                data = data + '\n' + replace_text
            else:
                data = re.sub(patter, replace_text, data)

        # save
        convert_file_path = os.path.join(save_path, file_path)
        os.makedirs(os.path.dirname(convert_file_path), exist_ok=True)
        with open(convert_file_path, 'w', encoding='UTF-8', newline='') as file:
            file.write(data)


def convert_special_rules_by_line(origin_path, save_path, package_name="megatron"):
    """convert_special_rules_by_line"""
    cur_special_rules = LINE_RULES[package_name]
    for file_path, rules in cur_special_rules.items():
        origin_file_path = os.path.join(origin_path, file_path)
        if rules[0] == "REMOVE":
            if os.path.exists(origin_file_path):
                os.remove(origin_file_path)
            continue

        if not os.path.exists(origin_file_path):
            "create new file"
            convert_file_path = os.path.join(save_path, file_path)
            os.makedirs(os.path.dirname(convert_file_path), exist_ok=True)
            with open(convert_file_path, 'w', encoding='UTF-8') as file:
                file.write(rules[0])
            continue
        # open
        with open(origin_file_path, 'r', encoding='UTF-8') as file:
            data = file.read()

        # replace
        for rule in rules:
            lines = [(line[0], line[1:]) for line in rule.split('\n') if line != '']
            pattern = '\n'.join([line for type, line in lines if type != '+'])
            replace = '\n'.join([line for type, line in lines if type != '-'])
            if pattern in data:
                data = replace.join(data.split(pattern))
            else:
                print(f"warning! {origin_file_path} replace fail")
                print(rule)
        # save
        convert_file_path = os.path.join(save_path, file_path)
        os.makedirs(os.path.dirname(convert_file_path), exist_ok=True)
        with open(convert_file_path, 'w', encoding='UTF-8') as file:
            file.write(data)


def convert_package(origin_path, save_path, package_name="megatron"):
    """convert_package"""
    if package_name == "MindSpeed-LLM":
        convert_special_rules_by_line(origin_path, save_path, package_name=package_name)
    else:
        convert_special_rules_by_line(origin_path, save_path, package_name=package_name)
        convert_special_rules(origin_path, save_path, package_name=package_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron_path", type=str, default=None,
                        help="origin megatron package path")
    parser.add_argument("--convert_megatron_path", type=str, default=None,
                        help="origin megatron package path")
    parser.add_argument("--mindspeed_path", type=str, default=None,
                        help="origin mindspeed package path")
    parser.add_argument("--convert_mindspeed_path", type=str, default=None,
                        help="origin mindspeed package path")
    parser.add_argument("--mindspeed_llm_path", type=str, default=None,
                        help="origin mindspeed package path")
    parser.add_argument("--convert_mindspeed_llm_path", type=str, default=None,
                        help="origin mindspeed package path")
    parser.add_argument("--mindspeed_rl_path", type=str, default=None,
                        help="origin mindspeed-rl package path")
    parser.add_argument("--convert_mindspeed_rl_path", type=str, default=None,
                        help="origin mindspeed-rl package path")    
    parser.add_argument("--vllm_path", type=str, default=None,
                        help="origin vllm package path")
    parser.add_argument("--convert_vllm_path", type=str, default=None,
                        help="origin vllm package path")    
    parser.add_argument("--vllm_ascend_path", type=str, default=None,
                        help="origin vllm-ascend package path")
    parser.add_argument("--convert_vllm_ascend_path", type=str, default=None,
                        help="origin vllm-ascend package path")    
    parser.add_argument("--transformers_path", type=str, default=None,
                        help="origin mindspeed package path")
    parser.add_argument("--convert_transformers_path", type=str, default=None,
                        help="origin mindspeed package path")
    parser.add_argument("--einops_path", type=str, default=None,
                        help="origin einops package path")
    parser.add_argument("--convert_einops_path", type=str, default=None,
                        help="convert einops package path")
    parser.add_argument("--is_rl", action="store_true",
                        help="is rl")
    args = parser.parse_args()

    if args.is_rl:
        from rules_rl.special_rules import SPECIAL_RULES
        from rules_rl.line_rules import LINE_RULES

    if args.megatron_path:
        origin_path = args.megatron_path
        save_path = origin_path if not args.convert_megatron_path else args.convert_megatron_path
        convert_package(origin_path, save_path)

    if args.mindspeed_path:
        origin_path = args.mindspeed_path
        save_path = origin_path if not args.convert_mindspeed_path else args.convert_mindspeed_path
        convert_package(origin_path, save_path, "mindspeed")

    if args.mindspeed_llm_path:
        origin_path = args.mindspeed_llm_path
        save_path = origin_path if not args.convert_mindspeed_llm_path else args.convert_mindspeed_llm_path
        convert_package(origin_path, save_path, "MindSpeed-LLM")

    if args.mindspeed_rl_path:
        origin_path = args.mindspeed_rl_path
        save_path = origin_path if not args.convert_mindspeed_rl_path else args.convert_mindspeed_rl_path
        convert_package(origin_path, save_path, "mindspeed-rl")

    if args.vllm_path:
        origin_path = args.vllm_path
        save_path = origin_path if not args.convert_vllm_path else args.convert_vllm_path
        convert_package(origin_path, save_path, "vllm")
        
    if args.vllm_ascend_path:
        origin_path = args.vllm_ascend_path
        save_path = origin_path if not args.convert_vllm_ascend_path else args.convert_vllm_ascend_path
        convert_package(origin_path, save_path, "vllm-ascend")

    if args.transformers_path:
        origin_path = args.transformers_path
        save_path = origin_path if not args.convert_transformers_path else args.convert_transformers_path
        convert_package(origin_path, save_path, "transformers")

    if args.einops_path:
        origin_path = args.einops_path
        save_path = origin_path if not args.convert_einops_path else args.convert_einops_path
        convert_special_rules(origin_path, save_path, package_name="einops")
