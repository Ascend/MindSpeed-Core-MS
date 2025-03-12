import argparse
import os
import re
from rules.general_rules import SHELL_RULES
from rules.line_rules import LINE_RULES

def getfiles(folder_path):
    filenames = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            filenames.append(file_path)
    return filenames


def convert_general_rules(orign_path, save_path, package_name="megatron"):

    filenames = getfiles(orign_path)
    # print(orign_path)
    # print(filenames)
    for file_path in filenames:
        # open
        if ".pyc" in file_path or ".sh" not in file_path:
            continue
        # print(file_path)
        with open(file_path, 'r', encoding='UTF-8') as file:
            data = file.read()

        # replace
        if ".sh" in file_path:
            all_rules = SHELL_RULES

        for rule in all_rules:
            search_text = rule[0]
            replace_text = rule[1]
            data = data.replace(search_text, replace_text)

        # save
        relative_path = file_path.split(orign_path)[-1]
        convert_file_path = os.path.join(save_path, relative_path)
        # if file_path != convert_file_path:
        #     print(file_path, convert_file_path)
        os.makedirs(os.path.dirname(convert_file_path), exist_ok=True)
        with open(convert_file_path, 'w', encoding='UTF-8') as file:
            file.write(data)


def convert_special_rules_by_line(orign_path, save_path, package_name="megatron"):
    cur_special_rules = LINE_RULES[package_name]
    for file_path, rules in cur_special_rules.items():
        # open
        orign_file_path = os.path.join(orign_path, file_path)
        
        if not os.path.exists(orign_file_path):
            "create new file"
            convert_file_path = os.path.join(save_path,file_path)
            os.makedirs(os.path.dirname(convert_file_path), exist_ok=True)

            filtered_content = rules[0].replace('+', '')
            
            with open(convert_file_path, 'w', encoding='UTF-8') as file:
                file.write(filtered_content)
            continue

        # open
        with open(orign_file_path, 'r', encoding='UTF-8') as file:
            data = file.read()

        # replace
        for rule in rules:
            # lines = [(line[0], line[1:]) for line in rule.split('\n') if line != '']
            lines = []
            for line in rule.split('\n'):
                if line != '':
                    if line[0] in ['+', '-', ' ']:
                        lines.append((line[0], line[1:]))
                    else:
                        lines.append((line[0], line[0:]))
                else:
                    lines.append((line, line))

            pattern = '\n'.join([line for type, line in lines if type != '+'])
            replace = '\n'.join([line for type, line in lines if type != '-'])
            # print("++++++++++++{}+++++++++++++++++\n{}\n+++++++++++++++++++++++++++++\n{}\n+++++++++++++++++++++++++++++".format(file_path, pattern, replace))
            data = replace.join(data.split(pattern))
        # save
        convert_file_path = os.path.join(save_path, file_path)
        os.makedirs(os.path.dirname(convert_file_path), exist_ok=True)
        with open(convert_file_path, 'w', encoding='UTF-8') as file:
            file.write(data)


def convert_package(orign_path, save_path, package_name="megatron"):
    if package_name == "MindSpeed-LLM" or package_name == "MindSpeed-MM":
        convert_special_rules_by_line(orign_path, save_path, package_name=package_name)
            
    # if package_name in LINE_RULES and LINE_RULES[package_name]:
    #     print("convert_special_rules_by_line {}".format(package_name))
    #     convert_special_rules_by_line(orign_path, save_path, package_name=package_name)

    # if package_name in SPECIAL_RULES and SPECIAL_RULES[package_name]:
    #     print("convert_special_rules {}".format(package_name))
    #     convert_special_rules(orign_path, save_path, package_name=package_name)

    print("convert_general_rules {}".format(package_name))
    convert_general_rules(orign_path, save_path, package_name=package_name)


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
    parser.add_argument("--mindspeed_mm_path", type=str, default=None,
                        help="origin mindspeed_mm package path")
    parser.add_argument("--convert_mindspeed_mm_path", type=str, default=None,
                        help="origin mindspeed_mm package path")
    parser.add_argument("--transformers_path", type=str, default=None,
                        help="origin mindspeed package path")
    parser.add_argument("--convert_transformers_path", type=str, default=None,
                        help="origin mindspeed package path")
    parser.add_argument("--einops_path", type=str, default=None,
                        help="origin einops package path")
    parser.add_argument("--convert_einops_path", type=str, default=None,
                        help="convert einops package path")
    parser.add_argument("--peft_path", type=str, default=None,
                        help="peft einops package path")
    parser.add_argument("--convert_peft_path", type=str, default=None,
                        help="convert peft package path")
    parser.add_argument("--mindspeed_type", type=str, default=None,
                        help="Type of mindspeed, LLM or MM")
    args = parser.parse_args()

    if args.megatron_path:
        orign_path = args.megatron_path
        save_path = orign_path if not args.convert_megatron_path else args.convert_megatron_path
        convert_special_rules_by_line(orign_path, save_path, "megatron")
    
    if args.mindspeed_path:
        orign_path = args.mindspeed_path
        mindspeed_type = args.mindspeed_type
        save_path = orign_path if not args.convert_mindspeed_path else args.convert_mindspeed_path
        if mindspeed_type == "LLM":
            convert_special_rules_by_line(orign_path, save_path, "mindspeed_llm")
        elif mindspeed_type == "MM":
            convert_special_rules_by_line(orign_path, save_path, "mindspeed_mm")
        
        
    if args.mindspeed_llm_path:
        orign_path = args.mindspeed_llm_path
        save_path = orign_path if not args.convert_mindspeed_llm_path else args.convert_mindspeed_llm_path
        convert_package(orign_path, save_path, "MindSpeed-LLM")

    if args.mindspeed_mm_path:
        orign_path = args.mindspeed_mm_path
        save_path = orign_path if not args.convert_mindspeed_mm_path else args.convert_mindspeed_mm_path
        convert_package(orign_path, save_path, "MindSpeed-MM")
    
    if args.transformers_path:
        orign_path = args.transformers_path
        save_path = orign_path if not args.convert_transformers_path else args.convert_transformers_path
        convert_special_rules_by_line(orign_path, save_path, "transformers")

    if args.peft_path:
        orign_path = args.peft_path
        save_path = orign_path if not args.convert_peft_path else args.convert_peft_path
        convert_special_rules_by_line(orign_path, save_path, "peft")
