import argparse
import os
import re
import logging
import ast
import configparser
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define rule types
Rule = Tuple[str, str]
FileRules = Dict[str, List[Rule]]


class FunctionAndClassReplacer(ast.NodeTransformer):
    """
    A class used to replace a function or class definition in an AST (Abstract Syntax Tree)
    with a new node if the name matches the target name.
    """
    def __init__(self, target_name, new_node):
        self.target_name = target_name
        self.new_node = new_node

    def visit_FunctionDef(self, node):
        if node.name == self.target_name:
            return self.new_node
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        if node.name == self.target_name:
            return self.new_node
        return self.generic_visit(node)


def replace_function_or_class(code, target_name, new_definition):
    """
    Replace a function or class with the specified name.
    :param code: Original code
    :param target_name: Name of the function or class to be replaced
    :param new_definition: Definition code of the new function or class
    :return: Code after replacement
    """
    tree = ast.parse(code)
    new_tree = ast.parse(new_definition)
    new_node = new_tree.body[0]
    replacer = FunctionAndClassReplacer(target_name, new_node)
    new_tree = replacer.visit(tree)
    new_code = ast.unparse(new_tree)
    return new_code


class PackageConverter:
    # List of supported package names
    SUPPORTED_PACKAGES = ["megatron", "mindspeed", "MindSpeed-LLM", "MindSpeed-MM",
                          "transformers", "einops", "peft", "diffusers"]

    def __init__(self):
        self.config = self.load_config()
        self.ast_convert = FunctionAndClassReplacer()

    def load_config(self) -> configparser.ConfigParser:
        """
        Load the configuration file.
        """
        config = configparser.ConfigParser()
        try:
            config.read('converter.config')
        except Exception as e:
            logging.warning(f"Failed to read the configuration file: {e}. Using default configuration.")
        return config

    def get_files(self, folder_path: str) -> List[str]:
        """
        Get all file paths under the specified folder.
        :param folder_path: Folder path.
        :return: List of file paths.
        """
        filenames = []
        try:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    filenames.append(file_path)
        except Exception as e:
            logging.error(f"Error getting files from {folder_path}: {e}")
        return filenames

    def apply_general_rules(self, data: str, file_path: str) -> str:
        """
        Apply general rules to replace file content.
        :param data: File content.
        :param file_path: File path.
        :return: Replaced file content.
        """
        general_rules = self.get_general_rules(file_path)
        for search_text, replace_text in general_rules:
            try:
                data = data.replace(search_text, replace_text)
            except Exception as e:
                logging.error(f"Error applying general rule, file: {file_path}, rule: {search_text} -> {replace_text}: {e}")
        return data

    def get_general_rules(self, file_path: str) -> List[Rule]:
        """
        Get general rules based on the file type.
        :param file_path: File path.
        :return: List of rules.
        """
        if ".sh" in file_path:
            try:
                return self.config.get('RULES', 'SHELL_RULES').splitlines()
            except:
                from rules.general_rules import SHELL_RULES
                return SHELL_RULES
        else:
            try:
                return self.config.get('RULES', 'GENERAL_RULES').splitlines()
            except:
                from rules.general_rules import GENERAL_RULES
                return GENERAL_RULES

    def apply_file_rules(self, relative_path: str) -> str:
        """
        Apply file rules to replace the relative path of the file.
        :param relative_path: Relative path of the file.
        :return: Replaced relative path of the file.
        """
        try:
            file_rules = self.config.get('RULES', 'FILE_RULES').splitlines()
        except:
            from rules.general_rules import FILE_RULES
            file_rules = FILE_RULES
        for search_text, replace_text in file_rules:
            try:
                relative_path = relative_path.replace(search_text, replace_text)
            except Exception as e:
                logging.error(f"Error applying file rule, path: {relative_path}, rule: {search_text} -> {replace_text}: {e}")
        return relative_path

    def save_file(self, data: str, convert_file_path: str) -> None:
        """
        Save the file.
        :param data: File content.
        :param convert_file_path: Save path.
        """
        try:
            os.makedirs(os.path.dirname(convert_file_path), exist_ok=True)
            with open(convert_file_path, 'w', encoding='UTF-8') as file:
                file.write(data)
            logging.info(f"File saved successfully: {convert_file_path}")
        except Exception as e:
            logging.error(f"Error saving file to {convert_file_path}: {e}")

    def read_file(self, file_path: str) -> str:
        """
        Read the file content.
        :param file_path: File path.
        :return: File content, or None if an error occurs.
        """
        try:
            with open(file_path, 'r', encoding='UTF-8') as file:
                return file.read()
        except UnicodeDecodeError as e:
            logging.error(f"Encoding error in file {file_path}: {e}")
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
        return None

    def convert_general_rules(self, orign_path: str, save_path: str, package_name: str = "megatron") -> None:
        """
        Convert general rules.
        :param orign_path: Original path.
        :param save_path: Save path.
        :param package_name: Package name.
        """
        filenames = self.get_files(orign_path)
        for file_path in filenames:
            if not file_path.endswith(('.sh', '.py')):
                logging.warning(f"The file {file_path} is not a .sh or .py file. Skipping.")
                continue
            data = self.read_file(file_path)
            if data is not None:
                data = self.apply_general_rules(data, file_path)
                relative_path = file_path.split(orign_path)[-1]
                convert_file_path = os.path.join(save_path, relative_path)
                self.save_file(data, convert_file_path)

    def convert_special_rules(self, orign_path: str, save_path: str, package_name: str = "megatron") -> None:
        """
        Convert special rules.
        :param orign_path: Original path.
        :param save_path: Save path.
        :param package_name: Package name.
        """
        self.convert_special_rules_by_line(orign_path, save_path, package_name)
        self.convert_special_rules_by_regular(orign_path, save_path, package_name)
        self.convert_special_rules_by_ast(orign_path, save_path, package_name)

    def convert_special_rules_by_ast(self, orign_path: str, save_path: str, package_name: str = "megatron") -> None:
        """
        Convert special rules using AST.
        :param orign_path: Original path.
        :param save_path: Save path.
        :param package_name: Package name.
        """
        logging.info(f"Start converting special rules for {package_name} using AST rules.")
        ast_rules = self.get_ast_rules(package_name)
        for file_path, rules in ast_rules.items():
            oeign_file_path = os.path.join(orign_path, file_path)
            data = self.read_file(oeign_file_path)
            if data is not None:
                new_data = replace_function_or_class(data, rules[0], rules[1])
                relative_path = file_path.split(orign_path)[-1]
                convert_file_path = os.path.join(save_path, relative_path)
                self.save_file(new_data, convert_file_path)

    def get_ast_rules(self, package_name: str) -> List[Rule]:
        """
        Get AST rules.
        :param package_name: Package name.
        :return: List of rules.
        """
        try:
            rules = self.config.get(f'AST_RULES_{package_name}', 'RULES').splitlines()
            rule_pairs = [tuple(rule.split('=')) for rule in rules]
            return rule_pairs
        except:
            from rules.ast_rules import AST_RULES
            rule_pairs = AST_RULES.get(package_name, {})
            return rule_pairs

    def convert_special_rules_by_regular(self, orign_path: str, save_path: str, package_name: str = "megatron") -> None:
        """
        Convert special rules using regular expressions.
        :param orign_path: Original path.
        :param save_path: Save path.
        :param package_name: Package name.
        """
        logging.info(f"Start converting special rules for {package_name} using regular expressions.")
        try:
            cur_special_rules = self.config.get(f'REGULAR_RULES_{package_name}', 'RULES').splitlines()
            rule_dict = {}
            for rule in cur_special_rules:
                parts = rule.split(':')
                if len(parts) == 2:
                    file_path, sub_rules = parts
                    sub_rule_pairs = [tuple(sub_rule.split('=')) for sub_rule in sub_rules.split(';')]
                    rule_dict[file_path] = sub_rule_pairs
        except:
            from rules.special_rules import REGULAR_RULES
            rule_dict = REGULAR_RULES.get(package_name, {})

        for file_path, rules in rule_dict.items():
            oeign_file_path = os.path.join(orign_path, file_path)
            data = self.read_file(oeign_file_path)
            if data is not None:
                for patter, replace_text in rules:
                    if patter == "":
                        data = data + '\n' + replace_text
                    else:
                        try:
                            data = re.sub(patter, replace_text, data)
                        except re.error as e:
                            logging.error(f"Error applying regular expression rule, file: {oeign_file_path}, rule: {patter} -> {replace_text}: {e}")
                convert_file_path = os.path.join(save_path, file_path)
                self.save_file(data, convert_file_path)

    def _parse_line_rule(self, rule: str) -> Tuple[str, str]:
        """
        Parse a line rule into a pattern and replacement text.
        """
        lines = []
        for line in rule.split('\n'):
            if line:
                if line[0] in ['+', '-', ' ']:
                    lines.append((line[0], line[1:]))
                else:
                    lines.append((line[0], line))
            else:
                lines.append((line, line))
        pattern = '\n'.join([line for type, line in lines if type != '+'])
        replace = '\n'.join([line for type, line in lines if type != '-'])
        return pattern, replace

    def convert_special_rules_by_line(self, orign_path: str, save_path: str, package_name: str = "megatron") -> None:
        """
        Convert special rules line by line.
        :param orign_path: Original path.
        :param save_path: Save path.
        :param package_name: Package name.
        """
        logging.info(f"Start converting special rules for {package_name} using line rules.")
        try:
            cur_special_rules = self.config.get(f'LINE_RULES_{package_name}', 'RULES').splitlines()
            rule_dict = {}
            for rule in cur_special_rules:
                parts = rule.split(':')
                if len(parts) == 2:
                    file_path, sub_rules = parts
                    sub_rules_list = sub_rules.split(';')
                    rule_dict[file_path] = sub_rules_list
        except:
            from rules.special_rules import LINE_RULES
            rule_dict = LINE_RULES.get(package_name, {})

        for file_path, rules in rule_dict.items():
            orign_file_path = os.path.join(orign_path, file_path)

            if not os.path.exists(orign_file_path):
                convert_file_path = os.path.join(save_path, file_path)
                os.makedirs(os.path.dirname(convert_file_path), exist_ok=True)
                filtered_content = rules[0].replace('+', '')
                self.save_file(filtered_content, convert_file_path)
                continue

            data = self.read_file(orign_file_path)
            if data is not None:
                for rule in rules:
                    pattern, replace = self._parse_line_rule(rule)
                    try:
                        data = replace.join(data.split(pattern))
                    except Exception as e:
                        logging.error(f"Error applying line rule, file: {orign_file_path}, rule: {rule}: {e}")
                convert_file_path = os.path.join(save_path, file_path)
                self.save_file(data, convert_file_path)

    def convert_package(self, orign_path: str, save_path: str, package_name: str = "megatron") -> None:
        """
        Convert a package.
        :param orign_path: Original path.
        :param save_path: Save path.
        :param package_name: Package name.
        """
        logging.info(f"Converting special rules for {package_name}")
        self.convert_special_rules(orign_path, save_path, package_name)

        logging.info(f"Converting general rules for {package_name}")
        self.convert_general_rules(orign_path, save_path, package_name)

    def parse_arguments(self) -> argparse.Namespace:
        """
        Parse command-line arguments.
        :return: Parsed arguments.
        """
        parser = argparse.ArgumentParser()
        for package in self.SUPPORTED_PACKAGES:
            parser.add_argument(f"--{package}_path", type=str, default=None,
                                help=f"Original path of the {package} package")
            parser.add_argument(f"--convert_{package}_path", type=str, default=None,
                                help=f"Converted path of the {package} package")
        return parser.parse_args()

    def start(self) -> None:
        args = self.parse_arguments()
        for package_name in self.SUPPORTED_PACKAGES:
            orign_path = getattr(args, f"{package_name}_path")
            if orign_path:
                if not os.path.exists(orign_path):
                    logging.warning(f"The specified original path {orign_path} for package {package_name} does not exist. Skipping this package.")
                    continue
                save_path = getattr(args, f"convert_{package_name}_path", orign_path)
                logging.info(f"Starting conversion of package {package_name}...")
                self.convert_package(orign_path, save_path, package_name)
                logging.info(f"Finished conversion of package {package_name}.")


if __name__ == "__main__":
    converter = PackageConverter()
    converter.start()