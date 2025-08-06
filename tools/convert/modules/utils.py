# Copyright (c) Huawei Technologies Co., Ltd 2025.  All rights reserved.
import os
import re
import libcst as cst
from libcst.metadata import MetadataWrapper


def source_file_iterator(path, ext='.py'):
    """
    get list of files from a path(could be dir or file) 
    """
    if os.path.isfile(path):
        if path.endswith(ext):
            yield path
    else:
        for entry in os.scandir(path):
            yield from source_file_iterator(entry.path)


def case_insensitive_replace(original_str, search_str, replacement_str):
    """
    replace search_str with replacement_str in original_str, ignore case
    """
    pattern = re.compile(re.escape(search_str), re.IGNORECASE)
    
    def replacer(match):
        matched_str = match.group()
        if matched_str.isupper():
            return replacement_str.upper()
        elif matched_str.islower():
            return replacement_str.lower()
        elif matched_str.istitle():
            return replacement_str.capitalize()
        else:
            formated_replacement = []
            for i in range(min(len(matched_str), len(replacement_str))):
                if matched_str[i].isupper():
                    formated_replacement.append(replacement_str[i].upper())
                else:
                    formated_replacement.append(replacement_str[i].lower())
            if len(replacement_str) > len(matched_str):
                formated_replacement.extend(replacement_str[len(matched_str):])
            return ''.join(formated_replacement)
    
    return pattern.sub(replacer, original_str)


def get_docstring(node):
    """
    get docstring from a cst func or class node
    """
    statements = node.body.body
    if statements is None:
        return None
    first_statement = statements[0]
    if not isinstance(first_statement, cst.SimpleStatementLine):
        return None
    if first_statement.body is None:
        return None
    if isinstance(first_statement.body[0], cst.Expr) and isinstance(first_statement.body[0].value, cst.BaseString):
        return first_statement.body[0].value
    return None


def create_nested_attribute_or_name(path):
    chain = path.split('.')
    if len(chain) == 1:
        return cst.Name(value=chain[0])
    current = cst.Name(value=chain[0])
    for part in chain[1:]:
        current = cst.Attribute(value=current, attr=cst.Name(value=part))
    return current


class FileConverter:
    def __init__(self, transformer, transformer_args):
        self.transformer = transformer
        self.transformer_args = transformer_args

    def convert(self, input_file):
        """
        convert single file
        args:
            input_file: path for a single file
        """
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                code = f.read()
            module = cst.parse_module(code)
            wrapper = MetadataWrapper(module)
            new_code = wrapper.visit(self.transformer(*self.transformer_args))
            with open(input_file, 'w') as f:
                f.write(new_code.code)
            return f'{input_file}: True'
        except Exception as e:
            return f'{input_file}: False\n------>{e}'
