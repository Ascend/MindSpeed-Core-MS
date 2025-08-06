# Copyright (c) Huawei Technologies Co., Ltd 2025.  All rights reserved.
import re
import libcst as cst
from libcst.metadata import PositionProvider, ScopeProvider, MetadataWrapper


class StringTransformer(cst.CSTTransformer):
    """
    map torch string to msadapter
    """
    METADATA_DEPENDENCIES = (PositionProvider, ScopeProvider)

    def __init__(self, string_mapping):
        self.string_mapping = string_mapping

    def leave_SimpleString(self, original_node, updated_node):
        new_value = original_node.value
        for (current_name, new_name) in self.string_mapping:
            if new_value.strip('\'') == current_name:
                new_value = re.sub(current_name, new_name, new_value)
        return updated_node.with_changes(value=new_value)


class PairTransformer(cst.CSTTransformer):

    def __init__(self, string_mapping):
        self.string_mapping = string_mapping
        
    def leave_DictElement(self, original_node: cst.DictElement, updated_node: cst.DictElement) -> cst.DictElement:
        if (
            isinstance(updated_node.key, cst.SimpleString) and
            isinstance(updated_node.value, cst.SimpleString)
        ):
            new_key_value = original_node.key.value
            new_value_value = original_node.value.value
            for ((current_name_key, current_name_val), (new_name_key, new_name_val)) in self.string_mapping:
                if new_key_value.strip('"\'') == current_name_key and new_value_value.strip('"\'') == current_name_val:
                    new_key_value = re.sub(current_name_key, new_name_key, new_key_value)
                    new_value_value = re.sub(current_name_val, new_name_val, new_value_value)
        new_key = updated_node.key.with_changes(value=new_key_value)
        new_value = updated_node.value.with_changes(value=new_value_value)
        return updated_node.with_changes(key=new_key, value=new_value)
