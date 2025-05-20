# Copyright (c) Huawei Technologies Co., Ltd 2012-2020.  All rights reserved.
import re
import libcst as cst
from libcst.metadata import PositionProvider, ScopeProvider, MetadataWrapper


class StringTransformer(cst.CSTTransformer):
    """
    map torch string to msadapter
    """
    METADATA_DEPENDENCIES = (PositionProvider, ScopeProvider)

    def __init__(self, current_name, new_name):
        self.current_name = current_name
        self.new_name = new_name

    def leave_FormattedStringText(self, original_node, updated_node):
        pattern = fr'\b{self.current_name}\.'
        new_value = re.sub(pattern, f'{self.new_name}.', original_node.value)
        return updated_node.with_changes(value=new_value)

    def leave_SimpleString(self, original_node, updated_node):
        pattern = fr'\b{self.current_name}\.'
        new_value = re.sub(pattern, f'{self.new_name}.', original_node.value)
        return updated_node.with_changes(value=new_value)
