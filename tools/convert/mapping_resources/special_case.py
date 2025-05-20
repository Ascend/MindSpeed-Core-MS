# Copyright (c) Huawei Technologies Co., Ltd 2012-2020.  All rights reserved.
from modules.string_transformer import StringTransformer

SPECIAL_CASE = {
    'MindSpeed-LLM/msadapter/library.py': {
        'converter': StringTransformer,
        'reason': 'torch in string, but affect module register'
    },
    'MindSpeed-LLM/mindspeed_llm/tasks/megatron_adaptor.py': {
        'converter': StringTransformer,
        'reason': 'torch in patch function call, affect module patch'
    },
    'MindSpeed-LLM/mindspeed/megatron_adaptor.py': {
        'converter': StringTransformer,
        'reason': 'torch in patch function call, affect module patch'
    }
}