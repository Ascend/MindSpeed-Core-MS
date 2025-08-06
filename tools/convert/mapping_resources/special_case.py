# Copyright (c) Huawei Technologies Co., Ltd 2025.  All rights reserved.
from modules.string_transformer import StringTransformer, PairTransformer

SPECIAL_CASE = {
    'msadapter/serialization.py': {
        'converter': StringTransformer,
        'mapping_list': [[('msadapter', 'torch'),]],
        'reason': "'torch' must not be replaced with 'msadapter' during model loading validation."
    },
    'msadapter/proxy.py': {
        'converter': PairTransformer,
        'mapping_list': [[(('msadapter', 'msadapter'), ('torch', 'msadapter')),]],
        'reason': 'Map "torch" to "msadapter" to enable compatibility mode.'
    },
}
