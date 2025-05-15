# Copyright (c) Huawei Technologies Co., Ltd 2012-2020.  All rights reserved.
import os
import re
import argparse
from tqdm import tqdm
import libcst as cst
from libcst.metadata import MetadataWrapper
from modules.api_transformer import APITransformer
from modules.utils import source_file_iterator


def convert_torch_api(input_file, new_name='msadapter'):
    """
    convert single file
    args:
        input_file: path for a single file
        new_name: new api name
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            code = f.read()
        module = cst.parse_module(code)
        wrapper = MetadataWrapper(module)
        new_code = wrapper.visit(APITransformer('torch', new_name))
        with open(input_file, 'w') as f:
            f.write(new_code.code)
        return f'{input_file}: True'
    except Exception as e:
        return f'{input_file}: False\n------>{e}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_change", type=str, default=None,
                        help="origin package path or file path")
    parser.add_argument("--multiprocess", type=int, default=32)
    parser.add_argument("--result_log", type=str, default='./result.log')
    args = parser.parse_args()

    file_iterator = source_file_iterator(args.path_to_change)

    from multiprocessing import Pool
    with Pool(processes=args.multiprocess) as pool:
        results = list(tqdm(pool.imap(convert_torch_api, file_iterator), desc="Processing"))
    with open(args.result_log, 'w') as f:
        f.write('\n'.join(results))
