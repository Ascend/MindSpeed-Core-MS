# Copyright (c) Huawei Technologies Co., Ltd 2012-2020.  All rights reserved.
import os
import re
import argparse
from multiprocessing import Pool
from tqdm import tqdm
import libcst as cst
from libcst.metadata import MetadataWrapper
from modules.api_transformer import APITransformer
from modules.utils import source_file_iterator, FileConverter
from mapping_resources.special_case import SPECIAL_CASE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_change", type=str, default=None,
                        help="origin package path or file path")
    parser.add_argument("--multiprocess", type=int, default=32)
    parser.add_argument("--result_log", type=str, default='./result.log')
    args = parser.parse_args()

    file_iterator = source_file_iterator(args.path_to_change)
    file_converter = FileConverter(APITransformer, ('torch', 'msadapter'))

    with Pool(processes=args.multiprocess) as pool:
        results = list(tqdm(pool.imap(file_converter.convert, file_iterator), desc="Processing"))
    
    for file, value in SPECIAL_CASE.items():
        file_converter = FileConverter(value['converter'], ('torch', 'msadapter'))
        results.append(file_converter.convert(file))
    with open(args.result_log, 'w') as f:
        f.write('\n'.join(results))