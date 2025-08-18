# Copyright (c) Huawei Technologies Co., Ltd 2025.  All rights reserved.
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
    args = parser.parse_args()
    
    file_iterator = source_file_iterator("./")
    string_mapping = [('torch', 'msadapter'), ('torch_npu', 'msadapter_npu')]
    file_converter = FileConverter(APITransformer, ('torch', 'msadapter', string_mapping))

    with Pool(processes=args.multiprocess) as pool:
        results = list(tqdm(pool.imap(file_converter.convert, file_iterator), desc="Processing"))

    if args.path_to_change not in ['MindSpeed-LLM/', 'MindSpeed-MM/']:
        raise ValueError(f"Unsupported path_to_change: {args.path_to_change}. "
                        "Expected 'MindSpeed-LLM/' or 'MindSpeed-MM/'.")

    for file, value in SPECIAL_CASE.items():
        file_converter = FileConverter(value['converter'], value['mapping_list'])
        results.append(file_converter.convert(file))

    result_log_path = os.path.join(os.getcwd(), "result.log")
    with open(result_log_path, 'w') as f:
        f.write('\n'.join(results))

    print(f"[INFO] Conversion completed. Log saved to: {result_log_path}")
