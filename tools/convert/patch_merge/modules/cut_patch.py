# Copyright (c) Huawei Technologies Co., Ltd 2025.  All rights reserved.
import json
import argparse
import os
from pathlib import Path

def split_json_file(data, output, left, right):
    """
    Select a portion of the patches in the json file
    """
    keys = list(data.keys())

    selected_keys = keys[left : right + 1]
    selected = {key: data[key] for key in selected_keys}

    json_indent = 4
    try:
        with open(Path(output), 'w', encoding='utf-8') as f:
            json.dump(selected, f, ensure_ascii=False, indent=json_indent)
    except IOError as e:
        print(f"Error: Unable to write to the output file {e}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Input json file')
    parser.add_argument('--output', default=None, help='Output json file')
    parser.add_argument('--count', default=False, action='store_true', help='Count the number of patches in the input file')
    parser.add_argument('--left', type=int, default=-1, help='Left bound')
    parser.add_argument('--right', type=int, default=-1, help='Right bound')
    args = parser.parse_args()
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File '{args.input}' not exist")
    except json.JSONDecodeError:
        print(f"File '{args.input}' is not a valid json file")

    if args.count:
        print(len(data))
    else:
        if args.output is None or args.left is None or args.right is None:
            raise Exception(f"Got wrong arguments: {args.output}, {args.left}, {args.right}")
        if args.left > args.right:
            raise Exception(f"Left bound {args.left} is greater than right bound {args.right}")
        split_json_file(data, args.output, args.left, args.right)