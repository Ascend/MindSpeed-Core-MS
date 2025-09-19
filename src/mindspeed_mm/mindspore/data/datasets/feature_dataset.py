# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""
get_data_from_feature_data.

mindspore is not support stack function with bloat16 input.
"""

import torch

def get_data_from_feature_data(self, feature_path: str) -> dict:
    """
    Load feature data from a specified file path.

    Args:
        feature_path (str): The path to the feature data file.

    Returns:
        dict: A dictionary containing the loaded feature data.
    """
    if feature_path.endswith(".pt"):
        feature_data = torch.load(feature_path, map_location=torch.device('cpu'))
        for key in feature_data.keys():
            if isinstance(feature_data[key], torch.Tensor):
                feature_data[key] = feature_data[key].to(torch.float) \
                    if feature_data[key].dtype == torch.bfloat16 else feature_data[key]
            else:
                tmp = []
                for tensor in feature_data[key]:
                    if tensor.dtype == torch.bfloat16:
                        tmp.append(tensor.to(torch.float))
                    else:
                        tmp.append(tensor)
                feature_data[key] = tuple(tmp) if isinstance(feature_data[key], tuple) else tmp
        return feature_data
    raise NotImplementedError("Unsupported file format. Only .pt files are currently supported.")
