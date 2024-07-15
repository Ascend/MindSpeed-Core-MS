# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch


def permute(tokens, indices, topk: int = 1):
    if topk > 1:
        assert indices.size(1) == topk
    flatten_indices = indices.view(-1)
    # sorted_indices = torch.argsort(flatten_indices, stable=True)  # argsort int64 will be run on host cpu
    sorted_indices = torch.sort(flatten_indices.float(), stable=True)[1]
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices
