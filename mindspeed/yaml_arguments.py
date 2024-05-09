# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

from functools import wraps


def core_transformer_config_from_yaml_wrapper(fn):
    @wraps(fn)
    def wrapper(args):
        config = fn(args)
        config.context_parallel_algo = args.context_parallel_algo
        config.batch_p2p_comm = False
        config.tp_comm_overlap = False
        return config

    return wrapper