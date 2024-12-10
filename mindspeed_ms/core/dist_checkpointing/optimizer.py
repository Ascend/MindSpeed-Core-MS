# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

""" Helpers for defining sharding for optimizer states based on existing sharding for model parameters. """


# pylint: disable=W0613
def get_optim_param_to_id_map(optim_params_iter):
    raise NotImplementedError

# pylint: disable=W0613
def get_param_id_to_sharded_param_map(
        model_sharded_state_dict, optim_params_iter
):
    raise NotImplementedError

# pylint: disable=W0613
def make_sharded_optimizer_tensor(model_param, optim_param, prefix):
    raise NotImplementedError

# pylint: disable=W0613
def optim_state_to_sharding_state(optim_state_dict, id_to_sharded_param_map, exclude_keys):
    raise NotImplementedError
