# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================
"""checkponiting"""

import glob
import json
import os
import mindspore as ms
from mindspore import _checkparam as validator
from mindspore import log as logger
from mindspore import nn
from mindspore.communication import get_group_size
from mindspore.communication import get_rank
from mindspore.ops import operations as P
from mindspore.train.serialization import _update_param

from mindspeed_ms.core.ddp.optimizer import MixedPrecisionOptimizer

_STRATEGY_DIR = "strategy"
_FORMAT = "ckpt"

# pylint: disable=W0622
def get_checkpoint_name(ckpt_path, format=_FORMAT, get_name_from_file=False,
                        prefix: str = "network", global_step=0):
    """
    Get checkpoint file name of model and optimizer.
    The layout of the ckpt_path will be like:
    ckpt_path/
    ├── rank_0
    │   ├── network_rank_0-0_0.ckpt
    │   └── network_rank_0-0_1.ckpt
    └── rank_1
        ├── network_rank_1-0_0.ckpt
        └── network_rank_1-0_1.ckpt
    The strategy file will be saved in a standalone dir for the possible subsequent merging.
    The checkpoint file will be separated in different dir for the possible subsequent transformation.
    """
    validator.check_value_type("ckpt_path", ckpt_path, [str])
    rank = get_rank()
    # ensure ckpt path exist
    ckpt_path = os.path.normpath(os.path.abspath(ckpt_path))
    ckpt_local_path = os.path.join(ckpt_path, f"rank_{rank}")
    os.makedirs(ckpt_local_path, exist_ok=True)
    # get default strategy file name
    strategy_local_path = os.path.join(ckpt_path, _STRATEGY_DIR)
    strategy_file = os.path.join(strategy_local_path, f"stratey{rank}.ckpt")
    # read ckpt name according to the ckpt path or return default name
    if get_name_from_file:
        rank_ckpts = glob.glob(os.path.join(ckpt_local_path, "*." + format))
        if not rank_ckpts:
            raise RuntimeError(f"{ckpt_local_path} has no .{format} ckpt file found")
        for checkpoint_file in rank_ckpts:
            if not os.path.isfile(checkpoint_file):
                ms.log.warning("{} is not a checkpoint file.".format(checkpoint_file))
                continue
            ckpt_file = checkpoint_file
    else:
        ckpt_file = os.path.join(ckpt_local_path, f"{prefix}_rank_{rank}_{global_step}.{format}")
    return ckpt_file, strategy_file


def _get_params_dict(model, optimizer):
    """ get params dict for saving checkpoint. """
    params_dict = {}
    if optimizer is None:
        params_dict = model.parameters_dict()
    elif isinstance(optimizer, MixedPrecisionOptimizer):
        params_dict = optimizer.state_dict()
        for _, param in model.parameters_and_names():
            if not param.requires_grad:
                params_dict[param.name] = param
    else:
        params_dict = optimizer.parameters_dict()
        for _, param in model.parameters_and_names():
            params_dict[param.name] = param
    if not params_dict:
        raise ValueError("None of params dict has been extract from model and optimizer.")
    return params_dict

def save_pre_process(model, optimizer):
    """ preprocess before saving, split qkv and handle pp embedding share """
    params_dict = _get_params_dict(model, optimizer)
    return params_dict

# pylint: disable=W0622
def ensure_total_ckpt_is_less_than_limit(ckpt_path: str, limit: int = 5, format: str = _FORMAT):
    """
    make sure the provided path contain less than limited number of checkpoint file
    Args:
        ckpt_path (str): Checkpoint file path.
        limit (int): limited number of checkpoint file. Default: 5
        format (str): checkpoint format. Default: '_format'
    """
    ckpt_list = [
        checkpoint for checkpoint in os.listdir(ckpt_path)
        if checkpoint.endswith(f'.{format}')
    ]
    # ckpt_list: [oldest, ..., newest]
    ckpt_list = sorted(ckpt_list, key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)))
    ckpt_num = len(ckpt_list)
    if ckpt_num > limit:
        for rm_ckpt_name in ckpt_list[: (ckpt_num - limit)]:
            logger.debug(f"Current checkpoint file exceed keep_checkpoint_max, removing {rm_ckpt_name}")
            rm_ckpt_path = os.path.join(ckpt_path, rm_ckpt_name)
            os.remove(rm_ckpt_path)


def record_last_ckpt_to_json(global_step: int, ckpt_file: str, meta_json: str):
    """record last ckpt info to json"""

    meta_data = {
        "last_global_step": global_step,
        "last_ckpt_file": ckpt_file
    }
    with open(meta_json, 'w', encoding="utf-8") as fp:
        json.dump(meta_data, fp)

# pylint: disable=W0612, W0613
def save_checkpoint(model, optimizer=None, opt_param_scheduler=None, ckpt_path="./", format=_FORMAT,
                    only_save_strategy=False, prefix: str = 'network', global_step=0,
                    crc_check: bool = False, keep_checkpoint_max: int = 5, untie_embeddings_and_output_weights=False,
                    append_dict=None, **kwargs):
    """ save checkpoint. """

    if crc_check and format == "safetensors":
        raise ValueError("crc_check does not support format 'safetensors' for now.")
    if keep_checkpoint_max < 1:
        raise ValueError(f"expect keep_checkpoint_max >= 1, but got {keep_checkpoint_max}")
    validator.check_value_type("model", model, [nn.Cell], "save_checkpoint")
    validator.check_value_type("optimizer", optimizer, [nn.Cell, type(None)], "save_checkpoint")
    rank_path = os.path.join(ckpt_path, f"rank_{get_rank()}")
    ckpt_file, strategy_file = get_checkpoint_name(ckpt_path, format=format, prefix=prefix, global_step=global_step)

    params_dict = save_pre_process(model, optimizer)
    if not only_save_strategy:
        append_dict = {} if append_dict is None else append_dict
        if opt_param_scheduler is not None:
            opt_state_dict = opt_param_scheduler.state_dict()
            append_dict.update(opt_state_dict)
        append_dict.update({"global_step": global_step})
        # ensure ckpt number is less than `keep_checkpoint_max` after saving,
        # so make 1 free space for incoming ckpt
        ensure_total_ckpt_is_less_than_limit(ckpt_path=rank_path, limit=keep_checkpoint_max - 1, format=format)
        ms.save_checkpoint(params_dict, ckpt_file, append_dict=append_dict, format=format, crc_check=crc_check)
        record_last_ckpt_to_json(global_step=global_step, ckpt_file=os.path.basename(ckpt_file),
                                 meta_json=os.path.join(rank_path, 'meta.json'))
    logger.info("ckpt saved")

# pylint: disable=W0212
def get_last_checkpoint(ckpt_path: str, format: str = _FORMAT):
    """Get last timestamp checkpoint under ckpt_path."""

    ckpt_list = [
        checkpoint for checkpoint in os.listdir(ckpt_path)
        if checkpoint.endswith(f'.{format}')
    ]
    if not ckpt_list:
        return None
    ckpt_list = sorted(ckpt_list, key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)))
    return os.path.join(ckpt_path, ckpt_list[-1])


def load_post_process(params_dict, optimizer=None):
    """ load post processing, concat qkv """

    if optimizer is not None and hasattr(optimizer, "zero_level") and optimizer.zero_level in ["z1", "z2", "z3"]:
        shard_id = get_rank()
        split = P.Split(0, get_group_size())
        for idx, param in enumerate(optimizer._parameters):
            if optimizer._status_splited[idx] or optimizer._parameter_splited[idx]:
                # moments1
                moments1_name = optimizer.moments1[idx].name
                moments1 = params_dict[moments1_name]
                splited_tensor = split(moments1)[shard_id]
                params_dict[moments1_name] = ms.Parameter(splited_tensor, name=moments1_name)
                # moments2
                moments2_name = optimizer.moments2[idx].name
                moments2 = params_dict[moments2_name]
                splited_tensor = split(moments2)[shard_id]
                params_dict[moments2_name] = ms.Parameter(splited_tensor, name=moments2_name)
            if optimizer.zero_level == "z3" and optimizer._parameter_splited[idx]:
                # param
                if "norm" in param.name or "embedding" in param.name:
                    continue
                cell_param = params_dict[param.name]
                splited_tensor = split(cell_param)[shard_id]
                params_dict[param.name] = ms.Parameter(splited_tensor, name=param.name)
    return params_dict

# pylint: disable=W0613
def load_checkpoint(model, optimizer=None, opt_param_scheduler=None, ckpt_path="./", format=_FORMAT,
                    crc_check=False, **kwargs):
    """load ckpt """

    if crc_check and format == "safetensors":
        raise ValueError("crc_check does not support format 'safetensors' for now.")
    validator.check_value_type("model", model, [nn.Cell], "load_checkpoint")
    validator.check_value_type("optimizer", optimizer, [nn.Cell, type(None)], "load_checkpoint")
    logger.info("ckpt loading")
    if os.path.isdir(ckpt_path):
        src_ckpt_file = get_last_checkpoint(os.path.join(ckpt_path, f"rank_{get_rank()}"), format=format)
    elif os.path.isfile(ckpt_path):
        src_ckpt_file = ckpt_path
    else:
        raise ValueError(f"There is no *.{format} in {ckpt_path}, load failed.")
    param_dict = ms.load_checkpoint(src_ckpt_file, format=format, crc_check=crc_check)

    if opt_param_scheduler is not None:
        opt_param_scheduler.load_state_dict(param_dict)
    if optimizer is not None and isinstance(optimizer, MixedPrecisionOptimizer):
        # restore distributed optimizer
        optimizer.load_state_dict(param_dict)
        # synchronize parameters in optimizer to model
        optimizer.reload_main_params()
        for _, param in model.parameters_and_names():
            if not param.requires_grad:
                new_param = param_dict[param.name]
                _update_param(param, new_param, False)
    else:
        # restore native optimizer/model
        param_dict = load_post_process(param_dict, optimizer)
        param_not_load, ckpt_not_load = ms.load_param_into_net(model, param_dict)
        if param_not_load:
            logger.warning(f"When loading into network, param_not_load:{param_not_load}")
        if ckpt_not_load:
            logger.warning(f"When loading into network, ckpt_not_load:{ckpt_not_load}")
        if optimizer is not None:
            param_not_load, ckpt_not_load = ms.load_param_into_net(optimizer, param_dict)
            if param_not_load:
                logger.warning(f"When loading into optimizer, param_not_load:{param_not_load}")
            if ckpt_not_load:
                logger.warning(f"When loading into optimizer, ckpt_not_load:{ckpt_not_load}")
    logger.info("ckpt loaded")

    return param_dict
