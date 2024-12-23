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
# ============================================================================

"""Model and parameters serialization."""

__all__ = [
    "get_checkpoint_name",
    "save_rng_state",
    "load_rng_state",
    'save_pre_process',
    'load_post_process',
    "save_checkpoint",
    "load_checkpoint",
    "detect_checkpoint_version_by_dir",
    "CkptVersion"
]

import re
import os
import sys
import stat
import glob
import shutil
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import _checkparam as validator
from mindspore.ops import operations as P
import mindspore.communication.comm_func as comm_func
try:
    from mindspore import default_generator, set_rng_state
except ImportError:
    from mindspore.nn.generator import default_generator, set_rng_state
from mindspore.communication import get_rank
from mindspore.train.serialization import _update_param
from mindspeed_ms.tools import logger
from mindspeed_ms.training.global_vars import get_args
from mindspeed_ms.core.utils import generate_state_dict, save_strategy_file, pp_layer_rename
from mindspeed_ms.core.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_pipeline_model_parallel_rank,
    get_expert_model_parallel_world_size
)
from mindspeed_ms.core.tensor_parallel.random import (
    get_rng_tracer,
    CANDIDATE_MODES
)
from mindspeed_ms.core.optimizer.optimizer import MixedPrecisionOptimizer

# Distribution configurations.
_STRATEGY_DIR = "strategy"
_FORMAT = "ckpt"
RE_ITER = re.compile("iter_([0-9]+)")


class CkptVersion:
    """Supported checkpoint versions"""
    # V1: Global Rank+Epoch+Step base
    # ie. rank_0/network_rank_0-0_1.ckpt
    V1 = 1.0

    # V2: Iteration+TP+PP+DP base
    # ie. iter_0000001/mp_rank_00_000/model_optim_rng_000.ckpt
    V2 = 2.0

    @classmethod
    def validate(cls, ckpt_version):
        if ckpt_version not in [cls.V1, cls.V2]:
            raise ValueError(f"Invalid checkpoint version:{ckpt_version}! "
                             f"Supports:{[cls.V1, cls.V2]}")


def get_checkpoint_tracker_filename(ckpt_path):
    """ Get the path of tracker file """
    return os.path.join(ckpt_path, "latest_checkpointed_iteration.txt")


# pylint: disable=W0622
def get_checkpoint_name(ckpt_path, format=_FORMAT, get_name_from_file=False,
                        prefix: str = "network", epoch_num: int = None, step_num: int = None,
                        ckpt_version: float = CkptVersion.V2,
                        base_name: str = "model_optim_rng", iteration: int = None,
                        release: bool = False, return_base_dir: bool = False,
                        dist_ckpt: bool = False, ensure_exist: bool = True):
    """
    Get checkpoint file name of model and optimizer.
    The strategy file will be saved in a standalone dir for the possible subsequent merging.
    The checkpoint file will be separated in different dir for the possible subsequent transformation.
    """
    validator.check_value_type("ckpt_path", ckpt_path, [str])
    CkptVersion.validate(ckpt_version)
    if ckpt_version == CkptVersion.V1:
        return _get_checkpoint_name_v1(
            ckpt_path, format, get_name_from_file,
            prefix, epoch_num, step_num, ensure_exist)
    return _get_checkpoint_name_v2(
        ckpt_path, format, prefix, base_name,
        iteration, release, return_base_dir,
        dist_ckpt, ensure_exist)


# pylint: disable=W0622
def _get_checkpoint_name_v2(ckpt_path, format=_FORMAT, prefix: str = "",
                            base_name: str = "model_optim_rng",
                            iteration: int = None, release: bool = False,
                            return_base_dir: bool = False,
                            dist_ckpt: bool = False,
                            ensure_exist: bool = True):
    """For checkpoint version: 2.0
    Get checkpoint file name of model and optimizer.
    The layout of the ckpt_path will be like:
    ckpt_path/
    ├── latest_checkpointed_iteration.txt
    ├── iter_0000001
    │   ├── mp_rank_00_000
    │   │   ├── model_optim_rng_000.ckpt
    │   │   └── model_optim_rng_001.ckpt
    │   ├── mp_rank_00_001
    │   ├── mp_rank_01_000
    │   └── mp_rank_01_001
    └── iter_0000002
    The strategy file will be saved in a standalone dir for the possible subsequent merging.
    The checkpoint file will be separated in different dir for the possible subsequent transformation.
    """
    ckpt_file = None
    strategy_file = None

    if release:
        directory = "release"
    elif iteration is not None:
        directory = f"iter_{iteration:07d}"
    else:
        directory = ""

    ckpt_path = os.path.normpath(os.path.abspath(ckpt_path))
    ckpt_local_path = os.path.join(ckpt_path, directory)
    if return_base_dir:
        return ckpt_local_path

    rank = get_rank()
    dp_rank = get_data_parallel_rank(with_context_parallel=True)
    tp_rank = get_tensor_model_parallel_rank()
    pp_rank = get_pipeline_model_parallel_rank()

    # Assemble path of ckpt file
    ckpt_local_path = os.path.join(ckpt_local_path, f"mp_rank_{tp_rank:02d}")
    if get_pipeline_model_parallel_world_size() > 1:
        ckpt_local_path += f"_{pp_rank:03d}"
    if ensure_exist:
        os.makedirs(ckpt_local_path, exist_ok=True)

    suffix = f"{dp_rank:03d}" \
        if dist_ckpt and get_data_parallel_world_size(with_context_parallel=True) > 1 else ""
    ckpt_file = os.path.join(ckpt_local_path, "_".join([
        part for part in [prefix, base_name, suffix] if part
    ]) + f".{format}")

    # Assemble path of strategy file
    strategy_local_path = os.path.join(ckpt_path, _STRATEGY_DIR)
    strategy_file = os.path.join(strategy_local_path, f"strategy{rank}.ckpt")
    return ckpt_file, strategy_file


# pylint: disable=W0622
def _get_checkpoint_name_v1(ckpt_path, format=_FORMAT, get_name_from_file=False,
                            prefix: str = "network", epoch_num: int = None, step_num: int = None,
                            ensure_exist: bool = True):
    """For checkpoint version: 1.0
    Get checkpoint file name of model and optimizer.
    The layout of the ckpt_path will be like:
    ckpt_path/
    ├── rank_0
    │   ├── network_rank_0-0_1.ckpt
    │   └── network_rank_0-0_2.ckpt
    └── rank_1
        ├── network_rank_1-0_1.ckpt
        └── network_rank_1-0_2.ckpt
    The strategy file will be saved in a standalone dir for the possible subsequent merging.
    The checkpoint file will be separated in different dir for the possible subsequent transformation.
    """
    rank = get_rank()
    # ensure ckpt path exist
    ckpt_path = os.path.normpath(os.path.abspath(ckpt_path))
    ckpt_local_path = os.path.join(ckpt_path, f"rank_{rank}")
    if ensure_exist:
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
        ckpt_file = os.path.join(ckpt_local_path, f"{prefix}_rank_{rank}-{epoch_num}_{step_num}.{format}")
    return ckpt_file, strategy_file


def save_rng_state():
    """ save random number generator state. """
    rng_state_dict = get_rng_tracer().get_state()
    rng_state_dict["default_generator"] = default_generator
    return rng_state_dict


def load_rng_state(param_dict):
    """ load random number generator state. """
    # set default rng tracer state
    target_state = {
        mode: param_dict.pop(mode)
        for mode in CANDIDATE_MODES
        if mode in param_dict
    }
    get_rng_tracer().set_state(target_state)
    loaded = list(target_state)
    no_loaded = list(filter(lambda x: x not in loaded, CANDIDATE_MODES))
    # set default generator state
    if "default_generator" in param_dict:
        default_generator_loaded = param_dict.pop("default_generator")
        set_rng_state(default_generator_loaded)
        loaded.append("default_generator")
    else:
        no_loaded.append("default_generator")
    if not loaded and no_loaded:
        raise KeyError(
            f"Unable to get the weight of {no_loaded} from state dict. "
            f"Specify --no-load-rng or --finetune to prevent "
            f"attempting to load the rng.")


def _update_zero(params_dict, shard_info, param, group):
    """ allgather param among dp region when using zero optimizer. """
    tensor_concat = comm_func.all_gather_into_tensor(param, group=group)[0]
    params_dict[param.name] = ms.Parameter(tensor_concat, name=param.name)
    shard_info[param.name]['opt_weight_shard_size'] = 0
    shard_info[param.name]['opt_weight_shard_step'] = 0


def _get_params_dict(model, optimizer, include_optim: bool = True):
    """ get params dict for saving checkpoint. """
    params_dict = None
    args = get_args()
    if optimizer is None:
        params_dict = model.parameters_dict()
    elif isinstance(optimizer, MixedPrecisionOptimizer):
        if args.use_dist_ckpt:
            params_dict = optimizer.state_dict(include_optim)
        else:
            params_dict = optimizer.get_parameter_state_dp_zero(include_optim)
        for _, param in model.parameters_and_names():
            if not param.requires_grad and "set_hidden_states" not in param.name:
                params_dict[param.name] = param
    else:
        params_dict = optimizer.parameters_dict() if include_optim else {}
        for _, param in model.parameters_and_names():
            if not param.requires_grad and "set_hidden_states" not in param.name:
                params_dict[param.name] = param
    if not params_dict:
        raise ValueError("None of params dict has been extract from model and optimizer.")
    return params_dict


def _save_process_pp_layers(shard_info, params_dict):
    """ preprocess pp layers before saving, drop pp suffix """
    if get_pipeline_model_parallel_world_size() == 1:
        return shard_info, params_dict
    model_shard_info = shard_info["model"]
    optimizer_shard_info = shard_info["optimizer"]
    need_process_dicts = [params_dict, model_shard_info, optimizer_shard_info]
    for process_dict in need_process_dicts:
        for name, _ in list(process_dict.items()):
            if "layers." not in name:
                continue
            new_name = pp_layer_rename(name, need_drop_suffix=True)
            local_param = process_dict.pop(name)
            process_dict[new_name] = local_param
    return shard_info, params_dict


# pylint: disable=W0212
def save_pre_process(shard_info, model, optimizer, config, include_optim: bool = True):
    """ preprocess before saving, split qkv and handle pp embedding share """
    model_shard_info = shard_info["model"]
    optimizer_shard_info = shard_info["optimizer"]
    params_dict = _get_params_dict(model, optimizer, include_optim)
    # ZeRO DP
    if optimizer is not None and hasattr(optimizer, "zero_level") and optimizer.zero_level in ["z1", "z2", "z3"]:
        dp_tp_group = get_data_parallel_group(with_context_parallel=optimizer.with_context_parallel)
        for idx, param in enumerate(optimizer._parameters):
            if include_optim and (optimizer._status_splited[idx] or optimizer._parameter_splited[idx]):
                _update_zero(params_dict, optimizer_shard_info, optimizer.moments1[idx], dp_tp_group)
                _update_zero(params_dict, optimizer_shard_info, optimizer.moments2[idx], dp_tp_group)
            if optimizer.zero_level == "z3" and optimizer._parameter_splited[idx]:
                _update_zero(params_dict, model_shard_info, param, dp_tp_group)

    # process qkv/moe/pp-share
    for name, param in list(params_dict.items()):
        target_shard_info = model_shard_info if name in model_shard_info else optimizer_shard_info
        ### moe layer
        if config.num_moe_experts is not None and config.num_moe_experts > 1 and "local_experts.0" in name:
            local_expert_num = config.num_moe_experts // get_expert_model_parallel_world_size()
            local_experts_list = []
            for idx in range(local_expert_num):
                local_expert_name = name.replace("local_experts.0", f"local_experts.{idx}")
                local_expert_param = params_dict.pop(local_expert_name).asnumpy()
                local_experts_list.append(local_expert_param)
                shard_dict = target_shard_info.pop(local_expert_name)
            local_experts_concat = np.stack(local_experts_list, axis=0)
            params_dict[name] = ms.Parameter(ms.Tensor(local_experts_concat))
            shard_dict['shape'] = local_experts_concat.shape
            shard_dict['shard'] = (get_expert_model_parallel_world_size(),) + shard_dict['shard']
            target_shard_info[name] = shard_dict

        ### handle pipeline head sharing
        args = get_args()
        if get_pipeline_model_parallel_world_size() == 1 and not args.untie_embeddings_and_output_weights:
            language_model_embedding = "language_model.embedding.word_embeddings.weight"
            language_model_head = "language_model.output_layer.weight"
            if language_model_embedding in name:
                new_name = name.replace(language_model_embedding, language_model_head)
                params_dict[new_name] = ms.Parameter(param, name=new_name)
                target_shard_info[new_name] = target_shard_info[name]

    shard_info, params_dict = _save_process_pp_layers(shard_info, params_dict)
    return shard_info, params_dict


# pylint: disable=W0212
def load_post_process(config, params_dict, optimizer=None, load_optim: bool = True):
    """ load post processing, concat qkv """
    for name, param in list(params_dict.items()):
        ### moe layer
        if config.num_moe_experts is not None and config.num_moe_experts > 1 and "local_experts.0" in name:
            local_expert_num = config.num_moe_experts // get_expert_model_parallel_world_size()
            params_dict.pop(name)
            for shard_id in range(local_expert_num):
                new_name = name.replace("local_experts.0", f"local_experts.{shard_id}")
                params_dict[new_name] = ms.Parameter(ms.Tensor(param[shard_id]))

    for name, param in list(params_dict.items()):
        ### pp layer
        new_name = pp_layer_rename(name, need_drop_suffix=False)
        params_dict.pop(name)
        params_dict[new_name] = ms.Parameter(ms.Tensor(param))

    ### ZeRO DP
    if optimizer is not None and hasattr(optimizer, "zero_level") and optimizer.zero_level in ["z1", "z2", "z3"]:
        shard_id = get_data_parallel_rank()
        split = P.Split(0, get_data_parallel_world_size())

        def _split_param(name, is_optim_state):
            if name not in params_dict:
                raise KeyError(
                    f"Unable to get the weight of '{name}' from state dict." + (
                        " Specify --no-load-optim or --finetune to prevent"
                        " attempting to load the optimizer state."
                        if is_optim_state else ""
                    )
                )
            splited_tensor = split(params_dict[name])[shard_id]
            params_dict[name] = ms.Parameter(splited_tensor, name=name)

        for idx, param in enumerate(optimizer._parameters):
            if load_optim and (optimizer._status_splited[idx] or optimizer._parameter_splited[idx]):
                _split_param(optimizer.moments1[idx].name, True)
                _split_param(optimizer.moments2[idx].name, True)
            if optimizer.zero_level == "z3" and optimizer._parameter_splited[idx]:
                # param
                if "norm" in param.name or "embedding" in param.name:
                    continue
                _split_param(param.name, False)
    return params_dict


# pylint: disable=W0622
def save_checkpoint(config, model, optimizer=None, opt_param_scheduler=None, ckpt_path="./", format=_FORMAT,
                    only_save_strategy=False, prefix: str = 'network', iteration: int = 0,
                    crc_check: bool = False, keep_checkpoint_max: int = 5, **kwargs):
    """
    Save checkpoint of distributed network to a specified file in the process of specified rank.

    Args:
        model (Cell): The network to be saved.
        ckpt_path (str): Checkpoint file path. Default: ``"./"``.
        optimizer (Cell): The optimizer to be saved. Default: ``None``.
        kwargs (dict): Configuration options dictionary.

    Raises:
        TypeError: If the type of parameter `model` is not nn.Cell.
        TypeError: If the type of parameter `optimizer` is not nn.Cell.
        TypeError: If the type of parameter `ckpt_path` is not str.
    """
    args = get_args()
    if crc_check and format == "safetensors":
        raise ValueError("crc_check does not support format 'safetensors' for now.")
    if keep_checkpoint_max < 1:
        raise ValueError(f"expect keep_checkpoint_max >= 1, but got {keep_checkpoint_max}")
    # validator check
    validator.check_value_type("model", model, [nn.Cell], "save_checkpoint")
    validator.check_value_type("optimizer", optimizer, [nn.Cell, type(None)], "save_checkpoint")
    for key in kwargs:
        logger.warning(f"The parameter '{key}' is not used in save_checkpoint.")

    file_saving = args.use_dist_ckpt or (get_data_parallel_rank(with_context_parallel=True) == 0)
    ckpt_file, strategy_file = get_checkpoint_name(ckpt_path, format=format, prefix=prefix, iteration=iteration,
                                                   dist_ckpt=args.use_dist_ckpt, ensure_exist=file_saving)
    if file_saving:
        logger.info(f"Saving model to {ckpt_file}")

    # generate sharded info
    shard_info = generate_state_dict(model, optimizer, not args.no_save_optim)
    shard_info, params_dict = save_pre_process(shard_info, model, optimizer, config, not args.no_save_optim)

    # saving
    save_strategy_file(shard_info, strategy_file)
    if not only_save_strategy:
        append_dict = dict()
        if not args.no_save_rng:
            append_dict.update(save_rng_state())
        if not args.no_save_optim and opt_param_scheduler is not None:
            append_dict.update(opt_param_scheduler.state_dict())
        append_dict.update({"iteration": iteration, "checkpoint_version": CkptVersion.V2})
        # ensure ckpt number is less than `keep_checkpoint_max` after saving,
        # so make 1 free space for incoming ckpt
        if file_saving:
            ensure_total_ckpt_is_less_than_limit(ckpt_path=ckpt_path, limit=keep_checkpoint_max - 1)
            ms.save_checkpoint(params_dict, ckpt_file, append_dict=append_dict, format=format, crc_check=crc_check)
            record_last_ckpt_to_json(iteration, get_checkpoint_tracker_filename(ckpt_path))
    logger.info("ckpt saved")


def ensure_total_ckpt_is_less_than_limit(ckpt_path: str, limit: int = 5):
    """
    make sure the provided path contain less than limited number of checkpoint file
    Args:
        ckpt_path (str): Checkpoint file path.
        limit (int): limited number of checkpoint file. Default: 5
        format (str): checkpoint format. Default: '_format'
    """
    if get_rank() != 0:
        return
    # Sorting ckpt by iteration and remove early iterations
    ckpt_list = sorted(
        glob.glob(os.path.join(ckpt_path, "iter_*")),
        key=lambda x: int(RE_ITER.findall(x)[0])
    )
    ckpt_num = len(ckpt_list)
    if ckpt_num > limit:
        for rm_ckpt_path in ckpt_list[: (ckpt_num - limit)]:
            logger.warning(f"Current checkpoint file exceed keep_checkpoint_max, "
                           f"removing {os.path.basename(rm_ckpt_path)}")
            shutil.rmtree(rm_ckpt_path, ignore_errors=True)


# pylint: disable=W0622
def load_checkpoint(config, model, optimizer=None, opt_param_scheduler=None, ckpt_path="./", format=_FORMAT,
                    crc_check=False, ckpt_version=CkptVersion.V2, release=False, **kwargs):
    """
    Load checkpoint info from a specified file in process of rank 0.

    Args:
        ckpt_path (str): Checkpoint file path.
        model (Cell): The network where the parameters will be loaded.
        optimizer (Cell): The optimizer where the parameters will be loaded.

    Raises:
        TypeError: If the type of parameter `ckpt_path` is not str.
        TypeError: If the type of parameter `model` is not nn.Cell.
        TypeError: If the type of parameter `optimizer` is not nn.Cell. Default: ``None``.
    """
    args = get_args()
    if crc_check and format == "safetensors":
        raise ValueError("crc_check does not support format 'safetensors' for now.")
    validator.check_value_type("model", model, [nn.Cell], "load_checkpoint")
    validator.check_value_type("optimizer", optimizer, [nn.Cell, type(None)], "load_checkpoint")
    CkptVersion.validate(ckpt_version)

    if os.path.isdir(ckpt_path):
        ckpt_version, src_ckpt_file, release = get_last_checkpoint(ckpt_path, format)
    elif os.path.isfile(ckpt_path):
        src_ckpt_file = ckpt_path
    else:
        src_ckpt_file = None

    if src_ckpt_file is None:
        logger.warning(f"There is no *.{format} in {ckpt_path}, load failed.")
        if args.exit_on_missing_checkpoint:
            logger.error("--exit-on-missing-checkpoint is set and exit now")
            sys.exit()
        logger.warning("Unable to load any checkpoint and will start from random.")
        args.resume_training = False
        return

    for key in kwargs:
        logger.warning(f"The parameter '{key}' is not used in load_checkpoint.")
    args = get_args()

    logger.info(f"Loading latest checkpoint: {src_ckpt_file}, this may take a while.")
    param_dict = ms.load_checkpoint(src_ckpt_file, format=format, crc_check=crc_check)

    if "checkpoint_version" in param_dict:
        tmp_ckpt_version = param_dict.get("checkpoint_version")
        if ckpt_version is not None and ckpt_version != tmp_ckpt_version:
            logger.warning(
                f"Loaded checkpoint version is not consistent with provided checkpoint version. "
                f"Loaded version is {tmp_ckpt_version} and "
                f"provided version is {ckpt_version}."
            )
        args.ckpt_version = tmp_ckpt_version
    else:
        if ckpt_version == CkptVersion.V2:
            logger.warning(
                f"Unable to get 'checkpoint_version' from ckpt "
                f"and it is expected to contained in ckpt "
                f"when provided ckpt_version is {ckpt_version}"
            )
        args.ckpt_version = CkptVersion.V1
    CkptVersion.validate(args.ckpt_version)

    if args.resume_training:
        if args.ckpt_version == CkptVersion.V1:
            # 'step_num' and 'epoch_num' are deprecated in the latest training progress
            args.epoch_num = int(param_dict.pop("epoch_num", 0))
            args.step_num = int(param_dict.pop("step_num", 0))
            logger.info(f"Checkpoint has trained {args.epoch_num} epochs, {args.step_num} steps.")
        else:
            args.iteration = int(param_dict.pop("iteration", 0))
            logger.info(f"Checkpoint has trained {args.iteration} iterations.")

    load_rng = not (release or args.finetune or args.no_load_rng)
    if load_rng:
        load_rng_state(param_dict)

    load_optim = not (release or args.finetune or args.no_load_optim)
    if load_optim and opt_param_scheduler is not None:
        opt_param_scheduler.load_state_dict(param_dict)

    # restore native optimizer/model
    param_dict = load_post_process(config, param_dict, optimizer, load_optim)
    if isinstance(optimizer, MixedPrecisionOptimizer):
        # restore distributed optimizer
        optimizer.load_state_dict(param_dict, load_optim)
        # synchronize parameters in optimizer to model
        optimizer.reload_main_params()
        if (args.fp16 or args.bf16) and not load_optim:
            optimizer.reload_model_params()
        for _, param in model.parameters_and_names():
            if not param.requires_grad and "set_hidden_states" not in param.name:
                new_param = param_dict.get(param.name)
                if new_param is None:
                    logger.warning(f"Fail to get the weight of '{param.name}' from state dict.")
                    continue
                _update_param(param, new_param, False)
    else:
        param_not_load, ckpt_not_load = ms.load_param_into_net(model, param_dict)
        if param_not_load:
            logger.warning(f"When loading ckpt into the model, param_not_load:{param_not_load}")
        if ckpt_not_load:
            logger.warning(f"When loading ckpt into the model, ckpt_not_load:{ckpt_not_load}")
        if load_optim and optimizer is not None:
            param_not_load, ckpt_not_load = ms.load_param_into_net(optimizer, param_dict)
            if param_not_load:
                logger.warning(f"When loading ckpt into the optimizer, param_not_load:{param_not_load}")
            if ckpt_not_load:
                logger.warning(f"When loading ckpt into the optimizer, ckpt_not_load:{ckpt_not_load}")

    logger.info(f"Checkpoint: {src_ckpt_file} is loaded successfully!")


def detect_checkpoint_version_by_dir(ckpt_path):
    """Detect checkpoint version based on directory arrangement"""
    logger.info(f"Detecting checkpoint versions of: {ckpt_path}")
    ckpt_versions = []
    # checkpoint version 2.0
    ckpt_list = glob.glob(os.path.join(ckpt_path, "iter_*"))
    if ckpt_list:
        ckpt_versions.append(CkptVersion.V2)
    # checkpoint version 1.0
    ckpt_list = glob.glob(os.path.join(ckpt_path, "rank_*"))
    if ckpt_list:
        ckpt_versions.append(CkptVersion.V1)
    logger.info(f"Detected  checkpoint versions: {ckpt_versions}")
    return ckpt_versions


def get_last_checkpoint(ckpt_path: str, format: str = _FORMAT, rank: int = None, ckpt_versions: list = None):
    """Get latest checkpoint under ckpt_path."""
    if not ckpt_versions:
        ckpt_versions = detect_checkpoint_version_by_dir(ckpt_path)
    # Get new version first
    if CkptVersion.V2 in ckpt_versions:
        ckpt_version = CkptVersion.V2
        ckpt_file, release = _get_last_checkpoint_v2(ckpt_path, format)
    elif CkptVersion.V1 in ckpt_versions:
        ckpt_version = CkptVersion.V1
        rank = get_rank() if rank is None else rank
        ckpt_path = os.path.join(ckpt_path, f"rank_{rank}")
        ckpt_file = _get_last_checkpoint_v1(ckpt_path, format)
        release = False
    else:
        raise ValueError(f"No a valid checkpoint directory: {ckpt_path}")
    return (ckpt_version, ckpt_file, release)


def _get_last_checkpoint_v2(ckpt_path: str, format: str = _FORMAT):
    """For checkpoint version: 2.0
    Get last iteration checkpoint under ckpt_path.
    """
    args = get_args()
    ckpt_file = None
    iteration = 0
    release = False
    tracker_file = get_checkpoint_tracker_filename(ckpt_path)

    if os.path.exists(tracker_file):
        with open(tracker_file, 'r') as fp:
            metastring = fp.read().strip()
            try:
                iteration = int(metastring)
            except ValueError:
                release = (metastring == 'release')
        assert iteration > 0 or release, \
            f"The content of tracker file is expected to be iteration or 'release', " \
            f"but got: '{metastring}'"
    else:
        ckpt_list = sorted(
            glob.glob(os.path.join(ckpt_path, "iter_*")),
            key=lambda x: int(RE_ITER.findall(x)[-1])
        )
        if not ckpt_list:
            logger.warning(f"No ckpts found in {ckpt_path}")
            return (ckpt_file, release)
        iteration = int(RE_ITER.findall(ckpt_list[-1])[-1])
        logger.warning(f"No tracker files found in {ckpt_path}. "
                       f"Using the latest ckpt: {os.path.basename(ckpt_list[-1])}")

    ckpt_file, _ = get_checkpoint_name(
        ckpt_path, format=format,
        iteration=iteration,
        release=release,
        dist_ckpt=args.use_dist_ckpt,
        ckpt_version=CkptVersion.V2,
        prefix=args.prefix
    )
    return (ckpt_file, release)


def _get_last_checkpoint_v1(ckpt_path: str, format: str = _FORMAT):
    """For checkpoint version: 1.0
    Get last timestamp checkpoint under ckpt_path.
    """
    ckpt_list = [
        checkpoint for checkpoint in os.listdir(ckpt_path)
        if checkpoint.endswith(f'.{format}')
    ]
    if not ckpt_list:
        return None
    ckpt_list = sorted(ckpt_list, key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)))
    return os.path.join(ckpt_path, ckpt_list[-1])


def record_last_ckpt_to_json(iteration: int, meta_file: str):
    """record last ckpt info to json"""
    flags = os.O_WRONLY | os.O_CREAT
    mode = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(meta_file, flags, mode), 'w', encoding="utf-8") as fp:
        fp.write(str(iteration))
