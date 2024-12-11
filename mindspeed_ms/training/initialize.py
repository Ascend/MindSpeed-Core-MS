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
"""mindspeed ms initialization."""

import os
import logging
from datetime import timedelta
import random
import numpy as np
import mindspore as ms
import mindspore.mint as mint
from mindspore.communication import get_rank, get_local_rank, get_group_size

from mindspeed_ms.training.arguments import parse_args, validate_args
from mindspeed_ms.training.yaml_arguments import validate_yaml
from mindspeed_ms.training.global_vars import set_global_variables, get_args
from mindspeed_ms.core import parallel_state
from mindspeed_ms.tools.logger import logger
from mindspeed_ms.core.tensor_parallel.random import parallel_mode_manual_seed, set_seed_states

# pylint: disable=W0102,W0613
def initialize_mindspeed_ms(
        extra_args_provider=None,
        args_defaults={},
        ignore_unknown_args=False,
        allow_no_cuda=False,
        skip_mpu_initialization=False,
):
    """ init mindspore distributed environment """
    # parse arguments
    args, defaults = parse_args(extra_args_provider, ignore_unknown_args)

    if args.yaml_cfg is not None:
        args = validate_yaml(args, defaults, args_defaults)
    else:
        setattr(defaults, "num_moe_experts", defaults.num_experts)
        setattr(defaults, "layernorm_epsilon", defaults.norm_epsilon)
        del defaults.num_experts
        del defaults.norm_epsilon
        validate_args(args, defaults, args_defaults)

    # set global args. build tokenizer,
    set_global_variables(args)

    # set logging level
    setup_logging()

    deterministic = "ON" if args.deterministic_mode else "OFF"
    ms.set_context(device_target='Ascend', mode=ms.PYNATIVE_MODE, deterministic=deterministic)

    # mindspore.distributed initialization
    def finish_mpu_init():
        args = get_args()

        # initialize distributed communication
        _initialize_distributed()

        # set random seed
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(args.seed, args.data_parallel_random_init)

    if skip_mpu_initialization:
        return None

    args = get_args()

    finish_mpu_init()

    return None


def _initialize_distributed():
    " initialize distributed communication and parallel. "
    args = get_args()

    device_count = ms.hal.device_count(device_target="Ascend")
    # pylint: disable=W0212
    if ms.communication._comm_helper._is_initialized():
        if args.rank == 0:
            print(
                "MindSpore NPU distribution is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
        args.rank = get_rank()

        # Here no args, so return world_size but not group size不等于worldsize
        args.world_size = get_group_size()
    else:
        if args.rank == 0:
            print("> initializing MindSpore NPU distributed...")
        # init mindspore device communication
        mint.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size,
            rank=args.rank,
            timeout=timedelta(minutes=args.distributed_timeout_minutes)
        )
        args.rank = get_rank()
        args.local_rank = get_local_rank()
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert (args.local_rank == device), "expected local-rank to be the same as rank % device-count."
            else:
                args.local_rank = device
            ms.set_context(device_id=device)

    # set the tensor model parallel, pipeline model parallel,
    # and data parallel communicators
    if device_count > 0:
        if parallel_state.model_parallel_is_initialized():
            logger.warning("model parallel is already initialized.")
        else:
            parallel_state.initialize_model_parallel(
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.virtual_pipeline_model_parallel_size,
                args.pipeline_model_parallel_split_rank,
                context_parallel_size=args.context_parallel_size,
                expert_model_parallel_size=args.expert_model_parallel_size,
                order='tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-pp-dp',
            )

            if args.rank == 0:
                print(
                    f"> initialized tensor model parallel with size "
                    f"{parallel_state.get_tensor_model_parallel_world_size()}"
                )
                print(
                    f"> initialized pipeline model parallel with size "
                    f"{parallel_state.get_pipeline_model_parallel_world_size()}"
                )


def _set_random_seed(seed_, data_parallel_random_init=False):
    """Set random seed for reproducibility."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * parallel_state.get_pipeline_model_parallel_rank())
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (10 * parallel_state.get_data_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        ms.set_seed(seed)
        ms.manual_seed(seed)
        if ms.hal.device_count(device_target="Ascend") > 0:
            parallel_mode_manual_seed(seed)
        set_seed_states()
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))


def write_args_to_tensorboard():
    """Write arguments to tensorboard."""
    raise NotImplementedError("write args to tensorboard is not supported now.")

# pylint: disable=W1203
def setup_logging() -> None:
    """ Sets the default logging level based on cmdline args and env vars.

    Precedence:
    1. Command line argument `--logging-level`
    2. Env var `MINDSPORE_LOGGING_LEVEL`
    3. Default logging level (INFO)

    Returns: None
    """
    args = get_args()
    logging_level = None
    env_logging_level = os.getenv('MINDSPORE_LOGGING_LEVEL', None)
    if env_logging_level is not None:
        logging_level = int(env_logging_level)
    if args.logging_level is not None:
        logging_level = args.logging_level

    if logging_level is not None:
        logger.info(f'Setting logging level to {logging_level}')
        logging.getLogger().setLevel(logging_level)
