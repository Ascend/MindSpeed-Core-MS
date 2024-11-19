# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron initialization."""
import logging
import random
import os

import numpy as np
import mindspore as ms
from mindspeed_ms.core.parallel_state import initialize_model_parallel
from mindspeed_ms.core.parallel_state import get_tensor_model_parallel_world_size
from mindspeed_ms.core.parallel_state import get_pipeline_model_parallel_world_size

from mindspeed_ms.core.parallel_state import get_pipeline_model_parallel_rank
from mindspeed_ms.core.parallel_state import get_data_parallel_rank
from mindspeed_ms.core.tensor_parallel.random import parallel_mode_manual_seed

from mindspeed_ms.training import get_args
from mindspeed_ms.training.arguments import parse_args, validate_args
from mindspeed_ms.training.global_vars import set_global_variables

logger = logging.getLogger(__name__)

# pylint: disable=W0102, W0613, W0105
def initialize_megatron(
        extra_args_provider=None,
        args_defaults={},
        ignore_unknown_args=False,
        allow_no_cuda=False,
        skip_mpu_initialization=False,
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    # if not allow_no_cuda:
    #     ms.run_check()
    #     #Make sure Ascend NPU is available.
    #     assert ms.hal.is_initialized("Ascend"), "MindFormers requires Ascend."

    # Parse arguments
    args = parse_args(extra_args_provider, args_defaults, ignore_unknown_args)
    '''
    if args.yaml_cfg is not None:
        args = validate_yaml(args, args_defaults)
    else:
        validate_args(args, args_defaults)
    '''
    if args.yaml_cfg is None:
        validate_args(args, args_defaults)

    # set global args, build tokenizer
    set_global_variables(args)

    # set logging level
    setup_logging()

    # mindspore.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Mindspore distributed.
        _initialize_distributed()
        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(args.seed, args.data_parallel_random_init)

    if skip_mpu_initialization:
        return None

    args = get_args()

    #  Complete initialization right away.
    finish_mpu_init()

    return None

# pylint: disable=W0212
def _initialize_distributed():
    """Initialize torch.distributed and core model parallel."""
    args = get_args()

    device_count = ms.hal.device_count(device_target="Ascend")
    if ms.communication._comm_helper._is_initialized():
        if args.rank == 0:
            print(
                "MindSpore NPU distribution is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
        args.rank = ms.communication.get_rank()

        # Here no args, so return world_size but not group size不等于worldsize
        args.world_size = ms.communication.get_group_size()
    else:

        print("> initializing MindSpore NPU distributed...", flush=True)
        ms.communication.init(backend_name="hccl")
        ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
        args.rank = ms.communication.get_rank()
        args.local_rank = ms.communication.get_local_rank()

        device = args.rank % device_count
        if args.local_rank is not None:
            assert (args.local_rank == device), "expected local-rank to be the same as rank % device-count."
        else:
            args.local_rank = device
        ms.set_context(device_id=device)

    if device_count > 0:
        initialize_model_parallel(
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank=args.pipeline_model_parallel_split_rank,
            context_parallel_size=args.context_parallel_size,
            expert_model_parallel_size=args.expert_model_parallel_size,
            order="tp-cp-ep-dp-pp" if not args.use_tp_pp_dp_mapping else 'tp-pp-dp',
            communicator_config_path=args.nccl_communicator_config_path
        )

        ## if you want to check the tp or pp group size:
        # from mindspeed_ms.core.parallel_state import get_pipeline_model_parallel_group
        # from mindspeed_ms.core.parallel_state import get_tensor_model_parallel_group
        # print("get_pipeline_model_parallel_group",get_pipeline_model_parallel_group())
        # print("get_tensor_model_parallel_group",get_tensor_model_parallel_group())

        if args.rank == 0:
            print(
                f"> initialized tensor model parallel with size "
                f"{get_tensor_model_parallel_world_size()}"
            )
            print(
                f"> initialized pipeline model parallel with size "
                f"{get_pipeline_model_parallel_world_size()}"
            )


def _set_random_seed(seed_, data_parallel_random_init=False):
    """Set random seed for reproducibility."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * get_pipeline_model_parallel_rank())
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (10 * get_data_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        ms.set_seed(seed)
        ms.manual_seed(seed)
        if ms.hal.device_count(device_target="Ascend") > 0:
            parallel_mode_manual_seed(seed)

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
