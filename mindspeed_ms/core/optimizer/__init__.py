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

"""optimizer init"""
import dataclasses
import os
import types
from mindspore import mint, ops
from mindspore.common import dtype as mstype
from mindspore.nn import Adam, SGD
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.experimental.optim.adamw import SpeedAdamW

from mindspeed_ms.training.global_vars import get_args
from mindspeed_ms.core.register import ModuleType, ModuleRegistry
from mindspeed_ms.core.distributed import DistributedDataParallel
from mindspeed_ms.core.dist_checkpointing import get_checkpoint_name
from mindspeed_ms.core.optimizer.lr_scheduler import get_learning_rate_scheduler
from mindspeed_ms.core.optimizer.optimizer_config import OptimizerConfig
from mindspeed_ms.core.parallel_state import (get_data_parallel_group,
                                              get_data_parallel_group_mccl,
                                              get_zero_full_shard_flag,
                                              get_zero_shard_group)

from . import zero
from . import lr_scheduler
from .distrib_optimizer import DistributedOptimizer
from .optimizer import MixedPrecisionOptimizer, Float16OptimizerWithFloat16Params, get_optimizer_param_scheduler

__all__ = [
    "DistributedOptimizer", "MixedPrecisionOptimizer", "Float16OptimizerWithFloat16Params", \
    "get_optimizer", "get_optimizer_param_scheduler"
]
__all__.extend(zero.__all__)
__all__.extend(lr_scheduler.__all__)


ModuleRegistry.register(SGD, ModuleType.OPTIMIZER)
ModuleRegistry.register(mint.optim.AdamW, ModuleType.OPTIMIZER, item_name='mint.AdamW')
ModuleRegistry.register(SpeedAdamW, ModuleType.OPTIMIZER, item_name='adam')


def get_ditributed_optimizer(optimizer, optimizer_config, model_chunks):
    " warp non-parallel optimizer with distributed optimizer. "
    if model_chunks is None:
        raise ValueError("When using DistributedOptimizer based on DDP, network instance should be passed "
                         "to get_optimizer method but got None.")
    per_model_buffers = {}
    per_model_ep_buffers = {}
    for model_idx, model_chunk in enumerate(model_chunks):
        if not isinstance(model_chunk, DistributedDataParallel):
            raise TypeError("When using DistribtedOptimizer, the network passed to get_optimizer should be "
                            "wrapped with DistributedDataParallel.")
        per_model_buffers[model_idx] = model_chunk.buffers
        per_model_ep_buffers[model_idx] = model_chunk.expert_parallel_buffers
    grad_scaler = None if not optimizer_config.loss_scale \
        else ops.Tensor(optimizer_config.loss_scale, mstype.float32)
    data_parallel_group = get_data_parallel_group(with_context_parallel=True)
    data_parallel_group_mccl = get_data_parallel_group_mccl(with_context_parallel=True)
    if not get_zero_full_shard_flag():
        data_parallel_group = get_zero_shard_group(with_context_parallel=True)
        data_parallel_group_mccl = get_zero_shard_group(with_context_parallel=True, is_mccl=True)
    distributed_optimizer = DistributedOptimizer(
        optimizer=optimizer,
        config=optimizer_config,
        grad_scaler=grad_scaler,
        init_state_fn=None,
        per_model_buffers=per_model_buffers,
        data_parallel_group=data_parallel_group,
        data_parallel_group_mccl=data_parallel_group_mccl,
    )
    args = get_args()
    save_path = args.save if args.save is not None else './output'
    ckpt_file, _ = get_checkpoint_name(os.path.join(save_path, 'opt_shard_info'),
                                       format='json', prefix='dist_opt_shard_info', epoch_num=0, step_num=0)

    # add freezed zero param to be combined
    extra_zero_list = []
    for _, param in model_chunks.parameters_and_names():
        if not param.requires_grad and hasattr(param, 'use_zero3') and param.use_zero3:
            extra_zero_list.append(param.name)

    distributed_optimizer.save_opt_shard_strategy(ckpt_file, extra_zero_list)

    return distributed_optimizer


def get_non_distributed_mixed_precision_optimizer(optimizer, optimizer_config):
    " warp non-parallel optimizer with Float16OptimizerWithFloat16Params optimizer. "
    grad_scaler = None if not optimizer_config.loss_scale \
        else ops.Tensor(optimizer_config.loss_scale, mstype.float32)
    optimizer = Float16OptimizerWithFloat16Params(
        optimizer,
        optimizer_config,
        grad_scaler=grad_scaler,
        init_state_fn=None,
    )
    return optimizer


def _set_group_lr_and_weight_decay(optimizer_config, params, lr, weight_decay):
    if isinstance(params[0], dict) and not optimizer_config.optimizer.startswith("mint") \
        and not optimizer_config.optimizer.startswith("adam"):
        using_group_lr = any("lr" in param for param in params)
        for param in params:
            if "order_params" not in param:
                if "lr" not in param and using_group_lr:
                    param["lr"] = lr
                if "weight_decay" not in param:
                    param["weight_decay"] = weight_decay


def _append_order_param_group(params, network, optimizer_cls):
    """
    Append 'order_params' parameter group to params when a user invokes
    'get_optimizer' with parameter groups and intends to create a
    subclass instance of mindspore.nn.optim.Optimizer.

    NOTE: mindspore.nn.optim.Optimizer assumes that 'order_params' contains
    the original parameter list of network and arranges its parameter list
    following the order of 'order_params'.
    """
    if issubclass(optimizer_cls, Optimizer) and \
        isinstance(params, list) and \
        all(isinstance(t, dict) and "params" in t for t in params):
        if network is None:
            raise ValueError("Network must be provided when using built-in "
                             "mindspore.nn.optim.Optimizer")
        params.append({"order_params": network.trainable_params()})
    return params


def _prepare_optimizer_kwargs(optimizer_config, params, network, optimizer_cls, kwargs):
    ''' prepare optimizer kwargs for optimizer '''
    args = get_args()
    weight_decay = optimizer_config.weight_decay

    if optimizer_config.lr_scheduler_kwargs is not None:
        learning_rate = get_learning_rate_scheduler(optimizer_config)
    else:
        learning_rate = optimizer_config.lr

    _set_group_lr_and_weight_decay(optimizer_config, params, learning_rate, weight_decay)

    optimizer_kwargs = optimizer_config.get_needed_params_for_class(optimizer_cls)
    if optimizer_config.optimizer.startswith("mint") or optimizer_config.optimizer.startswith("adam"):
        optimizer_kwargs["lr"] = learning_rate
        optimizer_kwargs["betas"] = tuple([optimizer_config.adam_beta1, optimizer_config.adam_beta2])
    else:
        optimizer_kwargs["learning_rate"] = learning_rate
    optimizer_kwargs["weight_decay"] = weight_decay
    optimizer_kwargs["params"] = params
    if "grad_allreduce_op" in kwargs:
        if optimizer_config.zero_without_ddp:
            optimizer_kwargs["grad_allreduce_op"] = kwargs["grad_allreduce_op"]
        kwargs.pop("grad_allreduce_op", None)
    if optimizer_config.zero_without_ddp:
        if network is None:
            raise ValueError("Network must be provided when get ZeRO optimizer instance.")
        optimizer_kwargs["zero_level"] = args.zero_level
        optimizer_kwargs["network"] = network
        if optimizer_config.zero_config is not None:
            optimizer_kwargs.update(optimizer_config.zero_config)
    optimizer_kwargs.update(kwargs)
    return optimizer_kwargs


def get_optimizer(optimizer_config, config, params=None, network=None, return_instance: bool = True, **kwargs):
    """
    Get an optimizer instance or class based on the provided optimizer configuration.

    Args:
        optimizer_config (OptimizerConfig): The configuration object for the optimizer.
        config (TransformerConfig): The configuration object for the training.
        params (list or dict, optional): The parameters to optimize. Default: ``None``.
        network (nn.Cell, optional): The network model, should be provided when use ZeRO optimizer. Default: ``None``.
        return_instance (bool): Whether to return an instance of the optimizer with extra optimizer arguments.
            Default: ``True``.
        **kwargs: Additional keyword arguments to be passed to the optimizer class.

    Returns:
        Optimizer instance, an instance of the optimizer class if `return_instance` is ``True``,
        otherwise the optimizer class itself.

    Raises:
        RuntimeError: If `zero_without_ddp` in `optimizer_config` is ``False`` and `zero_level` in
            `optimizer_config.parallel_config` is ``z3`` and `use_distributed_optimizer` in `training_config` is
            ``Fasle``.
        ValueError: If `return_instance` is ``True`` and `params` is ``None``.
        NotImplementedError: If `return_instance` is ``True`` and `weight_decay_kwargs` in `optimizer_config` is not
            ``None``.

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

            You need to save the following codes as a python file and run command:
            msrun --worker_num 1 --local_worker_num 1 --master_port 8848 --log_dir log --join True \
                  --cluster_time_out 300 example.py --micro-batch-size 8 --num-layers 4 --hidden-size 64 \
                  --num-attention-heads 4 --seq-length 32 --max-position-embeddings 32 --vocab-size 128 \
                  --tokenizer-type NullTokenizer --no-masked-softmax-fusion --lr 0.0001

        >>> from mindspore.communication import init
        >>> from mindspeed_ms.core.optimizer import get_optimizer, optimizer_config_from_args
        >>> from mindspeed_ms.training import get_model
        >>> from mindspeed_ms.legacy.model.language_model import get_language_model
        >>> from mindspeed_ms.training.utils import set_weight_decay
        >>> from mindspeed_ms.training import core_transformer_config_from_args, get_args
        >>> from mindspeed_ms.training.initialize import initialize_mindspeed_ms
        >>> def model_provider_func(pre_process=True, post_process=True):
        ...     config = core_transformer_config_from_args(get_args())
        ...     network_with_loss, _ = get_language_model(config=config, num_tokentypes=0,
        ...         add_pooler=False, encoder_attn_mask_type=None, pre_process=pre_process, post_process=post_process)
        ...     return network_with_loss
        >>> init()
        >>> initialize_mindspeed_ms()
        >>> args = get_args()
        >>> config = core_transformer_config_from_args(args)
        >>> optimizer_config = optimizer_config_from_args(args)
        >>> network_with_loss = get_model(model_provider_func, config)
        >>> group_params = set_weight_decay(network_with_loss.trainable_params(), optimizer_config.weight_decay)
        >>> optimizer = get_optimizer(optimizer_config, config, group_params, network_with_loss)
    """
    args = get_args()
    optimizer_config.zero_without_ddp = config.zero_level is not None and \
        not args.wrap_with_ddp

    optimizer_type = optimizer_config.optimizer

    if optimizer_config.zero_without_ddp:
        optimizer_type = optimizer_type + "ZeRO"

    elif config.zero_level == 'z3' and not optimizer_config.use_distributed_optimizer:
        raise RuntimeError("For zero3 with DDP, use_distributed_optimizer must be on. Please check the configuration.")

    optimizer_cls = ModuleRegistry.get_item(module_type=ModuleType.OPTIMIZER, item_name=optimizer_type)
    if not return_instance:
        return optimizer_cls

    if params is None:
        raise ValueError("params must be provided when return_instance is True.")

    params = _append_order_param_group(params, network, optimizer_cls)

    optimizer_kwargs = _prepare_optimizer_kwargs(optimizer_config, params, network, optimizer_cls, kwargs)
    return_item = optimizer_cls(**optimizer_kwargs)
    if args.wrap_with_ddp and optimizer_config.use_distributed_optimizer:
        return_item = get_ditributed_optimizer(
            return_item,
            optimizer_config,
            network,
        )
    elif config.fp16 or config.bf16:
        return_item = get_non_distributed_mixed_precision_optimizer(
            return_item,
            optimizer_config
        )
    return return_item


def optimizer_config_from_args(args, config_class=None):
    """Optimzer config from args"""
    # Config class.
    config_class = config_class or OptimizerConfig

    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(config_class):
        if hasattr(args, f.name):
            val = getattr(args, f.name)
            if isinstance(val, types.SimpleNamespace):
                val = vars(val)
            kw_args[f.name] = val

    config = config_class(**kw_args)
    return config
