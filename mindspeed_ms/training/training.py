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
"""For Training"""

import time
import contextlib
import os
import gc
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.communication.comm_func as comm_func
from mindspore.common.api import _pynative_executor

from mindspore import nn, Tensor, Parameter, mint, value_and_grad
from mindspore.train import Perplexity
from mindspore.communication.management import get_rank
from mindspore.amp import DynamicLossScaler, StaticLossScaler, all_finite
from mindspore.experimental.optim.optimizer import Optimizer as mintOptimizer

from mindspeed_ms.tools import logger
from mindspeed_ms.tools.resume_ckpt import get_resume_checkpoint
from mindspeed_ms.training import (
    get_args,
    core_transformer_config_from_args
)
from mindspeed_ms.training.yaml_arguments import core_transformer_config_from_yaml
from mindspeed_ms.core.optimizer import optimizer_config_from_args
from mindspeed_ms.core.transformer.transformer_config import TransformerConfig
from mindspeed_ms.core.parallel_state import (
    get_data_parallel_world_size,
    get_data_parallel_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_pipeline_model_parallel_world_size,
    get_pipeline_model_parallel_group,
    is_pipeline_last_stage,
    set_virtual_pipeline_model_parallel_rank,
    is_pipeline_first_stage,
    get_data_modulo_expert_parallel_group,
    get_expert_model_parallel_world_size,
    get_zero_full_shard_flag,
    get_zero_shard_group,
    get_zero_shard_grad_group,
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
    get_context_parallel_world_size
)
from mindspeed_ms.core.distributed import DistributedDataParallelConfig, \
    DistributedDataParallel
from mindspeed_ms.core.optimizer import MixedPrecisionOptimizer, DistributedOptimizer
from mindspeed_ms.core.pipeline_parallel.schedules import (
    forward_backward_pipelining_without_interleaving,
    forward_backward_pipelining_with_interleaving
)
from mindspeed_ms.core.dist_checkpointing import (
    save_checkpoint,
    load_checkpoint,
    get_last_checkpoint
)
from mindspeed_ms.legacy.model.moe.utils import MoEAuxLossAutoScaler
from mindspeed_ms.core.optimizer import get_optimizer, get_optimizer_param_scheduler
from mindspeed_ms.core.profiler import PynativeProfiler

from .utils import set_weight_decay
from .grad_handler import inplace_apply_to_tensor_list, get_grad_process_func, GradAccumulator


def get_sp_params(config: TransformerConfig):
    """get reduce parameters for sequence parallel"""
    use_lora = config.use_lora
    if use_lora:
        sp_params = [
            'qkv_proj.lora_a',
            'out_proj.lora_b',
            'mapping.lora_a',
            'projection.lora_b',
            'q_proj.lora_a',
            'kv_proj.lora_a',
            'gating.lora_a'
        ]
    else:
        sp_params = ["norm", "mlp.projection.bias", "attention.out_proj.bias",
                     "attention.linear_proj.bias", "linear_fc2.bias"]
    return sp_params


def rename_set_hidden_states_parameter(model, model_chunk_id=None):
    """ rename set_hidden_states parameter """
    weight_untrainable = model.untrainable_params()
    for param in weight_untrainable:
        if "set_hidden_states" in param.name:
            param.name = param.name + f"_{model_chunk_id}_chunk"


def model_zero_grad_buffer(model, wrap_with_ddp):
    """ zero grad buffer if wrap_with_ddp=True """
    if wrap_with_ddp:
        if isinstance(model, nn.CellList):
            for model_chunk_id, _ in enumerate(model):
                model[model_chunk_id].zero_grad_buffer()
        else:
            model.zero_grad_buffer()


class ParallelTrainingReducer:
    """The reducer for parallel training"""

    def __init__(self, params, config: TransformerConfig):
        super(ParallelTrainingReducer, self).__init__()

        args = get_args()
        self.enable_grad_reduce = {
            "dp": False,
            "pp": False,
            "tp": False,  # only valid in the case of sequence parallel
            "ep-dp": False
        }

        self.enable_loss_reduce = {
            "dp": False,
            "pp": False,
            "tp": False,
        }

        self.enable_grad_flag_reduce = {
            "dp": False,
            "pp": False,
            "tp": False,
        }

        self.sp_reduce_params = get_sp_params(config)
        self.expert_params = ["mlp.experts.local_experts"]

        self.batch_reduction = args.loss_reduction
        # dp
        if get_data_parallel_world_size() > 1:
            self.enable_loss_reduce["dp"] = True
            if config.zero_level is None \
                and not args.wrap_with_ddp:
                self.enable_grad_reduce["dp"] = True
            else:
                self.enable_grad_flag_reduce["dp"] = True

        # tp / sp
        if get_tensor_model_parallel_world_size() > 1:
            self.enable_grad_flag_reduce["tp"] = True
            if config.sequence_parallel:
                self.enable_grad_reduce["tp"] = True
                self.sp_reduce_filter = [
                    any([sp_param in param.name for sp_param in self.sp_reduce_params]) for param in params
                ]

        # pp
        if get_pipeline_model_parallel_world_size() > 1:
            self.enable_loss_reduce["dp"] = False
            self.enable_grad_flag_reduce["pp"] = True

        # ep
        if get_expert_model_parallel_world_size() > 1:
            self.enable_grad_reduce["ep-dp"] = True
            self.expert_filter = [
                any([ep_param in param.name for ep_param in self.expert_params]) for param in params
            ]

    def get_reduce_group(self, idx):
        if self.enable_grad_reduce["ep-dp"] and self.expert_filter[idx]:
            group = get_data_modulo_expert_parallel_group()
        else:
            group = get_data_parallel_group()
        return group

    def inplace_reduce_dp_grad(self, grads, params=None):
        """Reduce the gradients in data parallel mode."""
        if self.enable_grad_reduce["dp"]:
            if params is not None:
                for idx, param in enumerate(params):
                    if param.grad is None:
                        continue
                    group = self.get_reduce_group(idx)
                    param.grad = comm_func.all_reduce(param.grad, "sum", group)[0]
                    if self.batch_reduction == "mean":
                        param.grad = mint.div(param.grad, get_data_parallel_world_size())
            else:
                if self.batch_reduction == "mean":
                    for idx, grad in enumerate(grads):
                        group = self.get_reduce_group(idx)
                        grads[idx] = mint.div(
                            comm_func.all_reduce(grad, "sum", group)[0], get_data_parallel_world_size())
                elif self.batch_reduction == "sum":
                    for idx, grad in enumerate(grads):
                        group = self.get_reduce_group(idx)
                        grads[idx] = comm_func.all_reduce(grad, "sum", group)[0]

    def inplace_reduce_sp_grad(self, grads, params=None):
        """Reduce the gradients in sequence parallel mode over tp group."""
        if self.enable_grad_reduce["tp"]:
            if params is not None:
                for idx, param in enumerate(params):
                    if param.grad is None or not self.sp_reduce_filter[idx]:
                        continue
                    param.grad.copy_(comm_func.all_reduce(param.grad, "sum", get_tensor_model_parallel_group())[0])
            else:
                for idx, reduce_flag in enumerate(self.sp_reduce_filter):
                    if reduce_flag:
                        grads[idx] = comm_func.all_reduce(grads[idx], "sum", get_tensor_model_parallel_group())[0]

    def inplace_reduce_grad(self, grads, params=None):
        """Reduce the gradients in all parallel modes."""
        self.inplace_reduce_dp_grad(grads, params)
        self.inplace_reduce_sp_grad(grads, params)

    def reduce_dp_loss(self, loss):
        """Reduce the loss in data parallel mode."""
        if self.enable_loss_reduce["dp"]:
            if self.batch_reduction == "mean":
                loss = mint.div(
                    comm_func.all_reduce(loss, "sum", get_data_parallel_group())[0], get_data_parallel_world_size())
            else:
                loss = comm_func.all_reduce(loss, "sum", get_data_parallel_group())[0]
        return loss

    def reduce_overflow(self, overflow):
        """Reduce the overflow status in all parallel modes."""
        # logical or
        overflow = Tensor(overflow, dtype=mstype.int8)
        if self.enable_grad_flag_reduce["pp"]:
            overflow = comm_func.all_reduce(overflow, "max", get_pipeline_model_parallel_group())[0]
        if self.enable_grad_flag_reduce["dp"]:
            overflow = comm_func.all_reduce(overflow, "max", get_data_parallel_group())[0]
        if self.enable_grad_flag_reduce["tp"]:
            overflow = comm_func.all_reduce(overflow, "max", get_tensor_model_parallel_group())[0]

    def reduce_is_finite(self, is_finite):
        """Reduce the is_finite status in all parallel modes."""
        # logical and
        is_finite = Tensor(is_finite, dtype=mstype.int8)
        if self.enable_grad_flag_reduce["pp"]:
            is_finite = comm_func.all_reduce(is_finite, "prod", get_pipeline_model_parallel_group())[0]
        if self.enable_grad_flag_reduce["dp"]:
            is_finite = comm_func.all_reduce(is_finite, "prod", get_data_parallel_group())[0]
        if self.enable_grad_flag_reduce["tp"]:
            is_finite = comm_func.all_reduce(is_finite, "prod", get_tensor_model_parallel_group())[0]
        return is_finite.astype(mstype.bool_)


def get_model(model_provider_func, config: TransformerConfig):
    """
    Get a network model according to the config.

    Args:
        model_provider_func (Function): A function to get the network model.
        config (TransformerConfig): The configuration object for training the network model.

    Returns:
        nn.Cell, return a pre-configured network model with the config.

    Examples:
        >>> from mindspeed_ms.training import get_model
        >>> from mindspeed_ms.legacy.model.language_model import get_language_model
        >>> from mindspeed_ms.core.config import init_configs_from_yaml
        >>> def model_provider_func(model_config, pre_process=True, post_process=True):
        ...     network_with_loss, _ = get_language_model(config=model_config, num_tokentypes=0,
        ...         add_pooler=False, encoder_attn_mask_type=None, pre_process=pre_process, post_process=post_process)
        ...     return network_with_loss
        >>> config_file = "/path/to/config/file"
        >>> all_config = init_configs_from_yaml(config_file)
        >>> network_with_loss = get_model(model_provider_func, all_config.training_config)
    """
    args = get_args()
    model = nn.CellList(auto_prefix=False)
    if config.bf16 and args.wrap_with_ddp and \
            not args.accumulate_allreduce_grads_in_fp32:
        logger.warning("Using bf16 with ddp, automatically set 'accumulate_allreduce_grads_in_fp32=True'.")
        args.accumulate_allreduce_grads_in_fp32 = True
    if get_pipeline_model_parallel_world_size() > 1:
        if config.virtual_pipeline_model_parallel_size is not None and \
           config.virtual_pipeline_model_parallel_size > 1:
            for i in range(config.virtual_pipeline_model_parallel_size):
                set_virtual_pipeline_model_parallel_rank(i)
                pre_process = is_pipeline_first_stage()
                post_process = is_pipeline_last_stage()
                this_model = model_provider_func(pre_process=pre_process,
                                                 post_process=post_process)
                rename_set_hidden_states_parameter(this_model, i)
                model.append(this_model)
        else:
            pre_process = is_pipeline_first_stage()
            post_process = is_pipeline_last_stage()
            this_model = model_provider_func(pre_process=pre_process,
                                             post_process=post_process)
            # wrap with PP cell if pipeline parallelism is used
            model.append(this_model)
    else:
        this_model = model_provider_func(pre_process=True, post_process=True)
        model.append(this_model)

    if args.wrap_with_ddp:
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=args.accumulate_allreduce_grads_in_fp32,
            overlap_grad_reduce=args.overlap_grad_reduce,
            use_distributed_optimizer=args.use_distributed_optimizer,
            bucket_size=args.ddp_bucket_size,
            average_in_collective=(args.loss_reduction == 'mean'),
            check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad,
            enable_mem_align=args.enable_mem_align,
            use_zero3=(config.zero_level == "z3"),
        )
        logger.info(f"Wrap model with DistributedDataParallel. Config:\n{ddp_config}")
        zero_comm_group = None
        if not get_zero_full_shard_flag():
            zero_comm_group = {"zero_shard_group": get_zero_shard_group(with_context_parallel=True),
                               "zero_shard_grad_group": get_zero_shard_grad_group()}
        model = nn.CellList([DistributedDataParallel(config=config,
                                                     ddp_config=ddp_config,
                                                     module=model_chunck,
                                                     zero_comm_group=zero_comm_group) for model_chunck in model],
                            auto_prefix=False)

    return model


def get_forward_backward_func(network_with_loss, params, config: TransformerConfig):
    """
    Returns a forward-backward function for training a network with or without pipeline parallelism.

    Args:
        network_with_loss (callable): A function that takes inputs and returns the network output and loss.
        params (list): List of parameters to compute gradients for.
        config (TransformerConfig): Model configuration.

    Returns:
        callable: A forward-backward function that can be used for training the network.

    Raises:
        NotImplementedError: If pipeline parallelism is not implemented yet.
    """
    args = get_args()

    forward_backward_func = None
    seq_length = args.seq_length
    dp = get_data_parallel_world_size()
    micro_batch_num = args.global_batch_size // (args.micro_batch_size * dp)
    micro_batch_size = args.micro_batch_size
    data_layout = args.data_layout

    # no pipeline parallel
    if get_pipeline_model_parallel_world_size() == 1:

        def forward_with_loss_scale(*inputs_tuple, loss_scale=None, **inputs_dict):
            logits = None
            output = network_with_loss(*inputs_tuple, **inputs_dict)
            if isinstance(output, tuple):
                loss, logits = output[0], output[1]
            else:
                loss = output
            if loss_scale is not None:
                loss = mint.mul(loss, loss_scale.astype(loss.dtype))
            return loss, logits

        grad_position = None
        # If parallel_config.zero_level == z3, gradient with respect to inputs and weights
        if config.zero_level == "z3":
            grad_position = 0
        forward_backward_once_func = value_and_grad(
            forward_with_loss_scale, grad_position=grad_position, weights=params, has_aux=True
        )

        # if overlap_grad_reduce, grad will be accumulate in grad buffer
        if micro_batch_num > 1 and not args.wrap_with_ddp:
            grad_accumulator = GradAccumulator(micro_batch_num, op="sum")

        def forward_backward_func_with_grad_acc(
                *inputs_tuple, loss_scale=None, forward_only=False, **inputs_dict
        ):
            loss = None
            logits = None
            grads = None

            # reset grad buffer
            model_zero_grad_buffer(network_with_loss, args.wrap_with_ddp)

            # fuse loss scale and grad accumulation if do grad acc
            if args.loss_reduction == "mean" and micro_batch_num > 1:
                if loss_scale is None:
                    loss_scale = Tensor(1, mstype.float32)
                actual_loss_scale = mint.div(loss_scale, micro_batch_num)
            else:
                actual_loss_scale = loss_scale

            if args.wrap_with_ddp:
                no_sync_func = network_with_loss.no_sync
            else:
                no_sync_func = contextlib.nullcontext

            def forward_backward_on_microbatch(idx):
                nonlocal loss
                nonlocal logits
                nonlocal grads

                # slice inputs over batch size dimension
                inputs_tuple_micro = [
                    input_data[idx * micro_batch_size : (idx + 1) * micro_batch_size] for input_data in inputs_tuple
                ]
                inputs_dict_micro = {}
                for key, value in inputs_dict.items():
                    if value is not None:
                        inputs_dict_micro[key] = value[idx * micro_batch_size : (idx + 1) * micro_batch_size]
                    else:
                        inputs_dict_micro[key] = None
                # step on micro batch
                if forward_only:
                    loss_micro, logits_micro = forward_with_loss_scale(
                        *inputs_tuple_micro, loss_scale=actual_loss_scale, **inputs_dict_micro
                    )
                else:
                    (loss_micro, logits_micro), grads_micro = forward_backward_once_func(
                        *inputs_tuple_micro, loss_scale=actual_loss_scale, **inputs_dict_micro
                    )
                    if grad_position == 0:
                        grads_micro = grads_micro[1]
                    # accumulate grads
                    if micro_batch_num > 1 and not args.wrap_with_ddp:
                        grads = grad_accumulator(grads_micro)
                    else:
                        grads = grads_micro

                # process output, loss will be averaged in loss unscaling
                loss = loss_micro if loss is None else loss + loss_micro

                if logits is None:
                    logits = logits_micro
                else:
                    cat_dim = 0 if data_layout == "BSH" else 1
                    logits = mint.cat((logits, logits_micro), dim=cat_dim)

            # trigger dp reduce only on last step
            with no_sync_func():
                for idx in range(micro_batch_num - 1):
                    forward_backward_on_microbatch(idx)
            forward_backward_on_microbatch(micro_batch_num - 1)

            # unscale loss
            if loss_scale is not None:
                loss = mint.div(loss, loss_scale)

            if forward_only:
                return loss, logits

            # finalize ddp grad reduce
            if args.wrap_with_ddp:
                network_with_loss.final_grad_reduce()
            return (loss, logits), grads

        forward_backward_func = forward_backward_func_with_grad_acc

    else:
        def forward_backward_with_pipelining(
                *inputs_tuple, loss_scale=None, forward_only=False, **inputs_dict
        ):
            # reset grad buffer
            model_zero_grad_buffer(network_with_loss, args.wrap_with_ddp)

            if loss_scale is None:
                loss_scale = Tensor(1, mstype.float32)
            if config.virtual_pipeline_model_parallel_size is not None and \
               config.virtual_pipeline_model_parallel_size > 1:
                loss, logits, grads = forward_backward_pipelining_with_interleaving(
                    network_with_loss,
                    micro_batch_num,
                    seq_length,
                    micro_batch_size,
                    *inputs_tuple,
                    decoder_seq_length=None,
                    forward_only=forward_only,
                    collect_non_loss_data=False,
                    first_val_step=None,
                    config=config,
                    total_tokens_nums=None,
                    scale_sense=loss_scale,
                    **inputs_dict
                )
            else:
                loss, logits, grads = forward_backward_pipelining_without_interleaving(
                    network_with_loss,
                    micro_batch_num,
                    seq_length,
                    micro_batch_size,
                    *inputs_tuple,
                    decoder_seq_length=None,
                    forward_only=forward_only,
                    collect_non_loss_data=False,
                    first_val_step=None,
                    config=config,
                    total_tokens_nums=None,
                    scale_sense=loss_scale,
                    **inputs_dict
                )
            if forward_only:
                return loss, logits
            return (loss, logits), grads

        forward_backward_func = forward_backward_with_pipelining

    return forward_backward_func


class TrainOneStepCell(nn.Cell):
    r"""
    TrainOneStepCell with loss scaling, grad clipping, and grad accumulation.

    Args:
        network_with_loss (nn.Cell): The network with loss, output of the network should be loss,
            which is a scalar Tensor.
        optimizer (Optimizer): The optimizer used for training.
        opt_param_scheduler (OptimizerParamScheduler): Learning rate scheduler
        config (TransformerConfig): Transformer Configuration.
        **kwargs: Additional keyword arguments.

    Inputs:
        - **inputs_tuple** (tuple) - Tuple of input tensors, including input_ids, input_positions and attention_mask.
        - **inputs_dict** (dict) - Dict of input tensors.

    Outputs:
        Tuple of 5 Tensor, the loss, overflow flag, current loss scale value, learning rate, and gradients norm.

        - **loss** (Tensor) - A tensor means the train loss value, the shape is :math:`()`.
        - **is_finite** (Tensor) - A bool, indicates whether grads is finite.
        - **loss scale** (Union[Tensor, None]) - The loss scale value, if not using loss scaling, the value is ``None``.
        - **learning rate** (Union[Tensor, list[Tensor]) - The model learning rate.
        - **global_norm** (Tensor) - A tensor means the global norm of the gradients.

    Examples:
        >>> from mindspeed_ms.core.optimizer import get_optimizer
        >>> from mindspeed_ms.training import get_model, TrainOneStepCell
        >>> from mindspeed_ms.legacy.model.language_model import get_language_model
        >>> from mindspeed_ms.training.utils import set_weight_decay
        >>> from mindspeed_ms.core.config import init_configs_from_yaml
        >>> def model_provider_func(config, pre_process=True, post_process=True):
        ...     network_with_loss, _ = get_language_model(config=config, num_tokentypes=0,
        ...         add_pooler=False, encoder_attn_mask_type=None, pre_process=pre_process, post_process=post_process)
        ...     return network_with_loss
        >>> config_file = "/path/to/config/file"
        >>> all_config = init_configs_from_yaml(config_file)
        >>> training_config = all_config.training_config
        >>> optimizer_config = all_config.optimizer_config
        >>> config = all_config.config
        >>> network_with_loss = get_model(model_provider_func, training_config)
        >>> group_params = set_weight_decay(network_with_loss.trainable_params(), optimizer_config.weight_decay)
        >>> optimizer = get_optimizer(optimizer_config, training_config, group_params, network_with_loss,
        >>>                           grad_allreduce_op=training_config.loss_reduction)
        >>> train_one_step_cell = TrainOneStepCell(network_with_loss, optimizer, None, config)
    """

    # pylint: disable=W0613
    def __init__(self, network_with_loss, optimizer, opt_param_scheduler, config: TransformerConfig, **kwargs):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        args = get_args()
        if isinstance(network_with_loss, nn.CellList) and len(network_with_loss) == 1:
            network_with_loss = network_with_loss[0]
        self.network_with_loss = network_with_loss
        self.optimizer = optimizer
        self.config = config
        self.opt_param_scheduler = opt_param_scheduler
        self.increment = args.global_batch_size
        self.wrap_with_ddp = args.wrap_with_ddp
        self.use_mixed_precision_optimizer = isinstance(optimizer, MixedPrecisionOptimizer)
        if isinstance(optimizer, DistributedOptimizer) and args.overlap_param_gather:
            optimizer.enable_pre_hook(network_with_loss)

        if hasattr(optimizer, "parameters"):
            parameters = optimizer.parameters
        else:
            logger.warning(
                "Fail to get parameters from optimizer and will get parameters "
                "from network_with_loss.trainable_params() alternatively. "
                "But this may cause exception due to a mismatch between parameters and gradients."
            )
            parameters = network_with_loss.trainable_params()

        self.params_with_grad = parameters if self.use_mixed_precision_optimizer else None

        # init loss scaler
        if args.loss_scale is not None:
            self.loss_scaler = StaticLossScaler(scale_value=args.loss_scale)
        else:
            # dynamic loss scaler is used only if the model is computed in float16
            if config.compute_dtype == mstype.float16:
                self.loss_scaler = DynamicLossScaler(
                    scale_value=args.initial_loss_scale,
                    scale_factor=args.hysteresis,
                    scale_window=args.loss_scale_window,
                )
            else:
                logger.warning(
                    "Dynamic loss scale is only supported for float16 computation. Not using loss scaling."
                )
                self.loss_scaler = StaticLossScaler(scale_value=1)

        # init grad clip func
        self.use_grad_clip = config.grad_clip_kwargs is not None
        if self.use_grad_clip:
            self.grad_clip_func = get_grad_process_func(
                config, not args.untie_embeddings_and_output_weights,
                params=parameters
            )
        # init grad scale func
        self.grad_scale_func = inplace_apply_to_tensor_list(mint.mul)

        # init parallel reducer
        self.parallel_reducer = ParallelTrainingReducer(parameters, config)

        dp = get_data_parallel_world_size()
        self.micro_batch_num = args.global_batch_size // (args.micro_batch_size * dp)
        # init forward_backward_func
        self.forward_backward_func = get_forward_backward_func(
            network_with_loss, parameters, config
        )
        self.accumulate_allreduce_grads_in_fp32 = args.accumulate_allreduce_grads_in_fp32


    def unscale_and_clip_grads(self, grads, loss_scale=None):
        """Handle grads with scaling and clipping.

        Args:
            grads (tuple): The gradients.
            loss_scale (Tensor, optional): The scaling factor of loss. Defaults: None.
        """
        if loss_scale is not None:
            inv_scale = mint.reciprocal(loss_scale).astype(grads[0].dtype)
            self.grad_scale_func(grads, inv_scale)
        global_norm = None
        if self.use_grad_clip:
            global_norm = self.grad_clip_func(grads)
        return global_norm

    def construct(self, *inputs_tuple, **inputs_dict):
        """Forward, backward, grad process, and optimizer step."""
        # forward and backward
        if self.use_mixed_precision_optimizer:
            self.optimizer.zero_grad()

        if self.loss_scaler is not None:
            current_step_loss_scale = self.loss_scaler.scale_value
        else:
            current_step_loss_scale = None
        if self.config.num_moe_experts is not None and self.config.num_moe_experts > 1:
            MoEAuxLossAutoScaler.set_loss_scale(mint.div(current_step_loss_scale, self.micro_batch_num))

        # loss is scale and unscale in forward_backward_func
        (loss, _), grads = self.forward_backward_func(*inputs_tuple, loss_scale=current_step_loss_scale, **inputs_dict)

        # apply grad reducer
        grads = list(grads)
        if not self.use_mixed_precision_optimizer and not self.wrap_with_ddp \
            and self.accumulate_allreduce_grads_in_fp32:
            grads = [grad.to(mstype.float32) for grad in grads]
        self.parallel_reducer.inplace_reduce_grad(grads, self.params_with_grad)

        # check overflow. When using mixed precision optimizer,
        # this process will be done in optimizer
        is_finite = True
        if not self.use_mixed_precision_optimizer and not self.wrap_with_ddp:
            is_finite = all_finite(grads)
            # sync over tp and pp group
            is_finite = self.parallel_reducer.reduce_is_finite(is_finite)

            if self.loss_scaler is not None:
                self.loss_scaler.adjust(is_finite)

        global_norm = None
        if is_finite:
            global_norm = self._call_optimizer(grads, current_step_loss_scale)

        # Update learning rate.
        if self.opt_param_scheduler:
            self.opt_param_scheduler.step(increment=self.increment)
        if isinstance(self.optimizer, mintOptimizer):
            learning_rate = self.optimizer.lrs
        else:
            learning_rate = self.optimizer.get_lr()
        if isinstance(learning_rate, (Parameter, Tensor)):
            learning_rate = float(learning_rate.value())
        if isinstance(learning_rate, (tuple, list)):
            learning_rate = tuple(
                float(individual_learning_rate.value())
                if isinstance(individual_learning_rate, (Parameter, Tensor))
                else individual_learning_rate
                for individual_learning_rate in learning_rate
            )

        # reduce loss if dp
        loss = self.parallel_reducer.reduce_dp_loss(loss)

        return loss, is_finite, current_step_loss_scale, learning_rate, global_norm

    def _call_optimizer(self, grads, current_step_loss_scale):
        # scale grads and clip grads if enabled
        global_norm = None
        if not self.use_mixed_precision_optimizer:
            global_norm = self.unscale_and_clip_grads(grads, current_step_loss_scale)
            grads_tuple = tuple(grads)
            self.optimizer(grads_tuple)
        else:
            _, global_norm, _ = self.optimizer()
        return global_norm

def train(
        train_one_step_cell,
        train_dataloader,
        val_dataloader=None,
        metrics=None,
        evaluation_func=None,
        resume_dict=None,
        get_batch_func=None,
        **kwargs,
):
    """
    Train the model using the provided training configuration.

    Note:
        Before using this function, users need to define the `get_dataset` function first which generates training and
        validation sets.

    Args:
        train_one_step_cell (TrainOneStepCell): The training cell object.
        train_dataloader (Dataset): The iterator for the training dataset.
        val_dataloader (Dataset): The iterator for the validation dataset. Defaults: ``None``.
        metrics (dict[str, Metric], optional): A dictionary of metrics to track during training. Defaults: ``None``.
        evaluation_func (Function, optional): The evaluation function to use for validation. Defaults: ``None``.
        resume_dict (dict): Resume training parameters. Defaults: ``None``.
        get_batch_func (Function): A function to get the batch size in the dataset_config. Defaults: ``None``.
        **kwargs: Additional keyword arguments.

    Raises:
        ValueError: If `get_batch_func` is ``None`` and `train_dataloader` is ``None``.

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.
            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> from mindspeed_ms.core.optimizer import get_optimizer
        >>> from mindspeed_ms.training import get_model, TrainOneStepCell, train
        >>> from mindspeed_ms.legacy.model.language_model import get_language_model
        >>> from mindspeed_ms.training.utils import set_weight_decay
        >>> from mindspeed_ms.core.config import init_configs_from_yaml
        >>> def model_provider_func(model_config, pre_process=True, post_process=True):
        ...     network_with_loss, _ = get_language_model(config=model_config, num_tokentypes=0, add_pooler=False,
        ...         encoder_attn_mask_type=None, pre_process=pre_process, post_process=post_process)
        ...     return network_with_loss
        >>> config_file = "/path/to/config/file"
        >>> all_config = init_configs_from_yaml(config_file)
        >>> training_config = all_config.training_config
        >>> optimizer_config = all_config.optimizer_config
        >>> model_config = all_config.model_config
        >>> network_with_loss = get_model(model_provider_func, training_config)
        >>> group_params = set_weight_decay(network_with_loss.trainable_params(), optimizer_config.weight_decay)
        >>> optimizer = get_optimizer(optimizer_config, training_config, group_params, network_with_loss,
        >>>                           grad_allreduce_op=training_config.loss_reduction)
        >>> train_one_step_cell = TrainOneStepCell(network_with_loss, optimizer, None, training_config, model_config)
        >>> train_dataset_iterator, val_dataset_iterator = get_dataset(all_config.dataset_config)
        >>> train(train_one_step_cell=train_one_step_cell, train_dataloader=train_dataloader,
        >>>       training_config=training_config)
    """
    args = get_args()
    if args.resume_training and resume_dict is not None:
        initial_epoch = resume_dict.get("epoch_num")
        initial_step = resume_dict.get("step_num")
    else:
        initial_epoch = 0
        initial_step = 0

    dataset_size = None
    # broadcast only support [Int32], datasize limit [-2147483648, 2147483647]
    dataset_size_tensor = ms.Tensor(np.zeros(1), dtype=mstype.int32)
    # There is 2 way to fetch traing step data: 1. get_batch_func; 2. train_dataloader
    # both way need to know dataset size
    if get_batch_func is None and train_dataloader is None:
        # A ERROR SITUATION
        raise ValueError(f"`get_batch_func` and `train_dataloader` should not be `None` at the same time, " + \
                         f"but got {get_batch_func} and {train_dataloader}")
    if train_dataloader is not None:
        dataset_size = train_dataloader.get_dataset_size()
        dataset_size_tensor = ms.Tensor(dataset_size, dtype=mstype.int32)
    if get_batch_func is not None:
        # if using `get_batch_func`, train_dataloader is None when tp_rank != 0,
        # so we need to broadcast it to other tp_rank
        tp_world_size = get_tensor_model_parallel_world_size()
        src_rank = (get_rank() // tp_world_size) * tp_world_size
        dataset_size_tensor = comm_func.broadcast(dataset_size_tensor, src_rank, get_tensor_model_parallel_group())
        comm_func.barrier(group=get_tensor_model_parallel_group())
        _pynative_executor.sync()
    dataset_size = dataset_size_tensor.asnumpy().tolist()
    if isinstance(dataset_size, list):
        dataset_size = dataset_size[0]
    if dataset_size <= 0:
        raise ValueError(f"Expect dataset_size > 0, but got {dataset_size}")

    logger.info(f"dataset size is {dataset_size}")

    train_one_step_cell.set_train()
    config = train_one_step_cell.config

    global_step = 1
    epoch_step = 1
    current_epoch = 0
    if args.resume_training:
        global_step = initial_step + initial_epoch * dataset_size + 1
        epoch_step = global_step % dataset_size
        current_epoch = global_step // dataset_size
        if epoch_step == 0 and current_epoch > 0:
            epoch_step = dataset_size
            current_epoch -= 1
        logger.info(f"Resume training starts from global step {global_step}, epoch {current_epoch}, step {epoch_step}.")

    if epoch_step > 1:
        logger.info(f"Resume training will skip {epoch_step - 1} step data")

    evaluation_flag = (
        val_dataloader is not None
        and evaluation_func is not None
        and metrics is not None
        and args.eval_interval is not None
        and args.best_metric_comparison is not None
        and args.eval_metric is not None
    )
    save_ckpt_flag = (args.save_interval is not None and
                      args.train_iters != 0 and
                      args.save_interval <= args.train_iters)
    correct_metric_flag = is_pipeline_last_stage() # not use pp or pp last_stage

    if evaluation_flag:
        if args.best_metric_comparison == "less_equal":
            best_metric_compare_func = mint.less_equal
            best_metric = Tensor(float("inf"))
        elif args.best_metric_comparison == "greater_equal":
            best_metric_compare_func = mint.greater_equal
            best_metric = Tensor(float("-inf"))
        elif args.best_metric_comparison == "less":
            best_metric_compare_func = mint.less
            best_metric = Tensor(float("inf"))
        elif args.best_metric_comparison == "greater":
            best_metric_compare_func = mint.greater
            best_metric = Tensor(float("-inf"))
    profiler = PynativeProfiler()

    # both `get_batch_func` and `train_dataloader` need create train_dataloader
    if train_dataloader is not None:
        if args.resume_training:
            if not args.new_dataset:
                # when epoch_step > 1, means resume traing mode, will skip some data.
                train_data_dict_iterator = train_dataloader.skip(epoch_step - 1).create_dict_iterator()
            else:
                logger.warning(f"When `resume_training = True` and `new_dataset = True`, will use a new dataset.")
                train_data_dict_iterator = train_dataloader.create_dict_iterator()
        else:
            train_data_dict_iterator = train_dataloader.create_dict_iterator()
    else:
        train_data_dict_iterator = None

    # training loop
    while not (
            args.epochs is not None
            and current_epoch >= args.epochs
            or global_step > args.train_iters
    ):
        # we need to refresh train_dataloader every epoch
        # when resume training, epoch_step > 1, so train_data_dict_iterator will not be refreshed
        if epoch_step == 1 and train_dataloader is not None:
            train_data_dict_iterator = train_dataloader.create_dict_iterator(num_epochs=1)

        # get step data
        if get_batch_func is None:
            data = next(train_data_dict_iterator)
        else:
            data = get_batch_func(train_data_dict_iterator)
        logger.debug(f"step {global_step} input data are:\n{data}")
        start_time = time.time()
        profiler.step_begin(global_step)
        loss, is_finite, loss_scale, learning_rate, global_norm = train_one_step_cell(**data)
        cp_world_size = get_context_parallel_world_size()
        loss = loss / cp_world_size
        end_time = time.time()
        if args.log_interval is not None and global_step % args.log_interval == 0:
            if not correct_metric_flag:
                logger.warning("Metrics is only calculated on the last stage.")
            if isinstance(learning_rate, (tuple, list)):
                report_learning_rate = '('
                for lr in learning_rate:
                    report_learning_rate += "{:e},".format(lr)
                report_learning_rate += ')'
            else:
                report_learning_rate = "{:e}".format(learning_rate)
            logger.info(
                f"Epoch: {current_epoch}, Step: {epoch_step}, Loss: {loss}, "
                + f"Finite_grads: {is_finite}, "
                + f"Loss_scale: {loss_scale.value() if loss_scale is not None else None}, "
                + f"Learning_rate: {report_learning_rate}, Grad_norm: {global_norm}, "
                + f"Time: {(end_time - start_time) * 1000:.2f} ms"
            )

        if evaluation_flag and global_step % args.eval_interval == 0:
            is_best = Tensor(False, dtype=mstype.int8)
            results = evaluation_func(train_one_step_cell, val_dataloader, metrics, **kwargs)

            # update best_metrics only on last stage
            if correct_metric_flag and best_metric_compare_func(results[args.eval_metric], best_metric):
                best_metric = results[args.eval_metric]
                is_best = Tensor(True, dtype=mstype.int8)

            if get_pipeline_model_parallel_world_size() > 1:
                is_best = comm_func.all_reduce(is_best, "max", get_pipeline_model_parallel_group())[0]

            # save ckpt
            if is_best and save_ckpt_flag:
                logger.warning("saving best checkpoint")
                if save_ckpt_flag:
                    save_checkpoint(config,
                                    train_one_step_cell.network_with_loss,
                                    train_one_step_cell.optimizer,
                                    train_one_step_cell.opt_param_scheduler,
                                    args.save,
                                    format=args.dist_ckpt_format,
                                    prefix=args.prefix + "_best",
                                    epoch_num=current_epoch,
                                    step_num=epoch_step,
                                    crc_check=args.crc_check,
                                    keep_checkpoint_max=args.keep_checkpoint_max + 1)

        if save_ckpt_flag and global_step % args.save_interval == 0:
            save_checkpoint(config,
                            train_one_step_cell.network_with_loss,
                            train_one_step_cell.optimizer,
                            train_one_step_cell.opt_param_scheduler,
                            args.save,
                            format=args.dist_ckpt_format,
                            prefix=args.prefix,
                            epoch_num=current_epoch,
                            step_num=epoch_step,
                            crc_check=args.crc_check,
                            keep_checkpoint_max=args.keep_checkpoint_max)
            gc.collect()
        profiler.step_end(global_step)
        epoch_step += 1
        global_step += 1

        # update epoch_step and current_epoch
        if epoch_step > dataset_size:
            epoch_step = 1
            current_epoch += 1

    if isinstance(train_one_step_cell.optimizer, DistributedOptimizer) \
            and args.overlap_param_gather:
        train_one_step_cell.optimizer.sync_gather_all_model_params(True)

    if save_ckpt_flag:
        logger.info("Saving last step checkpoint.")
        # at the end of training loop, we use `global_step += 1`,
        # so the right global step should be 'global_step - 1',
        epoch_step = (global_step - 1) % dataset_size
        current_epoch = (global_step - 1) // dataset_size
        # to avoid situation like 'epoch 1, step 0'
        if epoch_step == 0:
            epoch_step = dataset_size
            current_epoch -= 1

        save_checkpoint(config,
                        train_one_step_cell.network_with_loss,
                        train_one_step_cell.optimizer,
                        train_one_step_cell.opt_param_scheduler,
                        args.save,
                        format=args.dist_ckpt_format,
                        prefix=args.prefix,
                        epoch_num=current_epoch,
                        step_num=epoch_step,
                        crc_check=args.crc_check,
                        keep_checkpoint_max=args.keep_checkpoint_max)
        gc.collect()
    logger.info("Training success!")


# pylint: disable=W0613, C0330
def pretrain(train_valid_test_datasets_provider,
             model_provider_func,
             model_type,
             forward_step_func=None,
             process_non_loss_data_func=None,
             **kwargs):
    """pretrain function"""

    args = get_args()
    if args.yaml_cfg is None:
        config = core_transformer_config_from_args(args)
    else:
        config = core_transformer_config_from_yaml(args)
    optimizer_config = optimizer_config_from_args(args)

    train_data_loader = kwargs.get("train_data_loader", None)
    get_batch_func = kwargs.get("get_batch_func", None)

    network_with_loss = get_model(model_provider_func, config)

    group_params = set_weight_decay(network_with_loss.trainable_params(), optimizer_config.weight_decay)
    optimizer = get_optimizer(
        optimizer_config,
        config,
        group_params,
        network_with_loss,
        grad_allreduce_op=optimizer_config.loss_reduction
    )
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    resume_dict = None
    if args.resume_training is True and \
       args.load is not None and \
       os.path.exists(args.load):

        rank_path = os.path.join(args.load, f"rank_{get_rank()}")
        if os.path.exists(rank_path):
            meta_path = os.path.join(rank_path, "meta.json")
            resume_by_meta = True
            if not os.path.exists(meta_path):
                logger.warning(f"Could not find meta.json in directory {rank_path}, using latest ckpt in {rank_path}")
                resume_by_meta = False
            resume_ckpt_name = get_resume_checkpoint(
                checkpoint_dir=args.load,
                resume_training=args.resume_training,
                resume_by_meta=resume_by_meta
                )
            logger.debug(f"resume_ckpt_name is {resume_ckpt_name}")
            if resume_ckpt_name is True:
                ckpt_path = args.load
            elif isinstance(resume_ckpt_name, str):
                ckpt_path = os.path.join(rank_path, resume_ckpt_name)
        else:
            pp_rank = get_pipeline_model_parallel_rank()
            dp_size = get_data_parallel_world_size()
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()
            local_rank_to_dp0_rank = pp_rank * dp_size * tp_size + tp_rank
            logger.warning(f"global rank_{get_rank()} ckpt not found, will load rank_{local_rank_to_dp0_rank} ckpt.")
            rank_path = os.path.join(args.load, f"rank_{local_rank_to_dp0_rank}")
            if not os.path.exists(rank_path):
                raise FileNotFoundError(f"Path {rank_path} not exists, please check your ckpt path.")
            ckpt_path = get_last_checkpoint(rank_path)
            if not ckpt_path or not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"File {ckpt_path} not exists, please check your ckpt path.")
        logger.debug(f"ckpt_path is {ckpt_path}")
        resume_dict = load_checkpoint(
            config=config,
            model=network_with_loss,
            optimizer=optimizer if ((not args.no_load_optim)
                                    or optimizer_config.use_distributed_optimizer) else None,
            opt_param_scheduler=opt_param_scheduler,
            ckpt_path=ckpt_path,
            format=args.dist_ckpt_format
            )
        logger.info(f"Checkpoint has trained {resume_dict.get('epoch_num', 0)} epochs, " + \
                    f"{resume_dict.get('step_num', 0)} steps.")
    train_one_step_cell = TrainOneStepCell(network_with_loss, optimizer, opt_param_scheduler,
                                           config)

    metrics = {
        "perplexity": Perplexity(),
    }
    train(
        train_one_step_cell=train_one_step_cell,
        train_dataloader=train_data_loader,
        val_dataloader=None,
        metrics=metrics,
        resume_dict=resume_dict,
        get_batch_func=get_batch_func,
        loss_func_type=optimizer_config.loss_func_kwargs.loss_func_type,
    )
