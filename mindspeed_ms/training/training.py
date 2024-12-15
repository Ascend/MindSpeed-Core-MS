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

import math
import time
import os
import gc
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.communication.comm_func as comm_func
from mindspore.common.api import _pynative_executor

from mindspore import nn, Tensor, Parameter, mint
from mindspore.communication.management import get_rank
from mindspore.amp import DynamicLossScaler, StaticLossScaler, all_finite
from mindspore.experimental.optim.optimizer import Optimizer as mintOptimizer

from mindspeed_ms.tools import logger
from mindspeed_ms.tools.resume_ckpt import get_resume_checkpoint
from mindspeed_ms.training import get_args
from mindspeed_ms.training.utils import set_weight_decay
from mindspeed_ms.core.optimizer import optimizer_config_from_args
from mindspeed_ms.core.transformer.transformer_config import TransformerConfig
import mindspeed_ms.core.parallel_state as mpu
from mindspeed_ms.core.distributed import DistributedDataParallelConfig, \
    DistributedDataParallel
from mindspeed_ms.core.optimizer import MixedPrecisionOptimizer, DistributedOptimizer
from mindspeed_ms.core.pipeline_parallel.schedules import (
    get_forward_backward_func
)
from mindspeed_ms.core.dist_checkpointing import (
    save_checkpoint,
    load_checkpoint,
    get_last_checkpoint
)
from mindspeed_ms.legacy.model.moe.utils import MoEAuxLossAutoScaler
from mindspeed_ms.core.optimizer import get_optimizer, get_optimizer_param_scheduler
from mindspeed_ms.core.profiler import PynativeProfiler

from mindspeed_ms.training.initialize import initialize_mindspeed_ms
from mindspeed_ms.core.utils import get_model_config

from .grad_handler import inplace_apply_to_tensor_list, get_grad_process_func
from .utils import print_rank_last, print_rank_0, is_last_rank
from .global_vars import get_timers, get_tensorboard_writer, get_wandb_writer

# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()


def get_train_valid_test_num_samples():
    """Train/valid/test num samples."""

    args = get_args()

    eval_iters = 0
    test_iters = 0
    # Number of train/valid/test samples.
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    if args.eval_interval is not None and args.eval_iters is not None:
        eval_iters = (args.train_iters // args.eval_interval + 1) * \
                    args.eval_iters
        test_iters = args.eval_iters

    return (
        train_samples,
        eval_iters * args.global_batch_size,
        test_iters * args.global_batch_size,
    )


def get_dataset_dict_iterator(train_dataloader, epoch_step):
    """ get dataset dict iterator """
    args = get_args()
    if args.resume_training:
        if not args.new_dataset:
            # when epoch_step > 1, means resume traing mode, will skip some data.
            factor = args.global_batch_size // mpu.get_data_parallel_world_size() // args.micro_batch_size
            train_data_dict_iterator = train_dataloader.skip((epoch_step - 1) * factor)\
                .create_dict_iterator(num_epochs=1)
        else:
            logger.warning(f"When `resume_training = True` and `new_dataset = True`, will use a new dataset.")
            train_data_dict_iterator = train_dataloader.create_dict_iterator(num_epochs=1)
    else:
        train_data_dict_iterator = train_dataloader.create_dict_iterator(num_epochs=1)
    return train_data_dict_iterator


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
                     "attention.linear_proj.bias", "mlp.linear_fc2.bias"]
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

        # dp
        if mpu.get_data_parallel_world_size() > 1:
            self.enable_loss_reduce["dp"] = True
            if config.zero_level is None \
                and not args.wrap_with_ddp:
                self.enable_grad_reduce["dp"] = True
            else:
                self.enable_grad_flag_reduce["dp"] = True

        # tp / sp
        if mpu.get_tensor_model_parallel_world_size() > 1:
            self.enable_grad_flag_reduce["tp"] = True
            if config.sequence_parallel:
                self.enable_grad_reduce["tp"] = True
                self.sp_reduce_filter = [
                    any([sp_param in param.name for sp_param in self.sp_reduce_params]) for param in params
                ]

        # pp
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            self.enable_loss_reduce["dp"] = False
            self.enable_grad_flag_reduce["pp"] = True

        # ep
        if mpu.get_expert_model_parallel_world_size() > 1:
            self.enable_grad_reduce["ep-dp"] = True
            self.expert_filter = [
                any([ep_param in param.name for ep_param in self.expert_params]) for param in params
            ]

    def get_reduce_group(self, idx):
        if self.enable_grad_reduce["ep-dp"] and self.expert_filter[idx]:
            group = mpu.get_data_modulo_expert_parallel_group()
        else:
            group = mpu.get_data_parallel_group()
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
                    param.grad = mint.div(param.grad, mpu.get_data_parallel_world_size())
            else:
                for idx, grad in enumerate(grads):
                    group = self.get_reduce_group(idx)
                    grads[idx] = mint.div(
                        comm_func.all_reduce(grad, "sum", group)[0], mpu.get_data_parallel_world_size())

    def inplace_reduce_sp_grad(self, grads, params=None):
        """Reduce the gradients in sequence parallel mode over tp group."""
        if self.enable_grad_reduce["tp"]:
            if params is not None:
                for idx, param in enumerate(params):
                    if param.grad is None or not self.sp_reduce_filter[idx]:
                        continue
                    param.grad.copy_(comm_func.all_reduce(param.grad,
                                                          "sum",
                                                          mpu.get_tensor_model_parallel_group())[0])
            else:
                for idx, reduce_flag in enumerate(self.sp_reduce_filter):
                    if reduce_flag:
                        grads[idx] = comm_func.all_reduce(grads[idx],
                                                          "sum",
                                                          mpu.get_tensor_model_parallel_group())[0]

    def inplace_reduce_grad(self, grads, params=None):
        """Reduce the gradients in all parallel modes."""
        self.inplace_reduce_dp_grad(grads, params)
        self.inplace_reduce_sp_grad(grads, params)

    def reduce_overflow(self, overflow):
        """Reduce the overflow status in all parallel modes."""
        # logical or
        overflow = Tensor(overflow, dtype=mstype.int8)
        if self.enable_grad_flag_reduce["pp"]:
            overflow = comm_func.all_reduce(overflow, "max", mpu.get_pipeline_model_parallel_group())[0]
        if self.enable_grad_flag_reduce["dp"]:
            overflow = comm_func.all_reduce(overflow, "max", mpu.get_data_parallel_group())[0]
        if self.enable_grad_flag_reduce["tp"]:
            overflow = comm_func.all_reduce(overflow, "max", mpu.get_tensor_model_parallel_group())[0]

    def reduce_is_finite(self, is_finite):
        """Reduce the is_finite status in all parallel modes."""
        # logical and
        is_finite = Tensor(is_finite, dtype=mstype.int8)
        if self.enable_grad_flag_reduce["pp"]:
            is_finite = comm_func.all_reduce(is_finite, "prod", mpu.get_pipeline_model_parallel_group())[0]
        if self.enable_grad_flag_reduce["dp"]:
            is_finite = comm_func.all_reduce(is_finite, "prod", mpu.get_data_parallel_group())[0]
        if self.enable_grad_flag_reduce["tp"]:
            is_finite = comm_func.all_reduce(is_finite, "prod", mpu.get_tensor_model_parallel_group())[0]
        return is_finite.astype(mstype.bool_)


def get_model(model_provider_func, model_type, wrap_with_ddp=True):
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
    args.model_type = model_type
    model = nn.CellList(auto_prefix=False)

    if mpu.get_pipeline_model_parallel_world_size() > 1:
        if args.virtual_pipeline_model_parallel_size is not None:
            for i in range(args.virtual_pipeline_model_parallel_size):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                pre_process = mpu.is_pipeline_first_stage()
                post_process = mpu.is_pipeline_last_stage()
                this_model = model_provider_func(pre_process=pre_process,
                                                 post_process=post_process)
                this_model.model_type = model_type
                rename_set_hidden_states_parameter(this_model, i)
                model.append(this_model)
        else:
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(pre_process=pre_process,
                                             post_process=post_process)
            this_model.model_type = model_type
            # wrap with PP cell if pipeline parallelism is used
            model.append(this_model)
    else:
        this_model = model_provider_func(pre_process=True, post_process=True)
        this_model.model_type = model_type
        model.append(this_model)

    # print number of parameters
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
                  mpu.get_tensor_model_parallel_rank(),
                  mpu.get_pipeline_model_parallel_rank(),
                  sum([sum([p.nelement() for p in module.get_parameters()])
                       for module in model])), flush=True)

    if wrap_with_ddp:
        config = get_model_config(model[0])
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=args.accumulate_allreduce_grads_in_fp32,
            overlap_grad_reduce=args.overlap_grad_reduce,
            use_distributed_optimizer=args.use_distributed_optimizer,
            bucket_size=args.ddp_bucket_size,
            average_in_collective=args.ddp_average_in_collective,
            check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad,
            enable_mem_align=args.enable_mem_align,
            use_zero3=(config.zero_level == "z3"),
        )
        logger.info(f"Wrap model with DistributedDataParallel. Config:\n{ddp_config}")
        zero_comm_group = None
        if not mpu.get_zero_full_shard_flag():
            zero_comm_group = {"zero_shard_group": mpu.get_zero_shard_group(with_context_parallel=True),
                               "zero_shard_grad_group": mpu.get_zero_shard_grad_group()}
        model = nn.CellList([DistributedDataParallel(config=config,
                                                     ddp_config=ddp_config,
                                                     module=model_chunck,
                                                     zero_comm_group=zero_comm_group) for model_chunck in model],
                            auto_prefix=False)

    return model


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
    def __init__(self,
                 network_with_loss,
                 optimizer,
                 opt_param_scheduler,
                 config: TransformerConfig,
                 **kwargs):
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
        loss_scaler = None
        if args.loss_scale is not None:
            loss_scaler = StaticLossScaler(scale_value=args.loss_scale)
        else:
            # dynamic loss scaler is used only if the model is computed in float16
            if config.compute_dtype == mstype.float16:
                loss_scaler = DynamicLossScaler(
                    scale_value=args.initial_loss_scale,
                    scale_factor=args.hysteresis,
                    scale_window=args.loss_scale_window,
                )
            else:
                logger.warning(
                    "Dynamic loss scale is only supported for float16 computation. Not using loss scaling."
                )
                loss_scaler = StaticLossScaler(scale_value=1)
        self.config.loss_scaler = loss_scaler

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

        # get args value
        dp = mpu.get_data_parallel_world_size()
        self.micro_batch_size = args.micro_batch_size
        self.seq_length = args.seq_length

        # calculate num_microbatches
        self.num_microbatches = args.global_batch_size // (self.micro_batch_size * dp)
        # init forward_backward_func
        self.forward_backward_func = get_forward_backward_func()
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

    def construct(self, forward_step_func, data_iterator, forward_only=False):
        """Forward, backward, grad process, and optimizer step."""
        # forward and backward
        if self.use_mixed_precision_optimizer:
            self.optimizer.zero_grad()

        if self.config.loss_scaler is not None:
            current_step_loss_scale = self.config.loss_scaler.scale_value
        else:
            current_step_loss_scale = None
        if self.config.num_moe_experts is not None and self.config.num_moe_experts > 1:
            MoEAuxLossAutoScaler.set_loss_scale(mint.div(current_step_loss_scale, self.num_microbatches))

        # reset grad buffer
        model_zero_grad_buffer(self.network_with_loss, self.wrap_with_ddp)

        # loss is scale and unscale in forward_backward_func
        losses_reduced, grads = self.forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=self.network_with_loss,
            num_microbatches=self.num_microbatches,
            seq_length=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            decoder_seq_length=None,
            forward_only=forward_only,
            collect_non_loss_data=False,
            first_val_step=None
        )

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

            if self.config.loss_scaler is not None:
                self.config.loss_scaler.adjust(is_finite)

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

        # reduce loss
        loss_reduced = self._loss_reduce(losses_reduced)

        return loss_reduced, is_finite, current_step_loss_scale, learning_rate, global_norm

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

    def _loss_reduce(self, losses_reduced):
        """ reduce losses """
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            loss_reduced = {}
            for key in losses_reduced[0].keys():
                numerator = 0
                denominator = 0
                for x in losses_reduced:
                    val = x[key]
                    # reduce loss base on per-token
                    if isinstance(val, (tuple, list)):
                        numerator += val[0]
                        denominator += val[1]
                    else:
                        # reduce loss base on num_microbatches
                        numerator += val
                        denominator += 1
                loss_reduced[key] = numerator / denominator
        else:
            loss_reduced = {'lm loss': Tensor(0, mstype.float32)}
        return loss_reduced

def train(
        forward_step_func,
        model,
        optimizer,
        opt_param_scheduler,
        train_data_iterator,
        valid_data_iterator,
        process_non_loss_data_func,
        config,
        metrics=None,
        evaluation_func=None,
        resume_dict=None,
        **kwargs
):
    """
    Train the model using the provided training configuration.

    Note:
        Before using this function, users need to define the `get_dataset` function first which generates training and
        validation sets.

    Args:
        train_one_step_cell (TrainOneStepCell): The training cell object.
        train_dataloader (Dataset): The iterator for the training dataset.
        valid_data_iterator (Dataset): The iterator for the validation dataset. Defaults: ``None``.
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
    if process_non_loss_data_func is not None:
        raise NotImplementedError("process_non_loss_data_func is not supported for now.")

    args = get_args()
    if args.resume_training and resume_dict is not None:
        initial_epoch = resume_dict.get("epoch_num")
        initial_step = resume_dict.get("step_num")
    else:
        initial_epoch = 0
        initial_step = 0

    step_num_per_epoch = None
    # broadcast only support [Int32], datasize limit [-2147483648, 2147483647]
    step_num_per_epoch_tensor = ms.Tensor(np.zeros(1), dtype=mstype.int32)

    if train_data_iterator is not None:
        train_dataloader_sample = train_data_iterator
        if isinstance(train_dataloader_sample, list):
            train_dataloader_sample = train_dataloader_sample[0]
        factor = args.global_batch_size // mpu.get_data_parallel_world_size() // args.micro_batch_size
        step_num_per_epoch = train_dataloader_sample.get_dataset_size() // factor
        step_num_per_epoch_tensor = ms.Tensor(step_num_per_epoch, dtype=mstype.int32)

    # if using `get_batch_func`, train_data_iterator is None when tp_rank != 0,
    # so we need to broadcast it to other tp_rank
    tp_world_size = mpu.get_tensor_model_parallel_world_size()
    src_rank = (get_rank() // tp_world_size) * tp_world_size
    step_num_per_epoch_tensor = comm_func.broadcast(step_num_per_epoch_tensor, src_rank,
                                                    mpu.get_tensor_model_parallel_group())
    comm_func.barrier(group=mpu.get_tensor_model_parallel_group())
    _pynative_executor.sync()

    step_num_per_epoch = step_num_per_epoch_tensor.asnumpy().tolist()
    if isinstance(step_num_per_epoch, list):
        step_num_per_epoch = step_num_per_epoch[0]
    if step_num_per_epoch <= 0:
        raise ValueError(f"Expect step_num_per_epoch > 0, but got {step_num_per_epoch}")

    logger.info(f"step number per epoch is {step_num_per_epoch}")

    train_one_step_cell = TrainOneStepCell(model, optimizer, opt_param_scheduler, config)
    train_one_step_cell.set_train()
    config = train_one_step_cell.config

    global_step = 1
    epoch_step = 1
    current_epoch = 0
    if args.resume_training:
        global_step = initial_step + initial_epoch * step_num_per_epoch + 1
        epoch_step = global_step % step_num_per_epoch
        current_epoch = global_step // step_num_per_epoch
        if epoch_step == 0 and current_epoch > 0:
            epoch_step = step_num_per_epoch
            current_epoch -= 1
        logger.info(f"Resume training starts from global step {global_step}, epoch {current_epoch}, step {epoch_step}.")

    if epoch_step > 1:
        logger.info(f"Resume training will skip {epoch_step - 1} step data")

    evaluation_flag = (
        valid_data_iterator is not None
        and evaluation_func is not None
        and metrics is not None
        and args.eval_interval is not None
        and args.best_metric_comparison is not None
        and args.eval_metric is not None
    )
    save_ckpt_flag = (args.save_interval is not None and
                      args.train_iters != 0 and
                      args.save_interval <= args.train_iters)
    correct_metric_flag = mpu.is_pipeline_last_stage() # not use pp or pp last_stage

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

    if train_data_iterator is not None:
        if isinstance(train_data_iterator, list):
            train_data_dict_iterator = []
            for cur_train_dataloader in train_data_iterator:
                train_data_dict_iterator.append(get_dataset_dict_iterator(cur_train_dataloader, epoch_step))
        else:
            train_data_dict_iterator = get_dataset_dict_iterator(train_data_iterator, epoch_step)
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
        if epoch_step == 1 and train_data_iterator is not None:
            if isinstance(train_data_iterator, list):
                train_data_dict_iterator = []
                for cur_train_dataloader in train_data_iterator:
                    train_data_dict_iterator.append(cur_train_dataloader.create_dict_iterator(num_epochs=1))
            else:
                train_data_dict_iterator = train_data_iterator.create_dict_iterator(num_epochs=1)

        start_time = time.time()
        profiler.step_begin(global_step)
        loss_reduced, is_finite, loss_scale, learning_rate, global_norm = train_one_step_cell(forward_step_func,
                                                                                              train_data_dict_iterator,
                                                                                              forward_only=False)
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
            print_log_str = f"Epoch: {current_epoch}, Step: {epoch_step}, "
            for loss_key, loss_value in loss_reduced.items():
                print_log_str += f"{loss_key}: {loss_value.item()}, "
            logger.info(
                print_log_str
                + f"Finite_grads: {is_finite}, "
                + f"Loss_scale: {loss_scale.value() if loss_scale is not None else None}, "
                + f"Learning_rate: {report_learning_rate}, Grad_norm: {global_norm}, "
                + f"Time: {(end_time - start_time) * 1000:.2f} ms"
            )

        if evaluation_flag and global_step % args.eval_interval == 0:
            is_best = Tensor(False, dtype=mstype.int8)
            results = evaluation_func(train_one_step_cell, valid_data_iterator, metrics, **kwargs)

            # update best_metrics only on last stage
            if correct_metric_flag and best_metric_compare_func(results[args.eval_metric], best_metric):
                best_metric = results[args.eval_metric]
                is_best = Tensor(True, dtype=mstype.int8)

            if mpu.get_pipeline_model_parallel_world_size() > 1:
                is_best = comm_func.all_reduce(is_best, "max", mpu.get_pipeline_model_parallel_group())[0]

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
        if epoch_step > step_num_per_epoch:
            epoch_step = 1
            current_epoch += 1

    if isinstance(train_one_step_cell.optimizer, DistributedOptimizer) \
            and args.overlap_param_gather:
        train_one_step_cell.optimizer.sync_gather_all_model_params(True)

    if save_ckpt_flag:
        logger.info("Saving last step checkpoint.")
        # at the end of training loop, we use `global_step += 1`,
        # so the right global step should be 'global_step - 1',
        epoch_step = (global_step - 1) % step_num_per_epoch
        current_epoch = (global_step - 1) // step_num_per_epoch
        # to avoid situation like 'epoch 1, step 0'
        if epoch_step == 0:
            epoch_step = step_num_per_epoch
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


# pylint: disable=W0613, C0330, W0102
def pretrain(train_valid_test_dataset_provider,
             model_provider,
             model_type,
             forward_step_func,
             process_non_loss_data_func=None,
             extra_args_provider=None,
             args_defaults={}):
    """pretrain function"""

        # initialize and get arguments
    initialize_mindspeed_ms(extra_args_provider=extra_args_provider,
                            args_defaults=args_defaults)

    # updata _TRAIN_START_TIME to the min
    global _TRAIN_START_TIME
    start_time_tensor = Tensor([_TRAIN_START_TIME], mstype.float32)
    start_time_tensor = comm_func.all_reduce(start_time_tensor, op='min')[0]
    _TRAIN_START_TIME = start_time_tensor.item()

    args = get_args()
    timers = get_timers()

    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    model, optimizer, opt_param_scheduler, resume_dict = setup_model_and_optimizer(model_provider, model_type)
    timers('model-and-optimizer-setup').stop()

    timers('train/valid/test-data-iterators-setup', log_level=0).start(barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            iterators = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
    timers('train/valid/test-data-iterators-setup').stop()

    timers.log(['model-and-optimizer-setup', 'train/valid/test-data-iterators-setup'], barrier=True)

    config = get_model_config(model[0])

    if not args.skip_train:
        logger.warning("training...")
        iteration = 0
        if args.do_train and args.train_iters > 0:
            train(
                forward_step_func=forward_step_func,
                model=model,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
                train_data_iterator=train_data_iterator,
                valid_data_iterator=valid_data_iterator,
                process_non_loss_data_func=process_non_loss_data_func,
                config=config,
                metrics=None,
                resume_dict=resume_dict,
            )
            iteration = args.train_iters
            logger.warning("training is done...")
    else:
        # TODO: Make iteration sense
        iteration = 0
        logger.warning("skip training...")

    if args.do_valid:
        prefix = f'on validation set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=False)

    if args.do_test:
        logger.warning("testing is not supported for now.")


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0):
    """ setup model, optimizer and opt_param_scheduler. """
    if no_wd_decay_cond is not None:
        logger.warning("For setup_model_and_optimizer, no_wd_decay_cond is not support for now.")
    if scale_lr_cond is not None:
        logger.warning("For setup_model_and_optimizer, scale_lr_cond is not support for now.")
    if lr_mult != 1.0:
        logger.warning("For setup_model_and_optimizer, lr_mult is not support for now.")
    args = get_args()

    model = get_model(model_provider_func, model_type, wrap_with_ddp=args.wrap_with_ddp)

    config = optimizer_config_from_args(args)
    group_params = set_weight_decay(model.trainable_params(), config.weight_decay)
    optimizer = get_optimizer(config,
                              args,
                              params=group_params,
                              network=model)
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    resume_dict = None
    load_dir = None
    if args.load is not None and \
       os.path.exists(args.load):
        load_dir = args.load
        args.finetune = False

    pretrain_dir = args.pretrained_checkpoint
    if load_dir is None and pretrain_dir is not None:
        if not os.path.exists(pretrain_dir):
            raise FileNotFoundError(f"Pretrained checkpoint not exists: {pretrain_dir}")
        load_dir = pretrain_dir
        args.finetune = True
        args.resume_training = False

    if load_dir is not None:
        rank_path = os.path.join(load_dir, f"rank_{get_rank()}")
        if os.path.exists(rank_path):
            meta_path = os.path.join(rank_path, "meta.json")
            resume_by_meta = True
            if not os.path.exists(meta_path):
                logger.warning(f"Could not find meta.json in directory {rank_path}, using latest ckpt in {rank_path}")
                resume_by_meta = False
            resume_ckpt_name = get_resume_checkpoint(
                checkpoint_dir=load_dir,
                resume_training=args.resume_training,
                resume_by_meta=resume_by_meta
                )
            logger.debug(f"resume_ckpt_name is {resume_ckpt_name}")
            if resume_ckpt_name is True:
                ckpt_path = load_dir
            elif isinstance(resume_ckpt_name, str):
                ckpt_path = os.path.join(rank_path, resume_ckpt_name)
        else:
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
            tp_size = mpu.get_tensor_model_parallel_world_size()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            local_rank_to_dp0_rank = pp_rank * dp_size * tp_size + tp_rank
            logger.warning(f"global rank_{get_rank()} ckpt not found, will load rank_{local_rank_to_dp0_rank} ckpt.")
            rank_path = os.path.join(load_dir, f"rank_{local_rank_to_dp0_rank}")
            if not os.path.exists(rank_path):
                raise FileNotFoundError(f"Path {rank_path} not exists, please check your ckpt path.")
            ckpt_path = get_last_checkpoint(rank_path)
            if not ckpt_path or not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"File {ckpt_path} not exists, please check your ckpt path.")
        logger.debug(f"ckpt_path is {ckpt_path}")
        resume_dict = load_checkpoint(
            config=get_model_config(model[0]),
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            ckpt_path=ckpt_path,
            format=args.dist_ckpt_format
        )
        logger.info(f"Checkpoint has trained {resume_dict.get('epoch_num', 0)} epochs, "
                    f"{resume_dict.get('step_num', 0)} steps.")
        if resume_dict is not None and args.resume_training is False:
            resume_dict = None
            logger.warning("resume_dict extract from checkpoint is not 'None', but resume_training is 'False', "
                           "so resume_dict will be set to 'None'")

    return model, optimizer, opt_param_scheduler, resume_dict


def build_train_valid_test_data_loaders(build_train_valid_test_datasets_provider):
    """ build the loader for train, validation and test set """
    args = get_args()
    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)
    train_dataloader, valid_dataloader, test_dataloader = \
        build_train_valid_test_datasets_provider(get_train_valid_test_num_samples())

    do_train = train_dataloader is not None and args.train_iters > 0
    do_valid = valid_dataloader is not None and args.eval_iters > 0
    do_test = test_dataloader is not None and args.eval_iters > 0

    args.do_train = getattr(args, "do_train", False) or do_train
    args.do_valid = getattr(args, "do_valid", False) or do_valid
    args.do_test = getattr(args, "do_test", False) or do_test

    return train_dataloader, valid_dataloader, test_dataloader


def build_train_valid_test_data_iterators(build_train_valid_test_datasets_provider):
    """ build the iterator of the data set """
    train_dataloader, valid_dataloader, test_dataloader = \
        build_train_valid_test_data_loaders(build_train_valid_test_datasets_provider)
    train_data_iterator = train_dataloader
    valid_data_iterator = valid_dataloader
    test_data_iterator = test_dataloader

    return train_data_iterator, valid_data_iterator, test_data_iterator


def evaluate(forward_step_func,
             data_iterator,
             model,
             process_non_loss_data_func,
             config,
             verbose=False):
    """ Evaluation func """
    if process_non_loss_data_func:
        raise NotImplementedError("process_non_loss_data_func is not supported for now.")
    args = get_args()
    timers = get_timers()
    timers('evaluation', log_level=0).start(barrier=True)
    eval_batch_size = args.global_batch_size
    eval_num_microbatch = eval_batch_size // (args.micro_batch_size * args.data_parallel_size)

    for submodel in model:
        submodel.set_train(False)
    if verbose:
        print_rank_0(f"Evaluation on {args.eval_iters * eval_batch_size} samples")
    if isinstance(data_iterator, list):
        valid_data_dict_iterator = []
        for cur_valid_dataloader in data_iterator:
            valid_data_dict_iterator.append(cur_valid_dataloader.create_dict_iterator())
    else:
        valid_data_dict_iterator = data_iterator.create_dict_iterator()
    iteration = 0
    total_loss_reduced = {}
    while iteration < args.eval_iters:
        iteration += 1
        if verbose:
            print_rank_0(f"Evaluating iter {iteration}/{args.eval_iters}")
        forward_backward_func = get_forward_backward_func()
        config.timers = None
        losses_reduced, _ = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=valid_data_dict_iterator,
            model=model,
            num_microbatches=eval_num_microbatch,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=True
        )
        config.timers = get_timers()

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            for key in losses_reduced[0].keys():
                for x in losses_reduced:
                    if key not in total_loss_reduced:
                        total_loss_reduced[key] = Tensor([0.0, 0.0], mstype.float32)
                    val = x[key]
                    if isinstance(val, (list, tuple)):
                        total_loss_reduced[key][0] += val[0]
                        total_loss_reduced[key][1] += val[1]
                    else:
                        total_loss_reduced[key][0] += val
                        total_loss_reduced[key][1] += 1
        else:
            total_loss_reduced = {"lm loss": [Tensor(0, mstype.float32), 1]}
        args.consumed_valid_samples += eval_batch_size

        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_each_device = Tensor(
                [train_time > args.exit_duration_in_mins],
                mstype.int8
            )
            done_each_device = comm_func.all_reduce(done_each_device, "max")[0]
            done = done_each_device.item()
            if done:
                logger.warning('Exiting during evaluation, timelimit reached')
                return None, None, True

    collected_non_loss_data = None

    for submodel in model:
        submodel.set_train(True)
    for key in total_loss_reduced:
        numerator, denominator = total_loss_reduced[key]
        total_loss_reduced[key] = numerator / denominator
    timers('evaluation').stop()
    timers.log(['evaluation'])

    return total_loss_reduced, collected_non_loss_data, False


def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, process_non_loss_data_func, config,
                               verbose=False, write_to_tensorboard=True):
    """Helper function to evaluate."""
    args = get_args()
    if write_to_tensorboard:
        writer = get_tensorboard_writer()
    else:
        writer = None

    wandb_writer = get_wandb_writer()

    total_loss_dict, _, timelimit = evaluate(
        forward_step_func, data_iterator, model,
        process_non_loss_data_func, config, verbose
    )
    if timelimit:
        return
    string = " validation loss at {} | ".format(prefix)
    for key in total_loss_dict:
        string += "{} value: {:.6E} | ".format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += "{} PPL: {:.6E} | ".format(key, ppl)
        if writer:
            writer.add_scalar('{} validation'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('{} validation vs samples'.format(key),
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar('{} validation ppl'.format(key), ppl,
                                  iteration)
                writer.add_scalar('{} validation ppl vs samples'.format(key),
                                  ppl, args.consumed_train_samples)
            if wandb_writer and is_last_rank():
                wandb_writer.log({
                    '{} validation'.format(key): total_loss_dict[key].item()},
                    iteration)

    if process_non_loss_data_func:
        raise NotImplementedError("process_non_loss_data_func is not supported for now.")

    length = len(string) + 1
    print_rank_last("-" * length)
    print_rank_last(string)
    print_rank_last("-" * length)
