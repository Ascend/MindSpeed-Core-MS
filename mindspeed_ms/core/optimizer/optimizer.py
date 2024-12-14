# Copyright 2024 Huawei Technologies Co., Ltd
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""optimizer registration and factory method"""
from collections import OrderedDict
from typing import Callable, List, Optional

import numpy as np
import mindspore as ms
from mindspore import mint, ops, Tensor, Parameter
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.communication.comm_func as comm_func
from mindspeed_ms.training.optimizer_param_scheduler import OptimizerParamScheduler

from mindspeed_ms.training.global_vars import get_args
from mindspeed_ms.tools import logger
from mindspeed_ms.training.grad_handler import inplace_apply_to_tensor_list, \
    get_grad_norm_fp32, clip_grad_by_total_norm_fp32, param_is_not_shared
from mindspeed_ms.core.parallel_state import get_tensor_model_parallel_rank, get_model_parallel_group
from mindspeed_ms.core.utils import (
    local_multi_tensor_applier,
    local_multi_tensor_l2_norm,
    local_multi_tensor_scale
)

from .. import tensor_parallel
from ..dist_checkpointing.mapping import ShardedStateDict
from ..dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    make_sharded_optimizer_tensor,
    optim_state_to_sharding_state,
)
from .optimizer_config import OptimizerConfig

multi_tensor_applier = local_multi_tensor_applier
l2_norm_impl = local_multi_tensor_l2_norm
multi_tensor_scale_impl = local_multi_tensor_scale


def get_optimizer_param_scheduler(optimizer):
    """ Build the learning rate scheduler."""
    # Iteration-based training.
    args = get_args()
    global_batch_size = args.global_batch_size
    if args.train_iters is not None and args.train_iters > 0:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters

        lr_decay_steps = args.lr_decay_iters * global_batch_size
        wd_incr_steps = args.train_iters * global_batch_size
        wsd_decay_steps = None
        if args.lr_wsd_decay_iters is not None:
            wsd_decay_steps = args.lr_wsd_decay_iters * global_batch_size
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_iters * global_batch_size
    # Sample-based training.
    elif args.train_samples:
        args.train_iters = args.train_samples // global_batch_size
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        lr_decay_steps = args.lr_decay_samples
        wd_incr_steps = args.train_samples
        wsd_decay_steps = args.lr_wsd_decay_samples
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_samples
    else:
        raise Exception(
            'either train-iters or train-samples should be positive number.')

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style
        )

    return opt_param_scheduler


def _zero_grad_group_helper(group: List[ms.Parameter], set_to_none: bool):
    """
    Zero out the gradient for a group of parameters.
    """
    for param in group:
        if param.grad is not None:
            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()


def _multi_tensor_copy_this_to_that(
        this: List[ms.Tensor], that: List[ms.Tensor], overflow_buf: Optional[ms.Tensor] = None
):
    """
    Use multi-tensor-applier to copy values from one list to another.
    We don't have a bfloat16 implementation so for now if the overflow_buf
    is not provided, we default back to simple loop copy to be compatible
    with bfloat16.
    """
    if overflow_buf:
        # overflow_buf.fill_(0)
        overflow_buf = ms.Tensor([0], dtype=ms.int32)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(multi_tensor_scale_impl, overflow_buf, [this, that], 1.0)
    else:
        for this_, that_ in zip(this, that):
            that_.copy_(this_)


class MixedPrecisionOptimizer(nn.Cell):
    """
    MixedPrecision Optimizer base class.

    Args:
        optimizer (mindspore.experimental.optim.optimizer): Base optimizer.
        config (OptimizerConfig): Configuration object for optimizer.
        grad_scaler (GradScaler): Gradient scaling. When `grad_scaler=None`, no scaler will be used for
            gradients.
        init_state_fn: Function to initialize state parameters of optimizer.
    """
    def __init__(
            self,
            optimizer,
            config,
            grad_scaler,
            init_state_fn
        ):
        super(MixedPrecisionOptimizer, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.config = config
        self.grad_scaler = grad_scaler
        if init_state_fn is not None:
            logger.warning("Float16OptimizerWithFloat16Params only support AdamW optimizer for now. "
                           "The 'init_state_fn' will not be used.")
        self.init_state_fn = init_state_fn

        if self.grad_scaler is None:
            self._scale_one = Tensor([1.0], dtype=mstype.float32)

        self.grad_scale_func = inplace_apply_to_tensor_list(mint.mul)

        self.grads = []
        self.found_inf = Tensor(False, dtype=mstype.bool_)
        self._scale_zero = Tensor([0.0], dtype=mstype.float32)

    def _get_lrs(self):
        """ get lrs. """
        return self.optimizer.lrs

    def _set_lrs(self, value):
        """ set lrs. """
        self.optimizer.lrs = value

    lrs = property(_get_lrs, _set_lrs)

    # pylint: disable=W0613
    def zero_grad(self, set_to_none=True):
        """ zero grad data. """
        return

    # pylint: disable=R1705
    def get_loss_scale(self):
        """ get loss scale. """
        if self.grad_scaler is None:
            return self._scale_one
        elif isinstance(self.grad_scaler, Tensor):
            return self.grad_scaler
        return self.grad_scaler.scale

    def reload_model_params(self):
        """ copy model params to its fp32 copy. """
        self._copy_model_params_to_main_params()

    def get_parameters_(self):
        """ get parameters registered to optimizer in order. """
        return self.optimizer.parameters

    def get_lr(self):
        """ get learning rate. """
        return tuple(self.optimizer.lrs)

    def get_model_parallel_group(self):
        """ return model_parallel_group for global norm allreduce. """
        return get_model_parallel_group()

    def get_main_grads_for_grad_norm(self):
        """ collect main gradients for grad norm compute. """
        params = self.get_parameters_()
        grads_for_norm = []
        for param in params:
            grad = param.grad
            grad_not_none = grad is not None
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = not (
                ("norm" in param.name)
                or ("mlp.projection.bias" in param.name)
                or ("attention.out_proj.bias" in param.name)
                or ("mlp.linear_fc2.bias" in param.name)
                or ("attention.linear_proj.bias" in param.name)
            ) or (get_tensor_model_parallel_rank() == 0)
            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(grad)
        return grads_for_norm

    def clip_grad_norm(self, clip_grad):
        """ clip gridients by global norm. """
        params = self.get_parameters_()
        grads_for_norm = self.get_main_grads_for_grad_norm()
        grad_norm = get_grad_norm_fp32(grads_for_norm, model_parallel_group=self.get_model_parallel_group())
        clip_grad_by_total_norm_fp32(params, clip_grad, grad_norm)
        return grad_norm

    def _unscale_main_grads_and_check_for_nan(self):
        """ check nan in main grads and unscale when using grad_scaler. """
        self._collect_main_grad_data()
        self.found_inf = Tensor(False, mstype.bool_)
        inv_scale = mint.reciprocal(self.grad_scaler).astype(mstype.float32)
        self.grad_scale_func(self.grads, inv_scale)
        for grad in self.grads:
            self.found_inf = mint.logical_not(mint.isfinite(grad).all())
        self.found_inf = comm_func.all_reduce(
            self.found_inf.astype(mstype.float32), 'max', get_model_parallel_group())[0]
        return mint.greater(self.found_inf, self._scale_zero)

    # pylint: disable=R1705
    def prepare_grads(self):
        """ grads overflow check and unscaling. """
        self._copy_model_grads_to_main_grads()

        if self.config.reuse_fp32_param:
            self.fp16_tensor_convert_to_fp32_tensor()

        if self.grad_scaler:
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()
            return found_inf_flag
        else:
            self._collect_main_grad_data()
        return False

    def step_with_ready_grads(self):
        """ optimizer update and copy from fp32 copy to model params. """
        success = self.optimizer(self.grads)

        if self.config.reuse_fp32_param:
            self.fp32_tensor_convert_to_fp16_tensor()
        else:
            self._copy_main_params_to_model_params()

        return success

    def construct(self):
        """ construct function. """
        found_inf_flag = self.prepare_grads()
        if found_inf_flag:
            return False, None, None
        grad_norm = None
        if self.config.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.config.clip_grad)
        num_zeros_in_grad = None
        success = self.step_with_ready_grads()

        return success, grad_norm, num_zeros_in_grad

    # Promote param_groups so it can be retrieved or set via
    # "optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)


class Float16OptimizerWithFloat16Params(MixedPrecisionOptimizer):
    """Float16 optimizer for fp16 and bf16 data types.

    Args:
        optimizer (mindspore.experimental.optim.Optimizer): base optimizer such as Adam or SGD.
        config (OptimizerConfig): configuration object for optimizer.
        grad_scaler (MegatronGradScaler): used for scaling gradients. Note that
            this can be None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constant gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
        init_state_fn (Callable, optional): function to initialize state in the optimizer.
    """

    def __init__(
            self,
            optimizer: ms.experimental.optim.Optimizer,
            config: OptimizerConfig,
            grad_scaler,
            init_state_fn: Callable,
    ):
        super().__init__(
            optimizer,
            config,
            grad_scaler,
            init_state_fn,
        )
        self.wrap_with_ddp = get_args().wrap_with_ddp

        if self.config.bf16:
            self._dummy_overflow_buf = None
        else:
            self._dummy_overflow_buf = ms.tensor([0], dtype=ms.int32)  # not in used for now

        # Handle main parameters.

        # Three groups of parameters:
        #   float16_groups: original float16 parameters
        #   fp32_from_float16_groups: fp32 copy of float16 parameters
        #   fp32_from_fp32_groups: original fp32 parameters
        self.float16_groups = []
        self.fp32_from_float16_groups = []
        self.fp32_from_fp32_groups = []

        # For all the groups in the original optimizer:
        for param_group in self.optimizer.param_groups:
            float16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_float16_params_this_group = []
            # For all the parameters in this group:
            for i, param in enumerate(param_group['params']):
                if param.requires_grad:
                    # float16 params:
                    if param.dtype in [ms.bfloat16, ms.float16]:
                        float16_params_this_group.append(param)
                        # Create a copy
                        # main_param = param.clone().float()
                        main_param = ms.Tensor(param.value().astype(ms.float32))
                        main_param.name = param.name
                        # Copy tensor model parallel attributes.
                        tensor_parallel.copy_tensor_model_parallel_attributes(main_param, param)
                        if hasattr(param, 'shared'):
                            main_param.shared = param.shared
                        # Replace the optimizer params with the new fp32 copy.
                        param_group['params'][i] = main_param

                        fp32_from_float16_params_this_group.append(main_param)
                        # Reset existing state dict key to the new main param.
                        if param in self.optimizer.state:
                            self.optimizer.state[main_param] = self.optimizer.state.pop(param)
                    # fp32 params.
                    elif param.dtype == ms.float32:
                        fp32_params_this_group.append(param)
                        param_group['params'][i] = param

                    else:
                        raise TypeError(
                            'Wrapped parameters dtype must be one of '
                            'mindspore.float32,  '
                            'mindspore.bfloat16, or '
                            'mindspore.float16. '
                            'Received {}'.format(param.dtype)
                        )

                    if not self.wrap_with_ddp:
                        # register hook function for parameters which sets param.grad attr
                        param.register_hook(self._make_param_hook(param))

            self.float16_groups.append(float16_params_this_group)
            self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)

        # adapt for mindspore
        self._update_optimizer_attr()
        # init reuse and release memories of original bf16 params (aka. model params)
        if self.config.reuse_fp32_param:
            self._reuse_fp32_param_init()

    # pylint: disable=W0221
    def zero_grad(self, set_to_none=True):
        """We only need to zero the model related parameters, i.e.,
        float16_groups & fp32_from_fp32_groups. We additionally zero
        fp32_from_float16_groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point."""
        if self.wrap_with_ddp:
            set_to_none = False
        for group in self.float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_fp32_groups:
            _zero_grad_group_helper(group, set_to_none)

        self.grads = []

    # pylint: disable=C0111
    def _collect_main_grad_data_for_unscaling(self):

        main_grads = []

        # fp32 params from float16 ones.
        for main_group in self.fp32_from_float16_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad)

        # Append fp32 parameters.
        for main_group in self.fp32_from_fp32_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad)

        return main_grads

    def _get_model_and_main_params_data_float16(self):
        model_data = []
        main_data = []
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param)
                main_data.append(main_param)
        return model_data, main_data

    # pylint: disable=C0111
    def _copy_model_grads_to_main_grads(self):
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                if hasattr(model_param, 'main_grad'):
                    main_param.grad = model_param.main_grad.float()
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.float()

                # Safe to deallocate model's grad/main_grad after copying.
                # (If using contiguous buffers, main_grad's memory should
                # persist and therefore should not be deallocated.)
                model_param.grad = None

        # For fp32 grads, we need to reset the grads to main grad.
        for model_group in self.fp32_from_fp32_groups:
            for model_param in model_group:
                if hasattr(model_param, 'main_grad'):
                    model_param.grad = model_param.main_grad

    def _copy_main_params_to_model_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(
            this=main_data, that=model_data, overflow_buf=self._dummy_overflow_buf
        )

    def _copy_model_params_to_main_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(
            this=model_data, that=main_data, overflow_buf=self._dummy_overflow_buf
        )

    def state_dict(self, include_optim: bool = True):
        """ get optimizer state dict for saving checkpoint. """
        param_dict = OrderedDict()

        for param in self.optimizer.parameters:
            if isinstance(param, Parameter):
                param_dict[param.name] = param
            elif isinstance(param, Tensor):
                param_dict[param.name] = Parameter(param, name=param.name)
            else:
                raise TypeError("Instance in optimizer.parameters should be mindspore.Parameter or "
                                "mindspore.Tensor, but got {}".format(type(param)))

        if not include_optim:
            return param_dict

        for param in self.optimizer.exp_avg:
            param_dict[param.name] = param

        for param in self.optimizer.exp_avg_sq:
            param_dict[param.name] = param

        # add state step to state_dict
        param_dict['state_step'] = self.optimizer.state_step

        # add learning rate and weight decay to state_dict
        for group_idx, lr in enumerate(self.optimizer.lrs):
            lr_name = lr.name
            param_dict[lr_name] = lr
            wd_name = lr_name.replace('learning_rate', 'weight_decay')
            param_dict[wd_name] = Parameter(
                ops.Tensor(
                    self.optimizer.param_groups[group_idx]['weight_decay'],
                    dtype=mstype.float64,
                ),
                name=wd_name,
                requires_grad=False,
            )
        return param_dict

    # pylint: disable=E1111
    def sharded_state_dict(
            self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False
    ):

        if is_loading:
            self.init_state_fn(self.optimizer)

        state_dict = self.state_dict()

        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict, chain.from_iterable(g for g in self.float16_groups)
        )

        # Convert fp32_from_fp16_params
        assert len(state_dict['fp32_from_fp16_params']) == len(
            state_dict['optimizer']['param_groups']
        )
        state_dict['fp32_from_fp16_params'] = [
            [
                make_sharded_optimizer_tensor(
                    id_to_sharded_param_map[param_id],
                    fp32_param,
                    prefix=f'optimizer.state.fp32_param',
                )
                for param_id, fp32_param in zip(state_group['params'], fp32_group)
            ]
            for fp32_group, state_group in zip(
                state_dict['fp32_from_fp16_params'], state_dict['optimizer']['param_groups']
            )
        ]

        step = self._extract_common_per_param_step(state_dict['optimizer'])

        # Convert regular optimizer state
        # all optimizer parameters passed to optim_state_to_sharding_state are
        # expected to have the same shape as the model parameters,
        # so we save the step separately and ignore it here
        optim_state_to_sharding_state(
            state_dict['optimizer'], id_to_sharded_param_map, exclude_keys="step"
        )
        # save step as a shared step among all parameters. Separate per-parameter
        # steps are not supported
        state_dict['optimizer']['state']['common_step'] = step
        return state_dict

    def load_state_dict(self, state_dict, load_optim: bool = True):
        """ load state dict into optimizer. """
        state_list = list(self.optimizer.exp_avg) + list(self.optimizer.exp_avg_sq)
        param_dict = list(self.optimizer.parameters) + state_list
        state_name = list(map(lambda x: x.name, state_list))
        for param in param_dict:
            if not load_optim and param.name in state_name:
                continue
            if param.name not in state_dict:
                logger.warning(
                    f"No state data found for '{param.name}' and it won't be loaded." + (
                        " Specify --no-load-optim or --finetune to prevent"
                        " attempting to load the optimizer state."
                        if param.name in state_name else ""
                    )
                )
                continue
            param.copy_(state_dict[param.name])

        if not load_optim:
            return

        if 'state_step' in state_dict.keys():
            self.optimizer.state_step.assign_value(state_dict['state_step'].value())

        # load learning rate
        for group_idx, lr in enumerate(self.optimizer.lrs):
            lr_name = lr.name
            if lr_name in state_dict.keys():
                lr = state_dict[lr_name]
                self.optimizer.param_groups[group_idx]['lr'] = lr.item()
            wd_name = lr_name.replace('learning_rate', 'weight_decay')
            if wd_name in state_dict.keys():
                self.optimizer.param_groups[group_idx]['weight_decay'] = state_dict.get(wd_name).item()

    def _collect_main_grad_data(self):
        """ collect main grad for unscaling """
        for param in self.optimizer.parameters:
            self.grads.append(param.grad)

    def _update_optimizer_attr(self):
        """Update parameter and state attributes for mindspore optimizer"""
        self.main_param_to_exp_avg = {}
        self.main_param_to_exp_avg_sq = {}

        self.optimizer.parameters = list(self.optimizer.parameters)
        self.optimizer.exp_avg = list(self.optimizer.exp_avg)
        self.optimizer.exp_avg_sq = list(self.optimizer.exp_avg_sq)

        for group_idx, param_group in enumerate(self.optimizer.param_groups):
            for param_idx, main_param in enumerate(param_group['params']):
                # update main param
                param_world_index = self.optimizer.group_start_id[group_idx] + param_idx
                self.optimizer.parameters[param_world_index] = main_param
                # update state
                if isinstance(self.optimizer, mint.optim.AdamW):
                    param_state_exp_avg = self.optimizer.exp_avg[param_world_index]
                    param_state_exp_avg_sq = self.optimizer.exp_avg_sq[param_world_index]
                    self.optimizer.exp_avg[param_world_index] = ms.Parameter(
                        param_state_exp_avg.asnumpy().astype(np.float32),
                        name=param_state_exp_avg.name,
                        requires_grad=param_state_exp_avg.requires_grad
                    )
                    self.optimizer.exp_avg_sq[param_world_index] = ms.Parameter(
                        param_state_exp_avg_sq.asnumpy().astype(np.float32),
                        name=param_state_exp_avg_sq.name,
                        requires_grad=param_state_exp_avg_sq.requires_grad
                    )

                    self.main_param_to_exp_avg[main_param] = self.optimizer.exp_avg[param_world_index]
                    self.main_param_to_exp_avg_sq[main_param] = self.optimizer.exp_avg_sq[param_world_index]

        # self.optimizer.parameters = ms.ParameterTuple(self.optimizer.parameters)
        if isinstance(self.optimizer, mint.optim.AdamW):
            self.optimizer.exp_avg = ms.ParameterTuple(self.optimizer.exp_avg)
            self.optimizer.exp_avg_sq = ms.ParameterTuple(self.optimizer.exp_avg_sq)

    def _reuse_fp32_param_init(self):
        """Reuse BF16 parameter init"""
        self.res_float16_groups = []
        self.float16_float32_groups = []
        self.int32_float32_groups = []
        for float16_params_this_group, fp32_from_float16_group in zip(
                self.float16_groups, self.fp32_from_float16_groups):
            res_float16_params_this_group = []
            float16_float32_params_this_group = []
            int32_float32_params_this_group = []
            for i, (_, fp32_from_fp16_param) in enumerate(zip(float16_params_this_group, fp32_from_float16_group)):
                res_float16_params_this_group.append(
                    ms.Tensor(np.empty(fp32_from_fp16_param.numel() * 1), dtype=ms.bfloat16))
                float16_float32_params_this_group.append(
                    ms.Tensor(np.empty(fp32_from_fp16_param.numel() * 2), dtype=ms.bfloat16))
                int32_float32_params_this_group.append(
                    ms.Tensor(np.empty(fp32_from_fp16_param.numel() * 1), dtype=ms.int32))
                self.init_and_reuse_storage_of_tensors(
                    fp32_from_float16_group[i],
                    float16_float32_params_this_group[-1],
                    res_float16_params_this_group[-1],
                    float16_params_this_group[i],
                    int32_float32_params_this_group[-1]
                )
            self.res_float16_groups.append(res_float16_params_this_group)
            self.float16_float32_groups.append(float16_float32_params_this_group)
            self.int32_float32_groups.append(int32_float32_params_this_group)

        args = get_args()
        if args.npu_deterministic:
            self.fp16_tensor_convert_to_fp32_tensor = self.fp16_tensor_convert_to_fp32_tensor_deterministic
            self.fp32_tensor_convert_to_fp16_tensor = self.fp32_tensor_convert_to_fp16_tensor_deterministic

    # pylint: disable=W0212
    def init_and_reuse_storage_of_tensors(
            self,
            fp32_tensor,
            bf16_fp32_tensor,
            res_tensor,
            bf16_tensor,
            int32_tensor
    ):
        """
        init a list of tensor with length of 2*fp32_tensor.numel() in bf16 to share the same storage.
        Args:
            fp32_tensor: original fp32 tensor.
            bf16_fp32_tensor: a bf16 tensor share the same storage with original list of fp32 tensors.
            res_tensor: a bf16 tensor that store the residual value of fp32 to bf16, shares a half of the
            storage with bf16_fp32_tensor.
            bf16_tensor: a bf16 tensor that store the value from fp32, shares another half of the
            storage with bf16_fp32_tensor.
            int32_tensors: a list of int32 tensors share the same storages with original list of fp32 tensors.
        """
        ms.utils._reuse_data_ptr(bf16_fp32_tensor, fp32_tensor, 0) # [0p 0p 0p] -> [0 p 0 p 0 p]
        ms.utils._reuse_data_ptr(int32_tensor, fp32_tensor, 0)
        self.fp32_tensors_to_bf16_tensors([int32_tensor], [bf16_fp32_tensor])  # [0 p 0 p 0 p] -> [0 0 0 p p p]
        ms.utils._reuse_data_ptr(res_tensor, bf16_fp32_tensor, 0)  # [0 0 0]
        ms.utils._reuse_data_ptr(bf16_tensor, bf16_fp32_tensor, res_tensor.numel())  # [p p p]

    # pylint: disable=E0202
    def fp16_tensor_convert_to_fp32_tensor(self):
        for int32_float32_group, float16_param_group in zip(
                self.int32_float32_groups, self.float16_float32_groups):
            self.bf16_tensors_to_fp32_tensors(int32_float32_group, float16_param_group)

    # pylint: disable=E0202
    def fp32_tensor_convert_to_fp16_tensor(self):
        for int32_float32_param_group, float16_param_group in zip(
                self.int32_float32_groups, self.float16_float32_groups):
            self.fp32_tensors_to_bf16_tensors(int32_float32_param_group, float16_param_group)

    def fp32_tensors_to_bf16_tensors(self, int32_tensors, bf16_fp32_tensors):
        """
        fp32(0p0p0p0p) -> bf16(pppp) + res(0000)
        rearrange the storage of bf16_fp32_tensor so that recover the fp32_tensor.
        Args:
            int32_tensor: int32 tensor that shares the same device ptr with original list of fp32 tensor.
            bf16_fp32_tensor: bf16 tensor that shares the same device ptr with original list of fp32 tensor.
        Returns:
            None
        """
        for int32_tensor, bf16_fp32_tensor in zip(int32_tensors, bf16_fp32_tensors):
            if bf16_fp32_tensor.numel() == 0:
                return
            int32_tensor.copy_((int32_tensor + 32768).contiguous())
            bf16_fp32_tensor.copy_(bf16_fp32_tensor.view(-1, 2).swapaxes(1, 0).reshape(-1).contiguous())

    def bf16_tensors_to_fp32_tensors(self, int32_tensors, bf16_fp32_tensors):
        """
        res(0000) + bf16(pppp) -> fp32(0p0p0p0p)
        rearrange the storage of bf16_fp32_tensor so that recover the fp32_tensor.
        Args:
            int32_tensor: int32 tensor that shares the same device ptr with original list of fp32 tensor.
            bf16_fp32_tensor: bf16 tensor that shares the same device ptr with original list of fp32 tensor.
        Returns:
            None
        """
        for int32_tensor, bf16_fp32_tensor in zip(int32_tensors, bf16_fp32_tensors):
            if bf16_fp32_tensor.numel() == 0:
                return
            bf16_fp32_tensor.copy_(bf16_fp32_tensor.view(2, -1).swapaxes(1, 0).reshape(-1).contiguous())
            int32_tensor.copy_((int32_tensor - 32768).contiguous())

    def fp16_tensor_convert_to_fp32_tensor_deterministic(self):
        for int32_float32_group, float16_param_group, fp32_from_float16_group in zip(
                self.int32_float32_groups, self.float16_float32_groups, self.fp32_from_float16_groups):
            self.bf16_tensors_to_fp32_tensors_deterministic(
                int32_float32_group, float16_param_group, fp32_from_float16_group, self.optimizer)

    def fp32_tensor_convert_to_fp16_tensor_deterministic(self):
        for int32_float32_param_group, float16_param_group, fp32_from_float16_group in zip(
                self.int32_float32_groups, self.float16_float32_groups, self.fp32_from_float16_groups):
            self.fp32_tensors_to_bf16_tensors_deterministic(
                int32_float32_param_group, float16_param_group, fp32_from_float16_group, self.optimizer)

    def fp32_tensors_to_bf16_tensors_deterministic(self, int32_tensors, bf16_fp32_tensors, fp32_tensors, optimizer):
        for int32_tensor, bf16_fp32_tensor, fp32_tensor in zip(int32_tensors, bf16_fp32_tensors, fp32_tensors):
            if bf16_fp32_tensor.numel() == 0:
                return
            odd_even_tensor = ((int32_tensor & 131071) == 32768).int()
            int32_tensor.copy_((int32_tensor + 32768).contiguous())
            self.optimizer_exp_avg_save_sign(optimizer, fp32_tensor, int32_tensor, odd_even_tensor)
            bf16_fp32_tensor.copy_(bf16_fp32_tensor.view(-1, 2).transpose(1, 0).reshape(-1).contiguous())

    def bf16_tensors_to_fp32_tensors_deterministic(
            self, int32_tensors, bf16_fp32_tensors, fp32_tensors, optimizer):
        for int32_tensor, bf16_fp32_tensor, fp32_tensor in zip(int32_tensors, bf16_fp32_tensors, fp32_tensors):
            if bf16_fp32_tensor.numel() == 0:
                return
            bf16_fp32_tensor.copy_(bf16_fp32_tensor.view(2, -1).transpose(1, 0).reshape(-1).contiguous())
            self.optimizer_exp_avg_load_sign(optimizer, fp32_tensor, int32_tensor)
            int32_tensor.copy_((int32_tensor - 32768).contiguous())

    # pylint: disable=W0613
    def optimizer_exp_avg_save_sign(self, optimizer, fp32_param, int32_tensor, odd_even_tensor):
        int32_tensor.copy_((int32_tensor - odd_even_tensor).contiguous())
        sign_tensor = mint.sign(odd_even_tensor - 0.5).reshape(self.main_param_to_exp_avg_sq[fp32_param].shape)
        self.main_param_to_exp_avg_sq[fp32_param].copy_(
            mint.mul(self.main_param_to_exp_avg_sq[fp32_param], sign_tensor)
        )

    # pylint: disable=W0613
    def optimizer_exp_avg_load_sign(self, optimizer, fp32_param, int32_tensor):
        odd_even_tensor = (mint.sign(self.main_param_to_exp_avg_sq[fp32_param]) > 0).reshape(-1)
        self.main_param_to_exp_avg_sq[fp32_param].copy_(
            mint.abs(self.main_param_to_exp_avg_sq[fp32_param])
        )
        int32_tensor.copy_((int32_tensor + odd_even_tensor).contiguous())

    def _make_param_hook(self, param):
        """ make closure function as the param hook. """
        def param_hook(grad):
            # when using bf16, gradients shuold be cast to fp32 for communication and optim
            if param.grad is not None:
                # grad accumulate
                param.grad = mint.add(param.grad, grad)
            else:
                if grad.dtype == mstype.bfloat16:
                    param.grad = ops.cast(grad, mstype.float32)
                else:
                    param.grad = grad
            return param.grad

        return param_hook
