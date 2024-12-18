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
"""optimizer registration and factory method"""
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import mint, Tensor
from mindspore.communication import GlobalComm
from mindspore.communication.comm_func import all_reduce
from mindspore.communication.management import get_rank, create_group

from mindspeed_ms.core.ddp.parallel_state import get_zero_full_shard_flag, get_zero_shard_group
from .grad_handler import inplace_apply_to_tensor_list, \
    get_grad_norm_fp32, clip_grad_by_total_norm_fp32


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
            print("Float16OptimizerWithFloat16Params only support AdamW optimizer for now. "
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

    def zero_grad(self):
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
        rank_id = int(get_rank())
        rank_group = "tp-pp-" + str(rank_id)
        create_group(rank_group, rank_id)
        return rank_group

    def get_main_grads_for_grad_norm(self):
        """ collect main gradients for grad norm compute. """
        params = self.get_parameters_()
        grads_for_norm = []
        for param in params:
            grad = param.grad
            grad_not_none = grad is not None
            is_not_tp_duplicate = not (
                    ("norm" in param.name)
                    or ("mlp.projection.bias" in param.name)
                    or ("attention.out_proj.bias" in param.name)
            )
            is_shard = (
                    ('x_embedder.' not in param.name) and ('t_embedder.' not in param.name) and (
                    'y_embedder.y_proj.fc1.' not in param.name))
            if grad_not_none and is_not_tp_duplicate:
                grads_for_norm.append(grad)
            else:
                print(f"111param.name{param.name},param.grad{param.grad.shape}", flush=True)

        return grads_for_norm

    def clip_grad_norm(self, clip_grad):
        """ clip gridients by global norm. """
        params = self.get_parameters_()
        grads_for_norm = self.get_main_grads_for_grad_norm()

        # as we do not need to consider mp, use zero_shard_group directly
        zero_shard_group = get_zero_shard_group(with_context_parallel=True) if not get_zero_full_shard_flag() \
            else GlobalComm.WORLD_COMM_GROUP
        grad_norm = get_grad_norm_fp32(grads_for_norm, parallel_group=zero_shard_group)
        clip_grad_by_total_norm_fp32(params, clip_grad, grad_norm)
        return grad_norm

    def _unscale_main_grads_and_check_for_nan(self):
        """ check nan in main grads and unscale when using grad_scaler. """
        self._collect_main_grad_data()
        self.found_inf = Tensor(False, mstype.bool_)
        inv_scale = mint.reciprocal(self.grad_scaler).astype(mstype.float32)
        self.grad_scale_func(self.grads, inv_scale)
        for grad in self.grads:
            self.found_inf = mint.logical_and(self.found_inf, mint.logical_not(mint.isfinite(grad)).all())
        self.found_inf = all_reduce(self.found_inf.astype(mstype.float32), 'max', get_model_parallel_group())[0]
        return mint.greater(self.found_inf, self._scale_zero)

    # pylint: disable=R1705
    def prepare_grads(self):
        """ grads overflow check and unscaling. """
        self._copy_model_grads_to_main_grads()
        if self.grad_scaler:
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()
            return found_inf_flag
        else:
            self._collect_main_grad_data()
        return False

    def step_with_ready_grads(self):
        """ optimizer update and copy from fp32 copy to model params. """
        success = self.optimizer(self.grads)
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
