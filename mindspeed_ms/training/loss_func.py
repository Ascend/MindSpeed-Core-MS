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
"""loss function"""
from mindspore import nn, mint
from mindspore.communication.management import get_rank

from mindspeed_ms.core.tensor_parallel import ReduceFromContextParallelRegion
from mindspeed_ms.core.parallel_state import get_context_parallel_world_size
from mindspeed_ms.training.global_vars import get_args
from mindspeed_ms.core.register import ModuleType, ModuleRegistry
from mindspeed_ms.core.optimizer.optimizer_config import OptimizerConfig

__all__ = ['get_loss_func']


class LossWithMask(nn.Cell):
    """
    Calculate the loss with mask and mean reduction.

    Args:
        - **loss_func** (Function) - Loss function.

    Inputs:
        - **logits** (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32. The output logits of
          the backbone.

        - **label** (Tensor) - Tensor of shape (N, ). The ground truth label of the sample.

        - **input_mask** (Tensor) - Tensor of shape (N, ). input_mask indicates whether there are padded inputs and for
          padded inputs it will not be counted into loss.

    Returns:
        The corresponding cross entropy loss.

    Examples:
        >>> import numpy as np
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor, nn
        >>> loss = LossWithMask(nn.CrossEntropyLoss())
        >>> logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), mstype.float32)
        >>> labels_np = np.array([1]).astype(np.int32)
        >>> input_mask = Tensor(np.ones(1).astype(np.float32))
        >>> labels = Tensor(labels_np)
        >>> output = loss(logits, labels, input_mask)
        >>> output.shape
        (1,)
    """
    # pylint: disable=W0613
    def __init__(self, loss_func, *args, **kwargs):
        super(LossWithMask, self).__init__()
        self.loss_func = loss_func
        self.reduce_from_context_parallel_region = ReduceFromContextParallelRegion()

    def construct(self, logits, label, loss_mask):
        "Calculate the loss with mask and mean reduction."
        args = get_args()
        output_tensor = self.loss_func(logits, label)
        if output_tensor.ndim == 2:
            output_tensor = output_tensor.swapaxes(0, 1).contiguous()
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        total_tokens = loss_mask.sum()
        loss = mint.cat([mint.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

        cp_world_size = get_context_parallel_world_size()
        if cp_world_size > 1:
            loss = self.reduce_from_context_parallel_region(loss)

        # Check individual rank losses are not NaN prior to DP all-reduce.
        if args.check_for_nan_in_loss_and_grad:
            global_rank = get_rank()
            if loss[0].isnan():
                raise ValueError(f"Rank {global_rank}: found NaN in local forward loss calculation")

        return loss[0] * cp_world_size / loss[1]


def get_loss_func(optimizer_config: OptimizerConfig, return_instance: bool = True, **kwargs):
    """
    Get the loss function based on the provided loss function configuration.

    Args:
        optimizer_config (OptimizerConfig): The configuration object for training.

    Returns:
        loss_fn (callable): The loss function based on the provided configuration.

    Raises:
        ValueError: If the specified loss function type is not supported.
    """
    loss_func_kwargs = optimizer_config.loss_func_kwargs
    loss_func_kwargs["reduction"] = "mean"
    loss_func_type = loss_func_kwargs['loss_func_type']
    if "CrossEntropyLoss" in loss_func_type:
        loss_func_kwargs["reduction"] = 'none'
    loss_func_cls = ModuleRegistry.get_item(module_type=ModuleType.LOSS_FUNC, item_name=loss_func_type)
    if return_instance:
        loss_func_kwargs.update(kwargs)
        loss_func_kwargs = ModuleRegistry.get_needed_params_for_init(loss_func_cls, loss_func_kwargs)
        return loss_func_cls(**loss_func_kwargs)
    return loss_func_cls
