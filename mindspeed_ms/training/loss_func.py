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

from mindspeed_ms.core.tensor_parallel import ReduceFromContextParallelRegion
from mindspeed_ms.core.parallel_state import get_context_parallel_world_size
from mindspeed_ms.core.register import ModuleType, ModuleRegistry

__all__ = ['get_loss_func']


class LossWithMask(nn.Cell):
    """
    Calculate the loss with mask and mean reduction.

    Args:
        loss_func (Function): Loss function.
        args (tuple): Input arguments.
        kwargs (dict): Keyword arguments.

    Inputs:
        - **logits** (Tensor) - The output logits of the backbone. Tensor of shape :math:`(N, C)`.
          Data type must be float16 or float32.
        - **label** (Tensor) - The ground truth label of the sample. Tensor of shape :math:`(N, )`
          or the same shape as `logits`.
        - **input_mask** (Tensor) - The `input_mask` indicates whether there are padded inputs
          and for padded inputs it will not be counted into loss. Tensor of shape :math:`(N, )`.

    Outputs:
        The corresponding cross entropy loss.

    Examples:
        .. note::
            Before running the following examples, you need to configure the environment variables.

            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore.communication.management import init
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor, nn
        >>> from mindspeed_ms.core.parallel_state import initialize_model_parallel
        >>> from mindspeed_ms.training.loss_func import LossWithMask
        >>> ms.set_context(device_target='Ascend', mode=ms.PYNATIVE_MODE)
        >>> ms.set_seed(2024)
        >>> init()
        >>> initialize_model_parallel()
        >>> loss = LossWithMask(nn.CrossEntropyLoss())
        >>> logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]),
        ...                 mstype.float32)
        >>> labels = Tensor(np.array([1]).astype(np.int32))
        >>> input_mask = Tensor(np.ones(1).astype(np.float32))
        >>> output = loss(logits, labels, input_mask)
        >>> print(output)
        67.0
    """
    # pylint: disable=W0613
    def __init__(self, loss_func, *args, **kwargs):
        super(LossWithMask, self).__init__()
        self.loss_func = loss_func
        self.reduce_from_context_parallel_region = ReduceFromContextParallelRegion()

    def construct(self, logits, label, loss_mask):
        "Calculate the loss with mask and mean reduction."
        output_tensor = self.loss_func(logits, label)
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        cp_world_size = get_context_parallel_world_size()
        if cp_world_size > 1:
            loss = mint.cat([mint.sum(losses * loss_mask).view(1), loss_mask.sum().view(1)])
            loss = self.reduce_from_context_parallel_region(loss)
            loss = loss[0] / loss[1]
        else:
            if loss_mask.sum() == 0:
                loss = mint.sum(losses * loss_mask) / 1
                print("==NAN, loss mask sum 0", flush=True)
            else:
                loss = mint.sum(losses * loss_mask) / loss_mask.sum()
            if loss.isnan():
                loss = mint.sum(losses * loss_mask) / 1
                print("==NAN, dd", flush=True)

        return loss * cp_world_size


def get_loss_func(training_config, return_instance: bool = True, **kwargs):
    """
    Get the loss function based on the provided loss function configuration.

    Args:
        training_config (TrainingConfig): The configuration object for training.

    Returns:
        loss_fn (callable): The loss function based on the provided configuration.

    Raises:
        ValueError: If the specified loss function type is not supported.
    """
    loss_func_kwargs = training_config.loss_func_kwargs
    loss_func_kwargs["reduction"] = training_config.loss_reduction
    loss_func_type = loss_func_kwargs['loss_func_type']
    if "CrossEntropyLoss" in loss_func_type:
        loss_func_kwargs["reduction"] = 'none'
    loss_func_cls = ModuleRegistry.get_item(module_type=ModuleType.LOSS_FUNC, item_name=loss_func_type)
    if return_instance:
        loss_func_kwargs.update(kwargs)
        loss_func_kwargs = ModuleRegistry.get_needed_params_for_init(loss_func_cls, loss_func_kwargs)
        return LossWithMask(loss_func=loss_func_cls(**loss_func_kwargs))
    return loss_func_cls
