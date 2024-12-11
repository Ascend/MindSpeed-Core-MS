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
"""Optimizer config."""

from dataclasses import dataclass
import inspect
from typing import Callable, Optional, Union

import mindspore.common.dtype as mstype
from mindspore._c_expression.typing import Float, BFloat


@dataclass
class OptimizerConfig:
    """Optimizer config class"""

    ##############
    # General
    ##############
    optimizer: str = "adam"
    """Optimizer to use (one of adam or SGD)."""

    lr: Optional[float] = None
    """Initial learning rate. Depending on decay style and initial warmup, the learning rate at each
       iteration would be different.
    """

    min_lr: Optional[float] = None
    """Minimum value for learning rate. The scheduler clip values below this threshold."""

    # Placeholder. Do not use
    decoupled_lr: Optional[float] = None
    """Separate learning rate for the input and output layer."""

    # Placeholder. Do not use
    decoupled_min_lr: Optional[float] = None
    """Minimum value for learning rate for the input and output layer. The scheduler clip values
       below this threshold.
    """

    weight_decay: float = 0.01
    """Weight decay coefficient for L2 regularization."""

    # Additional parameter
    lr_scheduler_kwargs: dict = None
    """Learning rate scheduler args, additional parameter"""

    ##############
    # Precision
    ##############
    fp16: bool = False
    """Use fp16, 'params_dtype' and 'compute_dtype' will be set to 'float16' automatically"""

    bf16: bool = False
    """Use bf16, 'params_dtype' and 'compute_dtype' will be set to 'bfloat16' automatically"""

    params_dtype: Union[Float, BFloat] = mstype.float32
    """Parameter initialize data type"""

    ###############
    # Loss scaling
    ###############
    loss_scale: Optional[float] = None
    """Static loss scaling, positive power of 2 values can improve fp16 convergence. If None,
       dynamic loss scaling is used.
    """

    initial_loss_scale: float = 2 ** 32
    """Initial loss-scale for dynamic loss scaling."""

    # Placeholder. Do not use
    min_loss_scale: float = 1.0
    """Minimum loss scale for dynamic loss scaling."""

    loss_scale_window: float = 1000.0
    """Window over which to raise/lower dynamic scale."""

    hysteresis: int = 2
    """Hysteresis for dynamic loss scaling."""

    # Additional parameter
    loss_func_kwargs: dict = None
    """Loss function args"""

    ##############
    # Optimizer
    ##############
    # Adam
    adam_beta1: float = 0.9
    """First coefficient for computing running averages of gradient and its square in Adam
    optimizer.
    """

    adam_beta2: float = 0.999
    """Second coefficient for computing running averages of gradient and its square in Adam
    optimizer.
    """

    # Placeholder. Do not use
    adam_eps: float = 1e-08
    """Term added to the denominator to improve numerical stability in Adam optimizer."""

    # SGD. Placeholder. Do not use
    sgd_momentum: float = 0.9
    """Momentum factor for SGD optimizer."""

    #######################
    # Distributed optimizer
    #######################
    use_distributed_optimizer: bool = False
    """Distribute optimizer state over data-parallel replicas."""

    overlap_grad_reduce: bool = False
    """If true, overlap grad reduce-scatter with backward compute in distributed optimizer."""

    overlap_param_gather: bool = False
    """If true, overlap param all-gather with forward compute in distributed optimizer."""

    ################
    # Miscellaneous
    ################
    clip_grad: float = 1.0
    """Gradient clipping based on global L2 norm."""

    # Placeholder. Do not use
    log_num_zeros_in_grad: bool = False
    """If true, calculate and log the number of zeros in gradient."""

    # Placeholder. Do not use
    barrier_with_L1_time: bool = False   # pylint: disable=C0103
    """If true, use barrier with level 1 time measurements."""

    # Placeholder. Do not use
    timers: Callable = None
    """Function to get timers."""

    # Additional arguments
    grad_clip_kwargs: dict = None
    """Gradient clip arguments"""

    ################
    # ZeRO
    ################

    # Additional arguments
    zero_config: dict = None
    """ZeRO optimizer config
    - param_resident (bool): After the forward propagation, the parameters are resident and not split. Default: Flase.
    - allreduce_after_grad_accumulation (bool): Use allreduce in optimizer after gradient accumulation. Default: Flase.
    - grad_allreduce_op (str): Gradient allreduce operator. like `sum`, `mean`. Default: sum.
    - opt_parallel_group (str): Name of communication group used by optimizer parallel. Default: None.
    - cpu_offload (bool): The process of optimizer will be offload to host. The gradients, parameters and optimizer
                          status will be offload to host. Default: Flase.
    """

    # Additional arguments
    zero_without_ddp: bool = False
    """"""

    ################
    # MindSpeed feature
    ################
    reuse_fp32_param: bool = False


    def get_needed_params_for_class(self, a_class):
        """
        Returns a dictionary of the needed parameters for a given class.

        Args:
            a_class: The class for which the needed parameters are to be retrieved.

        Returns:
            A dictionary containing the needed parameters and their corresponding values from the current instance.
        """
        needed_parameters = inspect.signature(a_class).parameters.keys()
        return {k: v for k, v in self.__dict__.items() if k in needed_parameters}


    def __str__(self):
        gap = 2 * " "
        attributes = vars(self)
        print_str = "\n" + self.__class__.__name__ + "\n"
        for name, val in attributes.items():
            new_str = str(val)
            new_str = new_str.replace("\n", "\n" + gap)
            print_str += f"{gap}{name}: {new_str}\n"

        return print_str
