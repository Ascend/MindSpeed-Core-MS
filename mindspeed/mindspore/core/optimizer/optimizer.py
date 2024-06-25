# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import mindtorch.torch as torch
from mindspore.amp import all_finite


def mixed_precision_optimizer_unscale_and_check_for_nan(self):
    # Collect main grads.
    main_grads = self._collect_main_grad_data_for_unscaling()

    # Reset found inf.
    self.found_inf.fill_(0.0)

    self.found_inf.assign_value((~all_finite(main_grads)).astype(self.found_inf.dtype))
    for grad in main_grads:
        grad.assign_value(grad * self.grad_scaler.inv_scale)

        # Update across all model parallel instances.
    torch.distributed.all_reduce(
        self.found_inf, op=torch.distributed.ReduceOp.MAX, group=self.get_model_parallel_group()
    )

    # Check for nan.
    found_inf_flag = self.found_inf.item() > 0

    return found_inf_flag


def float16_optimizer_collect_for_unscaling(self):
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


def float16_optimizer_init(self,
                           optimizer,
                           config,
                           grad_scaler,
                           init_state_fn):
    from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer
    from megatron.core import tensor_parallel
    MixedPrecisionOptimizer.__init__(self, optimizer, config, grad_scaler, init_state_fn)

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
                if param.type() in ['torch.cuda.HalfTensor',
                                    'torch.cuda.BFloat16Tensor']:
                    float16_params_this_group.append(param)
                    # Create a copy
                    main_param = torch.nn.Parameter(param.detach().clone().float())
                    # Copy tensor model parallel attributes.
                    tensor_parallel.copy_tensor_model_parallel_attributes(main_param,
                                                                          param)
                    if hasattr(param, 'shared'):
                        main_param.shared = param.shared
                    # Replace the optimizer params with the new fp32 copy.
                    param_group['params'][i] = main_param

                    fp32_from_float16_params_this_group.append(main_param)
                    # Reset existing state dict key to the new main param.
                    if param in self.optimizer.state:
                        # reduce memory , this state is not used
                        self.optimizer.state[main_param] = None
                        # fp32 params.
                elif param.type() == 'torch.cuda.FloatTensor':
                    fp32_params_this_group.append(param)
                    param_group['params'][i] = param
                else:
                    raise TypeError('Wrapped parameters must be one of '
                                    'torch.cuda.FloatTensor,  '
                                    'torch.cuda.HalfTensor, or '
                                    'torch.cuda.BFloat16Tensor. '
                                    'Received {}'.format(param.type()))

        self.float16_groups.append(float16_params_this_group)
        self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)
        self.fp32_from_fp32_groups.append(fp32_params_this_group)
