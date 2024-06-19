# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
from functools import wraps

import os
import warnings
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch_npu
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.parameter import Parameter

from megatron.training import get_args
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    _initialize_affine_weight_cpu,
    set_tensor_model_parallel_attributes,
    _initialize_affine_weight_gpu,
    _grad_accum_fusion_available,
    linear_with_grad_accumulation_and_async_allreduce,
    linear_with_frozen_weight,
    LinearWithGradAccumulationAndAsyncCommunication
)
from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.layers import (linear_with_grad_accumulation_and_async_allreduce, 
                                                  linear_with_frozen_weight)
from megatron.core.parallel_state import get_tensor_model_parallel_world_size
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.utils import VocabUtility, divide, split_tensor_along_last_dim
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint, prepare_input_tensors_for_wgrad_compute
from .ascend_turbo.mc2_linears_seq_parallel import RowSeqParallelLinear


def vocab_parallel_embedding_forward(self, input_):
    if self.tensor_model_parallel_size > 1:
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | \
                     (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input *= ~input_mask
    else:
        masked_input = input_
        # Get the embeddings.

    # For higher accumulation accuracy for bf16 on NPU.
    output_parallel = F.embedding(masked_input, self.weight)

    # Mask the output embedding.
    if self.tensor_model_parallel_size > 1:
        output_parallel *= ~input_mask[..., None]
    # Reduce across all the model parallel GPUs.
    output = reduce_from_tensor_model_parallel_region(output_parallel)
    return output


def row_parallel_nocomm_optimizer_wrapper(forward_func):
    @wraps(forward_func)
    def row_parallel_forward(*args, **kwargs):
        global_args = get_args()
        if global_args.optimize_recomp_communication_level == 0:
            output = forward_func(*args, **kwargs)
        else:
            output = row_parallel_nocomm(*args, **kwargs)
        return output
    return row_parallel_forward


def row_parallel_nocomm(self, input_):
    """Forward of RowParallelLinear w/o comm

            Args:
                input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

            Returns:
                - output
                - bias
    """
    # Set up backprop all-reduce.
    if self.input_is_parallel:
        input_parallel = input_
    else:
        assert not self.sequence_parallel
        input_parallel = scatter_to_tensor_model_parallel_region(input_)
    # Matrix multiply.
    if not self.weight.requires_grad:
        self._forward_impl = linear_with_frozen_weight
    else:
        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce
    output_parallel = self._forward_impl(
        input=input_parallel,
        weight=self.weight,
        bias=None,
        gradient_accumulation_fusion=self.gradient_accumulation_fusion,
        async_grad_allreduce=False,
        sequence_parallel=False,
    )

    global_args = get_args()
    output_ = output_parallel
    if global_args.optimize_recomp_communication_status < 2:
        if self.sequence_parallel:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        elif self.explicit_expert_comm:  # non-expert only tensor-parallelism
            assert self.skip_bias_add
            output_ = output_parallel
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)
    global_args.optimize_recomp_communication_status = global_args.optimize_recomp_communication_status + 1 \
        if global_args.optimize_recomp_communication_status > 0 \
        else global_args.optimize_recomp_communication_status
    if not self.skip_bias_add:
        output = (output_ + self.bias) if self.bias is not None else output_
        output_bias = None
    else:
        output = output_
        output_bias = self.bias
    return output, output_bias


class RowSeqParallelLinearNoComm(RowSeqParallelLinear):
    @staticmethod
    def forward(ctx, input_, weight, bias, group):
        global_args = get_args()
        world_size = get_tensor_model_parallel_world_size()
        rank = torch.distributed.get_rank(group)
        if global_args.optimize_recomp_communication_status < 2:
            global_args.optimize_recomp_communication_status = global_args.optimize_recomp_communication_status + 1 \
                if global_args.optimize_recomp_communication_status > 0 \
                else global_args.optimize_recomp_communication_status
            return RowSeqParallelLinear.forward(ctx, input_, weight, bias, group)
        else:
            if torch.__version__ > "2.0":
                global_rank = torch.distributed.get_global_rank(group, rank)
                hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(
                    global_rank
                )
            else:
                hcomm_info = group.get_hccl_comm_name(rank)
            ctx.save_for_backward(input_, weight)
            ctx.hcomm_info = hcomm_info
            ctx.world_size = world_size
            ctx.use_bias = bias is not None
            output_ = torch.matmul(input_, weight.t())
            global_args.optimize_recomp_communication_status = global_args.optimize_recomp_communication_status + 1 \
                if global_args.optimize_recomp_communication_status > 0 \
                else global_args.optimize_recomp_communication_status

            return output_[:output_.shape[0] // world_size]

    @staticmethod
    def backward(ctx, grad_output):
        return RowSeqParallelLinear.backward(ctx, grad_output)


class LinearWithGradAccumulationAndAsyncCommunicationPipeExperts(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel,
        grad_output_buffer,
        pipe_experts
    ):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel
        ctx.grad_output_buffer = grad_output_buffer
        ctx.pipe_experts = pipe_experts

        if sequence_parallel:
            global_args = get_args()
            if global_args.use_ascend_mc2 and not pipe_experts:
                from .ascend_turbo.ascend_turbo_cfg import ascend_turbo_cfg
                group = get_tensor_model_parallel_group()
                rank = get_tensor_model_parallel_rank()
                ascend_turbo_cfg.set_world_size(get_tensor_model_parallel_world_size)
                hcomm_info = None

                if torch.__version__ > "2.0":
                    global_rank = torch.distributed.get_global_rank(group, rank)
                    hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)
                else:
                    hcomm_info = group.get_hccl_comm_name(rank)

                x = input.reshape(input.shape[0] * input.shape[1], input.shape[2])
                world_size = ascend_turbo_cfg.get_world_size()
                output, all_gather_grad_output = torch_npu.npu_all_gather_base_mm(
                    x,
                    weight.t(),
                    hcomm_info,
                    world_size,
                    bias=bias,
                    gather_index=0,
                    gather_output=(not ascend_turbo_cfg.all_gather_recomputation)
                )
                output = output.view(
                    output.shape[0] // input.shape[1], input.shape[1], output.shape[1]
                )
                ctx.all_gather_output = all_gather_grad_output
                ctx.world_size = world_size
                ctx.group = group
            elif pipe_experts:
                from mindspeed.moe.async_comm_utils import get_fw_ag_output
                total_input = get_fw_ag_output()[0]
                output = torch.matmul(total_input, weight.t())
            else:
                world_size = get_tensor_model_parallel_world_size()
                dim_size = list(input.size())
                dim_size[0] = dim_size[0] * world_size

                all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
                torch.distributed._all_gather_base(
                    all_gather_buffer, input, group=get_tensor_model_parallel_group()
                )
                total_input = all_gather_buffer
                output = torch.matmul(total_input, weight.t())
        else:
            total_input = input
            output = torch.matmul(total_input, weight.t())

            if bias is not None:
                output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        grad_output_buffer = ctx.grad_output_buffer

        wgrad_compute = True
        if grad_output_buffer is not None:
            grad_output_buffer.append(grad_output)
            wgrad_compute = False

        if wgrad_compute:
            if ctx.sequence_parallel:
                world_size = get_tensor_model_parallel_world_size()
                dim_size = list(input.size())
                dim_size[0] = dim_size[0] * world_size

                if ctx.pipe_experts:
                    all_gather_buffer = torch.empty(dim_size, dtype=input.dtype, device=torch.cuda.current_device())
                else:
                    all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")

                handle = torch.distributed._all_gather_base(
                    all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
                )

                # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
                # gather is scheduled before the input gradient computation
                total_input = all_gather_buffer
            else:
                total_input = input
        grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel and wgrad_compute:
            handle.wait()

        if wgrad_compute:
            grad_output, total_input = prepare_input_tensors_for_wgrad_compute(
                grad_output, total_input
            )

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            assert not ctx.async_grad_allreduce
            dim_size = list(input.size())
            sub_grad_input = torch.empty(
                dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
            )
            # reduce_scatter
            handle = torch.distributed._reduce_scatter_base(
                sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation

        if ctx.gradient_accumulation_fusion:
            if wgrad_compute:
                if weight.main_grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                        total_input, grad_output, weight.main_grad
                    )
                elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                        total_input, grad_output, weight.main_grad
                    )
                else:
                    raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

            if hasattr(weight, 'grad_added_to_main_grad'):
                # When overlap_grad_reduce is True, need to ensure that backward hooks
                # are all run on the main backprop thread to prevent deadlocks. Setup
                # dummy grad_weight tensor to prevent backward hooks from being run
                # in a background thread.
                if getattr(weight, 'zero_out_wgrad', False):
                    grad_weight = torch.zeros(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    grad_weight = torch.empty(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None
        from mindspeed.moe.pipe_experts import get_async_bw_all_gather_count
        if ctx.pipe_experts and get_async_bw_all_gather_count() != 2:
            grad_output.storage().resize_(0)

        if ctx.sequence_parallel:
            handle.wait()
            # Need to return None's as gradient has to flow for all the input arguments
            # provided during forward
            return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None

        if ctx.async_grad_allreduce:
            handle.wait()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel: bool,
    pipe_experts=False,
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.

    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.

    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.

    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calcluation of the
    weight gradients.

    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.

    Args:

        input (torch.Tensor required): input like torch.nn.functional.linear

        weight (torch.Tensor required): weight like torch.nn.functional.linear

        bias (torch.Tensor optional): bias like torch.nn.functional.linear

        gradient_accumulation_fusion (bool required): Perform the gradient
            accumulation fusion, requires the custom CUDA extension
            fused_weight_gradient_mlp_cuda module. To use
            gradient_accumulation_fusion you must install APEX with
            --cpp_ext and --cuda_ext. For example: "pip install
            --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\"
            " Note that the extension requires CUDA>=11. Otherwise, you
            must turn off gradient accumulation fusion."

        async_grad_allreduce (bool required): Do the allreduce of input
            gradients asyncronously with the computation of weight
            gradients. If sequence_parallel is True, this must be
            False, as no all reduce is performed.

    sequence_parallel (bool required): Indicates that sequence
        parallelism is used and thus in the forward pass the input is
        all gathered, and the backward pass the input gradients are
        reduce scattered.

    grad_output_buffer (List[torch.Tensor] optional): Buffer used to save
        output gradients when embedding table wgrad compute is deferred.
        Defaults to None.
    """
    if not linear_with_grad_accumulation_and_async_allreduce.warned:
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
            if sequence_parallel:
                warnings.warn(
                    "When using sequence parallelism it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce.warned = True

            if async_grad_allreduce:
                warnings.warn(
                    "When using async grad allreduce it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup"
                )
                linear_with_grad_accumulation_and_async_allreduce.warned = True

    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel,
        grad_output_buffer,
        pipe_experts
    ]
    return LinearWithGradAccumulationAndAsyncCommunicationPipeExperts.apply(*args)


linear_with_grad_accumulation_and_async_allreduce.warned = False


def parallel_linear_init_wrapper(init_func):
    @wraps(init_func)
    def parallel_linear_init_func(self, *args, pipe_experts: bool = False, **kwargs):
        output = init_func(self, *args, **kwargs)
        self.pipe_experts = pipe_experts
        return output
    return parallel_linear_init_func


def row_parallel_moe(self, input_):
    """Forward of RowParallelLinear

    Args:
        input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

    Returns:
        - output
        - bias
    """

    if self.config._cpu_offloading_context is not None:
        if self.config._cpu_offloading_context.inside_context == True:
            assert (
                    self.config.cpu_offloading == False
            ), "CPU Offloading cannot be enabled while using non-TE modules"

    # Set up backprop all-reduce.
    global_args = get_args()
    if global_args.use_ascend_mc2 and not self.pipe_experts:
        output = Mc2RowSeqParallelLinear.apply(
            input_, self.weight, None, get_tensor_model_parallel_group()
        )

        if not self.skip_bias_add:
            output = output + self.bias if self.bias is not None else output
            output_bias = None
        else:
            output_bias = self.bias

        return output, output_bias

    if self.input_is_parallel:
        input_parallel = input_
    else:
        assert not self.sequence_parallel
        input_parallel = scatter_to_tensor_model_parallel_region(input_)
    # Matrix multiply.
    if not self.weight.requires_grad:
        self._forward_impl = linear_with_frozen_weight
    else:
        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce
    output_parallel = self._forward_impl(
        input=input_parallel,
        weight=self.weight,
        bias=None,
        gradient_accumulation_fusion=self.gradient_accumulation_fusion,
        async_grad_allreduce=False,
        sequence_parallel=False,
        pipe_experts=self.pipe_experts
    )

    # All-reduce across all the partitions or self.pipe_experts
    if self.explicit_expert_comm or self.pipe_experts:
        assert self.skip_bias_add
        output_ = output_parallel
    elif self.sequence_parallel:
        output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
    else:
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
    if not self.skip_bias_add:
        output = (output_ + self.bias) if self.bias is not None else output_
        output_bias = None
    else:
        output = output_
        output_bias = self.bias
    return output, output_bias


def column_parallel_moe(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None):
    """Forward of ColumnParallelLinear

    Args:
        input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        weight (optional): weight tensor to use, compulsory when
            skip_weight_param_allocation is True.

    Returns:
        - output
        - bias

    """
    if weight is None:
        if self.weight is None:
            raise RuntimeError(
                "weight was not supplied to ColumnParallelLinear forward pass "
                "and skip_weight_param_allocation is True."
            )
        weight = self.weight
    else:
        # Check the weight passed in is the correct shape
        expected_shape = (self.output_size_per_partition, self.input_size)
        if weight.shape != expected_shape:
            raise RuntimeError(
                f"supplied weight's shape is {tuple(weight.shape)}, "
                f"not {expected_shape} as expected"
            )

    if self.config._cpu_offloading_context is not None:
        if self.config._cpu_offloading_context.inside_context == True:
            assert (
                self.config.cpu_offloading == False
            ), "CPU Offloading cannot be enabled while using non-TE modules"

    bias = self.bias if not self.skip_bias_add else None

    if (
        self.async_tensor_model_parallel_allreduce
        or self.sequence_parallel
        or self.explicit_expert_comm
    ):
        input_parallel = input_
    else:
        input_parallel = copy_to_tensor_model_parallel_region(input_)

    if self.config.defer_embedding_wgrad_compute:
        self.embedding_activation_buffer.append(input_parallel)

    # Matrix multiply.
    if not weight.requires_grad:
        self._forward_impl = linear_with_frozen_weight
    else:
        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

    output_parallel = self._forward_impl(
        input=input_parallel,
        weight=weight,
        bias=bias,
        gradient_accumulation_fusion=self.gradient_accumulation_fusion,
        async_grad_allreduce=False
        if self.explicit_expert_comm
        else self.async_tensor_model_parallel_allreduce,
        sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
        grad_output_buffer=self.grad_output_buffer
        if self.config.defer_embedding_wgrad_compute
        else None,
        pipe_experts=self.pipe_experts
    )
    if self.gather_output:
        # All-gather across the partitions.
        assert not self.sequence_parallel
        output = gather_from_tensor_model_parallel_region(output_parallel)
    else:
        output = output_parallel
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias


class Mc2RowSeqParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, group):
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None

        from .ascend_turbo.ascend_turbo_cfg import ascend_turbo_cfg
        rank = get_tensor_model_parallel_rank()
        ascend_turbo_cfg.set_world_size(get_tensor_model_parallel_world_size)
        world_size = ascend_turbo_cfg.get_world_size()
        hcomm_info = None

        if torch.__version__ > "2.0":
            global_rank = torch.distributed.get_global_rank(group, rank)
            hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(
                global_rank
            )
        else:
            hcomm_info = group.get_hccl_comm_name(rank)

        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])

        output = torch_npu.npu_mm_reduce_scatter_base(
            x, weight.t(), hcomm_info, world_size, reduce_op="sum", bias=bias
        )

        ctx.hcomm_info = hcomm_info
        ctx.world_size = world_size

        output = output.view(
            output.shape[0] // input_.shape[1], input_.shape[1], output.shape[1]
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        hcomm_info = ctx.hcomm_info
        world_size = ctx.world_size

        grad_output_ = grad_output.reshape(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )

        grad_input, all_gather_grad_output = torch_npu.npu_all_gather_base_mm(
            grad_output_, weight, hcomm_info, world_size, bias=None, gather_index=0
        )
        grad_input = grad_input.view_as(input_)

        total_input = input_
        total_input = total_input.view(
            total_input.shape[0] * total_input.shape[1], total_input.shape[2]
        )
        grad_weight = all_gather_grad_output.t().matmul(total_input)

        is_grad_bias_needed = ctx.needs_input_grad[2]
        if is_grad_bias_needed and ctx.use_bias:
            grad_bias = (
                grad_output.sum(dim=0)
                if grad_output.is_contiguous()
                else grad_output.t().sum(dim=1)

            )
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None
