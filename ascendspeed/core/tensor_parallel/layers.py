from functools import wraps

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron.core.tensor_parallel.mappings import (
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region
)
from megatron.core.tensor_parallel.layers import (linear_with_grad_accumulation_and_async_allreduce, 
                                                  linear_with_frozen_weight)
from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


def VocabParallelEmbeddingForward(self, input_):

    if torch.any(
        (input_ < 0) | (input_ >= self.num_embeddings)
    ):
        raise ValueError("An input token is out of bounds of the embedding table")
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
    output_parallel = self.weight[masked_input]

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


def LinearWithGradAccumulationAndAsyncCommunication_backward_wrapper(org_func):
    @wraps(org_func)
    def wrapper_func(*args, **kwargs):
        global_args = get_args()
        use_unpad = global_args.use_unpad
        if not use_unpad:
            return org_func(*args, **kwargs)
        return LinearWithGradAccumulationAndAsyncCommunication_backward(*args, **kwargs)
    return wrapper_func


def LinearWithGradAccumulationAndAsyncCommunication_backward(ctx, grad_output):
    input_data, weight = ctx.saved_tensors
    use_bias = ctx.use_bias

    if ctx.sequence_parallel:
        world_size = get_tensor_model_parallel_world_size()
        dim_size = list(input_data.size())
        dim_size[0] = dim_size[0] * world_size

        all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input_data.dtype, "mpu")
        handle = torch.distributed._all_gather_base(
            all_gather_buffer, input_data, group=get_tensor_model_parallel_group(), async_op=True
        )

        # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
        # gather is scheduled before the input gradient computation
        total_input = all_gather_buffer
    else:
        total_input = input_data
    grad_input = grad_output.matmul(weight)

    if ctx.sequence_parallel:
        handle.wait()

    # Doing gather + slicing during the NeMo forward pass can make this tensor
    # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
    # clones it if it's not contiguous:
    # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
    grad_output = grad_output.contiguous()
    # Convert the tensor shapes to 2D for execution compatibility
    if grad_output.dim() != 2:
        grad_output = grad_output.view(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )
    if total_input.dim() != 2:
        total_input = total_input.view(
            total_input.shape[0] * total_input.shape[1], total_input.shape[2]
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

    if ctx.sequence_parallel:
        handle.wait()
        res = sub_grad_input, grad_weight, grad_bias, None, None, None
        return res

    if ctx.async_grad_allreduce:
        handle.wait()

    res = grad_input, grad_weight, grad_bias, None, None, None
    return res
