# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import operator
import queue
from functools import reduce
import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    is_pipeline_first_stage
)


def gather(input_slice):
    world_size = get_tensor_model_parallel_world_size()
    dim_size = list(input_slice.size())
    dim_size[0] = dim_size[0] * world_size

    required_len = reduce(operator.mul, dim_size, 1)
    all_gather_buffer = torch.empty(
        required_len, dtype=input_slice.dtype, device=torch.cuda.current_device(), requires_grad=False
    )
    all_gather_buffer = all_gather_buffer[0:required_len].view(*dim_size)

    handle = torch.distributed._all_gather_base(
        all_gather_buffer, input_slice, group=get_tensor_model_parallel_group(), async_op=True
    )

    # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
    # gather is scheduled before the input gradient computation
    total_input = all_gather_buffer
    return total_input, handle


class WeightGradStore:

    cache = []
    weight_grad_queue = queue.Queue()
    stored_grads = None
    store_grad_cache = []
    grad_store = []
    sequence_parallel = False

    @classmethod
    def put(cls, total_input, grad_output, weight, sequence_parallel, in_row=False, pipe_experts=False):
        cls.cache.append((total_input, grad_output, weight, sequence_parallel, in_row, pipe_experts))

    @classmethod
    def flush(cls):
        if not is_pipeline_first_stage(True):
            cls.weight_grad_queue.put(cls.cache)
            cls.cache = []

    @classmethod
    def save_grad_output(cls, grad):
        cls.grad_store.append(grad)

    @classmethod
    def overlap_all_gather(cls):
        # Used for grad_output all gather in RowParallel and input all gather in ColumnParallel.
        if len(cls.stored_grads) > 0:
            (input_slice, grad_output_slice, weight, sequence_parallel, in_row, pipe_experts) = cls.stored_grads.pop(0)
            if not sequence_parallel:
                return (input_slice, grad_output_slice, weight, sequence_parallel, in_row, pipe_experts), None
            if not in_row:
                total_input, handle = gather(input_slice)
                grad_output = grad_output_slice
            else:
                if pipe_experts:
                    grad_output_slice = cls.grad_store.pop(0)
                grad_output, handle = gather(grad_output_slice)
                total_input = input_slice
            return (total_input, grad_output, weight, sequence_parallel, in_row, pipe_experts), handle
        else:
            raise Exception("All Gather empty queue.")

    @classmethod
    def overlap_matmul(cls, grad_store_cache):
        total_input, grad_output, weight, _, _, _ = grad_store_cache
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )
        total_input = total_input.view(
            total_input.shape[0] * total_input.shape[1], total_input.shape[2]
        )
        weight.main_grad.data.add_(grad_output.t().matmul(total_input))

    @classmethod
    def pop(cls):
        if is_pipeline_first_stage(True):
            return
        if cls.stored_grads is None:
            cls.stored_grads = cls.weight_grad_queue.get()
            input_slice, grad_output_slice, weight, sequence_parallel, in_row, pipe_experts = cls.stored_grads.pop(0)
            cls.sequence_parallel = sequence_parallel
            if not cls.sequence_parallel:
                grad_output = grad_output_slice
            else:
                if pipe_experts:
                    grad_output_slice = cls.grad_store.pop(0)
                grad_output, handle = gather(grad_output_slice)
                handle.wait()
            cls.store_grad_cache = (input_slice, grad_output, weight, sequence_parallel, in_row, pipe_experts)

        while len(cls.stored_grads) > 0:
            next_grad_cache, handle = cls.overlap_all_gather()
            cls.overlap_matmul(cls.store_grad_cache)
            if handle is not None:
                handle.wait()
            cls.store_grad_cache = next_grad_cache
        if cls.sequence_parallel and cls.weight_grad_queue.qsize() > 0:
            cls.stored_grads = cls.weight_grad_queue.get()
            next_grad_cache, handle = cls.overlap_all_gather()
            cls.overlap_matmul(cls.store_grad_cache)
            if handle is not None:
                handle.wait()
            cls.store_grad_cache = next_grad_cache
        else:
            cls.overlap_matmul(cls.store_grad_cache)
            cls.stored_grads = None
            cls.store_grad_cache = None
        return
