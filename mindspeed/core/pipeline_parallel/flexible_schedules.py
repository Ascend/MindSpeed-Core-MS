# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
from functools import wraps
from typing import Callable, Iterator, List, Optional, Union, Tuple
import torch
import torch.distributed as dist
from megatron import core
from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.utils import get_model_config, get_model_type
from megatron.core.pipeline_parallel.schedules import (
    get_tensor_shapes, 
    forward_step, 
    backward_step, 
    deallocate_output_tensor,
    check_first_val_step
)
from megatron.core.pipeline_parallel.p2p_communication import (
    Shape, 
    _communicate_shapes, 
    _communicate, 
    _batched_p2p_ops,
    _p2p_ops
)
from megatron.core.parallel_state import get_pipeline_model_parallel_group
from mindspeed.core.parallel_state import get_pipeline_parallel_group_for_new_stream


forward_comm_stream = None
backward_comm_stream = None
default_stream = None
scheduler_plan = None


def recv_forward(tensor_shapes, config, group):
    input_tensors = []
    wait_handles = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None or core.parallel_state.is_pipeline_first_stage():
            input_tensor = None
            wait_handle = None
        else:
            if config.timers is not None:
                config.timers('forward-recv', log_level=2).start()
            input_tensor, _, wait_handle = _communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                tensor_shape=tensor_shape,
                config=config,
                group=group,
                wait_on_reqs=False
            )
            if config.timers is not None:
                config.timers('forward-recv').stop()
        input_tensors.append(input_tensor)
        wait_handles.append(wait_handle)
    return input_tensors, wait_handles


def recv_backward(tensor_shapes, config, group):
    output_tensor_grads = []
    wait_handlers = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None or core.parallel_state.is_pipeline_last_stage():
            output_tensor_grad = None
            wait_handle = None
        else:
            if config.timers is not None:
                config.timers('backward-recv', log_level=2).start()
            _, output_tensor_grad, wait_handle = _communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                tensor_shape=tensor_shape,
                config=config,
                group=group,
                wait_on_reqs=False
            )
            if config.timers is not None:
                config.timers('backward-recv').stop()
        output_tensor_grads.append(output_tensor_grad)
        wait_handlers.append(wait_handle)
    return output_tensor_grads, wait_handlers


def send_forward(output_tensors, tensor_shapes, config, group):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None or core.parallel_state.is_pipeline_last_stage():
            continue

        if config.timers is not None:
            config.timers('forward-send', log_level=2).start()
        _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            config=config,
            group=group,
            wait_on_reqs=False
        )
        if config.timers is not None:
            config.timers('forward-send').stop()


def send_backward(input_tensor_grads, tensor_shapes, config, group):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None or core.parallel_state.is_pipeline_first_stage():
            continue

        if config.timers is not None:
            config.timers('backward-send', log_level=2).start()
        _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            config=config,
            group=group,
            wait_on_reqs=False
        )
        if config.timers is not None:
            config.timers('backward-send').stop()


def _communicate(
    *,
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    wait_on_reqs: bool = True,
    group: dist.ProcessGroup = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Args:
        tensor_send_next (torch.Tensor, optional):
            Tensor to send to next rank (no tensor sent if None)

        tensor_send_prev (torch.Tensor, optional):
            Tensor to send to prev rank (no tensor sent if None)

        recv_prev (boolean, required):
            whether tensor should be received from previous rank.

        recv_next (boolean, required):
            whether tensor should be received from next rank.

        tensor_shape (List[int] or torch.Size, required):
            shape of tensor to receive (this method assumes that all
            tensors sent and received in a single function call are
            the same shape).

        wait_on_reqs (boolean, optional, default=False):
            For non-batched p2p communication, wait on each request
            before returning.

    Returns:
        tuple containing

        - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
        - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.

    """
    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    if not config.variable_seq_lengths:
        recv_prev_shape = tensor_shape
        recv_next_shape = tensor_shape
    else:
        recv_prev_shape, recv_next_shape = _communicate_shapes(
            tensor_send_next, tensor_send_prev, recv_prev, recv_next, config
        )

    if recv_prev:
        if config.pipeline_dtype is None:
            raise RuntimeError("pipeline_dtype must be provided if recv_prev is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_prev is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_prev = torch.empty(
            recv_prev_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        )
    if recv_next:
        if config.pipeline_dtype is None:
            raise RuntimeError("dtype must be provided if recv_next is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_next is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_next = torch.empty(
            recv_next_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        )

    # Send tensors in both the forward and backward directions as appropriate.
    if config.use_ring_exchange_p2p:

        def _ring_exchange_wrapper(**kwargs):
            torch.distributed.ring_exchange(**kwargs)
            return []

        p2p_func = _ring_exchange_wrapper
    elif config.batch_p2p_comm:
        if not wait_on_reqs:
            raise AssertionError('wait_on_reqs must be True when use batch_p2p_comm')
        p2p_func = _batched_p2p_ops
    else:
        p2p_func = _p2p_ops

    reqs = p2p_func(
        tensor_send_prev=tensor_send_prev,
        tensor_recv_prev=tensor_recv_prev,
        tensor_send_next=tensor_send_next,
        tensor_recv_next=tensor_recv_next,
        group=group
    )

    if wait_on_reqs and len(reqs) > 0:
        for req in reqs:
            req.wait()
        reqs = None

    if config.batch_p2p_comm and config.batch_p2p_sync:
        # To protect against race condition when using batch_isend_irecv().
        # User should assert that we have a modern enough PyTorch to not need this
        torch.cuda.synchronize()

    return tensor_recv_prev, tensor_recv_next, reqs


def generate_1f1b_scheduler_plan(pp_size, num_micro_batch):
    scheduler_plan_all_stages = {}

    num_warmup_microbatch = [pp_size - r - 1 for r in range(pp_size)]
    num_cooldown_microbatch = num_warmup_microbatch
    num_stable_microbatch = [(num_micro_batch * 2 - num_warmup_microbatch[r] - num_cooldown_microbatch[r]) // 2
                             for r in range(pp_size)]

    forward_count = [1 for _ in range(pp_size)]
    backward_count = [1 for _ in range(pp_size)]

    # warmup
    for pp_rank in range(pp_size):
        key = 'stage{}'.format(pp_rank)
        scheduler_plan_all_stages[key] = []
        for i in range(num_warmup_microbatch[pp_rank]):
            value = 'F{}'.format(forward_count[pp_rank])
            scheduler_plan_all_stages[key].append(value)
            forward_count[pp_rank] += 1

    # stable
    for pp_rank in range(pp_size):
        key = 'stage{}'.format(pp_rank)
        for i in range(num_stable_microbatch[pp_rank]):
            value = 'F{}'.format(forward_count[pp_rank])
            scheduler_plan_all_stages[key].append(value)
            forward_count[pp_rank] += 1

            value = 'B{}'.format(backward_count[pp_rank])
            scheduler_plan_all_stages[key].append(value)
            backward_count[pp_rank] += 1

    # cooldown
    for pp_rank in range(pp_size):
        key = 'stage{}'.format(pp_rank)
        for i in range(num_cooldown_microbatch[pp_rank]):
            value = 'B{}'.format(backward_count[pp_rank])
            scheduler_plan_all_stages[key].append(value)
            backward_count[pp_rank] += 1

    return scheduler_plan_all_stages


def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise.

    """

    if isinstance(model, list):
        if not len(model) == 1:
            raise AssertionError("non-interleaved pipeline parallelism does not support model chunking")
        model = model[0]
    if isinstance(data_iterator, list):
        if not len(data_iterator) == 1:
            raise AssertionError("non-pipeline-parallel schedule does not support model chunking")
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
        - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)

    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(
        rank=rank - 1,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
    )
    send_tensor_shapes = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    def wait_helper(wait_handlers):
        for reqs in wait_handlers:
            if reqs is not None:
                for req in reqs:
                    req.wait()

    global forward_comm_stream
    if forward_comm_stream is None:
        forward_comm_stream = torch.cuda.Stream()

    global backward_comm_stream
    if backward_comm_stream is None:
        backward_comm_stream = torch.cuda.Stream()

    global default_stream
    if default_stream is None:
        default_stream = torch.cuda.default_stream()

    global scheduler_plan
    if scheduler_plan is None:
        scheduler_plan = generate_1f1b_scheduler_plan(parallel_state.get_pipeline_model_parallel_world_size(),
                                                      num_microbatches)
        key = 'stage{}'.format(parallel_state.get_pipeline_model_parallel_rank())
        scheduler_plan = scheduler_plan.get(key)

    config.batch_p2p_comm = False
    fwd_wait_handles, bwd_wait_handles = None, None
    current_tag_id = -1
    for tag in scheduler_plan:
        current_tag_id += 1
        if tag.startswith('F'):
            # Decide to checkpoint all layers' activations of the current micro-batch
            if max_outstanding_backprops is not None:
                checkpoint_activations_microbatch = (
                    current_tag_id % max_outstanding_backprops >= config.num_microbatches_with_partial_activation_checkpoints
                )
            else:
                checkpoint_activations_microbatch = None

            with torch.cuda.stream(forward_comm_stream):
                input_tensor, fwd_wait_handles = recv_forward(
                    recv_tensor_shapes, config, get_pipeline_model_parallel_group()
                )

            wait_helper(fwd_wait_handles)
            output_tensor = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                checkpoint_activations_microbatch,
                check_first_val_step(first_val_step, forward_only, current_tag_id == 0)
            )

            with torch.cuda.stream(forward_comm_stream):
                forward_comm_stream.wait_stream(default_stream)
                send_forward(
                    output_tensor,
                    send_tensor_shapes,
                    config,
                    get_pipeline_model_parallel_group()
                )
                for tensor in output_tensor:
                    if tensor is not None:
                        tensor.record_stream(forward_comm_stream)

            if not forward_only:
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
        else:
            if current_tag_id == len(scheduler_plan) - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            with torch.cuda.stream(backward_comm_stream):
                output_tensor_grads, bwd_wait_handles = recv_backward(
                    send_tensor_shapes, config, get_pipeline_parallel_group_for_new_stream()
                )

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            wait_helper(bwd_wait_handles)
            input_tensor_grad = backward_step(
                input_tensor,
                output_tensor,
                output_tensor_grads,
                model_type,
                config
            )

            with torch.cuda.stream(backward_comm_stream):
                backward_comm_stream.wait_stream(default_stream)
                send_backward(
                    input_tensor_grad,
                    recv_tensor_shapes,
                    config,
                    get_pipeline_parallel_group_for_new_stream()
                )
                for tensor in input_tensor_grad:
                    if tensor is not None:
                        tensor.record_stream(backward_comm_stream)

    if not forward_only:
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func([model])

    return forward_data_store


