import os

import torch
from megatron import get_args, get_timers, get_num_microbatches
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.utils import unwrap_model
from megatron.core import mpu


def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Set grad to zero.
    for model_chunk in model:
        # If using distributed optimizer, don't zero buffer here; zeroing of buffer is
        # handled automatically by the optimizer after all-gathers finish.
        # Otherwise, zero the buffer.
        model_chunk.zero_grad_buffer(zero_buffer=(not args.use_distributed_optimizer))
    optimizer.zero_grad()

    # Forward pass.
    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False)

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Vision gradients.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    found_silent_flag, update_successful, grad_norm, num_zeros_in_grad = False, False, None, None
    if int(os.getenv('NPU_DETECT', '0')):
        from torch_npu.utils.silent_error import silent_fault_check
        silent_error = silent_fault_check()
        silent_error = torch.tensor(silent_error, dtype=torch.float32).npu()
        torch.distributed.all_reduce(silent_error, op=torch.distributed.ReduceOp.MAX)
        found_silent_flag = (silent_error.item() > 0)

    if not found_silent_flag or not (int(os.getenv('NPU_RECOVERY', '0'))):
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
    else:
        # The silent error is found, and skip the step, then call print_error_plog api to print log in plog.
        import torch_npu
        if hasattr(torch_npu.npu, "print_error_plog"):
            torch_npu.npu.print_error_plog("NPUCheckEvent:AICore Numerical error happen, skip this step!")
    timers('optimizer').stop()

    # Vision momentum.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad