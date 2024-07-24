# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import contextlib
import functools
import mindspore
import mindtorch

grad_fn_global = None


def ms_forward_and_backward(
        grad_fn,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        config,
        collect_non_loss_data,
        checkpoint_activations_microbatch
):
    """mindspore forward and backward method"""

    output_tensor, params_gradient = grad_fn(
        data_iterator,
        num_microbatches,
        input_tensor,
        collect_non_loss_data,
        checkpoint_activations_microbatch
    )

    params = model.trainable_params()

    for i in range(len(params)):
        if not hasattr(params[i], 'grad') or params[i].grad is None:
            params[i].grad = mindtorch.torch.cast_to_adapter_tensor(params_gradient[i])
        else:
            params[i].grad += mindtorch.torch.cast_to_adapter_tensor(params_gradient[i])
        params[i].main_grad.assign_value(params[i].grad)
    return output_tensor, params_gradient


def forward_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(data_iterator, num_microbatches, input_tensor, collect_non_loss_data,
                checkpoint_activations_microbatch, model):
        from megatron.core.utils import get_attr_wrapped_model
        from megatron.core import parallel_state

        set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
        set_input_tensor(input_tensor)

        if checkpoint_activations_microbatch is None:
            output_tensor, loss_func = fn(data_iterator, model)
        else:
            output_tensor, loss_func = fn(data_iterator, model, checkpoint_activations_microbatch)

        if parallel_state.is_pipeline_last_stage():
            if not collect_non_loss_data:
                output_tensor = loss_func(output_tensor)
                loss, _, loss_reduced = output_tensor
                if num_microbatches != 0:
                    output_tensor = loss / num_microbatches
                forward_data = loss_reduced
            else:
                data = loss_func(output_tensor, non_loss_data=True)
                forward_data = data
        return output_tensor, forward_data

    return wrapper


def forward_backward_no_pipelining(
        *,
        forward_step_func,
        data_iterator,
        model,
        num_microbatches: int,
        seq_length: int,  # unused
        micro_batch_size: int,  # unused
        decoder_seq_length: int = None,  # unused
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
        first_val_step: bool = None,
):
    from megatron.core.utils import get_attr_wrapped_model, get_model_config, get_model_type
    from megatron.core.pipeline_parallel.schedules import forward_step, check_first_val_step
    if isinstance(model, list):
        model = model[0]
    if isinstance(data_iterator, list):
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    forward_step_func = forward_wrapper(forward_step_func)

    def new_forward_step_func(data_iterator, num_microbatches, input_tensor, collect_non_loss_data,
                              checkpoint_activations_microbatch):
        loss, forward_data = forward_step_func(data_iterator, num_microbatches, input_tensor,
                                 collect_non_loss_data, checkpoint_activations_microbatch, model)
        if config.grad_scale_func is not None:
            loss = config.grad_scale_func(loss)
        forward_data_store.append(forward_data)
        return loss

    grad_fn_global = mindspore.grad(new_forward_step_func, 2, model.trainable_params(), has_aux=False)
    total_num_tokens = mindtorch.torch.tensor(0, dtype=mindtorch.torch.int).cuda()
    with no_sync_func():
        for i in range(num_microbatches - 1):
            if forward_only:
                output_tenso, num_tokensr = forward_step(
                    forward_step_func,
                    data_iterator,
                    model,
                    num_microbatches,
                    input_tensor,
                    forward_data_store,
                    config,
                    collect_non_loss_data,
                    is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
                    current_microbatch=i,
                )
            else:
                output_tensor, params_gradient = ms_forward_and_backward(
                    grad_fn_global,
                    data_iterator,
                    model,
                    num_microbatches,
                    input_tensor,
                    config,
                    collect_non_loss_data,
                    None,
                )
    if forward_only:
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            is_first_microbatch=check_first_val_step(first_val_step, forward_only, num_microbatches == 1),
            current_microbatch=num_microbatches - 1,
        )
    else:
        output_tensor, params_gradient = ms_forward_and_backward(
            grad_fn_global,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            config,
            collect_non_loss_data,
            None
        )

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism and layernorm all-reduce for sequence parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )
    if config.timers is not None:
        config.timers('forward-backward').stop()
    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism and layernorm all-reduce for sequence parallelism).
        config.finalize_model_grads_func([model])
    return forward_data_store
