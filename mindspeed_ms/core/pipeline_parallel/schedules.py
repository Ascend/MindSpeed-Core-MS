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
""" pipeline process function """

import contextlib
from typing import Union
import mindspore.common.dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore import Tensor, mint, nn, hal, value_and_grad
import mindspore.communication.comm_func as comm_func

from mindspeed_ms.training.global_vars import get_args
from mindspeed_ms.training.grad_handler import GradAccumulator
from mindspeed_ms.core.utils import get_model_config
import mindspeed_ms.core.parallel_state as parallel_state

from .p2p_communication import P2PPrimitive


def _accumulate_grads_func(accumulate_grads, current_micro_grads):
    """ Accumulate grad """
    for i, accumulate_grad in enumerate(accumulate_grads):
        accumulate_grads[i] = mint.add(accumulate_grad, current_micro_grads[i])
    return tuple(accumulate_grads)


# pylint: disable=R1710
def _get_set_hidden_states_parameter(model):
    """ Get the parameter which set by set_input_tensor() """
    param = None
    weight_untrainable = model.untrainable_params()
    for cur_param in weight_untrainable:
        if "set_hidden_states" in cur_param.name:
            param = cur_param
            param.requires_grad = True
            return param
    if param is None:
        raise ValueError("Parameter 'set_hidden_states' is not found.")


class ModelWithLoss(nn.Cell):
    """Model with loss cell """
    def __init__(self,
                 core_forward_func,
                 loss_func,
                 calculate_per_token_loss=False,
                 requires_grad=True):
        super().__init__(auto_prefix=False)
        self.set_grad(requires_grad)
        self.core_forward_func = core_forward_func
        self.loss_func = loss_func
        self.post_process = parallel_state.is_pipeline_last_stage()
        self.calculate_per_token_loss = calculate_per_token_loss
        self.loss_reduced = None

    def construct(self, *args):
        """ Complete forward process """
        output_tensor = self.core_forward_func(*args)
        if self.post_process:
            outputs = self.loss_func(output_tensor)
            if isinstance(outputs, tuple):
                outputs = list(outputs)
                output_length = len(outputs)
                self._set_loss_reduced(outputs.pop())
                if output_length == 3 and not self.calculate_per_token_loss:
                    num_tokens = outputs[1]
                    outputs[0] /= num_tokens
                outputs = tuple(outputs)
            return outputs
        return output_tensor

    def get_loss_reduced(self):
        """ get self.loss_reduced """
        loss_reduced = self.loss_reduced
        self.loss_reduced = None
        return loss_reduced

    def _set_loss_reduced(self, loss_reduced):
        """ set self.loss_reduced """
        if not isinstance(loss_reduced, dict):
            raise TypeError(f"The last output of 'loss_func' must be dict, but got type {type(loss_reduced)}.")
        self.loss_reduced = loss_reduced


def get_forward_backward_func():
    """
    Returns a forward-backward function for training a network with or without pipeline parallelism.

    Returns:
        callable: A forward-backward function that can be used for training the network.

    Raises:
        NotImplementedError: If pipeline parallelism is not implemented yet.
    """
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pipeline_model_parallel_size > 1:
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func


# pylint: disable=W0613
def forward_step(forward_step_func,
                 data_iterator,
                 model,
                 num_microbatches,
                 backward_loss,
                 print_loss,
                 backward_step_cell_list,
                 recv_data=None,
                 calculate_per_token_loss=False,
                 num_tokens_list=None,
                 requires_grad=True,
                 **kwargs):
    """
    Run forward step.
    In first stage, the input_ids will be used from data iterator.
    Otherwise, it will be changes to 'recv_data'.

    The forward process of the model should include the calculation of loss.
    In last stage, the output_tensor is defaulted to the loss value, which will be accumulated

    Outputs:
        Tuple of 3 Tensor, the output_tensor, accumulate_loss and input_data.
        - **output_tensor** (Tensor) -  forward output.
        - **accumulate_loss** (Tensor) -  model final loss value.
        - **input_data** (Tensor) -  The micro input data after correction by recv_data.
    """
    num_tokens = None

    if isinstance(recv_data, (list, tuple)):
        recv_data = recv_data[0]

    if not parallel_state.is_pipeline_first_stage() and recv_data is not None:
        model.set_input_tensor(recv_data)

    # get 'input_tensors'、'forward_step_func' and 'loss_func'
    forward_step_outputs = forward_step_func(data_iterator, model)
    if not isinstance(forward_step_outputs, tuple) or len(forward_step_outputs) != 3:
        raise RuntimeError("The output format of 'forward_step' function must be: "
                           "(input_tensors, core_forward_func, partial(loss_func, ...)), "
                           "where 'input_tensors' is supported as tuple, 'core_forward_func' and "
                           "'partial(loss_func, ...)' are supoprted as callable function.")
    input_tensors, core_forward_step, loss_func = forward_step_outputs

    # init core forward step cell
    core_forward_step_cell = ModelWithLoss(core_forward_step, loss_func, requires_grad)

    # run forward
    output_tensors = core_forward_step_cell(*input_tensors)

    if parallel_state.is_pipeline_last_stage():
        loss_reduced = core_forward_step_cell.get_loss_reduced()
        loss_reduced_is_not_none = 1 if loss_reduced is not None else 0
        # check output
        if isinstance(output_tensors, tuple):
            # pylint: disable=C1801
            outputs_length = len(output_tensors) + loss_reduced_is_not_none
            if outputs_length == 3:
                output_tensor, num_tokens = output_tensors
                if not calculate_per_token_loss:
                    output_tensor /= num_microbatches
            elif outputs_length == 2:
                output_tensor = output_tensors[0] / num_microbatches
            else:
                raise RuntimeError("The outputs number of 'loss_func' must be 2 or 3, "
                                   f"but got {outputs_length} outputs.")
        else:
            output_tensor = output_tensors / num_microbatches
            if not loss_reduced_is_not_none:
                loss_reduced = comm_func.all_reduce(output_tensors, group=parallel_state.get_data_parallel_group())[0]
                loss_reduced = {'lm loss': loss_reduced}

        if calculate_per_token_loss and num_tokens is None:
            raise RuntimeError("When 'calculate_per_token_loss=True', the output of loss_func must be: "
                               "(loss, num_tokens, loss_reduced). But now the 'tokens_nums' is missing.")
        # micro acc process
        backward_loss.append(output_tensor)
        print_loss.append(loss_reduced)
        if num_tokens is not None:
            num_tokens_list.append(num_tokens)
    else:
        output_tensor = output_tensors

    # append 'core_forward_step_cell' for running backward
    backward_step_cell_list.append(core_forward_step_cell)

    input_tensors += (recv_data,)
    return output_tensor, list(input_tensors)


# pylint: disable=W0613
def backward_step(*input_tensor,
                  backward_step_cell,
                  recv_grads,
                  model,
                  weight,
                  scale_sense,
                  backward_loss,
                  accumulate_grads,
                  num_microbatches=None,
                  calculate_per_token_loss=False,
                  num_tokens_list=None,
                  wrap_with_ddp=False,
                  **kwargs):
    """
    Run backward step.
    In last stage, recv_grads is None, and it will be init as all ones tensor for grad accumulation.
    In first stage, The return value of dout is None.

    Outputs:
        Tuple of 2 Tensor, the dout and accumulate_grads.
        - **dout** (Tensor) -  input_ids grads result for bprop.
        - **accumulate_grads** (Tuple) -  weight grads for optimize.
    """

    if isinstance(recv_grads, (list, tuple)):
        recv_grads = recv_grads[0]

    # init dout if in last stage
    if parallel_state.is_pipeline_last_stage():
        # scaling grad base on num_microbatches or tokens nums
        factor = 1.0
        if not calculate_per_token_loss:
            factor /= num_microbatches

        if scale_sense is None:
            scale_sense = Tensor(1.0, mstype.float32)

        # init grad
        recv_grads = mint.ones_like(backward_loss[0]) * F.cast(scale_sense * factor, backward_loss[0].dtype)

        # stop 'tokens_num' gradient bprop
        if num_tokens_list:
            num_tokens_grad = mint.zeros_like(num_tokens_list[0])
            recv_grads = (recv_grads, num_tokens_grad)

    # get grad function input
    # 'recv_data' need to set into the model
    recv_data = input_tensor[-1]
    input_tensor = tuple(input_tensor[:-1])

    # set input tensor for backpropagation
    if not parallel_state.is_pipeline_first_stage():
        model.set_input_tensor(recv_data)

    # get grad function
    grad_fn = C.GradOperation(get_by_list=True, sens_param=True)(backward_step_cell, weight)

    # calculate grads
    weight_grad = grad_fn(*input_tensor, sens=recv_grads)
    weight_grad = list(weight_grad)

    # get dout and weight_grad
    dout = weight_grad.pop(0)
    weight_grad_tuple = tuple(weight_grad)

    # the first stage do not require backpropagation
    if parallel_state.is_pipeline_first_stage():
        dout = None

    # accumulate grads between multi-micro input
    if not wrap_with_ddp and weight[-1].grad is None:
        if accumulate_grads is None:
            accumulate_grads = weight_grad_tuple
        else:
            accumulate_grads = _accumulate_grads_func(list(accumulate_grads), list(weight_grad_tuple))
    else:
        accumulate_grads = weight_grad_tuple

    return dout, accumulate_grads


def forward_backward_no_pipelining(
        *,
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        seq_length,
        micro_batch_size,
        decoder_seq_length=None,
        forward_only=False,
        collect_non_loss_data=False,
        first_val_step=None,
):
    """ Run forward and backward without pipelining. """
    if decoder_seq_length is not None:
        raise NotImplementedError("decoder_seq_length is not supported for now.")
    if collect_non_loss_data:
        raise NotImplementedError("collect_non_loss_data is not supported for now.")
    if first_val_step is not None:
        raise NotImplementedError("first_val_step is not supported for now.")

    if isinstance(model, nn.CellList):
        if len(model) > 1:
            raise TypeError("'forward_backward_no_pipeline' does not support model chunking.")
        model = model[0]
    if isinstance(data_iterator, (list, tuple)):
        if len(data_iterator) != 1:
            raise ValueError("non-pipeline-parallel schedule does not support model chunking.")
        data_iterator = data_iterator[0]

    # get config
    args = get_args()
    config = get_model_config(model)

    forward_data_store = []
    num_tokens_list = []
    grads = None

    grad_position = None
    # when using ZeRO3 optimizer parallelism, gradient with respect to inputs and weights
    if config.zero_level == "z3":
        grad_position = 0

    if args.wrap_with_ddp:
        no_sync_func = model.no_sync
    else:
        no_sync_func = contextlib.nullcontext

    def forward_with_loss_scale(forward_step_func, data_iterator, loss_scale=None):
        # get 'input_tensors', 'forward_step_func' and 'loss_func'
        input_tensors, core_forward_step, loss_func = forward_step_func(data_iterator, model)

        # run forward
        lm_output = core_forward_step(*input_tensors)
        outputs = loss_func(lm_output)
        if isinstance(outputs, tuple):
            outputs = list(outputs)
            if len(outputs) == 3 and not args.calculate_per_token_loss:
                num_tokens = outputs[1]
                outputs[0] /= num_tokens
            outputs[0] /= num_microbatches
            if loss_scale is not None:
                outputs[0] = mint.mul(outputs[0], loss_scale.astype(outputs[0].dtype))
            outputs = tuple(outputs)
        else:
            outputs /= num_microbatches
            if loss_scale is not None:
                outputs = mint.mul(outputs, loss_scale.astype(outputs.dtype))
        return outputs

    forward_backward_once_func = value_and_grad(
        forward_with_loss_scale, grad_position=grad_position, weights=model.trainable_params(), has_aux=True
    )

    # if overlap_grad_reduce, grad will be accumulate in grad buffer
    if num_microbatches > 1 and not args.wrap_with_ddp:
        grad_accumulator = GradAccumulator(num_microbatches, op="sum")

    # get loss scale
    scale_sense = config.loss_scaler.scale_value if config.loss_scaler is not None else Tensor(1, mstype.float32)

    if args.wrap_with_ddp:
        no_sync_func = model.no_sync
    else:
        no_sync_func = contextlib.nullcontext

    def foward_backward_on_microbatch():
        nonlocal grads

        # run forward and backward
        outputs, micro_grads = forward_backward_once_func(
            forward_step_func, data_iterator, loss_scale=scale_sense,
        )

        if isinstance(outputs, tuple):
            report_loss = outputs[-1]
            if len(outputs) == 3:
                num_tokens_list.append(outputs[1])
        else:
            report_loss = comm_func.allreduce(outputs, group=parallel_state.get_data_parallel_group())[0]
            report_loss = {'lm loss': report_loss}
        if scale_sense is not None:
            for key, values in report_loss.items():
                if isinstance(values, tuple):
                    values = list(values)
                    for idx in range(len(values)):
                        values[idx] = mint.div(values[idx], scale_sense)
                    values = tuple(values)
                else:
                    values = mint.div(values, scale_sense)
                report_loss[key] = values
        forward_data_store.append(report_loss)

        if grad_position == 0:
            micro_grads = micro_grads[1]

        # accumulate grads
        if num_microbatches > 1 and not args.wrap_with_ddp:
            grads = grad_accumulator(micro_grads)
        else:
            grads = micro_grads

    # trigger dp reduce only on last step
    with no_sync_func():
        for _ in range(num_microbatches - 1):
            foward_backward_on_microbatch()
    foward_backward_on_microbatch()

    if not forward_only:
        # finalize ddp grad reduce
        if args.wrap_with_ddp:
            model.final_grad_reduce()

        if args.calculate_per_token_loss:
            grads = scale_gradients_for_per_token(grads, model, num_tokens_list, args.wrap_with_ddp)

    return forward_data_store, grads


# pylint: disable=C0103
def forward_backward_pipelining_with_interleaving(
        *,
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        seq_length,
        micro_batch_size,
        decoder_seq_length=None,
        forward_only=False,
        collect_non_loss_data=False,
        first_val_step=None
):
    """ Pipeline with interleaving wrapper for split model run forward and backward """
    if not isinstance(model, nn.CellList):
        raise TypeError("The 'model' input of 'forward_backward_pipelining_with_interleaving' must be nn.CellList. "
                        "Here it is recommended to use the 'get_model' function to build the model")
    if not isinstance(data_iterator, (list, tuple)):
        raise TypeError("The 'data_iterator' input of 'forward_backward_pipelining_with_interleaving' "
                        "must be list or tuple.")
    if decoder_seq_length is not None:
        raise NotImplementedError("decoder_seq_length is not supported for now.")
    if collect_non_loss_data:
        raise NotImplementedError("collect_non_loss_data is not supported for now.")
    if first_val_step is not None:
        raise NotImplementedError("first_val_step is not supported for now.")
    if forward_only:
        raise NotImplementedError("'forward_only' input of pipeline interleaved is not supported for now.")

    args = get_args()
    config = get_model_config(model[0])

    # set grad
    requires_grad = True
    if forward_only:
        requires_grad = False
    for sub_model in model:
        sub_model.set_grad(requires_grad=requires_grad)

    # init p2p class
    p2p_primitive = P2PPrimitive(config=config)

    # record sync model chunk id
    synchronized_model_chunks = set()

    # get model weights and merge `set_hidden_states` parameter
    weights = [sub_model.trainable_params() for sub_model in model]
    set_hidden_states_parameters = []
    for i, _ in enumerate(weights):
        set_hidden_states_parameter = _get_set_hidden_states_parameter(model[i])
        weights[i].insert(0, set_hidden_states_parameter)
        set_hidden_states_parameters.append(set_hidden_states_parameter)

    # get config value
    scale_sense = config.loss_scaler.scale_value if config.loss_scaler is not None else None
    hidden_size = config.hidden_size
    use_sequence_parallel = config.sequence_parallel
    overlap_p2p_comm = config.overlap_p2p_comm
    calculate_per_token_loss = config.calculate_per_token_loss
    data_layout = args.data_layout
    wrap_with_ddp = args.wrap_with_ddp
    overlap_grad_reduce = args.overlap_grad_reduce
    delay_grad_reduce = args.delay_grad_reduce

    # correct tensor shape if use seq parallel or context parallel
    tensor_shape = correct_p2p_shape(seq_length, hidden_size, micro_batch_size, data_layout, use_sequence_parallel)[0]

    # save each forward process input data for running backward
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]
        accumulate_grads_list = [None] * len(model)
    input_tensors = [[] for _ in range(len(model))]
    backward_step_cell_list = [[] for _ in range(len(model))]
    backward_loss = []
    print_loss = []
    num_tokens_list = []

    # warm up process
    pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    num_model_chunks = len(model)
    total_num_microbatches = num_microbatches * num_model_chunks
    if num_microbatches % pp_world_size != 0:
        raise RuntimeError("When using pipeline with interleaved, "
                           "the 'num_microbatches' must be divisible by pipeline world size.")
    if total_num_microbatches == pp_world_size:
        warm_up_steps = total_num_microbatches
        all_warmup_steps = True
    else:
        warm_up_steps = 2 * (pp_world_size - pp_rank - 1)
        warm_up_steps += (num_model_chunks - 1) * pp_world_size
        warm_up_steps = min(warm_up_steps, total_num_microbatches)
        all_warmup_steps = False

    def disable_grad_sync():
        """ disable asynchronous grad reductions """
        for sub_model in model:
            sub_model.enable_sync(False)

    def enable_grad_sync():
        """ enable asynchronous grad reductions """
        for sub_model in model:
            sub_model.enable_sync(True)

    def forward_step_helper(microbatch_id):
        """ run forward helper function """
        # get model vpp rank
        model_chunk_id = get_model_chunk_id(microbatch_id, pp_world_size, num_model_chunks, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # get forward micro data and recv_data
        input_tensor = input_tensors[model_chunk_id][-1]

        # run forward
        output_tensor, micro_input_data = forward_step(forward_step_func,
                                                       data_iterator=data_iterator[model_chunk_id],
                                                       model=model[model_chunk_id],
                                                       num_microbatches=num_microbatches,
                                                       backward_loss=backward_loss,
                                                       print_loss=print_loss,
                                                       backward_step_cell_list=backward_step_cell_list[model_chunk_id],
                                                       recv_data=input_tensor,
                                                       calculate_per_token_loss=calculate_per_token_loss,
                                                       num_tokens_list=num_tokens_list,
                                                       requires_grad=requires_grad)
        input_tensors[model_chunk_id][-1] = micro_input_data

        if forward_only:
            input_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(microbatch_id):
        """ run forward helper function """
         # get model vpp rank
        model_chunk_id = get_model_chunk_id(microbatch_id, pp_world_size, num_model_chunks, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # get micro data and recv_grads
        input_tensor = input_tensors[model_chunk_id].pop(0)
        recv_grads = output_tensor_grads[model_chunk_id].pop(0)
        backward_step_cell = backward_step_cell_list[model_chunk_id].pop(0)

        # enable ddp grad sync before backward_step
        is_last_microbatch_cur_chunk = is_last_microbatch_for_model_chunk(microbatch_id,
                                                                          pp_world_size,
                                                                          num_model_chunks,
                                                                          total_num_microbatches)
        if wrap_with_ddp and not delay_grad_reduce and is_last_microbatch_cur_chunk:
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        # run backward
        dout, accumulate_grads = backward_step(*input_tensor,
                                               backward_step_cell=backward_step_cell,
                                               recv_grads=recv_grads,
                                               model=model[model_chunk_id],
                                               weight=weights[model_chunk_id],
                                               scale_sense=scale_sense,
                                               backward_loss=backward_loss,
                                               accumulate_grads=accumulate_grads_list[model_chunk_id],
                                               num_microbatches=num_microbatches,
                                               calculate_per_token_loss=calculate_per_token_loss,
                                               num_tokens_list=num_tokens_list,
                                               wrap_with_ddp=wrap_with_ddp)
        accumulate_grads_list[model_chunk_id] = accumulate_grads

        # if delay_grad_reduce=True, reduce all bucket grad after backward_step
        if wrap_with_ddp and overlap_grad_reduce:
            if delay_grad_reduce:
                grad_sync_microbatch_id = microbatch_id - pp_world_size
                is_last_microbatch_sync_chunk = is_last_microbatch_for_model_chunk(grad_sync_microbatch_id,
                                                                                   pp_world_size,
                                                                                   num_model_chunks,
                                                                                   total_num_microbatches)
                if grad_sync_microbatch_id >= 0 and is_last_microbatch_sync_chunk:
                    grad_sync_chunk_id = get_model_chunk_id(grad_sync_microbatch_id,
                                                            pp_world_size,
                                                            num_model_chunks,
                                                            forward=False)
                    enable_grad_sync()
                    model[grad_sync_chunk_id].issue_grad_reduce()
                    synchronized_model_chunks.add(grad_sync_chunk_id)
            disable_grad_sync()
        return dout

    forward_reqs = None
    backward_reqs = None
    input_tensor = None

    # no sync for ddp model
    if wrap_with_ddp:
        disable_grad_sync()

    # set virtual rank
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    # get first iteration micro data input
    input_tensors[0].append(p2p_primitive.recv_forward(tensor_shape))

    # warm up process
    for i in range(warm_up_steps):
        # if overlap_p2p_comm is True, wait for comm stream
        if forward_reqs is not None:
            # pylint: disable=E1133
            for req in forward_reqs:
                req.wait()

        # run warm up forward
        ouput_tensor = forward_step_helper(i)

        # decide communication operation
        recv_prev = True
        next_model_chunk_id = get_model_chunk_id(i + 1, pp_world_size, num_model_chunks, forward=True)
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True) and next_model_chunk_id == 0:
            recv_prev = False
        if i == total_num_microbatches - 1:
            recv_prev = False
        if parallel_state.is_pipeline_last_stage():
            ouput_tensor = None

        # warm up send and recv
        if not overlap_p2p_comm:
            if i == warm_up_steps - 1 and not forward_only and not all_warmup_steps:
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                else:
                    recv_next = True
                input_tensor, recv_grads = \
                    p2p_primitive.send_forward_backward_recv_forward_backward(ouput_tensor,
                                                                              None,
                                                                              recv_prev=recv_prev,
                                                                              recv_next=recv_next,
                                                                              tensor_shape=tensor_shape)
                output_tensor_grads[-1].append(recv_grads)
            else:
                input_tensor = p2p_primitive.send_forward_recv_forward(ouput_tensor,
                                                                       recv_prev=recv_prev,
                                                                       tensor_shape=tensor_shape)
        else:
            input_tensor, forward_reqs = p2p_primitive.send_forward_recv_forward(
                ouput_tensor,
                recv_prev,
                tensor_shape,
                overlap_p2p_comm=True
            )
            if i == warm_up_steps - 1 and not forward_only and not all_warmup_steps:
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                else:
                    recv_next = True
                recv_grads, backward_reqs = p2p_primitive.send_backward_recv_backward(
                    None,
                    recv_next,
                    tensor_shape,
                    overlap_p2p_comm
                )
                output_tensor_grads[-1].append(recv_grads)
        # slice micro input data for next iteration
        input_tensors[next_model_chunk_id].append(input_tensor)

    # 1F1B process
    steady_steps = total_num_microbatches - warm_up_steps
    for i in range(steady_steps):
        forward_i = i + warm_up_steps
        if overlap_p2p_comm:
            # wait for forward comm op complete
            if forward_reqs is not None:
                for req in forward_reqs:
                    req.wait()

            # run forward
            ouput_tensor = forward_step_helper(forward_i)

            # set forward virtual id
            forward_model_chunk_id = get_model_chunk_id(forward_i, pp_world_size, num_model_chunks, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
            if parallel_state.is_pipeline_last_stage():
                ouput_tensor = None

            # decide forward communication operation
            recv_prev = True
            next_forward_model_chunk_id = get_model_chunk_id(forward_i + 1,
                                                             pp_world_size,
                                                             num_model_chunks,
                                                             forward=True)
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                if next_forward_model_chunk_id == 0:
                    recv_prev = False

            # if running last micro batch data, do not recv anything
            if i == steady_steps - 1:
                recv_prev = False

            # send forward, recv forward
            input_tensor, forward_reqs = p2p_primitive.send_forward_recv_forward(
                ouput_tensor,
                recv_prev,
                tensor_shape,
                overlap_p2p_comm=True
            )

            # wait for backward comm op complete
            if backward_reqs is not None:
                for req in backward_reqs:
                    req.wait()

            # run backward
            dout = backward_step_helper(i)

            # set backward virtual id
            backward_model_chunk_id = get_model_chunk_id(i, pp_world_size, num_model_chunks, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            if parallel_state.is_pipeline_first_stage():
                dout = None

            # recv backward
            recv_next = True
            next_backward_model_chunk_id = get_model_chunk_id(i + 1,
                                                              pp_world_size,
                                                              num_model_chunks,
                                                              forward=False)
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == num_model_chunks - 1:
                    recv_next = False

            # send backward, recv backward
            recv_grads, backward_reqs = p2p_primitive.send_backward_recv_backward(
                dout,
                recv_next,
                tensor_shape,
                overlap_p2p_comm=True
            )
        else:
            # run forward
            ouput_tensor = forward_step_helper(forward_i)

            # run backward
            dout = backward_step_helper(i)

            # set forward virtual id
            forward_model_chunk_id = get_model_chunk_id(forward_i, pp_world_size, num_model_chunks, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
            if parallel_state.is_pipeline_last_stage():
                ouput_tensor = None

            # set backward virtual id
            backward_model_chunk_id = get_model_chunk_id(i, pp_world_size, num_model_chunks, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            if parallel_state.is_pipeline_first_stage():
                dout = None

            # decide communication operation
            # recv forward
            recv_prev = True
            next_forward_model_chunk_id = get_model_chunk_id(forward_i + 1,
                                                             pp_world_size,
                                                             num_model_chunks,
                                                             forward=True)
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                if next_forward_model_chunk_id == 0:
                    recv_prev = False

            # recv backward
            recv_next = True
            next_backward_model_chunk_id = get_model_chunk_id(i + 1,
                                                              pp_world_size,
                                                              num_model_chunks,
                                                              forward=False)
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == num_model_chunks - 1:
                    recv_next = False

            # if running last micro batch data, do not recv anything
            if i == steady_steps - 1:
                recv_prev = False

            input_tensor, recv_grads = \
                p2p_primitive.send_forward_backward_recv_forward_backward(ouput_tensor,
                                                                          dout,
                                                                          recv_prev=recv_prev,
                                                                          recv_next=recv_next,
                                                                          tensor_shape=tensor_shape)
        if not i == steady_steps - 1:
            # save comm tensor for next 1F1B iteration
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        output_tensor_grads[next_backward_model_chunk_id].append(recv_grads)

    # wait backward comm stream
    if overlap_p2p_comm and backward_reqs is not None:
        # pylint: disable=E1133
        for req in backward_reqs:
            req.wait()

    # recv grad for running cooldown
    if all_warmup_steps:
        recv_grads = p2p_primitive.recv_backward(tensor_shape)
        output_tensor_grads[-1].append(recv_grads)

    # cooldown process
    for i in range(steady_steps, total_num_microbatches):
        dout = backward_step_helper(i)
        next_backward_model_chunk_id = get_model_chunk_id(i + 1,
                                                          pp_world_size,
                                                          num_model_chunks,
                                                          forward=False)
        # decide communication operation
        recv_next = True
        if parallel_state.is_pipeline_last_stage(ignore_virtual=True) and \
            next_backward_model_chunk_id == num_model_chunks - 1:
            recv_next = False
        if i == total_num_microbatches - 1:
            recv_next = False
        recv_grads = p2p_primitive.send_backward_recv_backward(dout,
                                                               recv_next=recv_next,
                                                               tensor_shape=tensor_shape)
        output_tensor_grads[next_backward_model_chunk_id].append(recv_grads)

    # get all model chunk parameters
    merge_weights = []
    for model_chunk_params in weights:
        # remove 'set_hidden_states' parameter
        merge_weights.extend(model_chunk_params[1:])
    del weights

    # merge grads
    all_model_chunk_grads = None
    for model_chunk_grad in accumulate_grads_list:
        if all_model_chunk_grads is None:
            all_model_chunk_grads = model_chunk_grad
        else:
            all_model_chunk_grads += model_chunk_grad
    del accumulate_grads_list

    if wrap_with_ddp:
        # grad sync for remaining model chunks
        enable_grad_sync()
        if overlap_grad_reduce and delay_grad_reduce:
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    model[model_chunk_id].issue_grad_reduce()
                    synchronized_model_chunks.add(model_chunk_id)

        # finish grad sync
        for model_chunk_id in range(num_model_chunks):
            model[model_chunk_id].final_grad_reduce()

    # Reduce and scale grads
    all_model_chunk_grads = reduce_and_scale_grads(list(all_model_chunk_grads),
                                                   merge_weights,
                                                   model,
                                                   num_tokens_list,
                                                   wrap_with_ddp,
                                                   calculate_per_token_loss)

    # return forward data
    forward_data_store = print_loss

    # reset set_hidden_states attr
    for cur_set_hidden_states_param in set_hidden_states_parameters:
        cur_set_hidden_states_param.requires_grad = False

    return forward_data_store, all_model_chunk_grads


# pylint: disable=C0103
def forward_backward_pipelining_without_interleaving(
        *,
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        seq_length,
        micro_batch_size,
        decoder_seq_length=None,
        forward_only=False,
        collect_non_loss_data=False,
        first_val_step=None
):
    """ Pipeline 1F1B wrapper for split model run forward and backward """
    if decoder_seq_length is not None:
        raise NotImplementedError("decoder_seq_length is not supported for now.")
    if collect_non_loss_data:
        raise NotImplementedError("collect_non_loss_data is not supported for now.")
    if first_val_step is not None:
        raise NotImplementedError("first_val_step is not supported for now.")

    if isinstance(model, nn.CellList):
        if len(model) > 1:
            raise TypeError("'forward_backward_pipelining_without_interleaving' does not support model chunking.")
        model = model[0]
    if isinstance(data_iterator, (list, tuple)):
        if len(data_iterator) > 1:
            raise TypeError("The 'data_iterator' input of 'forward_backward_pipelining_without_interleaving' "
                            "can't be list or tuple.")
        data_iterator = data_iterator[0]

    args = get_args()
    config = get_model_config(model)

    # set grad
    requires_grad = True
    if forward_only:
        requires_grad = False
    model.set_grad(requires_grad=requires_grad)

    # init p2p class
    p2p_primitive = P2PPrimitive(config=config)

    # get model weights and merge `set_hidden_states` parameter
    weights = model.trainable_params()
    set_hidden_states_param = _get_set_hidden_states_parameter(model)
    weights.insert(0, set_hidden_states_param)

    # get config value
    scale_sense = config.loss_scaler.scale_value if config.loss_scaler is not None else None
    hidden_size = config.hidden_size
    use_sequence_parallel = config.sequence_parallel
    calculate_per_token_loss = config.calculate_per_token_loss
    data_layout = args.data_layout
    wrap_with_ddp = args.wrap_with_ddp
    overlap_grad_reduce = args.overlap_grad_reduce
    delay_grad_reduce = args.delay_grad_reduce

    # correct tensor shape if use seq parallel or context parallel
    recv_tensor_shapes = correct_p2p_shape(seq_length, hidden_size, \
                                           micro_batch_size, data_layout, use_sequence_parallel)
    send_tensor_shapes = correct_p2p_shape(seq_length, hidden_size, \
                                           micro_batch_size, data_layout, use_sequence_parallel)

    # save each forward process input data for running backward
    if not forward_only:
        input_tensors = []
    backward_step_cell_list = []
    backward_loss = []
    print_loss = []
    num_tokens_list = []

    # no sync for ddp model
    if wrap_with_ddp:
        model.enable_sync(False)

    # get warm up stage steps
    pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    warm_up_steps = min(pp_world_size - pp_rank - 1, num_microbatches)
    input_tensor = None

    # warm up process
    for i in range(warm_up_steps):
        # if is not first stage, get forward input tensor for model
        input_tensor = recv_forward(recv_tensor_shapes, p2p_primitive)

        # run forward
        output_tensor, micro_input_data = forward_step(forward_step_func=forward_step_func,
                                                       data_iterator=data_iterator,
                                                       model=model,
                                                       num_microbatches=num_microbatches,
                                                       backward_loss=backward_loss,
                                                       print_loss=print_loss,
                                                       backward_step_cell_list=backward_step_cell_list,
                                                       recv_data=input_tensor,
                                                       calculate_per_token_loss=calculate_per_token_loss,
                                                       num_tokens_list=num_tokens_list,
                                                       requires_grad=requires_grad)

        # save micro input data for backward
        if not forward_only:
            input_tensors.append(micro_input_data)

        # send forward result to next stage
        send_forward(output_tensor, send_tensor_shapes, p2p_primitive)

    # prepare input data for 1F1B
    steady_steps = num_microbatches - warm_up_steps
    if steady_steps > 0:
        input_tensor = recv_forward(recv_tensor_shapes, p2p_primitive)

    # 1F1B process
    accumulate_grads = None
    for i in range(steady_steps):
        output_tensor, micro_input_data = forward_step(forward_step_func=forward_step_func,
                                                       data_iterator=data_iterator,
                                                       model=model,
                                                       num_microbatches=num_microbatches,
                                                       backward_loss=backward_loss,
                                                       print_loss=print_loss,
                                                       backward_step_cell_list=backward_step_cell_list,
                                                       recv_data=input_tensor,
                                                       calculate_per_token_loss=calculate_per_token_loss,
                                                       num_tokens_list=num_tokens_list,
                                                       requires_grad=requires_grad)

        if forward_only:
            # only send forward result
            send_forward(output_tensor, send_tensor_shapes, p2p_primitive)

            # recv data from prev stage
            if not i == steady_steps - 1:
                input_tensor = recv_forward(recv_tensor_shapes, p2p_primitive)
        else:
            recv_grads = send_forward_recv_backward(output_tensor, send_tensor_shapes, p2p_primitive)
            input_tensors.append(micro_input_data)

            # bprop func need forward's input data
            input_tensor = input_tensors.pop(0)
            backward_step_cell = backward_step_cell_list.pop(0)

            # enable sync for ddp in the first pipeline stage
            if wrap_with_ddp and warm_up_steps == 0 and i == steady_steps - 1:
                if not delay_grad_reduce or pp_rank == 0:
                    model.enable_sync(True)

            # run backward
            dout, accumulate_grads = backward_step(*input_tensor,
                                                   backward_step_cell=backward_step_cell,
                                                   recv_grads=recv_grads,
                                                   model=model,
                                                   weight=weights,
                                                   scale_sense=scale_sense,
                                                   backward_loss=backward_loss,
                                                   accumulate_grads=accumulate_grads,
                                                   num_microbatches=num_microbatches,
                                                   calculate_per_token_loss=calculate_per_token_loss,
                                                   num_tokens_list=num_tokens_list,
                                                   wrap_with_ddp=wrap_with_ddp)

            if i == steady_steps - 1:
                input_tensor = None
                send_backward(dout, recv_tensor_shapes, p2p_primitive)
            else:
                input_tensor = send_backward_recv_forward(dout, recv_tensor_shapes, p2p_primitive)

    if not forward_only:
        # cooldown process
        cooldown_steps = warm_up_steps
        for i in range(cooldown_steps):
            # enable sync for ddp in the first pipeline stage
            if wrap_with_ddp and i == cooldown_steps - 1:
                if not delay_grad_reduce or pp_rank == 0:
                    model.enable_sync(True)

            # run backward
            input_tensor = input_tensors.pop(0)
            backward_step_cell = backward_step_cell_list.pop(0)
            recv_grads = recv_backward(send_tensor_shapes, p2p_primitive)
            dout, accumulate_grads = backward_step(*input_tensor,
                                                   backward_step_cell=backward_step_cell,
                                                   recv_grads=recv_grads,
                                                   model=model,
                                                   weight=weights,
                                                   scale_sense=scale_sense,
                                                   backward_loss=backward_loss,
                                                   accumulate_grads=accumulate_grads,
                                                   num_microbatches=num_microbatches,
                                                   calculate_per_token_loss=calculate_per_token_loss,
                                                   num_tokens_list=num_tokens_list,
                                                   wrap_with_ddp=wrap_with_ddp)
            send_backward(dout, recv_tensor_shapes, p2p_primitive)

        # ddp grad sync
        if wrap_with_ddp and not pp_rank == 0:
            model.enable_sync(True)
            if overlap_grad_reduce and delay_grad_reduce:
                model.issue_grad_reduce()
        if wrap_with_ddp:
            model.final_grad_reduce()

        # Reduce and scale grads
        weights.pop(0)
        accumulate_grads = reduce_and_scale_grads(list(accumulate_grads),
                                                  weights,
                                                  model,
                                                  num_tokens_list,
                                                  wrap_with_ddp,
                                                  calculate_per_token_loss)

    # return forward data
    forward_data_store = print_loss

    # reset set_hidden_states attr
    set_hidden_states_param.requires_grad = False

    return forward_data_store, accumulate_grads


def is_last_microbatch_for_model_chunk(microbatch_id, pp_world_size, num_model_chunks, total_num_microbatches):
    """ check if micro iteration is last by microbatch_id """
    microbatch_group_size = pp_world_size * num_model_chunks
    num_microbatch_groups = total_num_microbatches // microbatch_group_size
    microbatch_group_id = microbatch_id // microbatch_group_size
    microbatch_id_in_group = microbatch_id % microbatch_group_size
    if microbatch_group_id == num_microbatch_groups - 1:
        return microbatch_id_in_group % pp_world_size == pp_world_size - 1
    return False


def get_model_chunk_id(microbatch_id, pp_world_size, num_model_chunks, forward):
    """ get model chunk id by microbatch_id """
    micro_group = microbatch_id % (pp_world_size * num_model_chunks)
    model_chunk_id = micro_group // pp_world_size
    if not forward:
        model_chunk_id = num_model_chunks - model_chunk_id - 1
    return model_chunk_id


def correct_p2p_shape(seq_length, hidden_size, micro_batch_size, data_layout, use_sequence_parallel=False):
    """
    Correct right tensor shape under context parallel or sequence parallel.
    """
    seq_length = seq_length // parallel_state.get_context_parallel_world_size()
    if use_sequence_parallel:
        seq_length = seq_length // parallel_state.get_tensor_model_parallel_world_size()
    if data_layout == "BSH":
        return ((micro_batch_size, seq_length, hidden_size),)
    return ((seq_length, micro_batch_size, hidden_size),)


def recv_forward(tensor_shapes: Union[tuple, list],
                 p2p: P2PPrimitive):
    """ Recv forward output tensor from prev rank in pipeline. """
    tensor_shape = tensor_shapes
    if isinstance(tensor_shapes[0], (tuple, list)):
        tensor_shape = tensor_shape[0]

    if tensor_shape is None:
        recv_tensor = None
    else:
        recv_tensor = p2p.recv_forward(tensor_shape)
    recv_tensor = (recv_tensor,) if recv_tensor is not None else None
    return recv_tensor


def send_forward(input_tensors: Union[Tensor, list],
                 tensor_shapes: Union[tuple, list],
                 p2p: P2PPrimitive):
    """ Send forward output tensor to next rank in pipeline. """
    tensor_shape = tensor_shapes
    if isinstance(tensor_shapes[0], (tuple, list)):
        tensor_shape = tensor_shape[0]

    if tensor_shape is not None:
        p2p.send_forward(input_tensors)


def send_backward(input_grads: Union[Tensor, list],
                  tensor_shapes: Union[tuple, list],
                  p2p: P2PPrimitive):
    """ Send backward output tensor to next rank in pipeline. """
    tensor_shape = tensor_shapes
    if isinstance(tensor_shapes[0], (tuple, list)):
        tensor_shape = tensor_shape[0]

    if tensor_shape is not None:
        p2p.send_backward(input_grads)


def recv_backward(tensor_shapes: Union[tuple, list],
                  p2p: P2PPrimitive):
    """ recv backward dout tensor from next rank in pipeline. """
    tensor_shape = tensor_shapes
    if isinstance(tensor_shapes[0], (tuple, list)):
        tensor_shape = tensor_shape[0]

    recv_grads = ()
    if tensor_shape is None:
        recv_grads += (None,)
    else:
        recv_grads += (p2p.recv_backward(tensor_shape),)
    return recv_grads


def send_forward_recv_backward(input_tensors: Union[Tensor, list],
                               tensor_shapes: Union[tuple, list],
                               p2p: P2PPrimitive):
    """ Send forward output and recv backward dout from next rank in pipeline."""
    tensor_shape = tensor_shapes
    input_tensor = input_tensors

    if isinstance(tensor_shape[0], (tuple, list)):
        tensor_shape = tensor_shape[0]

    if isinstance(input_tensor, (tuple, list)):
        input_tensor = input_tensor[0]

    if tensor_shape is None:
        recv_grads = None
    else:
        recv_grads = p2p.send_forward_recv_backward(input_tensor, tensor_shape)

    recv_grads = (recv_grads,) if recv_grads is not None else None
    return recv_grads


def send_backward_recv_forward(input_grads: Union[Tensor, list],
                               tensor_shapes: Union[tuple, list],
                               p2p: P2PPrimitive):
    """ Send backward grad and recv forward output from prev rank in pipeline."""
    tensor_shape = tensor_shapes
    input_grad = input_grads
    if isinstance(tensor_shape[0], (tuple, list)):
        tensor_shape = tensor_shape[0]

    if isinstance(input_grad, (tuple, list)):
        input_grad = input_grad[0]

    input_tensors = ()
    if tensor_shape is None:
        input_tensors += (None,)
    else:
        input_tensor = p2p.send_backward_recv_forward(input_grad, tensor_shape)
        input_tensors += (input_tensor,)
    return input_tensors


def reduce_and_scale_grads(grads,
                           weights,
                           model,
                           num_tokens_list,
                           wrap_with_ddp=False,
                           calculate_per_token_loss=False):
    """ Reduce and scale model grads. """
    # AllReduce embedding (if shared embedding weight in first stage and last stage)
    grads = all_reduce_share_embedding(grads, weights, model, wrap_with_ddp)

    # Scaling grads base on per-token when using 'calculate_per_token_loss=True'
    if calculate_per_token_loss:
        grads = scale_gradients_for_per_token(grads, model, num_tokens_list, wrap_with_ddp)
    return grads


def scale_gradients_for_per_token(grads, model, num_tokens_list, wrap_with_ddp):
    """ scale gradients base on total tokens. """
    if not isinstance(model, nn.CellList):
        model = [model]
    if parallel_state.get_pipeline_model_parallel_last_rank():
        if not num_tokens_list:
            raise RuntimeError("When 'calculate_per_token_loss=True', num_tokens can not be None.")
        num_tokens = sum(num_tokens_list)
    else:
        num_tokens = Tensor(0, mstype.float32)
    num_tokens = comm_func.broadcast(num_tokens,
                                     src=parallel_state.get_pipeline_model_parallel_last_rank(),
                                     group=parallel_state.get_pipeline_model_parallel_group())
    num_tokens = comm_func.all_reduce(num_tokens, group=parallel_state.get_data_parallel_group())[0]
    if wrap_with_ddp:
        for model_chunk in model:
            if num_tokens > 0:
                scaling = 1.0 / num_tokens
                model_chunk.scale_gradients(scaling)
    else:
        for i in len(grads):
            grads[i] *= scaling
    return grads


def all_reduce_share_embedding(grads, weights, model, wrap_with_ddp):
    """ Reduce share embedding grads in embedding comm group """
    if parallel_state.is_rank_in_embedding_group(ignore_virtual=True):
        # get share weight name
        weight_name = [weight.name for weight in weights]
        if isinstance(model, nn.CellList):
            model = model[0]
        if wrap_with_ddp:
            model = model.module
        share_embeddings_and_output_weights = model.share_embeddings_and_output_weights
        # sync for share weight
        if parallel_state.get_pipeline_model_parallel_world_size() > 1 and share_embeddings_and_output_weights:
            shared_weight_index = []
            weight_grad = None
            for i, grad_i in enumerate(grads):
                if hasattr(weights[i], "shared_embedding") and weights[i].shared_embedding:
                    if wrap_with_ddp and weights[i].grad is None:
                        continue
                    weight_grad = weight_grad + grad_i if weight_grad is not None else grad_i
                    shared_weight_index.append(i)
            if len(shared_weight_index) > 1:
                shared_weight_name_list = [weight_name[idx] for idx in shared_weight_index]
                raise RuntimeError(f"Now, only support one share weight in the stage. "
                                   f"But got a couple of share weights: {shared_weight_name_list}")
            if shared_weight_index:
                if wrap_with_ddp:
                    hal.current_stream().wait_stream(hal.communication_stream())
                weight_grad.copy_(comm_func.all_reduce(weight_grad, group=parallel_state.get_embedding_group())[0])
    return grads
