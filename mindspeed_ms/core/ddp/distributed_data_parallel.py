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
""" Distributed data parallel wrapper. """

from collections import deque
from contextlib import contextmanager
import numpy as np
from mindspore import mint, ops, _no_grad, Parameter
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.common.api import _pynative_executor
from mindspore.common.initializer import Zero
from mindspore.communication import get_group_size, create_group
from mindspore.communication.comm_func import all_gather_into_tensor, reduce_scatter_tensor, all_reduce

from mindspeed_ms.core.ddp.create_comm import get_dp_world_size as get_data_parallel_world_size, \
    get_dp_group as get_data_parallel_group
from .parallel_state import get_zero_full_shard_flag
from .param_and_grad_buffer import ParamAndGradBuffer

__all__ = ['DistributedDataParallel']


@_no_grad()
def all_gather_param(cell, wait_buffer, data_parallel_group):
    # print("all_gather_param before: ",cell.weight.name,cell.cell_id, cell.weight.shape, _pynative_executor.enable_grad(), cell._has_config_recompute, cell.weight.gathered,flush=True)
    if not hasattr(cell, 'sharded_weight'):
        cell.sharded_weight = Parameter(0.0)
    cell.sharded_weight.assign_value(cell.weight)
    (param, comm_handle) = all_gather_into_tensor(cell.sharded_weight, group=data_parallel_group, async_op=True)
    param.name = cell.weight.name
    cell.weight.gathered = 1
    wait_buffer.append((param, comm_handle, cell))


@_no_grad()
def bp_all_gather_param(cell, bp_wait_buffer, data_parallel_group):
    ''' all gather param '''
    if cell.cell_id in bp_wait_buffer or (hasattr(cell.weight, 'gathered') and cell.weight.gathered == 1):
        return
    # print("all_gather_param before: ",cell.weight.name,cell.cell_id, cell.weight.shape, _pynative_executor.enable_grad(), cell._has_config_recompute, cell.weight.gathered,flush=True)
    if not hasattr(cell, 'sharded_weight'):
        cell.sharded_weight = Parameter(0.0)
    cell.sharded_weight.assign_value(cell.weight)
    (param, comm_handle) = all_gather_into_tensor(cell.sharded_weight, group=data_parallel_group, async_op=True)
    param.name = cell.weight.name
    cell.weight.gathered = 1
    bp_wait_buffer[cell.cell_id] = (param, comm_handle, cell)


@_no_grad()
def reduce_scatter_grad(param, wait_grad_buffer, grad_reduce_in_fp32, average_in_collective, data_parallel_group):
    ''' reduce scatter for param grad after grad mean reduction '''
    # param = cell.weight
    # print("reduce_scatter_grad: ", param.name, flush=True)
    if grad_reduce_in_fp32:
        param.full_grad = ops.cast(param.full_grad, mstype.float32)

    if average_in_collective:
        param.full_grad = mint.div(param.full_grad, get_data_parallel_world_size())

    (grad, comm_handle) = reduce_scatter_tensor(param.full_grad, group=data_parallel_group, async_op=True)
    wait_grad_buffer.append((grad, comm_handle, param))


@_no_grad()
def all_reduce_grad(grad, zero_shard_grad_group):
    reduced_grad, _ = all_reduce(grad, group=zero_shard_grad_group, async_op=False)
    return reduced_grad


def wait_grad(wait_grad_buffer, zero_comm_group):
    ''' wait for grad reduction, and do grad accumulation'''
    if wait_grad_buffer:
        (grad, handle, param) = wait_grad_buffer.popleft()
        handle.wait()
        param.full_grad = None
        if zero_comm_group is not None:
            reduced_grad = all_reduce_grad(grad, zero_comm_group.get("zero_shard_grad_group"))
            param.grad.copy_(reduced_grad)
        else:
            param.grad.copy_(grad)


z3_optim_cells = []

# pylint: disable=W0621, W0612
def set_model_fw_bw_hook(network, grad_reduce_in_fp32, average_in_collective, zero_comm_group, depth):
    ''' register fw bw hook for the zero3 params '''
    wait_buffer = deque()
    bp_wait_buffer = {}
    wait_grad_buffer = deque()
    data_parallel_group = get_data_parallel_group(with_context_parallel=True)
    if zero_comm_group is not None:
        if zero_comm_group.get("zero_shard_group") is None or zero_comm_group.get("zero_shard_grad_group") is None:
            raise ValueError("zero_comm_group is illegel, please pass the correct group info like"
                             "{'zero_shard_group': xxx, 'zero_shard_grad_group': xxx}")
        data_parallel_group = zero_comm_group.get("zero_shard_group")
        zero_shard_grad_group = zero_comm_group.get("zero_shard_grad_group")
    global z3_optim_cells

    def recursion_cells(cell):
        sub_cells_list = cell.cells()

        for sub_cell in sub_cells_list:
            if sub_cell.__class__.__name__ in ["ColumnParallelLinear",
                    "ParallelLinear"] and sub_cell.use_zero3 and sub_cell.weight.requires_grad:
                sub_cell.pre_cell_id = sub_cell.next_cell_id = None
                z3_optim_cells.append(sub_cell)
            else:
                recursion_cells(sub_cell)

    chunk_size = 4
    recursion_cells(network)
    comm_op_nums = len(z3_optim_cells)
    layer_nums = comm_op_nums
    if depth:
        layer_nums = depth
    layer_comm_op_size = comm_op_nums // layer_nums
    weight_size_list = []
    for i in range(layer_comm_op_size):
        weight_size_list += [np.prod(np.array(z3_optim_cells[i].weight.shape))]
    dispatch_index = np.argmax(weight_size_list)

    layer_begin_list = range(0, len(z3_optim_cells), layer_comm_op_size)
    chunk_begin_list = sorted(
        [x + dispatch_index for x in layer_begin_list] + [x + dispatch_index + 1 for x in layer_begin_list])
    single_begin_list = range(len(z3_optim_cells))
    actual_chunk_begin_id = chunk_begin_list[0]
    for begin_layer in chunk_begin_list:
        if begin_layer < actual_chunk_begin_id:
            begin_layer = actual_chunk_begin_id
        chunk_list = range(begin_layer, begin_layer + chunk_size)
        single_begin_list = [x for x in single_begin_list if x not in chunk_list]
        actual_chunk_begin_id = begin_layer + chunk_size
    actual_chunk_begin_id = 0

    # pylint: disable=W0622, W0212, W0601
    def _pre_forward_cell_hook(cell, input):
        # print("pre forward cell hook: ",cell.weight.name,cell.cell_id, cell.weight.shape, _pynative_executor.enable_grad(), cell._has_config_recompute, cell.weight.gathered,flush=True)
        cell_id = cell.cell_id
        if cell._has_config_recompute and _pynative_executor.enable_grad():
            bp_all_gather_param(cell, bp_wait_buffer, data_parallel_group)
            if cell_id in bp_wait_buffer:
                (full_param, handle, pre_cell) = bp_wait_buffer.pop(cell_id)
                handle.wait()
                pre_cell.weight.assign_value(full_param)
            next_cell_id = cell_id - layer_comm_op_size
            if next_cell_id >= 0:
                next_cell = z3_optim_cells[next_cell_id]
                bp_all_gather_param(next_cell, bp_wait_buffer, data_parallel_group)
        else:
            global actual_chunk_begin_id
            if hasattr(cell, 'zero_start'):
                actual_chunk_begin_id = chunk_begin_list[0]
                all_gather_param(cell, wait_buffer, data_parallel_group)
            if cell.cell_id in chunk_begin_list:
                dispatch_cell_id = cell.cell_id
                if dispatch_cell_id < actual_chunk_begin_id:
                    dispatch_cell_id = actual_chunk_begin_id
                if dispatch_cell_id < len(z3_optim_cells):
                    cell_ = z3_optim_cells[dispatch_cell_id + 1]
                    for _ in range(chunk_size):
                        all_gather_param(cell_, wait_buffer, data_parallel_group)
                        next_cell_id = cell_.next_cell_id
                        if next_cell_id is None:
                            break
                        cell_ = z3_optim_cells[cell_.next_cell_id]
                    actual_chunk_begin_id = dispatch_cell_id + chunk_size
            elif cell.cell_id in single_begin_list:
                if cell.cell_id < len(z3_optim_cells) - 1:
                    cell_ = z3_optim_cells[cell.next_cell_id]
                    all_gather_param(cell_, wait_buffer, data_parallel_group)

            if wait_buffer:
                (full_param, handle, post_cell) = wait_buffer.popleft()
                handle.wait()
                post_cell.weight.assign_value(full_param)
        return input
    # pylint: disable=W0613, W0212
    def _post_forward_cell_hook(cell, input, output):
        if cell._has_config_recompute and _pynative_executor.enable_grad():
            return output
        cell.weight.assign_value(cell.sharded_weight)
        cell.weight.gathered = 2
        return output

    # pylint: disable=W0622, W0613
    def _pre_backward_cell_hook(cell, grad_output):
        # print("_pre_backward_cell_hook ",cell.weight.name,cell.cell_id, cell.weight.shape, cell._has_config_recompute, flush=True)
        cell_id = cell.cell_id
        bp_all_gather_param(cell, bp_wait_buffer, data_parallel_group)
        if cell_id in bp_wait_buffer:
            (full_param, handle, post_cell) = bp_wait_buffer.pop(cell_id)
            handle.wait()
            post_cell.weight.assign_value(full_param)
        pre_cell_id = cell.cell_id - layer_comm_op_size
        if pre_cell_id > 0:
            pre_cell = z3_optim_cells[pre_cell_id]
            bp_all_gather_param(pre_cell, bp_wait_buffer, data_parallel_group)

    # pylint: disable=W0622, W0613
    def _post_backward_cell_hook(cell, grad_input, grad_output):
        # print("_post_backward_cell_hook: ", cell.weight.name, cell.cell_id, cell.weight.shape,flush=True)
        cell.weight.assign_value(cell.sharded_weight)
        cell.weight.gathered = 3

        if not hasattr(cell, 'zero_end'):
            wait_grad(wait_grad_buffer, zero_comm_group)
        if hasattr(cell, 'zero_start'):
            while wait_grad_buffer:
                wait_grad(wait_grad_buffer, zero_comm_group)
            check_post_hook()

    if z3_optim_cells:
        z3_optim_cells[0].zero_start = True
        z3_optim_cells[-1].zero_end = True

    for i in range(len(z3_optim_cells) - 1):
        z3_optim_cells[i].cell_id = i
        z3_optim_cells[i].next_cell_id = i + 1
        z3_optim_cells[i + 1].pre_cell_id = i
    z3_optim_cells[len(z3_optim_cells) - 1].cell_id = len(z3_optim_cells) - 1

    for i, sub_cell in enumerate(z3_optim_cells):
        sub_cell.register_forward_pre_hook(_pre_forward_cell_hook)
        sub_cell.register_forward_hook(_post_forward_cell_hook)
        sub_cell.register_backward_pre_hook(_pre_backward_cell_hook)
        sub_cell.register_backward_hook(_post_backward_cell_hook)

    def _make_param_hook_zero3(param):
        """ make closure function as the param hook. """

        def param_hook(grad):
            # print("_make_param_hook_zero3: ", param.name, flush=True)
            param.full_grad = grad
            if grad.shape != param.grad.shape:
                reduce_scatter_grad(param, wait_grad_buffer, grad_reduce_in_fp32, average_in_collective,
                                    data_parallel_group)
            return param.grad

        return param_hook

    for param in network.get_parameters():
        if param.requires_grad:
            if (hasattr(param, 'use_zero3') and param.use_zero3):
                param.register_hook(_make_param_hook_zero3(param))


def check_post_hook():
    global z3_optim_cells
    for cell in z3_optim_cells:
        # print("check: ", cell.weight.name, cell.cell_id, cell.weight.shape,cell.weight.gathered, flush=True)
        if cell.weight.gathered == 1:
            cell.weight.assign_value(cell.sharded_weight)
            cell.weight.gathered = 3


class DistributedDataParallel(nn.Cell):
    """
    DistributedDataParallel wrapper. DistributedDataParallel allocates contiguous memory buffer for parameters
    and gradients. It also support gradient back-propagation computation and communication. When enable overlapping,
    parameters and gradients will be break up into bucekts which is the unit to conduct all-reduce/reduce-scatter
    communication among data parallel group.

    Args:
        config (TrainingConfig): The TrainingConfig object containing the training related configurations.
        ddp_config (DistributedDataParallelConfig): The DistributedDataParallelConfig object containing the ddp
            related configurations.
        module (Module): The module to be wrapped with ddp.
        disable_bucketing (bool): Disable bucketing, which means all parameters and gradients will be assigned
            to one bucket. Default: False.

    Returns:
        Model wrapped with DistributedDataParallel.

    Examples:
        >>> from mindformers.experimental.distri_cores.distributed import DistributedDataParallel, \
        >>>     DistributedDataParallelConfig
        >>> network = Model()
        >>> ddp_config = DistributedDataParallelConfig()
        >>> network = DistributedDataParallel(trainig_config, ddp_config, network)
    """

    def __init__(
            self,
            ddp_config,
            module,
            disable_bucketing=False,
            zero_comm_group=None,
            depth=None
    ):
        super(DistributedDataParallel, self).__init__(auto_prefix=False)
        self.ddp_config = ddp_config
        self.all_parameters = None

        print("################ ddp update 1017 ###################")
        # self.module = module
        # self.netwithloss = module  # netwithclass object
        self.module = module  # dit_model object
        # self.module = self.netwithloss.network # dit_model object

        self.param_to_buffer = {}
        self.zero3_param = []
        invalid_zero_comm_group = get_zero_full_shard_flag() != (zero_comm_group is None)
        if invalid_zero_comm_group:
            raise ValueError("When set the zero_shard_size, the zero_comm_group should be set as well, but got"
                             f"get_zero_full_shard_flag={get_zero_full_shard_flag()},"
                             f"zero_comm_group={zero_comm_group}")
        if ddp_config.use_zero3:
            set_model_fw_bw_hook(self.module, \
                                 self.ddp_config.grad_reduce_in_fp32, \
                                 self.ddp_config.average_in_collective, \
                                 zero_comm_group,
                                 depth)

        # print("%%%", flush=True)
        dp_size = get_group_size()
        # print("DistributedDataParallel dp_size", dp_size, flush=True)
        rank_list = [i for i in range(dp_size)]
        rank_group = [str(i) for i in rank_list]
        rank_group = "-".join(rank_group)

        if self.ddp_config.bucket_size is None:
            # bucket_size elem consumes memory: if use fp32(4B), then one bucket ranges from 4M(dp_size=1) to 160M(max)
            self.ddp_config.bucket_size = max(40000000, 1000000 * dp_size)

        self.bucket_size = self.ddp_config.bucket_size
        if False or disable_bucketing or not self.ddp_config.overlap_grad_reduce:
            self.bucket_size = None

        dense_params = []
        expert_parallel_params = []
        for _, param in self.module.parameters_and_names():
            if not param.requires_grad:
                continue
            param.grad = None
            param.main_grad = None

            param.grad_accumulated = False
            if hasattr(param, 'use_zero3') and param.use_zero3:
                grad_dtype = mstype.float32 if self.ddp_config.grad_reduce_in_fp32 else param.dtype
                param.grad = ops.Tensor(shape=param.shape, dtype=grad_dtype, init=Zero())
                self.zero3_param.append(param)
            elif getattr(param, 'allreduce', True):
                dense_params.append(param)
            else:
                expert_parallel_params.append(param)

        # if config.calculate_per_token_loss:
        #     gradient_scaling_factor = 1.0
        #     expert_gradient_scaling_factor = 1.0
        # else:
        expert_gradient_scaling_factor = 1.0
        gradient_scaling_factor = 1.0

        # if self.ddp_config.average_in_collective:
        #     gradient_scaling_factor = 1.0
        #     expert_gradient_scaling_factor = 1.0
        # else:
        #     data_parallel_world_size = get_group_size()
        #     gradient_scaling_factor = 1.0 / data_parallel_world_size
        #     expert_gradient_scaling_factor = 1.0 / data_parallel_world_size

        # allocate buffer for common params and expert params
        # global get_data_parallel_group
        # get_data_parallel_group = "dp-cp-" + rank_group
        # # print("get_data_parallel_group ---", get_data_parallel_group, flush=True)
        # create_group(get_data_parallel_group, rank_list)
        # print("get_data_parallel_group created ---",get_group_size(group=get_data_parallel_group), flush=True)

        self.buffers = self.allocate_buffers_for_parameters(
            dense_params,
            group=get_data_parallel_group(with_context_parallel=True),
            gradient_scaling_factor=gradient_scaling_factor,
            zero_comm_group=zero_comm_group,
        )

        get_data_modulo_expert_parallel_group = "dp-independent_ep-" + rank_group
        # print("get_data_modulo_expert_parallel_group ---", get_data_modulo_expert_parallel_group, flush=True)
        create_group(get_data_modulo_expert_parallel_group, rank_list)
        # print("get_data_modulo_expert_parallel_group created ---", flush=True)
        self.expert_parallel_buffers = self.allocate_buffers_for_parameters(
            expert_parallel_params,
            group=get_data_modulo_expert_parallel_group,
            gradient_scaling_factor=expert_gradient_scaling_factor,
            zero_comm_group=zero_comm_group,
        )

        # register hook for bucket grad reduce
        self.register_hook_for_params()

    def allocate_buffers_for_parameters(self, input_params, group, gradient_scaling_factor, zero_comm_group):
        """ allocate buffers for parameters in different dtype group. """
        param_and_grad_dtype_to_params = {}
        # group all params by parameter's data type and their gradient's data type.
        for param in input_params:
            param_dtype = param.dtype
            grad_dtype = mstype.float32 if self.ddp_config.grad_reduce_in_fp32 else param.dtype

            if (param_dtype, grad_dtype) not in param_and_grad_dtype_to_params:
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = []
            param_and_grad_dtype_to_params[(param_dtype, grad_dtype)].append(param)
        buffers = []
        # print("buffers", buffers, flush=True)
        # allocate buffer for each group separately
        for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():
            # print("add---", flush=True)
            buffers.append(
                ParamAndGradBuffer(
                    ddp_config=self.ddp_config,
                    param_dtype=param_dtype,
                    grad_dtype=grad_dtype,
                    params=params,
                    data_parallel_group=group,
                    bucket_size=self.bucket_size,
                    param_to_name=None,
                    gradient_scaling_factor=gradient_scaling_factor,
                    zero_comm_group=zero_comm_group
                )
            )
            # print("added ##", flush=True)
            for param in params:
                self.param_to_buffer[param] = buffers[-1]
            # print("param ##", flush=True)
        # print("^^^^^^^^^", flush=True)
        return buffers

    def issue_grad_reduce(self):
        """ issue grad reduce for each buffer. """
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.issue_grad_reduce()

    def final_grad_reduce(self):
        """ finalize grad reduce for each buffer. """
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.final_grad_reduce()

    def register_hook_for_params(self):
        """ register backward hook for each params. """
        if self.all_parameters is None:
            self.all_parameters = self.module.get_parameters()
        for param in self.all_parameters:
            if param.requires_grad:
                if not (hasattr(param, 'use_zero3') and param.use_zero3):
                    param.register_hook(self._make_param_hook(param, self.param_to_buffer))

    def set_input_tensor(self, input_tensor):
        """ set input tensor for model"""
        self.module.set_input_tensor(input_tensor)

    def construct(self, *inputs, **inputs_dict):
        """ construct for DistributedDataParallel. """
        output = self.module(*inputs, **inputs_dict)
        return output

    def zero_grad_buffer(self):
        """ reset buffers for the next train iteration. """
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.reset()
        for param in self.zero3_param:
            param.gathered = 0

    def enable_sync(self, enable):
        """ enable grad buffer sync or not. """
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.sync_enabled = enable

    @contextmanager
    def no_sync(self):
        """ context manager helper function. """
        self.enable_sync(False)
        try:
            yield
        finally:
            self.enable_sync(True)

    def _make_param_hook(
            self,
            param,
            param_to_buffer,
    ):
        """ make closure function as the param hook. """

        def param_hook(grad):
            buffer = param_to_buffer[param]
            if not param.grad_accumulated:
                param.main_grad.add_(grad)
            if self.ddp_config.overlap_grad_reduce:
                buffer.register_grad_ready(param)
            if param.grad is None:
                return ops.Tensor(0, param.dtype)
            return param.grad

        return param_hook
