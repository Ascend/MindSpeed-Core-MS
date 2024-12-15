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
""" Distributed optimizer wrapper. """

from collections import OrderedDict
import json
import os
import stat
import numpy as np

import mindspore as ms
import mindspore.ops as ops
from mindspore import _no_grad, mint
from mindspore.common import dtype as mstype
from mindspore.common.initializer import Zero
from mindspore.communication.management import get_group_size, get_rank
import mindspore.communication.comm_func as comm_func

from mindspeed_ms.tools import logger
from mindspeed_ms.core.distributed import ParamAndGradBuffer
from mindspeed_ms.core.utils import pp_layer_rename
from mindspeed_ms.training.global_vars import get_args

from .optimizer import MixedPrecisionOptimizer
from .utils import _update_adamw_var


def shard_bucket(bucket, group):
    """ shard bucket to dp_size part. """
    dp_world_size = get_group_size(group=group)
    dp_rank = get_rank(group=group)
    bucket_size = bucket.grad_data.numel()

    # each rank processes a slice data of whole bucket
    size_per_dp_rank = bucket_size // dp_world_size

    # bucket range info of slice data processed by this rank
    shard_start = dp_rank * size_per_dp_rank

    shard_end = min(bucket_size, shard_start + size_per_dp_rank)
    return shard_start, shard_end


class DistributedOptimizer(MixedPrecisionOptimizer):
    r"""
    This class build the mapping between parameters' range take charge by this dp rank in
    the optimizer parallelism and their shard ranges in the buffer, sharded bucket, integrated bucket
    and integrated parameters. This mapping is needed for converting between model param indexes
    and main parameter shard indexes. This class also update the non-parallel optimizer attributes
    according to parameter shard information.

    Args:
        optimizer (mindspore.mint.optim): Non-parallel optimizer.
        config (dict): The OptimizerConfig object containing the optimizer related configurations.
        grad_scaler (GradScaler): Gradient scaling. When it is ``None``, no scaler will be used for gradients.
        init_state_fn (Function): Function to initialize state parameters of optimizer.
        per_model_buffers (List): List of biffers of all model chunks.
        data_parallel_group (str): Data parallel group name.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.
            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> import mindspore.common.dtype as mstype
        >>> import mindspore.dataset as ds
        >>> from mindspore.communication.management import init
        >>> from mindspore.nn import SoftmaxCrossEntropyWithLogits
        >>> from mindspore.mint.optim import AdamW
        >>> from mindspeed_ms.core.tensor_parallel import (ColumnParallelLinear,
        ... RowParallelLinear)
        >>> from mindspeed_ms.core.parallel_state import (initialize_model_parallel,
        ... get_data_parallel_world_size, get_data_parallel_rank, get_data_parallel_group)
        >>> from mindspeed_ms.core.config import (OptimizerConfig, ModelParallelConfig,
        ... TransformerConfig, TrainingConfig)
        >>> from mindspeed_ms.core.distributed import (DistributedDataParallel,
        ... DistributedDataParallelConfig)
        >>> from mindspeed_ms.core.optimizer.distrib_optimizer import DistributedOptimizer
        >>> from tests.st.test_distri_core.utils import TestData, train
        >>> class TestNet2(nn.Cell):
        ...     def __init__(self, config):
        ...         super(TestNet2, self).__init__()
        ...         hidden_size = config.hidden_size
        ...         self.columnlinear = ColumnParallelLinear(input_size=hidden_size, output_size=hidden_size,
        ...                                                  config=config, init_method=config.init_method,
        ...                                                  bias=config.add_mlp_bias, gather_output=False,
        ...                                                  skip_bias_add=False, bias_init=config.bias_init)
        ...         self.rowlinear = RowParallelLinear(input_size=hidden_size, output_size=hidden_size, config=config,
        ...                                            init_method=config.init_method, bias=config.add_mlp_bias,
        ...                                            input_is_parallel=True, skip_bias_add=False,
        ...                                            bias_init=config.bias_init)
        ...         self.loss = SoftmaxCrossEntropyWithLogits()
        ...     def construct(self, input_, label_):
        ...         output, _ = self.columnlinear(input_)
        ...         output, _ = self.rowlinear(output)
        ...         output = ops.sum(output, dim=-1, keepdim=False)
        ...         output = ops.cast(output, mstype.float32)
        ...         loss = self.loss(output, label_)
        ...         return loss
        ...
        >>> ms.set_context(device_target='Ascend', mode=ms.PYNATIVE_MODE)
        >>> ms.set_seed(2024)
        >>> init()
        >>> initialize_model_parallel(tensor_model_parallel_size=2)
        >>> batch_size = 1
        >>> dataset_size = 6
        >>> seq_length = 8
        >>> hidden_size = 4
        >>> tensor_parallel = 1
        >>> bucket_size = 10
        >>> input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(np.float32)
        >>> label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
        >>> dataset = TestData(input_data=input_data, label_data=label_data)
        >>> dataset = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'],
        >>>                               num_shards=get_data_parallel_world_size(),
        >>>                               shard_id=get_data_parallel_rank())
        >>> dataset = dataset.batch(batch_size)
        >>> parallel_config = ModelParallelConfig()
        >>> training_config = TrainingConfig(parallel_config=parallel_config)
        >>> optimizer_config = OptimizerConfig(parallel_config=parallel_config)
        >>> model_config = TransformerConfig(vocab_size=40000, num_layers=1, num_attention_heads=1, add_mlp_bias=True,
        >>>                                  gated_linear_unit=False, hidden_size=hidden_size,
        >>>                                  ffn_hidden_size=4*hidden_size, hidden_act='gelu',
        >>>                                  parallel_config=parallel_config, params_dtype='float32',
        >>>                                  compute_dtype='float32')
        >>> ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, use_distributed_optimizer=True,
        >>>     bucket_size=bucket_size, average_in_collective=True, enable_mem_align=True)
        >>> network = TestNet2(config=model_config)
        >>> network_with_ddp = DistributedDataParallel(config=training_config, ddp_config=ddp_config, module=network)
        >>> optimizer = AdamW(params=network_with_ddp.get_parameters(), lr=1.0)
        >>> optimizer = DistributedOptimizer(optimizer=optimizer, config=optimizer_config, grad_scaler=None,
        >>>     init_state_fn=None, per_model_buffers=network_with_ddp.buffers,
        >>>     data_parallel_group=get_data_parallel_group(with_context_parallel=True))
        >>> losses = train(epoch_num=1, dataset=dataset, network=network_with_ddp, optimizer=optimizer)
    """
    @classmethod
    def _build_param_ranges_map(
            cls,
            param_index_map,
            bucket_world_range,
            bucket_offset
        ):
        """ Build parameter range map. """
        # param_range_map
        param_range_map = {}
        bucket_world_start, bucket_world_end = bucket_world_range
        shard_size = bucket_world_end - bucket_world_start

        for param, param_world_indexes in param_index_map.items():
            param_world_start, param_world_end, _ = param_world_indexes
            param_local_start = max(0, param_world_start - bucket_world_start)
            param_local_end = min(param_world_end - bucket_world_start, shard_size)

            # param_range_map only record shard info for parameters
            # in the buffer shard processed by this dp rank
            if param_local_start < param_local_end:
                # range in bucket shard of this param
                range_in_shard = (param_local_start, param_local_end)
                local_size_in_shard = param_local_end - param_local_start
                # range in buffer of this param
                range_in_buffer = (
                    bucket_world_start + param_local_start,
                    bucket_world_start + param_local_start + local_size_in_shard
                )
                # range in bucket of this param
                range_in_whole_bucket = (
                    range_in_buffer[0] - bucket_offset,
                    range_in_buffer[1] - bucket_offset
                )
                # range in integrated param of this param slice
                sub_param_start = max(0, bucket_world_start - param_world_start)
                sub_param_end = sub_param_start + local_size_in_shard
                sub_param_range = (sub_param_start, sub_param_end)
                # build param range map
                param_range_map[param] = {
                    'range_in_buffer': range_in_buffer,
                    'range_in_bucket': range_in_whole_bucket,
                    'range_in_shard': range_in_shard,
                    'range_in_param': sub_param_range
                }

        return param_range_map


    @classmethod
    def _build_bucket_ranges_map(cls, param_and_grad_buffer, bucket_index):
        """ Build parameter range map for bucket. """
        bucket = param_and_grad_buffer.buckets[bucket_index]
        data_parallel_group = param_and_grad_buffer.data_parallel_group
        if param_and_grad_buffer.zero_comm_group is not None:
            data_parallel_group = param_and_grad_buffer.zero_comm_group["zero_shard_group"]
        shard_start, shard_end = shard_bucket(bucket, group=data_parallel_group)

        # buffer range info of slice data processed by this rank
        bucket_world_range = (shard_start + bucket.offset, shard_end + bucket.offset)

        param_range_map = cls._build_param_ranges_map(param_and_grad_buffer.param_index_map,
                                                      bucket_world_range,
                                                      bucket.offset)
        return param_range_map

    @classmethod
    def _build_buffer_ranges_map(cls, param_and_grad_buffer):
        """ Build parameter range map for buffer. """
        return [
            cls._build_bucket_ranges_map(param_and_grad_buffer, bucket_idx)
            for bucket_idx in range(len(param_and_grad_buffer.buckets))
        ]

    @classmethod
    def _get_optimizer_group_ranges(cls, param_groups, param_ranges_map):
        """ Build optimizer group info for distributed optimizer. """
        world_param_group_map = {}
        # build map of parameters to their optimizer group in the original optimizer
        for group_idx, group in enumerate(param_groups):
            for param in group['params']:
                if not param.requires_grad:
                    raise ValueError("param.requires_grad should be True but got False!")
                world_param_group_map[param] = group_idx

        # In distributed optimizer, each dp rank only update a part of parameters,
        # thus only those parameters' group information is required. local_param_group_map mapping
        # parameters to their optimizer group index and its index in the groups' 'param' list.
        local_param_group_map = {}
        sharded_param_groups = [{'params': []} for _ in range(len(param_groups))]
        for all_bucket_range_list in param_ranges_map:
            for param_range_map in all_bucket_range_list:
                for param in param_range_map:
                    group_idx = world_param_group_map[param]
                    sharded_group = sharded_param_groups[group_idx]
                    sharded_group['params'].append(param)
                    local_param_group_map[param] = (group_idx, len(sharded_group['params']) - 1)

        # add zero3 params which are not in the param_ranges_map
        for group_index, group in enumerate(param_groups):
            for param in group["params"]:
                if hasattr(param, 'use_zero3') and param.use_zero3:
                    sharded_group = sharded_param_groups[group_index]
                    sharded_group["params"].append(param)
                    local_param_group_map[param] = (group_index, len(sharded_group["params"]) - 1)

        return local_param_group_map, sharded_param_groups

    @classmethod
    def _build_sharded_params_and_grads(
            cls,
            param_ranges_map,
            param_to_bucket_map,
            sharded_param_groups,
            buffers,
        ):
        """ Build shards of param and grad buffer. """
        param_fp16_groups = []
        param_fp32_groups = []
        # sharded_param_fp16_groups and sharded_param_fp32_groups
        # are view tensors on corresponding param_and_grad_buffer.
        # sharded_param_fp32_from_fp16_groups is a float32 copy of parameters,
        # which will apply for a new contiguous memory.
        sharded_param_fp16_groups = []
        sharded_param_fp32_from_fp16_groups = []
        sharded_param_fp32_groups = []
        zero3_param = []

        for sharded_group in sharded_param_groups:
            param_fp16_this_group = []
            param_fp32_this_group = []
            sharded_param_fp16_this_group = []
            sharded_param_fp32_from_fp16_this_group = []
            sharded_param_fp32_this_group = []

            # the param is the integrated parameter object
            for param in sharded_group['params']:
                if hasattr(param, 'use_zero3') and param.use_zero3:
                    if param.dtype in [mstype.float16, mstype.bfloat16]:
                        zero3_main_param = ops.cast(param, mstype.float32)
                        param_fp16_this_group.append(param)
                        zero3_main_param.name = param.name
                        zero3_main_param.origin_shape = param.shape
                        zero3_main_param.use_zero3 = True
                        sharded_param_fp16_this_group.append(param)
                        sharded_param_fp32_from_fp16_this_group.append(zero3_main_param)
                        zero3_param.append(zero3_main_param)
                    elif param.dtype == mstype.float32:
                        param_fp32_this_group.append(param)
                        zero3_fp32_main_param = param
                        zero3_fp32_main_param.name = param.name
                        zero3_fp32_main_param.origin_shape = param.shape
                        zero3_fp32_main_param.use_zero3 = True
                        sharded_param_fp32_this_group.append(zero3_fp32_main_param)
                        zero3_param.append(zero3_fp32_main_param)
                    else:
                        raise TypeError("Invalid parameter dtype. Supported parameter dtypes are"
                                        "`mindspore.float16`, `mindspore.bfloat16` and `mindspore.float32`,"
                                        " but got {}.".format(param.dtype))

                else:
                    buffer_idx, bucket_idx = param_to_bucket_map[param]
                    param_range = param_ranges_map[buffer_idx][bucket_idx][param]
                    param_start_in_buffer, param_end_in_buffer = param_range['range_in_buffer']
                    # for float16 and bfloat16 parameters, clone an float32 copy
                    if param.dtype in [mstype.float16, mstype.bfloat16]:
                        sharded_param_fp16 = buffers[buffer_idx].param_data[param_start_in_buffer:param_end_in_buffer]
                        sharded_param_fp32_from_fp16 = ops.cast(sharded_param_fp16, mstype.float32)
                        sharded_param_fp16.name = param.name
                        sharded_param_fp32_from_fp16.name = param.name
                        param.main_param = sharded_param_fp32_from_fp16
                        sharded_grad_fp32_from_fp16 = ops.cast(buffers[buffer_idx].grad_data[param_start_in_buffer: \
                                                               param_end_in_buffer],
                                                               mstype.float32)
                        param.grad = sharded_grad_fp32_from_fp16
                        sharded_param_fp32_from_fp16.grad = sharded_grad_fp32_from_fp16

                        param_fp16_this_group.append(param)
                        sharded_param_fp16_this_group.append(sharded_param_fp16)
                        sharded_param_fp32_from_fp16_this_group.append(sharded_param_fp32_from_fp16)

                    elif param.dtype == mstype.float32:
                        sharded_param_fp32 = buffers[buffer_idx].param_data[param_start_in_buffer:param_end_in_buffer]
                        sharded_param_fp32.name = param.name
                        param.main_param = sharded_param_fp32
                        sharded_grad_fp32 = buffers[buffer_idx].grad_data[param_start_in_buffer: param_end_in_buffer]
                        param.grad = sharded_grad_fp32
                        sharded_param_fp32.grad = sharded_grad_fp32
                        param_fp32_this_group.append(param)
                        sharded_param_fp32_this_group.append(sharded_param_fp32)

                    else:
                        raise TypeError("Invalid parameter dtype. Supported parameter dtypes are"
                                        "`mindspore.float16`, `mindspore.bfloat16` and `mindspore.float32`,"
                                        " but got {}.".format(param.dtype))
            param_fp16_groups.append(param_fp16_this_group)
            param_fp32_groups.append(param_fp32_this_group)
            sharded_param_fp16_groups.append(sharded_param_fp16_this_group)
            sharded_param_fp32_from_fp16_groups.append(sharded_param_fp32_from_fp16_this_group)
            sharded_param_fp32_groups.append(sharded_param_fp32_this_group)

        return (
            param_fp16_groups,
            param_fp32_groups,
            sharded_param_fp16_groups,
            sharded_param_fp32_from_fp16_groups,
            sharded_param_fp32_groups,
            zero3_param
        )

    def __init__(
            self,
            optimizer,
            config,
            grad_scaler,
            init_state_fn,
            per_model_buffers,
            data_parallel_group,
            data_parallel_group_mccl,
        ):
        super().__init__(
            optimizer,
            config,
            grad_scaler,
            init_state_fn
        )

        self.buffers = []
        if isinstance(per_model_buffers[0], ParamAndGradBuffer):
            per_model_buffers = {0: per_model_buffers}
        self.per_model_buffers = per_model_buffers
        self.data_parallel_group = data_parallel_group
        self.data_parallel_group_mccl = data_parallel_group_mccl
        self.update_success = False

        self.buffer_idx_to_model_idex_map = {}
        self.param_to_bucket_map = {}
        buffer_idx = 0

        for model_idx, buffers in self.per_model_buffers.items():
            for buffer in buffers:
                self.buffer_idx_to_model_idex_map[buffer_idx] = model_idx
                self.buffers.append(buffer)
                buffer_idx += 1

        self.zero3_param = []
        self.param_ranges_map = []
        # build param ranges
        for buffer in self.buffers:
            self.param_ranges_map.append(self._build_buffer_ranges_map(buffer))

        # build param to bucket map
        for buffer_idx, all_bucket_range_list in enumerate(self.param_ranges_map):
            for bucket_index, param_range_map in enumerate(all_bucket_range_list):
                for param, _ in param_range_map.items():
                    if param not in self.param_to_bucket_map:
                        self.param_to_bucket_map[param] = (buffer_idx, bucket_index)
                    else:
                        raise ValueError("Parameter should only belongs to a single bucket.")

        # get optimizer group info
        (
            self.model_param_group_map,
            self.sharded_param_groups
        ) = self._get_optimizer_group_ranges(self.optimizer.param_groups, self.param_ranges_map)

        # group parameters and gradients
        (
            self.param_fp16_groups,
            self.param_fp32_groups,
            self.sharded_param_fp16_groups,
            self.sharded_param_fp32_from_fp16_groups,
            self.sharded_param_fp32_groups,
            self.zero3_param
        ) = self._build_sharded_params_and_grads(self.param_ranges_map,
                                                 self.param_to_bucket_map,
                                                 self.sharded_param_groups,
                                                 self.buffers)

        self.param_buffer_dp_views = self._get_model_param_buffer_dp_views()
        self.overlap_param_gather = False

        # update self.optimizer attributions according to shard info
        self._update_optimizer_attr()

        # build map between parameter name and its index in optimizer.parameters after sharding
        self.param_idx_in_opt = {}
        for idx, param in enumerate(self.optimizer.parameters):
            self.param_idx_in_opt[param.name] = idx

        # build fp32 copy for fp16/bf16 params
        self.reload_model_params()

        # AllGather param overlap data structs
        self.overlap_param_gather = self.config.overlap_param_gather
        self.remove_cell_param_gather_handles = []
        self.all_gather_handles = []
        self.all_gather_handle_index_to_bucket_index_map = []
        self.model_index_to_all_gather_handle_index_map = {}
        self.all_gather_handle_indices = []
        self.param_to_all_gather_handle_index_map = {}
        self.bucket_allgather_param_data_map = {}
        self.buffer_bucket_index_list = sorted(self.param_buffer_dp_views, key=lambda x: (x[0], -x[1]))
        for buffer_index, bucket_index, _, _ in self.buffer_bucket_index_list:
            self.all_gather_handle_index_to_bucket_index_map.append((buffer_index, bucket_index))
            all_gather_handle_index = len(self.all_gather_handle_index_to_bucket_index_map) - 1
            # placeholder for handles
            self.all_gather_handles.append(None)
            if buffer_index not in self.buffer_idx_to_model_idex_map:
                raise RuntimeError(f"buffer_index {buffer_index} not in buffer_idx_to_model_idex_map"
                                   f" {self.buffer_idx_to_model_idex_map}")
            model_index = self.buffer_idx_to_model_idex_map[buffer_index]
            if model_index not in self.model_index_to_all_gather_handle_index_map:
                self.model_index_to_all_gather_handle_index_map[model_index] = []
            self.model_index_to_all_gather_handle_index_map[model_index].append(all_gather_handle_index)
            for param in self.buffers[buffer_index].buckets[bucket_index].params_list:
                self.param_to_all_gather_handle_index_map[param] = all_gather_handle_index
        self.num_all_gather_handles = len(self.all_gather_handle_index_to_bucket_index_map)


    # pylint: disable=W0613
    def zero_grad(self, set_to_none=True):
        """ reset grads data. """
        self.grads = []
        if self.overlap_param_gather:
            self._dispatch_gather_model_params(all_gather_handle_index=0)

    def get_model_parallel_group(self):
        """ return model_parallel_group for global norm allreduce. """
        return None

    def reload_main_params(self):
        """ reload main params to model params. """
        self._copy_main_params_to_model_params()
        self.sync_gather_all_model_params(force_sync=True)

    def step_with_ready_grads(self):
        """ optimizer update and synchronize updated parameters among dp group. """
        self.update_success = super().step_with_ready_grads()
        if self.update_success:
            # allgather updated buckets' data among dp group
            self.sync_gather_all_model_params()

    def _reassemble_param(self, buffer_data, state_dict):
        """ reassemble params from buffers into key-value dict """
        for buffer_idx, this_buffer in enumerate(self.buffers):
            for param, value in this_buffer.param_index_map.items():
                start_idx, end_idx, bucket_idx = value
                local_start_idx = start_idx - this_buffer.buckets[bucket_idx].offset
                local_end_idx = end_idx - this_buffer.buckets[bucket_idx].offset
                for key in buffer_data:
                    name = key + '.' + param.name if key else param.name
                    full_param = ms.Parameter(
                        buffer_data[key][buffer_idx][bucket_idx][local_start_idx:local_end_idx].reshape(param.shape),
                        name=name,
                        requires_grad=True,
                    )
                    state_dict[name] = full_param
        return state_dict

    def get_parameter_state_dp_zero(self, include_optim):
        """ gather param into dp zero """
        # Data parallelism variables.
        data_parallel_group_mccl = self.data_parallel_group_mccl
        data_parallel_world_size = mint.distributed.get_world_size(self.data_parallel_group_mccl)
        data_parallel_rank = mint.distributed.get_rank(group=self.data_parallel_group_mccl)
        data_parallel_global_ranks = mint.distributed.get_process_group_ranks(
            self.data_parallel_group_mccl
        )

        state_dict = self.state_dict()
        if include_optim:
            buffer_data = {"": {}, "exp_avg": {}, "exp_avg_sq": {}}
        else:
            buffer_data = {"": {}}

        for key in buffer_data:
            buffer_data[key] = [{} for _ in range(len(self.buffers))]
        for buffer_idx, bucket_idx, _, _ in self.param_buffer_dp_views:
            shard_name = 'buffer_{}_bucket_{}'.format(buffer_idx, bucket_idx)
            for key in buffer_data:
                name = key + '.' + shard_name if key else shard_name
                if name in state_dict:
                    bucket_data_view = state_dict.pop(name)
                    # Gather tensor list.
                    if data_parallel_rank == 0:
                        recv_tensors = [
                            ms.Tensor(np.zeros(bucket_data_view.shape, dtype=np.float32))
                            for _ in range(data_parallel_world_size)
                        ]
                    else:
                        recv_tensors = []

                    mint.distributed.gather(
                        bucket_data_view,
                        recv_tensors,
                        data_parallel_global_ranks[0],
                        group=data_parallel_group_mccl)

                    if data_parallel_rank == 0:
                        recv_tensors_concatenated = mint.cat(recv_tensors)
                        buffer_data[key][buffer_idx][bucket_idx] = recv_tensors_concatenated

        # reassemble parameters
        if data_parallel_rank == 0:
            state_dict = self._reassemble_param(buffer_data, state_dict)
            buffer_data = None

        for param in self.zero3_param:
            for key in buffer_data:
                name = key + '.' + param.name if key else param.name
                if name in state_dict:
                    param_data_view = state_dict[name]
                    if data_parallel_rank == 0:
                        recv_tensors = [
                            ms.Tensor(np.zeros(param_data_view.shape, dtype=np.float32))
                            for _ in range(data_parallel_world_size)
                        ]
                    else:
                        recv_tensors = []

                    mint.distributed.gather(
                        param_data_view,
                        recv_tensors,
                        data_parallel_global_ranks[0],
                        group=data_parallel_group_mccl)

                    if data_parallel_rank == 0:
                        recv_tensors_concatenated = mint.cat(recv_tensors)
                        state_dict[name] = ms.Parameter(recv_tensors_concatenated, name=name, requires_grad=False)
        return state_dict


    def _load_state_from_fs_model_space(self, state_dict, load_optim: bool = True):
        """ load state from fully sharded parameter state. """
        # Build a mapping from sharded param to sharded state
        param_to_state = zip(self.optimizer.parameters, self.optimizer.exp_avg, self.optimizer.exp_avg_sq)
        shard_param_to_state_map = {
            param.name: [exp_avg, exp_avg_sq]
            for param, exp_avg, exp_avg_sq in param_to_state
        }

        # Load optimizer state from the state dict
        for param, bucket_info in self.param_to_bucket_map.items():
            # Get param range info
            buffer_idx, bucket_idx = bucket_info
            param_range = self.param_ranges_map[buffer_idx][bucket_idx][param]
            param_start, param_end = param_range["range_in_param"]

            # Get sharded param & state
            shard_param = param.main_param
            if load_optim:
                shard_state = shard_param_to_state_map.get(shard_param.name, list())
                if len(shard_state) != 2:
                    raise ValueError(f"Fail to get sharded states of '{param.name}'.")
            else:
                shard_state = []
            state_name = [x.name for x in shard_state]

            # Get weight from state dict
            for ele in [shard_param] + shard_state:
                if not load_optim and ele.name in state_name:
                    continue
                weight = state_dict.get(ele.name)
                if weight is None:
                    logger.warning(
                        f"Fail to get the weight of '{ele.name}' from state dict." + (
                            " Specify --no-load-optim or --finetune to prevent"
                            " attempting to load the optimizer state."
                            if ele.name in state_name else ""
                        )
                    )
                    continue
                ele.copy_(
                    ms.Tensor(
                        weight.asnumpy().reshape(-1)[param_start:param_end],
                        dtype=ele.dtype,
                    )
                )

    def _load_state_dict_from_dp_zero(self, state_dict, load_optim: bool = True):
        """ load state dict from dp splited bucket state. """
        state_type = ["exp_avg.", "exp_avg_sq."]
        param_type = [""] + state_type if load_optim else [""]
        for buffer_index, bucket_index, _, _ in self.param_buffer_dp_views:
            param_range_map_this_bucket = self.param_ranges_map[buffer_index][bucket_index]
            shard_name = 'buffer_{}_bucket_{}'.format(buffer_index, bucket_index)
            for param, range_map in param_range_map_this_bucket.items():
                start_idx, end_idx = range_map['range_in_shard']
                param_id_in_opt = self.param_idx_in_opt[param.name]
                # check data exists in state dict
                for ele in param_type:
                    key = ele + shard_name
                    if key not in state_dict:
                        raise KeyError(
                            f"No shard data for '{key}' found in checkpoint state dict. "
                            f"Please check that parallel strategy and bucket sharding "
                            f"are not changed before loading state dict from dp zero." + (
                                " Or specify --no-load-optim or --finetune to prevent"
                                " attempting to load the optimizer state."
                                if ele in state_type else ""
                            )
                        )
                self.optimizer.parameters[param_id_in_opt].copy_(
                    ms.Tensor(
                        state_dict[shard_name].asnumpy()[start_idx:end_idx],
                        dtype=self.optimizer.parameters[param_id_in_opt].dtype,
                    )
                )
                if load_optim:
                    self.optimizer.exp_avg[param_id_in_opt].copy_(
                        ms.Tensor(
                            state_dict['exp_avg.' + shard_name].asnumpy()[start_idx:end_idx],
                            dtype=self.optimizer.exp_avg[param_id_in_opt].dtype,
                        )
                    )
                    self.optimizer.exp_avg_sq[param_id_in_opt].copy_(
                        ms.Tensor(
                            state_dict['exp_avg_sq.' + shard_name].asnumpy()[start_idx:end_idx],
                            dtype=self.optimizer.exp_avg_sq[param_id_in_opt].dtype,
                        )
                    )

    def load_state_dict(self, state_dict, load_optim: bool = True):
        """ load the state dict. """
        sharding_type = 'fully_sharded_model_space'
        for key in state_dict.keys():
            if 'buffer' in key and 'bucket' in key:
                sharding_type = 'dp_zero'
        if sharding_type == 'dp_zero':
            self._load_state_dict_from_dp_zero(state_dict, load_optim)
        elif sharding_type == 'fully_sharded_model_space':
            self._load_state_from_fs_model_space(state_dict, load_optim)
        else:
            raise NotImplementedError('Unknow sharding_type: {}'.format(sharding_type))

        args = get_args()
        for param in self.zero3_param:
            loaded_param = state_dict.get(param.name)
            if not args.use_dist_ckpt:
                dp_world_size = get_group_size(group=self.data_parallel_group)
                dp_rank = get_rank(group=self.data_parallel_group)
                splited_param = mint.split(loaded_param, loaded_param.shape[0] // dp_world_size)[dp_rank]
                param.copy_(splited_param)
            else:
                param.copy_(loaded_param)

            # saved shape is 2d, need to reshape to 1d
            if not load_optim:
                continue
            state_name = [f"exp_avg.{param.name}", f"exp_avg_sq.{param.name}"]
            no_exists = list(filter(lambda x: x not in state_dict, state_name))
            if no_exists:
                raise KeyError(
                    f"Fail to get the weight of {no_exists} from state dict. "
                    f"Specify --no-load-optim or --finetune to prevent "
                    f"attempting to load the optimizer state."
                )

            if not args.use_dist_ckpt:
                loaded_exp_avg = state_dict['exp_avg.' + param.name]
                loaded_exp_avg_sq = state_dict['exp_avg_sq.' + param.name]
                splited_exp_avg = mint.split(loaded_exp_avg, loaded_exp_avg.shape[0] // dp_world_size)[dp_rank]
                splited_exp_avg_sq = mint.split(loaded_exp_avg_sq, loaded_exp_avg_sq.shape[0] // dp_world_size)[dp_rank]

                param_id_in_opt = self.param_idx_in_opt[param.name]
                self.optimizer.exp_avg[param_id_in_opt].copy_(splited_exp_avg)
                self.optimizer.exp_avg_sq[param_id_in_opt].copy_(splited_exp_avg_sq)
            else:
                param_id_in_opt = self.param_idx_in_opt[param.name]
                self.optimizer.exp_avg[param_id_in_opt].copy_(state_dict['exp_avg.' + param.name])
                self.optimizer.exp_avg_sq[param_id_in_opt].copy_(state_dict['exp_avg_sq.' + param.name])

        # if no load optim, skip load lr, wd and state_step
        if not load_optim:
            return

        if 'state_step' in state_dict.keys():
            self.optimizer.state_step.assign_value(state_dict['state_step'].value())

        # load learning rate
        for group_idx, lr in enumerate(self.optimizer.lrs):
            lr_name = lr.name
            if lr_name in state_dict.keys():
                lr = state_dict[lr_name]
                self.optimizer.param_groups[group_idx]['lr'] = lr.item()
            wd_name = lr_name.replace('learning_rate', 'weight_decay')
            if wd_name in state_dict.keys():
                self.optimizer.param_groups[group_idx]['weight_decay'] = state_dict.get(wd_name).item()

    def _update_optimizer_attr(self):
        """ update attributes of self.optimizer according to shard information. """
        # reset optimizer.parameter
        self.optimizer.ori_parameters = self.optimizer.parameters
        self.optimizer.parameters = []
        for group_idx, _ in enumerate(self.optimizer.param_groups):
            # update params in this group
            self.optimizer.param_groups[group_idx]['params'] = [
                *self.sharded_param_fp32_groups[group_idx], \
                *self.sharded_param_fp32_from_fp16_groups[group_idx]
            ]
            self.optimizer.group_start_id[group_idx + 1] = self.optimizer.group_start_id[group_idx] + \
                len(self.optimizer.param_groups[group_idx]['params'])
            self.optimizer.lrs[group_idx] = self.optimizer.param_groups[group_idx]['lr']
            self.optimizer.parameters += tuple(self.optimizer.param_groups[group_idx]['params'])

        self.parameters = self.optimizer.ori_parameters
        self.defaults = self.optimizer.defaults

        # update non-parallel optimizer attributes
        _update_adamw_var(self.optimizer)

    def _collect_main_grad_data(self):
        """ collect grads this dp rank takes into account. """
        for param in self.optimizer.parameters:
            self.grads.append(param.grad)

    def _get_model_param_buffer_dp_views(self):
        """ get shard metadata for each bucket among dp group. """
        view_items = []
        for buffer_index, buffer in enumerate(self.buffers):
            for bucket_index, bucket in enumerate(buffer.buckets):
                shard_start, shard_end = shard_bucket(bucket, group=self.data_parallel_group)
                view_items.append((buffer_index, bucket_index, shard_start, shard_end))

        return view_items

    def _copy_model_grads_to_main_grads(self):
        """
        Before distributed optimizer update, copy the the gradient elements this dp rank take charge to sharded gradient
        group. For fp16 gradients, a fp32 copy will be created and optimizer will update using fp32 gradients instead of
        original fp16 gradients.
        """
        def copy_group_grads(model_groups, main_groups):
            for model_group, main_group in zip(model_groups, main_groups):
                for model_param, main_param in zip(model_group, main_group):
                    if hasattr(model_param, 'use_zero3') and model_param.use_zero3:
                        main_param.grad = ops.cast(model_param.grad, mstype.float32)
                    else:
                        buffer_idx, bucket_idx = self.param_to_bucket_map.get(model_param)
                        range_map = self.param_ranges_map[buffer_idx][bucket_idx][model_param]
                        param_start, param_end = range_map['range_in_param']
                        main_param.grad.copy_(ops.cast(model_param.main_grad.view(-1)[param_start: param_end],
                                                       mstype.float32))

        copy_group_grads(self.param_fp32_groups, self.sharded_param_fp32_groups)
        copy_group_grads(self.param_fp16_groups, self.sharded_param_fp32_from_fp16_groups)

    def _copy_model_params_to_main_params(self):
        """
        Before distributed optimizer update, copy the the elements this dp rank take charge to sharded param
        group. For fp16 params, a fp32 copy will be created and optimizer will update the fp32 params instead of
        original fp16 params.
        """
        def copy_group_params(model_groups, main_groups):
            for model_group, main_group in zip(model_groups, main_groups):
                for model_param, main_param in zip(model_group, main_group):
                    if hasattr(model_param, 'use_zero3') and model_param.use_zero3:
                        main_param.copy_(ops.cast(model_param, mstype.float32))
                    else:
                        buffer_idx, bucket_idx = self.param_to_bucket_map.get(model_param)
                        range_map = self.param_ranges_map[buffer_idx][bucket_idx][model_param]
                        param_start, param_end = range_map['range_in_param']
                        main_param.copy_(ops.cast(model_param.view(-1)[param_start:param_end], mstype.float32))

        copy_group_params(self.param_fp32_groups, self.sharded_param_fp32_groups)
        copy_group_params(self.param_fp16_groups, self.sharded_param_fp32_from_fp16_groups)

    def _copy_main_params_to_model_params(self):
        """
        After distributed optimizer update, update result need to be copied back to section in param_data buffer.
        For parameters with fp16 data type, a fp32 copy is used for optimizer update, their update result will be cast
        to original param data type and be copied back to param_data buffer with fp16 data type.
        """
        def copy_group_params(main_groups, model_groups):
            for main_group, model_group in zip(main_groups, model_groups):
                for main_param, model_param in zip(main_group, model_group):
                    if hasattr(model_param, 'use_zero3') and model_param.use_zero3:
                        model_param.copy_(ops.cast(main_param, model_param.dtype))
                    else:
                        buffer_idx, bucket_idx = self.param_to_bucket_map.get(model_param)
                        range_map = self.param_ranges_map[buffer_idx][bucket_idx][model_param]
                        param_start, param_end = range_map['range_in_param']
                        model_param.view(-1)[param_start:param_end].copy_(ops.cast(main_param, model_param.dtype))

        copy_group_params(self.sharded_param_fp32_groups, self.param_fp32_groups)
        copy_group_params(self.sharded_param_fp32_from_fp16_groups, self.param_fp16_groups)

    def sync_gather_all_model_params(self, force_sync=False):
        """
        After distributed optimizer update, only the elements this dp rank take charge has been updated.
        This function conducts all-gather on data parallel group to get all updated parameters.
        """
        if not self.overlap_param_gather or force_sync:
            for buffer_index, bucket_index, shard_start, shard_end in self.param_buffer_dp_views:
                bucket = self.buffers[buffer_index].buckets[bucket_index]
                param_data_view = bucket.param_data[shard_start: shard_end]
                group = bucket.data_parallel_group
                if bucket.zero_comm_group is not None:
                    group = bucket.zero_comm_group["zero_shard_group"]
                param_data = comm_func.all_gather_into_tensor(param_data_view,
                                                              group=group)[0].reshape(-1)
                bucket.param_data.copy_(param_data)

    def _dispatch_gather_model_params(self, all_gather_handle_index):
        """ dispatch allgather for model params"""
        (buffer_index, bucket_index, shard_start, shard_end) = self.buffer_bucket_index_list[all_gather_handle_index]
        bucket = self.buffers[buffer_index].buckets[bucket_index]
        param_data_view = bucket.param_data[shard_start: shard_end]
        group = bucket.data_parallel_group
        if bucket.zero_comm_group is not None:
            group = bucket.zero_comm_group["zero_shard_group"]
        param_data, param_all_gather_handle = comm_func.all_gather_into_tensor(param_data_view,
                                                                               group=group,
                                                                               async_op=True)
        self.bucket_allgather_param_data_map[all_gather_handle_index] = param_data.reshape(-1)
        self.all_gather_handles[all_gather_handle_index] = param_all_gather_handle

    # pylint: disable=W0622
    @_no_grad()
    def _pre_forward_cell_hook(self, cell, input):
        for cell_param in cell.get_parameters(False):
            if not cell_param.requires_grad:
                continue
            if cell_param in self.param_to_all_gather_handle_index_map:
                all_gather_handle_index = self.param_to_all_gather_handle_index_map[cell_param]
                self._finish_param_sync_helper(all_gather_handle_index)
        return input

    def enable_pre_hook(self, module):
        """
        Enable pre hook for every cell in module to overlap allgather in fsdp.

        Inputs:
            module (Cell): network for training.
        """
        optim_cells = []

        def recursion_cells(cell):
            sub_cells_list = cell.cells()
            for sub_cell in sub_cells_list:
                optim_cells.append(sub_cell)
                recursion_cells(sub_cell)

        recursion_cells(module)
        for sub_cell in optim_cells:
            remove_cell_param_gather_handle = sub_cell.register_forward_pre_hook(self._pre_forward_cell_hook)
            self.remove_cell_param_gather_handles.append(remove_cell_param_gather_handle)

    def disable_pre_hook(self):
        """ disable pre hook for every cell in module to overlap allgather in fsdp."""
        while self.remove_cell_param_gather_handles:
            remove_cell_param_gather_handle = self.remove_cell_param_gather_handles.pop()
            remove_cell_param_gather_handle.remove()
            remove_cell_param_gather_handle = None


    def _finish_param_sync_helper(self, all_gather_handle_index):
        """ sycn allgather"""
        all_gather_handle = self.all_gather_handles[all_gather_handle_index]
        if all_gather_handle is not None:
            all_gather_handle.wait()
            buffer_index, bucket_index = self.all_gather_handle_index_to_bucket_index_map[all_gather_handle_index]
            bucket = self.buffers[buffer_index].buckets[bucket_index]
            param_data = self.bucket_allgather_param_data_map[all_gather_handle_index]
            bucket.param_data.copy_(param_data)
            self.bucket_allgather_param_data_map[all_gather_handle_index] = None
            self.all_gather_handles[all_gather_handle_index] = None
            next_all_gather_handle_index = all_gather_handle_index + 1
            if next_all_gather_handle_index < self.num_all_gather_handles:
                self._dispatch_gather_model_params(next_all_gather_handle_index)

    def finish_param_sync(self, model_index):
        """ sync allgather in vpp with delay param gather"""
        # model_index indicates the model id in vpp
        if model_index in self.model_index_to_all_gather_handle_index_map:
            return
        all_gather_handle_indices = self.model_index_to_all_gather_handle_index_map[model_index]
        for all_gather_handle_index in all_gather_handle_indices:
            self._finish_param_sync_helper(all_gather_handle_index)

    def save_opt_shard_strategy(self, file, extra_zero_list):
        """
        Save distributed optimizer shard info as json file.

        Inputs:
            file (str): Path to save optimizer shard strategy.
        """
        # only rank 0 of each data parallel group need save strategy file.
        if get_rank(self.data_parallel_group) != 0:
            return
        strategy = {}
        strategy['dp_rank_list'] = [int(s) for s in self.data_parallel_group.split('-') if s.isdigit()]
        # build parameter info
        strategy['param_info'] = {}
        for buffer_idx, buffer in enumerate(self.buffers):
            for param, value in buffer.param_index_map.items():
                new_name = pp_layer_rename(param.name, need_drop_suffix=True)
                start_idx, end_idx, bucket_idx = value
                this_buffer = self.buffers[buffer_idx]
                start_idx, end_idx, bucket_idx = this_buffer.param_index_map[param]
                strategy['param_info'][new_name] = {}
                strategy['param_info'][new_name]['range_map'] = (buffer_idx, bucket_idx, start_idx, end_idx)
                strategy['param_info'][new_name]['shape'] = param.shape
                strategy['param_info'][new_name]['dtype'] = str(param.dtype).lower()

        zero_list = []
        for param in self.zero3_param:
            new_name = pp_layer_rename(param.name, need_drop_suffix=True)
            zero_list.append(new_name)
            zero_list.append('exp_avg.' + new_name)
            zero_list.append('exp_avg_sq.' + new_name)
        strategy['zero3_params'] = zero_list + extra_zero_list

        # build buffer info
        strategy['buffer_info'] = {}
        for buffer_idx, buffer in enumerate(self.buffers):
            strategy['buffer_info'][buffer_idx] = {}
            strategy['buffer_info'][buffer_idx]['buckets'] = {
                bucket_idx: (
                    bucket.grad_data_numel,
                    bucket.numel_unpadded
                ) for bucket_idx, bucket in enumerate(buffer.buckets)
            }
            strategy['buffer_info'][buffer_idx]['buffer_size'] = buffer.numel
            strategy['buffer_info'][buffer_idx]['buffer_numel_unpadded'] = buffer.numel_unpadded
            strategy['buffer_info'][buffer_idx]['bucket_num'] = len(buffer.buckets)
        # save as json file
        flags = os.O_WRONLY | os.O_CREAT
        mode = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(file, flags, mode), 'w') as f:
            json.dump(strategy, f, indent=4)

    def state_dict(self, include_optim: bool = True):
        """ get parameter dict for save checkpoint. """
        param_dict = OrderedDict()

        for buffer_index, bucket_index, shard_start, shard_end in self.param_buffer_dp_views:
            param_range_map_this_bucket = self.param_ranges_map[buffer_index][bucket_index]
            # create dummy tensor for this bucket shard, which will be used to save checkpoint
            param_shard = np.zeros(shape=(shard_end - shard_start), dtype=np.float32)
            if include_optim:
                exp_avg_shard = np.zeros(shape=(shard_end - shard_start), dtype=np.float32)
                exp_avg_sq_shard = np.zeros(shape=(shard_end - shard_start), dtype=np.float32)
            # copy param data into dummy tensor
            for param, range_map in param_range_map_this_bucket.items():
                start_idx, end_idx = range_map['range_in_shard']
                param_id_in_opt = self.param_idx_in_opt.get(param.name)
                param_shard[start_idx:end_idx] = ms.Tensor(
                    shape=self.optimizer.parameters[param_id_in_opt].shape,
                    dtype=self.optimizer.parameters[param_id_in_opt].dtype,
                    init=Zero(),
                ).assign_value(self.optimizer.parameters[param_id_in_opt]).copy().asnumpy()
                if not include_optim:
                    continue
                exp_avg_shard[start_idx:end_idx] = ms.Tensor(
                    shape=self.optimizer.exp_avg[param_id_in_opt].shape,
                    dtype=self.optimizer.exp_avg[param_id_in_opt].dtype,
                    init=Zero(),
                ).assign_value(self.optimizer.exp_avg[param_id_in_opt]).copy().asnumpy()
                exp_avg_sq_shard[start_idx:end_idx] = ms.Tensor(
                    shape=self.optimizer.exp_avg_sq[param_id_in_opt].shape,
                    dtype=self.optimizer.exp_avg_sq[param_id_in_opt].dtype,
                    init=Zero(),
                ).assign_value(self.optimizer.exp_avg_sq[param_id_in_opt]).copy().asnumpy()
            shard_name = 'buffer_{}_bucket_{}'.format(buffer_index, bucket_index)
            param_dict[shard_name] = ms.Parameter(
                ms.Tensor(param_shard),
                name=shard_name,
                requires_grad=False
            )
            if not include_optim:
                continue
            param_dict['exp_avg.' + shard_name] = ms.Parameter(
                ms.Tensor(exp_avg_shard),
                name='exp_avg.' + shard_name,
                requires_grad=False
            )
            param_dict['exp_avg_sq.' + shard_name] = ms.Parameter(
                ms.Tensor(exp_avg_sq_shard),
                name='exp_avg_sq.' + shard_name,
                requires_grad=False
            )

        for param in self.zero3_param:
            param_dict[param.name] = ms.Parameter(param)
            if not include_optim:
                continue
            param_id_in_opt = self.param_idx_in_opt.get(param.name)
            exp_avg_param = self.optimizer.exp_avg[param_id_in_opt]
            exp_avg_sq_param = self.optimizer.exp_avg_sq[param_id_in_opt]
            param_dict['exp_avg.' + param.name] = ms.Parameter(exp_avg_param,
                                                               name='exp_avg.' + param.name, requires_grad=False)
            param_dict['exp_avg_sq.' + param.name] = ms.Parameter(exp_avg_sq_param,
                                                                  name='exp_avg_sq.' + param.name, requires_grad=False)

        if not include_optim:
            return param_dict

        # add state step to state_dict
        param_dict['state_step'] = self.optimizer.state_step

        # add learning rate and weight decay to state_dict
        for group_idx, lr in enumerate(self.optimizer.lrs):
            lr_name = lr.name
            param_dict[lr_name] = lr
            wd_name = lr_name.replace('learning_rate', 'weight_decay')
            param_dict[wd_name] = ms.Parameter(
                ops.Tensor(
                    self.optimizer.param_groups[group_idx]['weight_decay'],
                    dtype=ms.float64,
                ),
                name=wd_name,
                requires_grad=False,
            )
        return param_dict

    def sharded_state_dict(self, model_sharded_state_dict):
        """provide optim's sharded state dict based on the model's sharding info"""
        state_dict = {}
        for _, param in enumerate(self.parameters):
            exp_avg_name = "exp_avg." + param.name
            exp_avg_sq_name = "exp_avg_sq." + param.name
            model_name = param.name
            if model_name in model_sharded_state_dict and 'shard' in model_sharded_state_dict[model_name]:
                shard = list(model_sharded_state_dict[model_name]['shard'])
                shape = list(model_sharded_state_dict[model_name]['shape'])
            else:
                raise Exception(f"the input dict has no shard info for '{model_name}'.")
            state_dict[exp_avg_name] = {
                'shape': shape,
                'shard': tuple(shard),
                'opt_weight_shard_step': 0,
                'opt_weight_shard_size': 0
            }
            state_dict[exp_avg_sq_name] = {
                'shape': shape,
                'shard': tuple(shard),
                'opt_weight_shard_step': 0,
                'opt_weight_shard_size': 0
            }

        return state_dict
