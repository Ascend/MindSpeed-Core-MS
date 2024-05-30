import os
import types
from functools import wraps
import torch
import torch_npu
import torch.distributed

from megatron.training import get_args

TRANSPOSE_BF16_BLOCK_SIZE = 4096 * 4096


def reuse_fp32_param_distrib_optimizer_init_wrapper(init_func):
    @wraps(init_func)
    def reuse_fp32_param_distrib_optimizer_init(*args, **kwargs):
        init_func(*args, **kwargs)
        self = args[0]
        global_args = get_args()
        self.reuse_fp32_param = global_args.reuse_fp32_param if hasattr(global_args, "reuse_fp32_param") else False
        # A flag that disables the value subtraction when the `fp16_tensor_convert_to_fp32_tensor` function is invoked for the first time.
        self.first_sub_flag = True
        if self.reuse_fp32_param:
            from mindspeed.op_builder import AlgorithmOpBuilder
            reuse_data_ptr = AlgorithmOpBuilder().load().reuse_data_ptr
            data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)
            data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
            self.model_param_bucket_and_res_map = {}
            self.model_param_bucket_and_shard_main_param_int32_view_map = {}
            self.shard_main_param_res_buffers = []
            self.bucket_num_groups = []
            for buffer in self.buffers:
                self.bucket_num_group = []
                bucket_res_numel = 0
                res_numel = buffer.numel // data_parallel_world_size
                shard_main_param_res_buffer = torch.zeros(res_numel, dtype=torch.bfloat16, device=buffer.param_data.device)
                self.shard_main_param_res_buffers.append(shard_main_param_res_buffer)
                for bucket in buffer.buckets:
                    self.bucket_num_group.append(bucket.param_data.numel())
                    param_data_dp_numel = bucket.param_data.numel() // data_parallel_world_size
                    shard_main_param_int32_view_bucket = torch.empty(param_data_dp_numel, dtype=torch.int32, device=bucket.param_data.device)
                    reuse_data_ptr(
                        shard_main_param_int32_view_bucket,
                        buffer.param_data,
                        (bucket_res_numel * data_parallel_world_size) // 2 + max(0, data_parallel_rank - 1) * param_data_dp_numel // 2)
                    self.model_param_bucket_and_res_map[bucket.param_data] = self.shard_main_param_res_buffers[-1][bucket_res_numel: bucket_res_numel + param_data_dp_numel]
                    self.model_param_bucket_and_shard_main_param_int32_view_map[bucket.param_data] = shard_main_param_int32_view_bucket
                    bucket_res_numel += param_data_dp_numel
                self.bucket_num_groups.append(self.bucket_num_group)

            for model_fp16_params_this_group, shard_fp32_from_float16_group in zip(
                self.model_float16_groups, self.shard_fp32_from_float16_groups):
                for i, (model_param, shard_fp32_main_param) in enumerate(
                    zip(model_fp16_params_this_group, shard_fp32_from_float16_group)):
                    world_range = self._get_model_param_range_map(model_param)["gbuf_world_in_bucket"]
                    gbuf_index, _, bucket_id = self.model_param_gbuf_map[model_param]
                    model_param_buffer = self.buffers[gbuf_index].param_data
                    bucket_offset_in_buffer = sum(self.bucket_num_groups[gbuf_index][:bucket_id]) // 2
                    model_param_bucket = self.buffers[gbuf_index].buckets[bucket_id].param_data
                    model_param_bucket_numel_per_dp = model_param_bucket.numel() // data_parallel_world_size
                    shard_fp32_param_bucket_offset = world_range.start if data_parallel_rank == 0 else \
                        world_range.start - model_param_bucket_numel_per_dp * (1 + data_parallel_rank) // 2
                    shard_main_param_buffer_start = bucket_offset_in_buffer + shard_fp32_param_bucket_offset
                    reuse_data_ptr(shard_fp32_from_float16_group[i], model_param_buffer, shard_main_param_buffer_start)
            torch_npu.npu.empty_cache()
            self._copy_model_params_to_main_params = _copy_model_params_to_main_params
            self.fp16_tensor_convert_to_fp32_tensor = types.MethodType(fp16_tensor_convert_to_fp32_tensor, self)
            self.fp32_tensor_convert_to_fp16_tensor = types.MethodType(fp32_tensor_convert_to_fp16_tensor, self)    
    return reuse_fp32_param_distrib_optimizer_init


def _copy_model_params_to_main_params():
    pass


def fp16_tensor_convert_to_fp32_tensor(self):
    """
    res(0000) + bf16(pppp) -> fp32(0p0p0p0p)

    Transform the bf16 data and residuals data in the continuous memory block
    into the fp32 tensor through view transposition.
    """
    data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
    iteration = getattr(get_args(), "iteration", 0)
    for buffer in self.buffers:
        for bucket in buffer.buckets:
            bucket_param_data = bucket.param_data
            param_data_dp_numel = bucket_param_data.numel() // data_parallel_world_size
            bucket_res = self.model_param_bucket_and_res_map[bucket_param_data]
            if data_parallel_rank == 0:
                bucket_param_data[param_data_dp_numel:param_data_dp_numel * 2].copy_(bucket_param_data[:param_data_dp_numel])
            bucket_res_position = max(0, data_parallel_rank - 1) * param_data_dp_numel
            shard_fp32_main_param_view = bucket_param_data[bucket_res_position: bucket_res_position + param_data_dp_numel * 2]
            shard_main_param_int32_view_bucket = self.model_param_bucket_and_shard_main_param_int32_view_map[bucket_param_data]

            loops = param_data_dp_numel // TRANSPOSE_BF16_BLOCK_SIZE
            remain = param_data_dp_numel % TRANSPOSE_BF16_BLOCK_SIZE
            workspace = torch.zeros(
                TRANSPOSE_BF16_BLOCK_SIZE * 2, dtype=torch.bfloat16, device=bucket_res.device)
            residual_space = bucket_res
            bf16_space_dp_rank = max(1, data_parallel_rank)
            bf16_space = bucket_param_data[param_data_dp_numel * bf16_space_dp_rank :param_data_dp_numel * (bf16_space_dp_rank + 1)]
           
            for loop in range(loops):
                copy_start = loop * TRANSPOSE_BF16_BLOCK_SIZE
                copy_end = (loop + 1) * TRANSPOSE_BF16_BLOCK_SIZE
                workspace_convert_view = workspace[:TRANSPOSE_BF16_BLOCK_SIZE * 2]
                workspace[:TRANSPOSE_BF16_BLOCK_SIZE].copy_(residual_space[copy_start: copy_end])
                workspace[TRANSPOSE_BF16_BLOCK_SIZE:TRANSPOSE_BF16_BLOCK_SIZE * 2].copy_(bf16_space[copy_start: copy_end])
                shard_fp32_main_param_view[copy_start * 2: copy_end * 2].copy_(
                    workspace_convert_view.view(2, -1).transpose(1, 0).reshape(-1).contiguous())

            if remain > 0:
                workspace_convert_view = workspace[:remain * 2]
                workspace[:remain].copy_(residual_space[-remain:])
                workspace[remain:remain * 2].copy_(bf16_space[-remain:])
                shard_fp32_main_param_view[-remain * 2:].copy_(
                    workspace_convert_view.view(2, -1).transpose(1, 0).reshape(-1).contiguous())
           
            if not self.first_sub_flag or iteration != 0:
                shard_main_param_int32_view_bucket[:param_data_dp_numel].sub_(32768)


def fp32_tensor_convert_to_fp16_tensor(self):
    """
    fp32(0p0p0p0p) -> fp32(0'p0'p0'p0'p) -> res(0000) + bf16(pppp)

    Transform the fp32 tensor in the continuous memory block
    into the bf16 data and residual through view transposition.
    """
    data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
    for buffer in self.buffers:
        for bucket in buffer.buckets:
            bucket_param_data = bucket.param_data
            param_data_dp_numel = bucket_param_data.numel() // data_parallel_world_size
            bucket_res = self.model_param_bucket_and_res_map[bucket_param_data]
            shard_main_param_int32_view_bucket = self.model_param_bucket_and_shard_main_param_int32_view_map[bucket_param_data]
            shard_main_param_int32_view_bucket[:param_data_dp_numel].add_(32768)
            self.first_sub_flag = False
           
            bucket_res_position = max(0, data_parallel_rank - 1) * param_data_dp_numel
            shard_fp32_main_param_view = bucket_param_data[bucket_res_position: bucket_res_position + param_data_dp_numel * 2]

            loops = param_data_dp_numel // TRANSPOSE_BF16_BLOCK_SIZE
            remain = param_data_dp_numel % TRANSPOSE_BF16_BLOCK_SIZE
            workspace = torch.zeros(
                TRANSPOSE_BF16_BLOCK_SIZE * 2, dtype=torch.bfloat16, device=bucket_res.device)
            bf16_space_dp_rank = max(0, data_parallel_rank - 1)
            residual_space = bucket_res
            bf16_space = bucket_param_data[
                param_data_dp_numel * bf16_space_dp_rank :param_data_dp_numel * (bf16_space_dp_rank + 1)]
           
            for loop in range(loops):
                workspace_convert_view = workspace[:TRANSPOSE_BF16_BLOCK_SIZE * 2]
                workspace_convert_view.copy_(
                    shard_fp32_main_param_view[loop * TRANSPOSE_BF16_BLOCK_SIZE * 2: (loop + 1) * TRANSPOSE_BF16_BLOCK_SIZE * 2])
                temp = workspace_convert_view.view(-1, 2).transpose(1, 0).reshape(-1).contiguous()
                residual_space[loop * TRANSPOSE_BF16_BLOCK_SIZE: (loop + 1) * TRANSPOSE_BF16_BLOCK_SIZE].copy_(
                    temp[:TRANSPOSE_BF16_BLOCK_SIZE])
                bf16_space[loop * TRANSPOSE_BF16_BLOCK_SIZE: (loop + 1) * TRANSPOSE_BF16_BLOCK_SIZE].copy_(
                    temp[TRANSPOSE_BF16_BLOCK_SIZE: TRANSPOSE_BF16_BLOCK_SIZE * 2])
           
            if remain > 0:
                workspace_convert_view = workspace[:remain * 2]
                workspace_convert_view.copy_(shard_fp32_main_param_view[-remain * 2:])
                temp = workspace_convert_view.view(-1, 2).transpose(1, 0).reshape(-1).contiguous()
                residual_space[-remain:].copy_(temp[:remain])
                bf16_space[-remain:].copy_(temp[remain: remain * 2])

            if data_parallel_rank != 0:
                shard_fp32_main_param_view[param_data_dp_numel:param_data_dp_numel * 2].copy_(shard_fp32_main_param_view[:param_data_dp_numel])
