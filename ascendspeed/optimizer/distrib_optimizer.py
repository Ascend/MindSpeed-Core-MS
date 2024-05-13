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
            from ascendspeed.op_builder import AlgorithmOpBuilder
            reuse_data_ptr = AlgorithmOpBuilder().load().reuse_data_ptr
            data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)
            data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
            self.shard_res_and_model_param_map = {}
            self.shard_main_param_res_and_model_param_buckets = []
            self.shard_main_param_int32_view_buckets = []
            for buffer in self.buffers:
                for bucket in buffer.buckets:
                    old_param_data = bucket.param_data
                    old_param_data_dp_numel = old_param_data.numel() // data_parallel_world_size
                    shard_main_param_res_and_model_param_bucket = torch.zeros(
                        old_param_data_dp_numel * (data_parallel_world_size + 1), dtype=torch.bfloat16, device=old_param_data.device)
                    shard_main_param_int32_view_bucket = torch.empty(
                        old_param_data_dp_numel, dtype=torch.int32, device=old_param_data.device)
                    shard_main_param_res_and_model_param_bucket[old_param_data_dp_numel:].copy_(old_param_data)
                    reuse_data_ptr(bucket.param_data, shard_main_param_res_and_model_param_bucket, old_param_data_dp_numel)
                    reuse_data_ptr(shard_main_param_int32_view_bucket, shard_main_param_res_and_model_param_bucket, 0)
                    self.shard_main_param_res_and_model_param_buckets.append(shard_main_param_res_and_model_param_bucket)
                    self.shard_main_param_int32_view_buckets.append(shard_main_param_int32_view_bucket)
                    self.shard_res_and_model_param_map[bucket.param_data] = shard_main_param_res_and_model_param_bucket
                    del old_param_data

            for model_fp16_params_this_group, shard_fp32_from_float16_group in zip(
                self.model_float16_groups, self.shard_fp32_from_float16_groups):
                for i, (model_param, shard_fp32_main_param) in enumerate(
                    zip(model_fp16_params_this_group, shard_fp32_from_float16_group)):
                    param_range_map = self._get_model_param_range_map(model_param)
                    world_range = param_range_map["gbuf_world_in_bucket"]
                    gbuf_index, _, bucket_id = self.model_param_gbuf_map[model_param]
                    model_param_buffer = self.buffers[gbuf_index].buckets[bucket_id].param_data
                    shard_main_param_res_and_model_param_bucket = self.shard_res_and_model_param_map[model_param_buffer]
                    model_param_buffer_numel = model_param_buffer.numel()
                    model_param_buffer_numel_per_dp = model_param_buffer_numel // data_parallel_world_size
                    shard_main_param_world_range_start = (world_range.start - model_param_buffer_numel_per_dp * data_parallel_rank)
                    reuse_data_ptr(
                        shard_fp32_from_float16_group[i], 
                        shard_main_param_res_and_model_param_bucket, 
                        shard_main_param_world_range_start
                    )
            torch_npu.npu.empty_cache()
            self.fp16_tensor_convert_to_fp32_tensor = types.MethodType(fp16_tensor_convert_to_fp32_tensor, self)
            self.fp32_tensor_convert_to_fp16_tensor = types.MethodType(fp32_tensor_convert_to_fp16_tensor, self)    
    return reuse_fp32_param_distrib_optimizer_init



def fp16_tensor_convert_to_fp32_tensor(self):
    """
    res(0000) + bf16(pppp) -> fp32(0p0p0p0p)

    Transform the bf16 data and residuals data in the continuous memory block 
    into the fp32 tensor through view transposition. 
    """
    data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
    for shard_main_param_res_and_model_param_bucket, shard_main_param_int32_view_bucket in zip(
        self.shard_main_param_res_and_model_param_buckets, self.shard_main_param_int32_view_buckets):
        per_dp_numel = shard_main_param_res_and_model_param_bucket.numel() // (data_parallel_world_size + 1)
        
        if data_parallel_rank == 0:
            shard_main_param_res_and_model_param_bucket[
                per_dp_numel * data_parallel_world_size: per_dp_numel * (data_parallel_world_size + 1)].copy_(
                shard_main_param_res_and_model_param_bucket[per_dp_numel: per_dp_numel * 2])
        shard_fp32_main_param_view = shard_main_param_res_and_model_param_bucket[:per_dp_numel * 2]
        loops = per_dp_numel // TRANSPOSE_BF16_BLOCK_SIZE
        remain = per_dp_numel % TRANSPOSE_BF16_BLOCK_SIZE
        workspace = torch.zeros(
            TRANSPOSE_BF16_BLOCK_SIZE * 2, dtype=torch.bfloat16, device=shard_main_param_res_and_model_param_bucket.device)
        residual_space = shard_main_param_res_and_model_param_bucket[:per_dp_numel]
        bf16_space_dp_rank = data_parallel_world_size if data_parallel_rank == 0 else data_parallel_rank + 1
        bf16_space = shard_main_param_res_and_model_param_bucket[
            per_dp_numel * bf16_space_dp_rank :per_dp_numel * (bf16_space_dp_rank + 1)]
        
        for loop in range(loops):
            copy_start = per_dp_numel - (loop + 1) * TRANSPOSE_BF16_BLOCK_SIZE
            copy_end = per_dp_numel - loop * TRANSPOSE_BF16_BLOCK_SIZE
            workspace_convert_view = workspace[:TRANSPOSE_BF16_BLOCK_SIZE * 2]
            workspace[:TRANSPOSE_BF16_BLOCK_SIZE].copy_(residual_space[copy_start: copy_end])
            workspace[TRANSPOSE_BF16_BLOCK_SIZE:TRANSPOSE_BF16_BLOCK_SIZE * 2].copy_(bf16_space[copy_start: copy_end])
            shard_fp32_main_param_view[copy_start * 2: copy_end * 2].copy_(
                workspace_convert_view.view(2, -1).transpose(1, 0).reshape(-1).contiguous())

        if remain > 0:
            workspace_convert_view = workspace[:remain * 2]
            workspace[:remain].copy_(residual_space[:remain])
            workspace[remain:remain * 2].copy_(bf16_space[:remain])
            shard_fp32_main_param_view[:remain * 2].copy_(
                workspace_convert_view.view(2, -1).transpose(1, 0).reshape(-1).contiguous())
        
        if not self.first_sub_flag:
            shard_main_param_int32_view_bucket[:per_dp_numel].sub_(32768)
    

def fp32_tensor_convert_to_fp16_tensor(self):
    """
    fp32(0p0p0p0p) -> fp32(0'p0'p0'p0'p) -> res(0000) + bf16(pppp) 

    Transform the fp32 tensor in the continuous memory block 
    into the bf16 data and residual through view transposition.
    """
    data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
    for shard_main_param_res_and_model_param_bucket, shard_main_param_int32_view_bucket in zip(
        self.shard_main_param_res_and_model_param_buckets, self.shard_main_param_int32_view_buckets):
        per_dp_numel = shard_main_param_res_and_model_param_bucket.numel() // (data_parallel_world_size + 1)
        shard_fp32_main_param_view = shard_main_param_res_and_model_param_bucket[:per_dp_numel * 2]
        shard_main_param_int32_view_bucket[:per_dp_numel].add_(32768)
        self.first_sub_flag = False
        loops = per_dp_numel // TRANSPOSE_BF16_BLOCK_SIZE
        remain = per_dp_numel % TRANSPOSE_BF16_BLOCK_SIZE
        workspace = torch.zeros(
            TRANSPOSE_BF16_BLOCK_SIZE * 2, dtype=torch.bfloat16, device=shard_main_param_res_and_model_param_bucket.device)
        residual_space = shard_main_param_res_and_model_param_bucket[:per_dp_numel]
        bf16_space_dp_rank = data_parallel_world_size if data_parallel_rank == 0 else data_parallel_rank + 1
        bf16_space = shard_main_param_res_and_model_param_bucket[
            per_dp_numel * bf16_space_dp_rank :per_dp_numel * (bf16_space_dp_rank + 1)]
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

        if data_parallel_rank == 0:
            shard_main_param_res_and_model_param_bucket[per_dp_numel: per_dp_numel * 2].copy_(
                shard_main_param_res_and_model_param_bucket[
                    per_dp_numel * data_parallel_world_size: per_dp_numel * (data_parallel_world_size + 1)])
