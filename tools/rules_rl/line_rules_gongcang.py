LINE_RULES = {
    "MindSpeed-LLM": {
    "mindspeed_llm/tasks/megatron_adaptor.py": [
"""                optimizer_config_init_wrapper
-            from mindspeed.optimizer.distrib_optimizer import reuse_fp32_param_distrib_optimizer_init_wrapper
+            from mindspeed.mindspore.optimizer.distrib_optimizer import reuse_fp32_param_distrib_optimizer_init_wrapper""",
"""
-        MegatronAdaptation.register(
-            'megatron.core.parallel_state.get_nccl_options', get_nccl_options_wrapper)
"""
    ],
    "mindspeed_llm/mindspore/mindspore_adaptor.py":[
        """-        MegatronAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel', GPTModel)
+        MegatronAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel', GPTModel, force_patch=True)"""
    ],
    "mindspeed_llm/core/datasets/blended_megatron_dataset_builder.py": [""" from ..parallel_state import get_pipeline_model_parallel_node_info
+from mindspore.communication import get_local_rank
 
 logger = logging.getLogger(__name__)""","""     if share_save:
         return rank == 0
     gpus_per_node = torch.cuda.device_count()
-    current_rank = torch.cuda.current_device()
+    current_rank = get_local_rank()
     if args.tensor_model_parallel_size > gpus_per_node:
         return mpu.get_tensor_model_parallel_rank() == 0
     return mpu.get_tensor_model_parallel_rank() == 0 and current_rank % gpus_per_node == 0"""],
    "mindspeed_llm/core/models/common/embeddings/rotary_pos_embedding.py": [
        """     for freq in freqs:
-        wavelen = 2 * math.pi / freq
+        wavelen = 2 / freq * math.pi
         if wavelen < high_freq_wavelen:""",
        """     inv_freq_mask = 1.0 - YarnRotaryPositionEmbedding.yarn_linear_ramp_mask(low, high, dim // 2).to(
-        device=freqs.device, dtype=torch.float32
+        dtype=torch.float32
     )""",
     """-    if self.inv_freq.device.type == 'cpu':
-        # move `inv_freq` to GPU once at the first micro-batch forward pass
-        self.inv_freq = self.inv_freq.to(device=torch.cuda.current_device())
""","""         mode = 1 if rotary_interleaved else 0
-        t = npu_rotary_position_embedding(t.contiguous(), cos_, sin_, mode).to(t.dtype)
+        t = torch_npu.npu_rotary_position_embedding(t.contiguous(), cos_, sin_, mode).to(t.dtype)"""
    ], 
    "mindspeed_llm/tasks/checkpoint/models.py": ["""         def _func_generator_set_weight(value):
             def func(self, **kwargs):
-                return _get_dst_obj(self, value, **kwargs).weight.data.copy_(kwargs.get('data'))
+                set_tensor = _get_dst_obj(self, value, **kwargs)
+                set_tensor.weight.data = kwargs.get('data')
+                return set_tensor.weight.data
             return func""",
             """         def _func_generator_set_bias(value):
             def func(self, **kwargs):
-                return _get_dst_obj(self, value, **kwargs).bias.data.copy_(kwargs.get('data'))
+                set_tensor = _get_dst_obj(self, value, **kwargs)
+                set_tensor.bias.data = kwargs.get('data')
+                return set_tensor.bias.data
             return func""",
     """             self.module = [AutoModelForCausalLM.from_pretrained(
-                load_dir, device_map=device_map, trust_remote_code=trust_remote_code, local_files_only=True
+                load_dir, trust_remote_code=trust_remote_code, local_files_only=True, low_cpu_mem_usage=False"""],
    "mindspeed_llm/tasks/models/transformer/multi_head_latent_attention.py":["""-        output = torch.matmul(input_, self.weight.t())
+        output = torch.matmul(input_.squeeze(1), self.weight.t())
+        output = output.unsqueeze(1)"""],
    },
    "megatron":{
        "core/tensor_parallel/cross_entropy.py":[
            """                 grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output
             )
 
-        return grad_input, None, None
+        return grad_input.to(torch.bfloat16), None, None"""
        ],
    },
    "mindspeed":{
    },

"mindspeed-rl": {
        "mindspeed_rl/config_cls/megatron_config.py": [
"""         self.swap_attention = False
+        self.ai_framework = \"pytorch\"""",
        ],
        "mindspeed_rl/models/base/base_training_engine.py": [
"""     def _get_forward_batch_info(batch_iter):
         batch = next(batch_iter)
         input_ids = batch['input_ids']
-        attention_mask_1d = generate_mask(input_ids, batch['prompt_length'] + batch['response_length']).to(
-            input_ids.device)
-        position_ids = torch.tensor(generate_position_ids(input_ids)).to(input_ids.device)
+        attention_mask_1d = generate_mask(input_ids, batch['prompt_length'] + batch['response_length'])
+        position_ids = torch.tensor(generate_position_ids(input_ids))
         attention_mask = get_tune_attention_mask(attention_mask_1d)
         return input_ids, attention_mask, position_ids, batch""",
"""         \"\"\"
         for k, v in data.items():
             if v is not None:
-                data[k] = v.to(next(self.model[0].parameters()).device)
+                data[k] = v #.to(next(self.model[0].parameters()).device)
         for model_module in self.model:
             model_module.eval()
         with torch.no_grad():""",
"""         grad_norm_list = []
         for k, v in data.items():
             if v is not None:
-                data[k] = v.to(next(self.model[0].parameters()).device)
+                data[k] = v #.to(next(self.model[0].parameters()).device)
         mini_batches = self._split_batches(data, batch_size=self.mini_batch_size_per_dp,
                                            shuffle_mini_batch=self.shuffle_mini_batch, dim=0)
         for model_module in self.model:"""
        
        ],
        "mindspeed_rl/models/loss/grpo_actor_loss_func.py": [
"""         pg_losses = -advantages * ratio
         pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
 
-        pg_mean_loss = F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
+        # pg_mean_loss = F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
+        tmp_mask = (pg_losses - pg_losses2) > 0
+        tmp_mask = tmp_mask.to(torch.int)
+        tmp = pg_losses * tmp_mask + pg_losses2 * (1 - tmp_mask)
+        pg_mean_loss = F.masked_mean(tmp, eos_mask)
         pg_mean_clipfrac = F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
 
         ref_approx_kl = ref_log_prob - log_prob"""
        ],
        "mindspeed_rl/trainer/base.py": [
""" from torch.utils.data import DataLoader
-from torch.utils.tensorboard import SummaryWriter
+# from torch.utils.tensorboard import SummaryWriter
""",
"""-        if kwargs.get("use_tensorboard", "") and self.wandb is None:
-            self.tensorboard = SummaryWriter()
+        # if kwargs.get("use_tensorboard", "") and self.wandb is None:
+        #     self.tensorboard = SummaryWriter()"""
       
        ],
        "mindspeed_rl/trainer/grpo_trainer_hybrid.py": [
"""         self.kwargs = kwargs
+        self.blocking = True""",
        ],
        "mindspeed_rl/trainer/utils/transfer_dock.py": [
"""             torch.stack([self.experience_data_status[single_column] == 1 for single_column in experience_columns]),
             dim=0,
         )
+        not_consumed_indexes = not_consumed_indexes.astype(torch.int32)
+        data_ready_indexes = data_ready_indexes.astype(torch.int32)
         usable_indexes = (not_consumed_indexes & data_ready_indexes).nonzero(as_tuple=True)[0]
 
         if len(usable_indexes) < mbs:""", 
        ],
        "mindspeed_rl/utils/compute.py": [
"""import torch
+from megatron.core import mpu
+from megatron.core.tensor_parallel import mappings""",
"""     global _ParallelState
     return _ParallelState
 
+class ReduceFromContextParallelRegionDPO(torch.autograd.Function):
+
+    @staticmethod
+    def symbolic(graph, input_):
+        torch.distributed.all_reduce(input_, op=torch.distributed.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
+        return input_
+
+    @staticmethod
+    def forward(ctx, input_):
+        torch.distributed.all_reduce(input_, op=torch.distributed.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
+        return input_
+
+    @staticmethod
+    def backward(ctx, grad_output):
+        return grad_output"""],
        "mindspeed_rl/workers/base_worker.py": [
""" 
             if get_tensor_model_parallel_rank(self.parallel_state, use_vllm) != 0 or \\
                     get_pipeline_model_parallel_rank(self.parallel_state, use_vllm) != 0:
+                batch_data_shape = batch_data_shape.numpy().tolist()
+                batch_data_dtype = batch_data_dtype.item()
                 if batch_data_dtype == 1:
                     batch_data[key] = torch.empty(batch_data_shape[0], batch_data_shape[1],
                                                   device=torch.cuda.current_device(),""",
"""-        index = index.cpu().numpy().tolist()
+        index = index.asnumpy().tolist()"""],
        "mindspeed_rl/workers/resharding/megatron_sharding_manager.py": [
"""     def offload_optimizer(self):
         for param_group in self.optimizer.optimizer.param_groups:
             for param in param_group['params']:
-                param.data = param.data.to(\"cpu\", non_blocking=False)
+                param.data = param.data.cpu(non_blocking=False) #to(\"cpu\", non_blocking=False)
         self.optimizer.optimizer.state = self._move_to_device(self.optimizer.optimizer.state, \"cpu\")
 
     def onload_optimizer(self):
         for param_group in self.optimizer.optimizer.param_groups:
             for param in param_group['params']:
-                param.data = param.data.to(torch.cuda.current_device(), non_blocking=False)
+                param.data = param.data.npu(non_blocking=False) #to(torch.cuda.current_device(), non_blocking=False)
         self.optimizer.optimizer.state = self._move_to_device(self.optimizer.optimizer.state,
                                                               torch.cuda.current_device())""",
"""         elif isinstance(data, dict):
             return {key: self._move_to_device(value, device) for key, value in data.items()}
         elif isinstance(data, torch.Tensor):
-            return data.to(device, non_blocking=False)
+            # return data.to(device, non_blocking=False)
+            if device == \"cpu\":
+                return data.cpu(non_blocking=False)
+            else:
+                return data.npu(non_blocking=False)
         else:
             return data""",
"""         if tensor.storage().size() != 0:
             cpu_state = self.tensor_to_cpu_states_map[tensor]
-            cpu_state.copy_(tensor, non_blocking=False)
+            self.tensor_to_cpu_states_map[tensor] = tensor.move_to("CPU", blocking = True)""",
"""         if tensor.storage().size() == 0:
             cpu_state = self.tensor_to_cpu_states_map[tensor]
-            tensor.storage().resize_(cpu_state.storage().size())
-            tensor.copy_(cpu_state, non_blocking=False)
+            tensor.storage().resize_(cpu_state.nbytes)
+            tmp = torch.Tensor(cpu_state.asnumpy()).npu()
+            tensor.copy_(tmp, non_blocking=False)
+            tmp = None"""       
        ],
        "mindspeed_rl/workers/resharding/memory_buffer.py": [
""" 
 def calc_padded_numel(shape: torch.Size, dtype: torch.dtype):
     \"\"\"for cuda memory alignment, make sure alignment by 128-bits\"\"\"
+    if dtype==torch.bfloat16:
+        dtype = torch.float32
     align_numel = 128 // torch.finfo(dtype).bits
     numel = shape.numel()
     return (numel + align_numel - 1) // align_numel * align_numel""",
""" 
     def offload(self):
         for memory_buffer in self.memory_buffers.values():
-            memory_buffer.data = memory_buffer.data.to(\"cpu\", non_blocking=False)
+            memory_buffer.data = memory_buffer.data.cpu(non_blocking=False) #.to(\"cpu\", non_blocking=False)
 
     def onload(self):
         for memory_buffer in self.memory_buffers.values():
-            memory_buffer.data = memory_buffer.data.to(torch.cuda.current_device(), non_blocking=False)
+            memory_buffer.data = memory_buffer.data.npu(non_blocking=False) #.to(torch.cuda.current_device(), non_blocking=False)"""
        ],
        "mindspeed_rl/workers/scheduler/launcher.py": [
""" from mindspeed_rl.workers.actor_hybrid_worker import ActorHybridWorker
 from mindspeed_rl.workers.reference_woker import ReferenceWorker
 from mindspeed_rl.workers.reward_woker import RewardWorker
-
+from mindspeed_rl.workers.scheduler.scheduler import create_worker_group_scheduler
+import socket
 
 def get_rl_resource_by_worker_type(rl_config: RLConfig, worker: Type[BaseWorker]):
     if (worker.__ray_actor_class__.__name__ ==""",
"""                 \"MASTER_PORT\": str(param.master_port) if param.master_port else \"\",
                 \"WORLD_SIZE\": str(param.world_size),
                 \"RANK\": str(param.rank_index),
+                \"MS_ROLE\": \"MS_WORKER\",
+                \"MS_WORKER_NUM\": str(param.world_size),
+                \"MS_NODE_ID\": str(param.rank_index), 
+                \"MS_SCHED_HOST\": param.master_addr if param.master_addr else \"localhost\",
+                \"MS_SCHED_PORT\": str(param.master_port) if param.master_port else \"\",
+                \"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES\": \"1\",
+                \"USE_RAY\":\"true\",
             }
         }
         return self.worker.options(""",
"""             **self.kwargs
         )
 
+    @staticmethod
+    def _get_free_port():
+        with socket.socket() as sock:
+            sock.bind((\"\", 0))
+            return sock.getsockname()[1]
+    @staticmethod
+    def _get_current_node_ip():
+        address = ray._private.services.get_node_ip_address()
+        # strip ipv6 address
+        return address.strip(\"[]\")
+
     def build_master_actor(self, placement_group, world_size) -> ray.actor.ActorHandle:
+        self.ms_sched_host = self._get_current_node_ip()
+        self.ms_sched_port= self._get_free_port()
+        _scheduler_name = f\"my_scheduler_{self.ms_sched_port}\"  # TODO 每个资源池要不一样的name
+        scheduler_actor = create_worker_group_scheduler(
+                name=_scheduler_name,
+                world_size=world_size, 
+            )
+        self.ms_sched_host = ray.get(scheduler_actor._get_current_node_ip.remote())
+        self.ms_sched_port = ray.get(scheduler_actor._get_free_port.remote())
+        scheduler_actor.init_process_group.remote()
+        scheduler_actor.get_status.remote()
+
         actor_handle = self.create_actor_handlers(
-            ActorHandlerParams(placement_group[0], world_size, 0, 0, None, None))
+            ActorHandlerParams(placement_group[0], world_size, 0, 0, self.ms_sched_host, self.ms_sched_port))
         self.actor_handlers.append(actor_handle)
         return actor_handle"""
        ],
        "mindspeed_rl/workers/resharding/utils.py": [
"""-        param_bytes = memory_buffer.data.detach().to(torch.float32).cpu().numpy().tobytes()
+        param_bytes = memory_buffer.data.detach().to(torch.float32).asnumpy().tobytes()""",
        ],
        "mindspeed_rl/workers/scheduler/scheduler.py": [
"""
import ray
 
import mindspore as ms
from mindspore import mint

from .worker import Worker
import socket
import os
 
@ray.remote
class WorkerGroupScheduler(Worker):
    def __init__(self):
        self.success = False
        with socket.socket() as sock:
            sock.bind((\"\", 0))
            self.port_ = sock.getsockname()[1]
        self.host_ = ray._private.services.get_node_ip_address()
        rank_zero_info = {
                \"MS_SCHED_HOST\": str(self.host_),
                \"MS_SCHED_PORT\": str(self.port_),
            }
        os.environ.update(rank_zero_info)

    def init_process_group(self):
        if not ms.communication._comm_helper._is_initialized():
            mint.distributed.init_process_group(
                backend=\"hccl\"
            )
            self.success = True
    
    def get_status(self):
        return self.success

    def _get_free_port(self):
        return self.port_

    def _get_current_node_ip(self):
        return self.host_
 
 
def create_worker_group_scheduler(name, world_size):
    env_vars: dict[str, str] = {
        \"MS_ROLE\": \"MS_SCHED\",
        \"MS_WORKER_NUM\": str(world_size),
        \"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES\": \"1\",
        'WORLD_SIZE': str(world_size),
        'WG_BACKEND': 'ray',
    }
    
    options = {'runtime_env': {'env_vars': env_vars}, 'name': name}
    return WorkerGroupScheduler.options(**options).remote()"""
        ],
        "mindspeed_rl/utils/pad_process.py": [
"""         # 获取当前行的截断索引
         trunc_idx = index_tensor[i].item()
         # 截断当前行
-        truncated_row = tensor[i, :trunc_idx].cpu()
+        truncated_row = tensor[i, :trunc_idx] #.cpu() 
         # 将截断后的行添加到列表中
         truncated_tensors.append(truncated_row)"""
        ],
        "mindspeed_rl/workers/scheduler/decorator.py": [
            """# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from enum import Enum
from functools import wraps, partial
from typing import Dict, List, Tuple
from types import FunctionType


# here we add a magic number of avoid user-defined function already have this attribute
MAGIC_ATTR = 'attrs_3141562937'


class Dispatch(Enum):
    RANK_ZERO = 0
    ONE_TO_ALL = 1
    ALL_TO_ALL = 2
    MEGATRON_COMPUTE = 3
    MEGATRON_PP_AS_DP = 4
    MEGATRON_PP_ONLY = 5
    MEGATRON_COMPUTE_PROTO = 6
    MEGATRON_PP_AS_DP_PROTO = 7
    DP_COMPUTE = 8
    DP_COMPUTE_PROTO = 9
    DP_COMPUTE_PROTO_WITH_FUNC = 10
    DP_COMPUTE_METRIC = 11
    DP_ALL_GATHER_TRAIN = 12
    DP_ALL_GATHER_INFER = 13
    


class Execute(Enum):
    ALL = 0
    RANK_ZERO = 1
    INFER = 2
    TRAIN = 3


def _split_args_kwargs_data_proto(chunks, *args, **kwargs):
    from mindspeed_llm.tasks.posttrain.rlxf.utils.protocol import DataProto, DataProtoFuture
    splitted_args = []
    for arg in args:
        if not isinstance(arg, (DataProto, DataProtoFuture)):
            raise TypeError(f"Argument {arg} must be an instance of DataProto or DataProtoFuture. Got {type(arg)}")
        splitted_args.append(arg.chunk(chunks=chunks))

    splitted_kwargs = {}
    for key, val in kwargs.items():
        if not isinstance(val, (DataProto, DataProtoFuture)):
            raise TypeError(f"Value for key {key} must be an instance of DataProto or DataProtoFuture. Got {type(val)}")
        splitted_kwargs[key] = val.chunk(chunks=chunks)

    return splitted_args, splitted_kwargs


def dispatch_one_to_all(worker_group, *args, **kwargs):
    args = tuple([arg] * worker_group.world_size for arg in args)
    kwargs = {k: [v] * worker_group.world_size for k, v in kwargs.items()}
    return args, kwargs


def dispatch_all_to_all(worker_group, *args, **kwargs):
    return args, kwargs


def collect_all_to_all(worker_group, output):
    return output


def dispatch_megatron_compute(worker_group, *args, **kwargs):
    \"\"\"
    User passes in dp data. The data is dispatched to all tp/pp ranks with the same dp
    \"\"\"
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    if not isinstance(worker_group, MegatronWorkerGroup):
        raise TypeError(f'worker_group must be MegatronWorkerGroup, Got {type(worker_group)}')
    
    all_args = []
    for arg in args:
        if not isinstance(arg, (Tuple, List)) or len(arg) != worker_group.dp_size:
            raise ValueError(f'Each argument must be a Tuple or List of length {worker_group.dp_size}, Got length {len(arg)}')
        transformed_args = []
        for i in range(worker_group.world_size):
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            transformed_args.append(arg[local_dp_rank])
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for k, v in kwargs.items():
        if not isinstance(v, (Tuple, List)) or len(v) != worker_group.dp_size:
            raise ValueError(f'Each argument in kwargs must be a Tuple or List of length {worker_group.dp_size}, Got length {len(v)}')
        transformed_v = []
        for i in range(worker_group.world_size):
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            transformed_v.append(v[local_dp_rank])
        all_kwargs[k] = transformed_v
    return all_args, all_kwargs


def collect_megatron_compute(worker_group, output):
    \"\"\"
    Only collect the data from the tp=0 and pp=last and every dp ranks
    \"\"\"
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    if not isinstance(worker_group, MegatronWorkerGroup):
        raise TypeError(f'worker_group must be MegatronWorkerGroup, Got {type(worker_group)}')
    output_in_dp = []
    pp_size = worker_group.get_megatron_global_info().pp_size
    for global_rank in range(worker_group.world_size):
        local_rank_info = worker_group.get_megatron_rank_info(rank=global_rank)
        if local_rank_info.tp_rank == 0 and local_rank_info.pp_rank == pp_size - 1:
            output_in_dp.append(output[global_rank])
    return output_in_dp


def dispatch_megatron_compute_data_proto(worker_group, *args, **kwargs):
    \"\"\"
    All the args and kwargs must be DataProto. The batch will be chunked by dp_size and passed to each rank
    \"\"\"
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    if not isinstance(worker_group, MegatronWorkerGroup):
        raise TypeError(f'worker_group must be an instance of MegatronWorkerGroup. Got {type(worker_group)}')

    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.dp_size, *args, **kwargs)
    return dispatch_megatron_compute(worker_group, *splitted_args, **splitted_kwargs)


def _concat_data_proto_or_future(output: List):
    from mindspeed_llm.tasks.posttrain.rlxf.utils.protocol import DataProto, DataProtoFuture
    import ray

    # make sure all the elements in output has the same type
    for single_output in output:
        if not isinstance(single_output, type(output[0])):
            raise TypeError(f"All elements in output must have the same type. Found {type(single_output)} and {type(output[0])}")

    output_prime = output[0]

    if isinstance(output_prime, DataProto):
        return DataProto.concat(output)
    elif isinstance(output_prime, ray.ObjectRef):
        return DataProtoFuture.concat(output)
    else:
        raise NotImplementedError


def collect_megatron_compute_data_proto(worker_group, output):
    \"\"\"
    Each output must be a DataProto. We concat the dim=0 of output
    \"\"\"
    from mindspeed_llm.tasks.posttrain.rlxf.utils.protocol import DataProto
    import ray

    output = collect_megatron_compute(worker_group, output)
    for single_output in output:
        if not isinstance(single_output, (DataProto, ray.ObjectRef)):
            raise TypeError(f"Expecting {single_output} to be DataProto or ray.ObjectRef, but got {type(single_output)}")

    return _concat_data_proto_or_future(output)


def dispatch_megatron_pp_as_dp(worker_group, *args, **kwargs):
    \"\"\"
    treat pp as dp.
    \"\"\"
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    if not isinstance(worker_group, MegatronWorkerGroup):
        raise TypeError(f'worker_group must be an instance of MegatronWorkerGroup. Got {type(worker_group)}')

    pp_size = worker_group.pp_size
    dp_size = worker_group.dp_size

    pp_dp_size = pp_size * dp_size

    all_args = []
    for arg in args:
        if not isinstance(arg, (List, Tuple)) or len(arg) != pp_dp_size:
            raise ValueError(f'Each argument in args must be a List or Tuple of length {pp_dp_size}, but got length {len(arg)}')
        transformed_args = []
        for i in range(worker_group.world_size):
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            local_pp_rank = worker_group.get_megatron_rank_info(rank=i).pp_rank
            # compute the rank in arg. Note that the order is dp then pp
            # Also note that the outputs within a pp group will be firstly allgathered, then only the output of pp0 will be collected.
            # For pp=2 dp=4, a batch of data "ABCDEFGH" should be dispatched and collected in below order:
            #    dispatch:       pp_allgther:        collect:
            #   dp 0 1 2 3      dp  0  1  2  3
            # pp +---------+  pp +-------------+
            #  0 | A C E G |   0 | AB CD EF GH |     ABCDEFGH
            #  1 | B D F H |   1 | AB CD EF GH |
            #    +---------+     +-------------+
            arg_rank = local_dp_rank * worker_group.pp_size + local_pp_rank

            transformed_args.append(arg[arg_rank])
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for k, v in kwargs.items():
        if not isinstance(v, (List, Tuple)) or len(v) != pp_dp_size:
            raise ValueError(f'Each argument in kwargs must be a List or Tuple of length {pp_dp_size}, but got length {len(v)}')
        transformed_v = []
        for i in range(worker_group.world_size):
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            local_pp_rank = worker_group.get_megatron_rank_info(rank=i).pp_rank
            # compute the rank in arg. Note that the order is dp then pp
            arg_rank = local_dp_rank * worker_group.pp_size + local_pp_rank
            transformed_v.append(v[arg_rank])
        all_kwargs[k] = transformed_v
    return all_args, all_kwargs


def collect_megatron_pp_as_dp(worker_group, output):
    \"\"\"
    treat pp as dp. Only collect data on tp=0
    \"\"\"
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    if not isinstance(worker_group, MegatronWorkerGroup):
        raise TypeError(f'worker_group must be an instance of MegatronWorkerGroup. Got {type(worker_group)}')
    output_in_dp = []
    for global_rank in range(worker_group.world_size):
        local_rank_info = worker_group.get_megatron_rank_info(rank=global_rank)
        if local_rank_info.tp_rank == 0 and local_rank_info.pp_rank == 0:
            output_in_dp.append(output[global_rank])
    return output_in_dp


def collect_megatron_pp_only(worker_group, output):
    \"\"\"
    Only collect output of megatron pp. This is useful when examine weight names as they are identical in tp/dp
    \"\"\"
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    if not isinstance(worker_group, MegatronWorkerGroup):
        raise TypeError(f'worker_group must be an instance of MegatronWorkerGroup. Got {type(worker_group)}')
    output_in_pp = []
    for global_rank in range(worker_group.world_size):
        local_rank_info = worker_group.get_megatron_rank_info(rank=global_rank)
        if local_rank_info.tp_rank == 0 and local_rank_info.dp_rank == 0:
            output_in_pp.append(output[global_rank])
    return output_in_pp


def dispatch_megatron_pp_as_dp_data_proto(worker_group, *args, **kwargs):
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    if not isinstance(worker_group, MegatronWorkerGroup):
        raise TypeError(f'worker_group must be an instance of MegatronWorkerGroup. Got {type(worker_group)}')

    pp_dp_size = worker_group.dp_size * worker_group.pp_size
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(pp_dp_size, *args, **kwargs)
    return dispatch_megatron_pp_as_dp(worker_group, *splitted_args, **splitted_kwargs)


def collect_megatron_pp_as_dp_data_proto(worker_group, output):
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    if not isinstance(worker_group, MegatronWorkerGroup):
        raise TypeError(f'worker_group must be an instance of MegatronWorkerGroup. Got {type(worker_group)}')

    output = collect_megatron_pp_as_dp(worker_group, output)
    return _concat_data_proto_or_future(output)


def dispatch_dp_compute(worker_group, *args, **kwargs):
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.worker_group import WorkerGroup
    if not isinstance(worker_group, WorkerGroup):
        raise TypeError(f'worker_group must be an instance of WorkerGroup. Got {type(worker_group)}')
    for arg in args:
        if not isinstance(arg, (Tuple, List)) or len(arg) != worker_group.world_size:
            raise ValueError(f'Each argument in args must be a Tuple or List of length {worker_group.world_size}')
    for _, v in kwargs.items():
        if not isinstance(v, (Tuple, List)) or len(v) != worker_group.world_size:
            raise ValueError(f'Each argument in kwargs must be a Tuple or List of length {worker_group.world_size}')
    return args, kwargs


def collect_dp_compute(worker_group, output):
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.worker_group import WorkerGroup
    if not isinstance(worker_group, WorkerGroup):
        raise TypeError(f'worker_group must be an instance of WorkerGroup. Got {type(worker_group)}')

    if len(output) != worker_group.world_size:
        raise ValueError(f'Output must have a length equal to world_size. Got length {len(output)}')
    return output


def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.worker_group import WorkerGroup
    if not isinstance(worker_group, WorkerGroup):
        raise TypeError(f'worker_group must be an instance of WorkerGroup. Got {type(worker_group)}')
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.world_size, *args, **kwargs)
    return splitted_args, splitted_kwargs


def dispatch_dp_compute_data_proto_with_func(worker_group, *args, **kwargs):
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.worker_group import WorkerGroup
    if not isinstance(worker_group, WorkerGroup):
        raise TypeError(f'worker_group must be an instance of WorkerGroup. Got {type(worker_group)}')

    if type(args[0]) != FunctionType:
        raise TypeError(f'The first argument must be a callable function. Got {type(args[0])}')  # NOTE: The first one args is a function!

    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.world_size, *args[1:], **kwargs)
    splitted_args_with_func = [[args[0]] * worker_group.world_size] + splitted_args
    return splitted_args_with_func, splitted_kwargs


def collect_dp_compute_data_proto(worker_group, output):
    import ray
    from mindspeed_llm.tasks.posttrain.rlxf.utils.protocol import DataProto
    for single_output in output:
        if not isinstance(single_output, (DataProto, ray.ObjectRef)):
            raise TypeError(f"Expecting {single_output} to be DataProto or ray.ObjectRef, but got {type(single_output)}")

    output = collect_dp_compute(worker_group, output)
    return _concat_data_proto_or_future(output)


def collect_dp_all_gather(worker_group, output, is_train):
    \"\"\"
    collect data in DP groups, in each DP group, only use the output return on TP_0 PP_last.
    \"\"\"
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    if not isinstance(worker_group, MegatronWorkerGroup):
        raise TypeError(f'worker_group must be an instance of MegatronWorkerGroup. Got {type(worker_group)}')
    output_in_dp = []
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.ray.base import get_actor_train_world_size
    actor_train_world_size = get_actor_train_world_size()
    pp_size = worker_group.get_megatron_global_info().pp_size if is_train else 1
    rank_offset = 0 if is_train else actor_train_world_size
    for global_rank in range(worker_group.world_size):
        is_train_node = global_rank < actor_train_world_size
        if is_train_node and not is_train:
            continue
        elif not is_train_node and is_train:
            continue
        local_rank_info = worker_group.get_megatron_rank_info(rank=global_rank)
        if local_rank_info.tp_rank == 0 and local_rank_info.pp_rank == pp_size - 1:
            output_in_dp.append(output[global_rank - rank_offset])
    return _concat_data_proto_or_future(output_in_dp)

collect_dp_train = partial(collect_dp_all_gather, is_train=True)
collect_dp_infer = partial(collect_dp_all_gather, is_train=False)



def get_predefined_dispatch_fn(dispatch_mode):
    predefined_dispatch_mode_fn = {
        Dispatch.ONE_TO_ALL: {
            'dispatch_fn': dispatch_one_to_all,
            'collect_fn': collect_all_to_all,
        },
        Dispatch.ALL_TO_ALL: {
            'dispatch_fn': dispatch_all_to_all,
            'collect_fn': collect_all_to_all,
        },
        Dispatch.MEGATRON_COMPUTE: {
            'dispatch_fn': dispatch_megatron_compute,
            'collect_fn': collect_megatron_compute,
        },
        Dispatch.MEGATRON_PP_AS_DP: {
            'dispatch_fn': dispatch_megatron_pp_as_dp,
            'collect_fn': collect_megatron_pp_as_dp,
        },
        Dispatch.MEGATRON_PP_ONLY: {
            'dispatch_fn': dispatch_one_to_all,
            'collect_fn': collect_megatron_pp_only
        },
        Dispatch.MEGATRON_COMPUTE_PROTO: {
            'dispatch_fn': dispatch_megatron_compute_data_proto,
            'collect_fn': collect_megatron_compute_data_proto
        },
        Dispatch.MEGATRON_PP_AS_DP_PROTO: {
            'dispatch_fn': dispatch_megatron_pp_as_dp_data_proto,
            'collect_fn': collect_megatron_pp_as_dp_data_proto
        },
        Dispatch.DP_COMPUTE: {
            'dispatch_fn': dispatch_dp_compute,
            'collect_fn': collect_dp_compute
        },
        Dispatch.DP_COMPUTE_PROTO: {
            'dispatch_fn': dispatch_dp_compute_data_proto,
            'collect_fn': collect_dp_compute_data_proto
        },
        Dispatch.DP_COMPUTE_PROTO_WITH_FUNC: {
            'dispatch_fn': dispatch_dp_compute_data_proto_with_func,
            'collect_fn': collect_dp_compute_data_proto
        },
        Dispatch.DP_COMPUTE_METRIC: {
            'dispatch_fn': dispatch_dp_compute_data_proto,
            'collect_fn': collect_dp_compute
        },
        Dispatch.DP_ALL_GATHER_TRAIN: {
            'dispatch_fn': dispatch_one_to_all,
            'collect_fn': collect_dp_train,
        },
        Dispatch.DP_ALL_GATHER_INFER: {
            'dispatch_fn': dispatch_one_to_all,
            'collect_fn': collect_dp_infer,
        },
    }
    return predefined_dispatch_mode_fn.get(dispatch_mode)


def get_predefined_execute_fn(execute_mode):
    \"\"\"
    Note that here we only asks execute_all and execute_rank_zero to be implemented
    Leave the choice of how these two functions handle argument 'blocking' to users
    \"\"\"
    predefined_execute_mode_fn = {
        Execute.ALL: {
            'execute_fn_name': 'execute_all'
        },
        Execute.RANK_ZERO: {
            'execute_fn_name': 'execute_rank_zero'
        },
        Execute.INFER: {
            'execute_fn_name': 'execute_infer'
        },
        Execute.TRAIN: {
            'execute_fn_name': 'execute_train'
        }
    }
    return predefined_execute_mode_fn.get(execute_mode)


def _check_dispatch_mode(dispatch_mode):
    if not isinstance(dispatch_mode, (Dispatch, Dict)):
        raise TypeError(f'dispatch_mode must be a Dispatch or a Dict. Got {type(dispatch_mode)}')
    if isinstance(dispatch_mode, Dict):
        necessary_keys = ['dispatch_fn', 'collect_fn']
        for key in necessary_keys:
            if key not in dispatch_mode:
                raise KeyError(f'key {key} should be in dispatch_mode if it is a dictionary')


def _check_execute_mode(execute_mode):
    if not isinstance(execute_mode, Execute):
        raise TypeError(f'execute_mode must be an instance of Execute. Got {type(execute_mode)}')


def _materialize_futures(*args, **kwargs):
    from mindspeed_llm.tasks.posttrain.rlxf.utils.protocol import DataProtoFuture
    new_args = []
    for arg in args:
        if isinstance(arg, DataProtoFuture):
            arg = arg.get()
        # add more type to materialize
        new_args.append(arg)
    for k, v in kwargs.items():
        if isinstance(v, DataProtoFuture):
            kwargs[k] = v.get()

    new_args = tuple(new_args)
    return new_args, kwargs


def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True, materialize_futures=True):
    _check_dispatch_mode(dispatch_mode=dispatch_mode)
    _check_execute_mode(execute_mode=execute_mode)

    def decorator(func):

        @wraps(func)
        def inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return func(*args, **kwargs)

        attrs = {'dispatch_mode': dispatch_mode, 'execute_mode': execute_mode, 'blocking': blocking}
        setattr(inner, MAGIC_ATTR, attrs)
        return inner

    return decorator
"""
        ],
        "mindspeed_rl/workers/scheduler/ray.py" :["""# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import ray


@ray.remote
class WorkerGroupRegisterCenter:

    def __init__(self, rank_zero_info):
        self.rank_zero_info = rank_zero_info

    def get_rank_zero_info(self):
        return self.rank_zero_info


def create_worker_group_register_center(name, info):
    return WorkerGroupRegisterCenter.options(name=name).remote(info)
"""],
"mindspeed_rl/workers/scheduler/worker.py": ["""# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
\"\"\"
the class for Worker
\"\"\"
import os
import socket
from dataclasses import dataclass
from .decorator import register, Dispatch


@dataclass
class DistRankInfo:
    tp_rank: int
    dp_rank: int
    pp_rank: int


@dataclass
class DistGlobalInfo:
    tp_size: int
    dp_size: int
    pp_size: int


class WorkerHelper:

    def _get_node_ip(self):

        def get_node_ip_by_sdk():
            if os.getenv("WG_BACKEND", None) == "ray":
                import ray
                return ray._private.services.get_node_ip_address()
            return None

        host_ipv4 = os.getenv("MY_HOST_IP", None)
        host_ipv6 = os.getenv("MY_HOST_IPV6", None)
        host_ip_by_env = host_ipv4 or host_ipv6
        host_ip_by_sdk = get_node_ip_by_sdk()

        host_ip = host_ip_by_env or host_ip_by_sdk
        return host_ip

    def _get_free_port(self):
        with socket.socket() as sock:
            sock.bind(('', 0))
            return sock.getsockname()[1]

    def get_availale_master_addr_port(self):
        return self._get_node_ip(), str(self._get_free_port())

    def _get_pid(self):
        return


class WorkerMeta:
    keys = [
        "WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK", "CUDA_VISIBLE_DEVICES",
        "MS_WORKER_NUM", "MS_ROLE", "MS_SCHED_HOST", "MS_SCHED_PORT"
    ]

    def __init__(self, store) -> None:
        self._store = store

    def to_dict(self):
        return {f"_{key.lower()}": self._store.get(f"_{key.lower()}", None) for key in WorkerMeta.keys}
from .ray import create_worker_group_register_center


# we assume that in each WorkerGroup, there is a Master Worker
class Worker(WorkerHelper):

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)

        # note that here we use int to distinguish
        disable_worker_init = int(os.environ.get('DISABLE_WORKER_INIT', 0))
        if disable_worker_init:
            return instance

        rank = os.environ.get("RANK", None)
        worker_group_prefix = os.environ.get("WG_PREFIX", None)

        # when decorator @ray.remote applies, __new__ will be called while we don't want to apply _configure_before_init
        if None not in [rank, worker_group_prefix] and 'ActorClass(' not in cls.__name__:
            instance._configure_before_init(f"{worker_group_prefix}_register_center", int(rank))

        return instance

    def _configure_before_init(self, register_center_name: str, rank: int):
        if not isinstance(rank, int):
            raise TypeError(f"rank must be int, instead of {type(rank)}")

        if rank == 0:
            master_addr, master_port = self.get_availale_master_addr_port()
            rank_zero_info = {
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
            }

            if os.getenv("WG_BACKEND", None) == "ray":
                from .ray import create_worker_group_register_center
                self.register_center = create_worker_group_register_center(name=register_center_name,
                                                                           info=rank_zero_info)

            os.environ.update(rank_zero_info)

    def __init__(self, cuda_visible_devices=None) -> None:
        # construct a meta from environment variable. Note that the import must be inside the class because it is executed remotely
        import os
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        self._rank = rank
        self._world_size = world_size

        ms_sched_host = os.environ["MS_SCHED_HOST"]
        ms_sched_port = os.environ["MS_SCHED_PORT"]

        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", world_size))
        local_rank = int(os.getenv("LOCAL_RANK", rank))

        store = {
            '_world_size': world_size,
            '_rank': rank,
            '_local_world_size': local_world_size,
            '_local_rank': local_rank,
            '_ms_worker_num': world_size,
            '_ms_sched_host': ms_sched_host,
            '_ms_sched_port': ms_sched_port
        }
        if cuda_visible_devices is not None:
            store['_cuda_visible_devices'] = cuda_visible_devices

        meta = WorkerMeta(store=store)
        self._configure_with_meta(meta=meta)

    def _configure_with_meta(self, meta: WorkerMeta):
        \"\"\"
        This function should only be called inside by WorkerGroup
        \"\"\"
        if not isinstance(meta, WorkerMeta):
            raise TypeError(
                f"Invalid meta type: expected WorkerMeta, got {type(meta).__name__}. "
                f"(Received value: {repr(meta)})"
            )
        self.__dict__.update(meta.to_dict())  # this is hacky
        for key in WorkerMeta.keys:
            val = self.__dict__.get(f"_{key.lower()}", None)
            if val is not None:
                os.environ[key] = str(val)
        os.environ["REDIS_STORE_SERVER_HOST"] = ""

    def get_master_addr_port(self):
        return self._master_addr, self._master_port

    def get_cuda_visible_devices(self):
        import os
        cuda_visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "not set")
        return cuda_visible_devices

    @property
    def world_size(self):
        return self._world_size

    @property
    def rank(self):
        return self._rank

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO_WITH_FUNC)
    def execute_with_func_generator(self, func, *args, **kwargs):
        ret_proto = func(self, *args, **kwargs)
        return ret_proto
"""]
    },

    "vllm": {
        "vllm/compilation/decorators.py": [
"""-from torch._dynamo.symbolic_convert import InliningInstructionTranslator"""
        ],
        "vllm/distributed/parallel_state.py": ["""-            cpu_group = torch.distributed.new_group(ranks, backend=\"gloo\")
+            cpu_group = torch.distributed.new_group(ranks, backend=\"hccl\")"""],
        "vllm/model_executor/layers/rotary_embedding.py": [
"""from vllm.model_executor.custom_op import CustomOp
+from vllm.platforms import current_platform""",
"""             query_pass = query[..., self.rotary_dim:]
             key_pass = key[..., self.rotary_dim:]
 
-        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(
-            positions.device)
+        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache #.to(positions.device)
         cos_sin = self.cos_sin_cache[torch.add(positions, offsets)
                                      if offsets is not None else positions]
         cos, sin = cos_sin.chunk(2, dim=-1)"""
        ],
        "vllm/model_executor/layers/sampler.py": [
"""-    scaled_min_p = min_p.unsqueeze_(dim=1) * top_probs
+    min_p = min_p.unsqueeze(dim=1)
+    scaled_min_p = min_p * top_probs
     tokens_to_remove = probs < scaled_min_p""",
"""             top_logprobs, top_token_ids = torch.topk(logprobs,
                                                      largest_num_logprobs,
                                                      dim=-1)
-            top_logprobs = top_logprobs.to('cpu')
-            top_token_ids = top_token_ids.to('cpu')
+            top_logprobs = top_logprobs
+            top_token_ids = top_token_ids
 
-        selected_logprobs = selected_logprobs.to('cpu')
-        ranks = ranks.to('cpu')
+        selected_logprobs = selected_logprobs
+        ranks = ranks"""
        ],
        "vllm/model_executor/layers/vocab_parallel_embedding.py": [
"""         added_vocab_end_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
     # torch.compile will fuse all of the pointwise ops below
     # into a single kernel, making it very fast
-    org_vocab_mask = (input_ >= org_vocab_start_index) & (
+    org_vocab_mask = torch.bitwise_and(input_ >= org_vocab_start_index,
         input_ < org_vocab_end_index)
-    added_vocab_mask = (input_ >= added_vocab_start_index) & (
+    added_vocab_mask = torch.bitwise_and(input_ >= added_vocab_start_index,
         input_ < added_vocab_end_index)
     added_offset = added_vocab_start_index - (
         org_vocab_end_index - org_vocab_start_index) - num_org_vocab_padding"""
        ],
        "vllm/model_executor/sampling_metadata.py": [
"""-            temperatures=temperatures_t.to(device=device, non_blocking=True),
-            top_ps=top_ps_t.to(device=device, non_blocking=True),
-            top_ks=top_ks_t.to(device=device, non_blocking=True),
-            min_ps=min_ps_t.to(device=device, non_blocking=True),
-            presence_penalties=presence_penalties_t.to(device=device,
-                                                       non_blocking=True),
-            frequency_penalties=frequency_penalties_t.to(device=device,
-                                                         non_blocking=True),
-            repetition_penalties=repetition_penalties_t.to(device=device,
-                                                           non_blocking=True),
-            prompt_tokens=prompt_t.to(device=device, non_blocking=True),
-            output_tokens=output_t.to(device=device, non_blocking=True),
+            temperatures=temperatures_t,
+            top_ps=top_ps_t,
+            top_ks=top_ks_t,
+            min_ps=min_ps_t,
+            presence_penalties=presence_penalties_t,
+            frequency_penalties=frequency_penalties_t,
+            repetition_penalties=repetition_penalties_t,
+            prompt_tokens=prompt_t,
+            output_tokens=output_t,
         )"""
        ],
        "vllm/utils.py": [
"""-    tensor = torch.from_numpy(padded_x).to(device)
+    tensor = torch.from_numpy(padded_x)""",
"""-    return t.to(device=target_device, non_blocking=True)
+    return t"""
        ],
        "vllm/worker/cache_engine.py": [
"""-            kv_cache.append(layer_kv_cache.view(kv_cache_shape))
+            if device == "cpu":
+                # kv_cache.append(layer_kv_cache.view(kv_cache_shape).cpu())
+                kv_cache.append(layer_kv_cache.numpy().reshape(kv_cache_shape))
+            else:
+                kv_cache.append(layer_kv_cache.view(kv_cache_shape))"""
        ],
    },
    "vllm-ascend":{
        "vllm_ascend/attention.py": [
"""         mask_value = torch.finfo(torch.float32).min
     else:
         mask_value = 1
-    attn_mask = torch.masked_fill(torch.zeros(size=(max_seq_len, max_seq_len)),
+    attn_mask = torch.masked_fill(torch.zeros((max_seq_len, max_seq_len)),
                                   mask_flag, mask_value).to(dtype)
     return attn_mask""",
"""             self._seq_len_cached = seqlen
             self.attn_mask_cache = generate_attn_mask(seqlen, dtype)
         if self.attn_mask_cache.device != device:
-            self.attn_mask_cache = self.attn_mask_cache.to(device)
+            self.attn_mask_cache = self.attn_mask_cache#.to(device)""",
"""         src_indices = src_to_dst[:, 0]
         dst_indices = src_to_dst[:, 1]
 
-        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(
-            dst_key_cache.device)
-        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(
-            dst_key_cache.device)
+        dst_key_cache[dst_indices] = src_key_cache[src_indices] # .to(dst_key_cache.device)
+        dst_value_cache[dst_indices] = src_value_cache[src_indices] # .to(dst_key_cache.device)""",
"""-        if self.w_kc is None or self.w_vc is None:
-            kv_b_proj_weight = self.kv_b_proj.weight.reshape(
-                self.num_heads, self.qk_nope_head_dim + self.v_head_dim,
-                self.kv_lora_rank)
-            self.w_kc = kv_b_proj_weight[:, :self.
-                                         qk_nope_head_dim, :].contiguous()
-            self.w_vc = kv_b_proj_weight[:,
-                                         self.qk_nope_head_dim:, :].transpose(
-                                             1, 2).contiguous()
+        # if self.w_kc is None or self.w_vc is None:
+        # print("self.kv_b_proj.weight shape is******", self.kv_b_proj.weight.shape, flush=True)
+        # print("reshape out is******", self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank, flush=True)
+        kv_b_proj_weight = self.kv_b_proj.weight.reshape(
+            self.num_heads, self.qk_nope_head_dim + self.v_head_dim,
+            self.kv_lora_rank)
+        self.w_kc = kv_b_proj_weight[:, :self.
+                                        qk_nope_head_dim, :].contiguous()
+        self.w_vc = kv_b_proj_weight[:,
+                                        self.qk_nope_head_dim:, :].transpose(
+                                            1, 2).contiguous()""",
"""                                       self.v_head_dim,
                                       dtype=query.dtype,
                                       device=query.device)
+            attn_output = torch.ones_like(attn_output, dtype=query.dtype)
             if (attn_metadata.block_tables is None
                     or attn_metadata.block_tables.numel() == 0):
                 assert attn_metadata.attn_mask is not None"""],
        "vllm_ascend/communicator.py": [
""" from typing import Optional
 
 import torch
+import torch.distributed as dist
 from torch.distributed import ProcessGroup
 from vllm.distributed.device_communicators.base_device_communicator import \\
     DeviceCommunicatorBase""",
"""                  device_group: Optional[ProcessGroup] = None,
                  unique_name: str = \"\"):
         super().__init__(cpu_group, device, device_group, unique_name)
-        # init device according to rank
-        self.device = torch.npu.current_device()
+        # init device according to local rank
+        local_rank = dist.get_rank(device_group)
+        self.device = torch.device(f\"npu:{local_rank}\")"""
        ],
        "vllm_ascend/ops/fused_moe.py": [
""" def fused_experts(hidden_states: torch.Tensor, w1: torch.Tensor,
                   w2: torch.Tensor, topk_weights: torch.Tensor,
                   topk_ids: torch.Tensor, top_k: int):""",
"""     gate_up_out_list = torch_npu.npu_grouped_matmul(x=[expanded_x],
-                                                    weight=[w1],
-                                                    split_item=2,
-                                                    group_list_type=0,
-                                                    group_type=0,
-                                                    group_list=expert_tokens)
+                                                    weight=[w1],
+                                                    split_item=3,
+                                                    group_list_type=0,
+                                                    group_type=0,
+                                                    group_list=expert_tokens)""",
"""     down_out_list = torch_npu.npu_grouped_matmul(x=[gate_up_out],
-                                                 weight=[w2],
-                                                 split_item=2,
-                                                 group_list_type=0,
-                                                 group_type=0,
-                                                 group_list=expert_tokens)
+                                                 weight=[w2],
+                                                 split_item=3,
+                                                 group_list_type=0,
+                                                 group_type=0,
+                                                 group_list=expert_tokens)""",
"""         original_scores = scores
         scores = scores + e_score_correction_bias.unsqueeze(0)
 
-    topk_group = 0 if topk_group is None else topk_group
-    num_expert_group = 0 if num_expert_group is None else num_expert_group
-
-    # TODO: Replace this piece of code to npu_group_topk when CANN and NNAL version is update
-    num_token = scores.shape[0]
-    group_scores = scores.view(num_token, num_expert_group,
-                               -1).max(dim=-1).values
-    group_idx = torch.topk(group_scores.to(torch.float32),
-                           k=topk_group,
-                           dim=-1,
-                           sorted=False)[1]
-    group_mask = torch.zeros_like(group_scores)
-    group_mask.scatter_(1, group_idx, 1)
-    score_mask = group_mask.unsqueeze(-1).expand(
-        num_token, num_expert_group,
-        scores.shape[-1] // num_expert_group).reshape(num_token, -1)
-    scores = scores.masked_fill(~score_mask.bool(), 0.0)
+    torch_npu.npu_group_topk(input=scores,
+                             out=scores,
+                             group_num=num_expert_group,
+                             k=topk_group)
 
     if e_score_correction_bias is not None:
         topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)[1]""",
"""     if renormalize:
         topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
 
-    return topk_weights, topk_ids.to(torch.int32)
-
+    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)""",
"""     down_out_list = torch.cat(down_out_list, dim=0)
     # TODO: Reorder device memory 2 times here, replace the current
     # implementation here when suitable operators become available.
+    routing_weights = topk_weights.to(down_out_list.dtype)
+    _, h_skip1 = down_out_list.shape
+    N_skip1, _ = topk_ids.shape
+    skip1_val = torch.zeros(N_skip1, h_skip1, dtype=down_out_list.dtype)
+    bias_val = torch.zeros(E, h_skip1, dtype=down_out_list.dtype)""",
"""     hidden_states = torch_npu.npu_moe_finalize_routing(
-        down_out_list,
-        skip1=None,
-        skip2=None,
-        bias=None,
-        scales=topk_weights,
-        expanded_src_to_dst_row=expanded_row_idx,
-        export_for_source_row=topk_ids)
+        down_out_list,
+        skip1=skip1_val,
+        skip2=None,
+        bias=bias_val,
+        scales=routing_weights,
+        expanded_src_to_dst_row=expanded_row_idx,
+        export_for_source_row=topk_ids)
""",
        ],
        "vllm_ascend/ops/layernorm.py": [
"""     import torch_npu
 
     if residual is not None:
-        x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight,
-                                                    self.variance_epsilon)
+        #x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight,
+        #                                            self.variance_epsilon)
+        residual = x + residual
+        x, _ = torch_npu.npu_rms_norm(residual, self.weight, self.variance_epsilon)
         return x, residual"""
        ],
        "vllm_ascend/ops/rotary_embedding.py": [
"""-    if self.cos_sin_cache.device != query.device:
-        self.cos_sin_cache = self.cos_sin_cache.to(query.device)
-    if self.cos_sin_cache.dtype != query.dtype:
-        self.cos_sin_cache = self.cos_sin_cache.to(query.dtype)""",
        ],

        "vllm_ascend/worker/worker.py": [
"""                              cache_block_size)
         num_npu_blocks = max(num_npu_blocks, 0)
         num_cpu_blocks = max(num_cpu_blocks, 0)
+        num_npu_blocks //= 100
         gc.collect()
         # TODO: don`t need impl this func after empty_cache in
         # Worker.determine_num_available_blocks() unified`""",
"""                         self.parallel_config, self.device_config)
             for _ in range(self.parallel_config.pipeline_parallel_size)
         ]
-        import torch_npu
-        for ve in range(self.parallel_config.pipeline_parallel_size):
-            num_layers = len(self.cache_engine[ve].gpu_cache)
-            for i in range(num_layers):
-                torch_npu.npu_format_cast(self.cache_engine[ve].gpu_cache[i],
-                                          2)
+        # import torch_npu
+        # for ve in range(self.parallel_config.pipeline_parallel_size):
+        #     num_layers = len(self.cache_engine[ve].gpu_cache)
+        #     for i in range(num_layers):
+        #         torch_npu.npu_format_cast(self.cache_engine[ve].gpu_cache[i],
+        #                                   2)
         self.gpu_cache = [
             self.cache_engine[ve].gpu_cache
             for ve in range(self.parallel_config.pipeline_parallel_size)"""
        ],
        "vllm_ascend.egg-info/entry_points.txt":["""[vllm.general_plugins]
ascend_enhanced_model = vllm_ascend:register_model

[vllm.platform_plugins]
ascend = vllm_ascend:register"""],
        "vllm_ascend.egg-info/PKG-INFO":["""Metadata-Version: 2.2
Name: vllm_ascend
Version: 0.1.dev68+g806235f.d20250308
Summary: vLLM Ascend backend plugin
Home-page: https://github.com/vllm-project/vllm-ascend
Author: vLLM-Ascend team
License: Apache 2.0
Project-URL: Homepage, https://github.com/vllm-project/vllm-ascend"""],
    }
}

SPECIAL_RULES = {
    "megatron":{
        "core/tensor_parallel/cross_entropy.py":
        [(r"masked_target\[target_mask\] = 0", "masked_target *= (1-target_mask)"),
         (r"predicted_logits\[target_mask\] = 0\.0", "predicted_logits *= (1-target_mask)"),
        ],
        "core/transformer/moe/moe_utils.py":
        [(r"index_copy_", "index_add_")],
        "core/transformer/moe/token_dispatcher.py":
        [(r"\.to\(\n?torch\.device\(\"cpu\"\)\)\n?", ""),
         (r"\.to\(\n?.*torch\.device\(\"cpu\"\),.*\n?.*\)", "")
         ],
        "legacy/model/module.py":
        [(r"val = float16_convertor\(val\)", "if val_typecheck.dtype == torch.float32:\n                val = float16_convertor(val)"),
         (r"val = val.float\(\)", "if val_typecheck.dtype in (torch.float16, torch.bfloat16):\n                val = val.float()")
        ],
        "training/training.py":
        [(r"log_string \+= \' \{\}\: \{\:\.6E\} \|\'\.format\(key, avg\)", "log_string += ' {}: {:.16f} |'.format(key, avg)"),
         (r"log_string \+= \' grad norm\: \{\:\.3f\} \|\'\.format\(grad_norm\)", "log_string += ' grad norm: {:.16f} |'.format(grad_norm)")   
        ]
    },
    "mindspeed":{
        "core/transformer/moe/token_dispatcher.py":
        [(r"\.to\(\n?torch\.device\(\"cpu\"\)\)\n?", ""),
         (r"\.to\(\n?.*torch\.device\(\"cpu\"\),.*\n?.*\)", ""),
         ]
    },
    "MindSpeed-LLM": {},
    "mindspeed_mm": {},
    "transformers": {
        "pytorch_utils.py":
        [(r"from safetensors\.torch import storage_ptr\, storage_size", "")]
    },
    "mindspeed-rl":{
        "mindspeed_rl/datasets/dataloader.py":
            [("pin_memory=True", "pin_memory=False")],
        "mindspeed_rl/datasets/prompt_dataset.py":
            [("pin_memory=True", "pin_memory=False")],
        # "":
        #     [("", "")],
    },
    "vllm":{
        
    },
    "vllm-ascend": {
        
    },
}

GENERAL_RULES = [
    ["@jit_fuser", ""],
    ["import transformer_engine as te", "import msadaptor.transformer_engine as te"],
    ["apex.", "msadaptor.apex."],
    ["_torch_npu_", "_msadaptor_npu_"],
    ["_torchvision_", "_msadaptorvision_"],
    ["torch_npu", "msadaptor.msadaptor_npu"],
    ["torchvision", "msadaptor.msadaptorvision"],
    ["torchair", "msadaptor.msadaptorair"],
    ["pytorch", "msadaptor"],
    ["torch", "msadaptor"],
    ["safetensors.msadaptor", "safetensors.torch"]
]

SHELL_RULES = [
    ["torchrun", "msrun"],
    ["--nproc_per_node", "--local_worker_num"],
    ["--nnodes $NNODES", "--worker_num $WORLD_SIZE"]
]

FILE_RULES = [
["torchvision", "msadaptorvision"],
["pytorch", "msadaptor"],
["torch", "msadaptor"]
]
