LINE_RULES = {
    "MindSpeed-LLM": {
    "core/datasets/blended_megatron_dataset_builder.py": [""" from megatron.core import mpu
+from mindspore.communication import get_local_rank""","""-    current_rank = torch.cuda.current_device()
+    current_rank = get_local_rank()"""],
    "convert_ckpt.py":["""if __name__ == '__main__':
+    import mindspore as ms
+    ms.set_context(device_target = "CPU", pynative_synchronize=True)
+    import torch
+    torch.configs.set_pyboost(False)"""],
"mindspeed_llm/core/transformer/dot_product_attention.py":["""-from megatron.core.packed_seq_params import PackedSeqParams
+# from megatron.core.packed_seq_params import PackedSeqParams""",
"""-from mindspeed.core.context_parallel.ulysses_context_parallel import ulyssesattn_context_parallel
+# from mindspeed.core.context_parallel.ulysses_context_parallel import ulyssesattn_context_parallel""",
"""-from mindspeed.core.context_parallel.context_parallel_kv_cache import get_cache_policy
-from mindspeed.utils import get_actual_seq_len, compute_qkv_index, get_position_ids
+# from mindspeed.core.context_parallel.context_parallel_kv_cache import get_cache_policy
+# from mindspeed.utils import get_actual_seq_len, compute_qkv_index, get_position_ids""",
"""-def do_ring_context_parallel(q, k, v, head_num, softmax_scale, attn_mask, dropout_p=0., pse=None, pse_type=None, packed_seq_params=None):
+def do_ring_context_parallel(q, k, v, head_num, softmax_scale, attn_mask, dropout_p=0., pse=None, pse_type=None):#, packed_seq_params=None):""",
"""-    if args.shape_order == "TND":
-        packed_seq_params = PackedSeqParams(
-            cu_seqlens_q=torch.tensor(actual_seq_len, dtype=torch.int64, device=torch.cuda.current_device()),
-            cu_seqlens_kv=torch.tensor(actual_seq_len, dtype=torch.int64, device=torch.cuda.current_device())
-        )
+    # if args.shape_order == "TND":
+    #     packed_seq_params = PackedSeqParams(
+    #         cu_seqlens_q=torch.tensor(actual_seq_len, dtype=torch.int64, device=torch.cuda.current_device()),
+    #         cu_seqlens_kv=torch.tensor(actual_seq_len, dtype=torch.int64, device=torch.cuda.current_device())
+    #     )""",
"""-        q_index, kv_index = compute_qkv_index(torch.tensor(actual_seq_len, dtype=torch.int64, device=torch.cuda.current_device()).clone().tolist())
-        packed_seq_params.q_index = q_index
-        packed_seq_params.kv_index = kv_index
-        packed_seq_params.position_ids = get_position_ids()
+    #     q_index, kv_index = compute_qkv_index(torch.tensor(actual_seq_len, dtype=torch.int64, device=torch.cuda.current_device()).clone().tolist())
+    #     packed_seq_params.q_index = q_index
+    #     packed_seq_params.kv_index = kv_index
+    #     packed_seq_params.position_ids = get_position_ids()""",
"""         output = ringattn_context_parallel(q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p,
-                                           packed_seq_params)
+                                            actual_seq_len, actual_seq_len)""",
"""-def do_ulyssesattn_context_parallel(self,
-        query: Tensor,
-        key: Tensor,
-        value: Tensor,
-        attention_mask,
-        attn_mask_type,
-        packed_seq_params):
-    args = get_args()
+# def do_ulyssesattn_context_parallel(self,
+#         query: Tensor,
+#         key: Tensor,
+#         value: Tensor,
+#         attention_mask,
+#         attn_mask_type,
+#         packed_seq_params):
+#     args = get_args()""",
"""-    sparse_mode = args.sparse_mode
-    if attn_mask_type == AttnMaskType.no_mask:
-        sparse_mode = 0  # default mask
+#     sparse_mode = args.sparse_mode
+#     if attn_mask_type == AttnMaskType.no_mask:
+#         sparse_mode = 0  # default mask""",
"""-    scale = 1.0 / math.sqrt(
-        self.hidden_size_per_attention_head) if self.scale_mask_softmax.scale is None else self.softmax_scale
+#     scale = 1.0 / math.sqrt(
+#         self.hidden_size_per_attention_head) if self.scale_mask_softmax.scale is None else self.softmax_scale""",
"""-    self.ulysses_comm_para['cache_policy'] = get_cache_policy(
-        self.layer_number, args.context_parallel_kv_cache_policy, args.context_parallel_cache_interval
-    )
-    self.ulysses_comm_para['use_ulysses_allgather_kv'] = args.use_ulysses_allgather_kv
-    attn_para = dict()
-    attn_para['packed_seq_params'] = packed_seq_params
-    attn_para['attention_mask'] = attention_mask
-    attn_para['scale'] = scale
-    attn_para['pre_tokens'] = args.pre_tockens
-    attn_para['next_tokens'] = args.next_tockens
-    attn_para['keep_prob'] = 1 - self.attention_dropout.p
-    attn_para['sparse_mode'] = sparse_mode
-    output = ulyssesattn_context_parallel(query, key, value, attn_para, self.ulysses_comm_para)
+#     self.ulysses_comm_para['cache_policy'] = get_cache_policy(
+#         self.layer_number, args.context_parallel_kv_cache_policy, args.context_parallel_cache_interval
+#     )
+#     self.ulysses_comm_para['use_ulysses_allgather_kv'] = args.use_ulysses_allgather_kv
+#     attn_para = dict()
+#     attn_para['packed_seq_params'] = packed_seq_params
+#     attn_para['attention_mask'] = attention_mask
+#     attn_para['scale'] = scale
+#     attn_para['pre_tokens'] = args.pre_tockens
+#     attn_para['next_tokens'] = args.next_tockens
+#     attn_para['keep_prob'] = 1 - self.attention_dropout.p
+#     attn_para['sparse_mode'] = sparse_mode
+#     output = ulyssesattn_context_parallel(query, key, value, attn_para, self.ulysses_comm_para)
     
-    return output
+#     return output""",
"""-        if self.config.context_parallel_size > 1 and args.context_parallel_algo == "ulysses_cp_algo":
-            return do_ulyssesattn_context_parallel(self, query, key, value, attention_mask, attn_mask_type, packed_seq_params)
+        # if self.config.context_parallel_size > 1 and args.context_parallel_algo == "ulysses_cp_algo":
+        #     return do_ulyssesattn_context_parallel(self, query, key, value, attention_mask, attn_mask_type, packed_seq_params)""",
"""-            pse_type=self.pse_type, packed_seq_params=packed_seq_params)
+            pse_type=self.pse_type)#, packed_seq_params=packed_seq_params)"""],
    "mindspeed_llm/core/datasets/blended_megatron_dataset_builder.py": [""" from ..parallel_state import get_pipeline_model_parallel_node_info
+from mindspore.communication import get_local_rank
 
 logger = logging.getLogger(__name__)""","""     if share_save:
         return rank == 0
     gpus_per_node = torch.cuda.device_count()
-    current_rank = torch.cuda.current_device()
+    # current_rank = torch.cuda.current_device()
+    current_rank = get_local_rank()
     if args.tensor_model_parallel_size > gpus_per_node:
         return mpu.get_tensor_model_parallel_rank() == 0
     return mpu.get_tensor_model_parallel_rank() == 0 and current_rank % gpus_per_node == 0"""],
    "mindspeed_llm/core/models/common/embeddings/rotary_pos_embedding.py": [
        """     for freq in freqs:
-        wavelen = 2 * math.pi / freq
+        wavelen = 2 / freq * math.pi
         if wavelen < high_freq_wavelen:""",
        """     inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
-        device=freqs.device, dtype=torch.float32
+        dtype=torch.float32
     )""",
        """         sin = torch.sin(freqs)
-        return torch_npu.npu_rotary_mul(t, cos, sin).to(t.dtype)
+        mode = 1 if rotary_interleaved else 0
+        return torch_npu.npu_rotary_position_embedding(t, cos.to(t.dtype), sin.to(t.dtype), mode=mode).to(t.dtype)
+        # return torch_npu.npu_rotary_mul(t, cos, sin).to(t.dtype)
 
     rot_dim = freqs.shape[-1]""",
     """-    if self.inv_freq.device.type == 'cpu':
-        # move `inv_freq` to GPU once at the first micro-batch forward pass
-        self.inv_freq = self.inv_freq.to(device=torch.cuda.current_device())
+    # if self.inv_freq.device.type == 'cpu':
+    #     # move `inv_freq` to GPU once at the first micro-batch forward pass
+    #     self.inv_freq = self.inv_freq.to(device=torch.cuda.current_device())""",
"""     if args.use_fused_rotary_pos_emb:
-        t = torch_npu.npu_rotary_mul(t, cos_, sin_).to(t.dtype)
+        # t = torch_npu.npu_rotary_mul(t, cos_, sin_).to(t.dtype)
+        mode = 1 if rotary_interleaved else 0
+        t = torch_npu.npu_rotary_position_embedding(t.contiguous(), cos_, sin_, mode=mode).to(t.dtype)
     elif args.use_fused_rotary_pos_emb_new:
         mode = 1 if rotary_interleaved else 0
-        t = npu_rotary_position_embedding(t.contiguous(), cos_, sin_, mode).to(t.dtype)
+        t = torch_npu.npu_rotary_position_embedding(t.contiguous(), cos_, sin_, mode=mode).to(t.dtype)"""
    ], 
    "mindspeed_llm/legacy/data/data_samplers.py": [
        """                                        num_workers=args.num_workers,
-                                       generator=torch.Generator().manual_seed(args.seed),
                                        collate_fn=collator,
-                                       pin_memory=True)
+                                       pin_memory=False)"""
    ],

    "mindspeed_llm/tasks/checkpoint/loader_hf.py": [
        """     def queue_put(name, msg):
         logger.info(f"sending {name}")
+        for k, v in msg.items():
+            if isinstance(v, torch.Tensor):
+                msg[k] = v.asnumpy()
         msg["name"] = name
         queue.put(msg)"""
    ],
    "mindspeed_llm/tasks/checkpoint/models.py": [
"""-                return _get_dst_obj(self, value, **kwargs).weight.data.copy_(kwargs.get('data'))
+                set_tensor = _get_dst_obj(self, value, **kwargs)
+                set_tensor.weight.data = kwargs.get('data')
+                return set_tensor.weight.data""",
"""-                return _get_dst_obj(self, value, **kwargs).bias.data.copy_(kwargs.get('data'))
+                set_tensor = _get_dst_obj(self, value, **kwargs)
+                set_tensor.bias.data = kwargs.get('data')
+                return set_tensor.bias.data""",
"""         self.layers_self_attention_linear_qkv_caches = {"layer_idx": -1, "weight": None, "bias": None}
+        # self.__register_functions()
 
     def initialize_args(self):""",
"""             self.module = [AutoModelForCausalLM.from_pretrained(
-                load_dir, device_map=device_map, trust_remote_code=trust_remote_code, local_files_only=True
+                load_dir, trust_remote_code=trust_remote_code, local_files_only=True, low_cpu_mem_usage=False"""
    ],
    "mindspeed_llm/tasks/checkpoint/saver.py": [
        """+import numpy as np
 import torch""",
        """         val = queue.get()
+        if isinstance(val, dict):
+            for k, v in val.items():
+                if isinstance(v, np.ndarray):
+                    val[k] = torch.Tensor(v)
         if val == "exit":
             logger.error("Loader exited, exiting saver")
             exit(1)"""
    ],
    "mindspeed_llm/tasks/megatron_adaptor.py": [
        """         # For torch >= 2.2.0
-        torch.compile = torch.jit.script""",
        """-        megatron.core.jit.jit_fuser = dummy_jit""",
 """         MegatronAdaptation.register('megatron.core.transformer.transformer_config.TransformerConfig.__post_init__',
                                     transformer_config_post_init_wrapper)
-        MegatronAdaptation.register('torch.cuda.get_device_capability', get_device_capability)
+
         megatron.core.transformer.transformer_block.LayerNormImpl = PTNorm
         MegatronAdaptation.register('megatron.core.transformer.transformer_block.TENorm', PTNorm)""",
"""             'megatron.core.transformer.transformer_layer.TransformerLayer._get_layer_offset',
             get_layer_offset_wrapper)
-        MegatronAdaptation.register(
-            'megatron.core.parallel_state.get_nccl_options', get_nccl_options_wrapper)
+        # MegatronAdaptation.register(
+        #     'megatron.core.parallel_state.get_nccl_options', get_nccl_options_wrapper)""",
    ],
    "mindspeed_llm/tasks/preprocess/decoder_packed_mtf_dataset.py":[
        """-        position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
+        position_ids_tensor = torch.arange(seq_length, dtype=torch.long, device=data.device)
+        position_ids = position_ids_tensor.numpy()""",
"""-        eod_index = eod_index.clone()
+        eod_index = eod_index.copy()
""","""-        for j in range(eod_index.numel()):
+        for j in range(eod_index.size):""","""-        return position_ids.clone()
+        position_ids_tensor = torch.from_numpy(position_ids)
+        return position_ids_tensor.clone()"""
    ],
    "mindspeed_llm/tasks/models/transformer/multi_head_latent_attention.py":["""-        output = torch.matmul(input_, self.weight.t())
+        output = torch.matmul(input_.squeeze(1), self.weight.t())
+        output = output.unsqueeze(1)"""],
    "mindspeed_llm/tasks/models/transformer/multi_token_predication.py":["""             rotary_seq_len *= self.config.context_parallel_size
             rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)
+        
+        def self_enorm(decoder_input):
+            return self.enorm(decoder_input)
+        
+        def self_hnorm(hidden_input_ids):
+            return self.hnorm(hidden_input_ids)
+
         if self.recompute_layer_norm:
-            self.enorm_ckpt = CheckpointWithoutOutput()
-            enorm_output = self.enorm_ckpt.checkpoint(self.enorm, False, decoder_input)
-            self.hnorm_ckpt = CheckpointWithoutOutput()
-            hnorm_output = self.hnorm_ckpt.checkpoint(self.hnorm, False, hidden_input_ids)
+            enorm_output = tensor_parallel.random.checkpoint(self_enorm, False, decoder_input)
+            hnorm_output = tensor_parallel.random.checkpoint(self_hnorm, False, hidden_input_ids)""",
"""-        if self.recompute_layer_norm:
-            self.enorm_ckpt.discard_output()
-            self.hnorm_ckpt.discard_output()
-            hidden_states.register_hook(self.enorm_ckpt.recompute)
-            hidden_states.register_hook(self.hnorm_ckpt.recompute)
         # hidden_states -> [s, b, h]""","""+        def self_final_layernorm(hidden_states):
+            return self.final_layernorm(hidden_states)
         # Final layer norm.
         if self.final_layernorm is not None:
             if self.recompute_layer_norm:
-                self.finalnorm_ckpt = CheckpointWithoutOutput()
-                finalnorm_output = self.finalnorm_ckpt.checkpoint(self.final_layernorm, False, hidden_states)
+                finalnorm_output = tensor_parallel.random.checkpoint(self_final_layernorm, False, hidden_states)""",
"""-        if self.recompute_layer_norm:
-            self.finalnorm_ckpt.discard_output()
-            logits.register_hook(self.finalnorm_ckpt.recompute)
         if args.output_multiplier_scale:
             logits = logits * args.output_multiplier_scale"""],
    "mindspeed_llm/tasks/posttrain/orm/orm_model.py":[
        """         # we sometimes want to run our RM head in FP32, this allows it
-        autocast_context = torch.autocast(device_type=hidden_states.device.type, dtype=self.dtype)
+        #autocast_context = torch.autocast(device_type=hidden_states.device, dtype=self.dtype)""","""         # hidden_size is S x B x D
-        with autocast_context:
-            output = super().forward(hidden_states.to(self.weight.dtype))[0]  # [S x B x self.output_size]
+        # with autocast_context:
+        output = super().forward(hidden_states.to(self.weight.dtype))[0]  # [S x B x self.output_size]
         return output.to(torch.float32).transpose(0, 1).contiguous()  # [B x S x self.output_size]"""
    ],
    "mindspeed_llm/tasks/posttrain/rlxf/ray_trainer/grpo_trainer.py":[
        """+        port = 9112
         for resource_pool, class_dict in self.resource_pool_to_cls.items():
             worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
-            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
+            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, detached=True, port=port)
             spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
             all_wg.update(spawn_wg)
+            port += 1"""
    ],
    "mindspeed_llm/tasks/posttrain/rlxf/single_controller/base/register_center/ray.py":[
        """ def create_worker_group_register_center(name, info):
-    return WorkerGroupRegisterCenter.options(name=name).remote(info)
+    return WorkerGroupRegisterCenter.options(name=name).remote(info)"""
    ],
    "mindspeed_llm/tasks/posttrain/rlxf/single_controller/base/worker.py":[
        """                 import ray
                 return ray._private.services.get_node_ip_address()
-            elif os.getenv("WG_BACKEND", None) == "torch_rpc":
-                from mindspeed_llm.tasks.posttrain.rlxf.single_controller import get_ip_addr
-                return get_ip_addr()""",
"""     keys = [
-        "WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "CUDA_VISIBLE_DEVICES"
+        "WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK", "CUDA_VISIBLE_DEVICES",
+        "MS_WORKER_NUM", "MS_ROLE", "MS_SCHED_HOST", "MS_SCHED_PORT\"""",
"""     def to_dict(self):
         return {f"_{key.lower()}": self._store.get(f"_{key.lower()}", None) for key in WorkerMeta.keys}

+from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.register_center.ray import create_worker_group_register_center""",
"""         if disable_worker_init:
             return instance
 
-        rank = os.environ.get("RANK", None)
+        rank = os.environ.get("MS_ROLE", None)
         worker_group_prefix = os.environ.get("WG_PREFIX", None)
 
         # when decorator @ray.remote applies, __new__ will be called while we don't want to apply _configure_before_init
         if None not in [rank, worker_group_prefix] and 'ActorClass(' not in cls.__name__:
-            instance._configure_before_init(f"{worker_group_prefix}_register_center", int(rank))
+            instance._configure_before_init(f"{worker_group_prefix}_register_center", rank)
 
         return instance
 
-    def _configure_before_init(self, register_center_name: str, rank: int):
-        assert isinstance(rank, int), f"rank must be int, instead of {type(rank)}"
+    def _configure_before_init(self, register_center_name: str, rank: str):
 
-        if rank == 0:
-            master_addr, master_port = self.get_availale_master_addr_port()
+        self.local_ip = self._get_node_ip()
+        local_ip_info = {
+                "HCCL_IF_IP": self.local_ip,
+            }
+        os.environ.update(local_ip_info)
+        if rank == "MS_SCHED":
+            ms_sched_host, ms_sched_port = self.get_availale_master_addr_port()
             rank_zero_info = {
-                "MASTER_ADDR": master_addr,
-                "MASTER_PORT": master_port,
+                "MS_SCHED_HOST": ms_sched_host,
+                "MS_SCHED_PORT": ms_sched_port,""",
"""         self._rank = rank
         self._world_size = world_size
 
-        master_addr = os.environ["MASTER_ADDR"]
-        master_port = os.environ["MASTER_PORT"]
+        ms_sched_host = os.environ["MS_SCHED_HOST"]
+        ms_sched_port = os.environ["MS_SCHED_PORT"]
 
-        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
-        local_rank = int(os.getenv("LOCAL_RANK", "0"))
+        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", world_size))
+        local_rank = int(os.getenv("LOCAL_RANK", rank))
 
         store = {
             '_world_size': world_size,
             '_rank': rank,
             '_local_world_size': local_world_size,
             '_local_rank': local_rank,
-            '_master_addr': master_addr,
-            '_master_port': master_port
+            '_ms_worker_num': world_size,
+            '_ms_sched_host': ms_sched_host,
+            '_ms_sched_port': ms_sched_port""",
"""             if val is not None:
                 os.environ[key] = str(val)
-        os.environ["REDIS_STORE_SERVER_HOST"] = str(self._master_addr).replace("[", "").replace(
-            "]", "") if self._master_addr else ""
+        os.environ["REDIS_STORE_SERVER_HOST"] = ""
 
     def get_master_addr_port(self):
         return self._master_addr, self._master_port
 
     def get_cuda_visible_devices(self):
         import os
-        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
+        cuda_visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "not set")"""

    ],
    "mindspeed_llm/tasks/posttrain/rlxf/single_controller/ray/base.py":[
"""                  detached=False,
                  worker_names=None,
+                 port=None,""",
"""                                           bin_pack=bin_pack,
-                                          detached=detached)
+                                          detached=detached,
+                                          port=port)""",
"""         self._worker_names = worker_names
         self._world_size = len(worker_names)
 
-    def _init_with_resource_pool(self, resource_pool, ray_cls_with_init, bin_pack, detached):
+    def _init_with_resource_pool(self, resource_pool, ray_cls_with_init, bin_pack, detached, port):""",
"""             for local_rank in range(local_world_size):
                 rank += 1
 
+                import re
+                if rank == 0:
+                    cia_name = type(ray_cls_with_init.cls).__name__
+                    match = re.search(r"ActorClass\(([^)]+)\)", cia_name)  # ray.remote(Obj) -> "ActorClass(Obj)"
+                    cia_name = match.group(1) if match else cia_name  # "ActorClass(Obj)" -> "Obj"
+                    self._scheduler_name = f"{self.name_prefix}{cia_name}_scheduler"  # e.g. Worker_2:5
+                    scheduler_actor = create_worker_group_scheduler(name=self._scheduler_name, world_size=world_size,
+                                                                    name_prefix=self.name_prefix)
+                    scheduler_actor.get_status.remote()
+                
+                    register_center_actor = None
+                    for _ in range(120):
+                        if f"{self.name_prefix}_register_center" not in list_named_actors():
+                            time.sleep(1)
+                        else:
+                            register_center_actor = ray.get_actor(f"{self.name_prefix}_register_center")
+                            break
+                    assert register_center_actor is not None, f"failed to get register_center_actor: {self.name_prefix}_register_center in {list_named_actors(all_namespaces=True)}"
+                    rank_zero_info = ray.get(register_center_actor.get_rank_zero_info.remote())
+                    self._ms_sched_host, self._ms_sched_port = rank_zero_info['MS_SCHED_HOST'], rank_zero_info['MS_SCHED_PORT']""",
"""                     'WG_PREFIX': self.name_prefix,
                     'WG_BACKEND': 'ray',
+                    'MS_NODE_ID': str(rank),
+                    'RANK_CUSTOMIZE': str(rank),
                     'RAY_LOCAL_WORLD_SIZE': str(local_world_size),
                     'RAY_LOCAL_RANK': str(local_rank),
+                    "MS_ROLE": "MS_WORKER",
+                    "MS_WORKER_NUM": str(world_size),
+                    "MS_SCHED_HOST": self._ms_sched_host,
+                    "MS_SCHED_PORT": self._ms_sched_port,
+                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
+                    "USE_RAY": "true"
                 }
-                if rank != 0:
-                    env_vars['MASTER_ADDR'] = self._master_addr
-                    env_vars['MASTER_PORT'] = self._master_port
-
-                import re""",
"""                 if detached:
                     ray_cls_with_init.update_options({'lifetime': 'detached'})
 
+                os.system(f"export RANK={str(rank)}")
+                os.environ['RANK']=str(rank)""",
"""                 self._workers.append(worker)
                 self._worker_names.append(name)
 
-                if rank == 0:
-                    register_center_actor = None
-                    for _ in range(120):
-                        if f"{self.name_prefix}_register_center" not in list_named_actors():
-                            time.sleep(1)
-                        else:
-                            register_center_actor = ray.get_actor(f"{self.name_prefix}_register_center")
-                    assert register_center_actor is not None, f"failed to get register_center_actor: {self.name_prefix}_register_center in {list_named_actors(all_namespaces=True)}"
-                    rank_zero_info = ray.get(register_center_actor.get_rank_zero_info.remote())
-                    self._master_addr, self._master_port = rank_zero_info['MASTER_ADDR'], rank_zero_info['MASTER_PORT']""",
"""                 if method_name.startswith(prefix):
-                    original_method_name = remove_prefix(method_name, prefix)
+                    # only valid when Python >= 3.9
+                    original_method_name = method_name.removeprefix(prefix)"""
    ],
    "mindspeed_llm/tasks/posttrain/rlxf/training/core_algos.py":[
        """     pg_losses = -advantages * ratio
     pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
 
-    pg_loss = F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
+    # pg_loss = F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
+    tmp_mask = (pg_losses - pg_losses2) > 0
+    tmp_mask = tmp_mask.to(torch.int)
+    tmp = pg_losses * tmp_mask + pg_losses2 * (1 - tmp_mask)
+    pg_loss = F.masked_mean(tmp, eos_mask)""",
"""     reward = reward.reshape(-1, n_sample_batch)
-    reward = (reward - reward.mean(dim=1, keepdim=True)) / (reward.std(dim=1, keepdim=True) + 1e-8)
+
+    reward_np = reward.cpu().numpy()
+    reward_np = (reward_np - reward_np.mean(axis=1, keepdims=True)) / (reward_np.std(axis=1, keepdims=True) + 1e-8)
+    reward = torch.from_numpy(reward_np).to(reward.dtype)
+
     reward = reward.reshape(-1)"""
    ],
    "mindspeed_llm/tasks/posttrain/rlxf/training/parameter_mapping.py":["""         param_info_data = param.data
         torch.distributed.broadcast(param_info_data, group=cur_group[1], src=cur_group[0], async_op=False)
+        param.data = param_info_data"""],
    "mindspeed_llm/tasks/posttrain/rlxf/utils/protocol.py":[
        """         batch.append(data.batch)
         non_tensor_batch.append(data.non_tensor_batch)
-    batch = torch.stack(batch).contiguous()
+    # batch = torch.stack(batch).contiguous()
+    res_dict = {}
+    for k in batch[0].keys():
+        res_dict[k] = torch.stack([val[k] for val in batch]).contiguous()
+        new_batch_size = res_dict[k].shape[:-1]
+    batch = TensorDict(res_dict, batch_size=new_batch_size)""",
"""         for batch in data:
             batch_lst.append(batch.batch)
         if batch_lst[0] is not None:
-            new_batch = torch.cat(batch_lst, dim=0)
+            # new_batch = torch.cat(batch_lst, dim=0)
+            res_dict = {}
+            for k in batch_lst[0].keys():
+                res_dict[k] = torch.cat([batch[k] for batch in batch_lst])
+            new_batch_size = sum([int(batch.batch_size[0]) for batch in batch_lst])
+            new_batch = TensorDict(res_dict, batch_size=new_batch_size)""",
    ],
    "mindspeed_llm/tasks/posttrain/rlxf/workers/megatron_hybrid_engine.py": [
"""-            return data.to(device, non_blocking=True)
+            # return data.to(device, non_blocking=True)
+            # return data.to(device)
+            if device == "cpu":
+                return data.cpu()
+            else:
+                return data.cuda()""",
"""-                param.data = param.data.to(device, non_blocking=True)
+                # param.data = param.data.to(device, non_blocking=True)
+                # param.data = param.data.to(device)
+                if device == "cpu":
+                    param.data = param.data.cpu()
+                else:
+                    param.data = param.data.cuda()""",
    ],
    "mindspeed_llm/tasks/posttrain/rlxf/workers/actor_train_infer.py":[
"""                 tokens = batch["input_ids"]
-                tokens_list = tokens.view(-1).cpu().numpy().tolist()
+                tokens_list = tokens.view(-1).asnumpy().tolist()""",
"""                 for additional_key in self.args.dataset_additional_keys:
-                    additional_val = batch.get(additional_key).view(-1).cpu().numpy().tolist()
+                    additional_val = batch.get(additional_key).view(-1).asnumpy().tolist()
 
                     for _ in range(args.n_samples_per_prompt):""",
        """         device = next(self.node.actor.model[0].parameters()).device
-        data = data.to(device)
 
         dataloader = self.node.actor.make_minibatch_iterator(data=data)""",
         """         output = DataProto(meta_info={'metrics': metrics})
-        output = output.to('cpu')
         torch.cuda.empty_cache()""",
         """             data.batch['old_log_probs'] = old_log_probs
-            data = data.to('cpu')
         else:  # pp intermediate stage, no useful results
             data = None""",
             """     max_length = max_length if max_length % pad_multi_of == 0 else (max_length // pad_multi_of + 1) * pad_multi_of
     torch.distributed.all_reduce(max_length, op=torch.distributed.ReduceOp.MAX)
+    max_length = max_length.item()
 
     tokenizer = get_tokenizer()""",
                     """         # We make recompute_old_log_prob by default here.
-        data = data.to(next(self.model[0].parameters()).device)
         with torch.no_grad():""","""         # TODO: actually, we just need to control the sampling order.
 
-        data.batch['attention_mask'] = data.batch['attention_mask'].to(bool)
+        data.batch['attention_mask'] = data.batch['attention_mask'].to(torch.bool)
 
         batch_size = self.args.micro_batch_size""","""+        # TODO check
+        self.args.use_kv_cache = False
         metrics = {}""","""         torch.cuda.empty_cache()
-
+        self.args.use_kv_cache = True
         return metrics"""],
    "mindspeed_llm/tasks/posttrain/rlxf/workers/reference.py": [
        """         output = self.reference.compute_log_prob(data=data)
         if output is not None:
             output = DataProto.from_dict(tensors={'ref_log_prob': output})
-            output = output.to('cpu')
         torch.cuda.empty_cache()
         return output"""],
    "mindspeed_llm/tasks/posttrain/rlxf/workers/reward.py":[
        """     @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
     def compute_rm_score(self, data: DataProto):
-        data = data.to('cuda')
         output = self.rm.compute_rm_score(data=data)
         output = DataProto.from_dict(tensors={'rm_scores': output})
-        output = output.to('cpu')
         torch.cuda.empty_cache()
         return output""","""             self.args.iteration, self.args.num_floating_point_operations_so_far = load_checkpoint(
-                model, None, None, strict=False)
+                model, None, None, strict=True)#, strict=False)"""
    ],
    "mindspeed_llm/tasks/posttrain/rlxf/utils/megatron_memory_buffer.py":["""-                buffer.param_data.to('cpu', non_blocking=True)
+                # buffer.param_data.to('cpu', non_blocking=True)
+                pass""",
"""-                buffer.param_data.to(torch.cuda.current_device(), non_blocking=True)
+                # buffer.param_data.to(torch.cuda.current_device(), non_blocking=True)
+                pass"""],
"mindspeed_llm/tasks/posttrain/rlxf/workers/actor_hybrid.py": ["""-        return data.to(device, non_blocking=True)
+        # return data.to(device, non_blocking=True)
+        if device == "cpu":
+            return data.cpu()
+        else:
+            return data.cuda()""",
"""-            param.data = param.data.to(device, non_blocking=True)
+            # param.data = param.data.to(device, non_blocking=True)
+            if device == "cpu":
+                param.data = param.data.cpu()
+            else:
+                param.data = param.data.cuda()""",
"""         super().__init__()
+        # import mindspore
+        # mindspore.context.set_context(pynative_synchronize=True)
         self.config = config""",
         """-        data = data.to(device)
+        # data = data.to(device)""",
"""-        output = output.to('cpu')
+        # output = output.to('cpu')""",
"""-                nonzeros = torch.nonzero(prompt_token_ids == pad_id, as_tuple=False)
+                # nonzeros = torch.nonzero(prompt_token_ids == pad_id, as_tuple=False)
+                import mindspore as ms
+                nonzeros = ms.mint.nonzero(prompt_token_ids == pad_id, as_tuple=False)""",
"""-                token_ids = prompt_token_ids[:first_pad_index].cpu().numpy().tolist()
+                token_ids = prompt_token_ids[:first_pad_index].asnumpy().tolist()""",
"""-                    additional_val = additional_dict[additional_key][i].cpu().numpy().tolist()
+                    additional_val = additional_dict[additional_key][i].asnumpy().tolist()""",
"""-        responses = [response.cpu().numpy().tolist() for response in responses]
+        responses = [response.asnumpy().tolist() for response in responses]""",
"""-                    idx_list_per_step.append(tokens.view(-1).cpu().numpy().tolist())
+                    idx_list_per_step.append(tokens.view(-1).asnumpy().tolist())""",
"""-        responses = [response.cpu().numpy().tolist() for response in responses]
+        responses = [response.asnumpy().tolist() for response in responses]""",
"""-            data = data.to('cpu')
+            # data = data.to('cpu')"""],
"mindspeed_llm/tasks/posttrain/rlxf/workers/vllm_rollout/vllm_rollout.py": ["""-from torch.nn.utils.rnn import pad_sequence
+from torch.nn.utils.rnn_beta import pad_sequence"""],
    "mindspeed_llm/tasks/posttrain/utils.py":[
        """ from mindspeed_llm.tasks.preprocess.decoder_packed_mtf_dataset import build_train_valid_test_datasets as build_instruction_dataset
+from megatron.core.tensor_parallel import mappings""",
"""         data[key].append(val)
 
 
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
+        return grad_output""",
"""     # Step 1: Compute the local max value for numerical stability
-    z_max = logits.max(dim=-1, keepdim=True).values
+    z_max = logits.max(dim=-1, keepdim=True)[0]""",
"""     # Step 2: Perform all-reduce to get the global max across all processes
-    torch.distributed.all_reduce(
-        z_max,
-        op=torch.distributed.ReduceOp.MAX,
-        group=mpu.get_tensor_model_parallel_group()
-    )
+    z_max = ReduceFromContextParallelRegionDPO.apply(z_max)""",
"""     # Step 5: Perform all-reduce to get the global sum of exp(x - z_max) across all processes
-    torch.distributed.all_reduce(
-        local_sum_exp,
-        op=torch.distributed.ReduceOp.SUM,
-        group=mpu.get_tensor_model_parallel_group()
-    )
+    local_sum_exp = mappings.reduce_from_tensor_model_parallel_region(local_sum_exp)
 
     # Step 6: Compute the log of the global sum of exp(x - z_max)""",
     """         valid_length = loss_mask.sum(-1)
 
-        torch.distributed.all_reduce(
-            all_log_probs,
-            op=torch.distributed.ReduceOp.SUM,
-            group=mpu.get_tensor_model_parallel_group()
-        )
+        all_log_probs = mappings.reduce_from_tensor_model_parallel_region(all_log_probs)""",
"""         )
 
         if per_token:
-            torch.distributed.all_reduce(
-                per_token_log_probs,
-                op=torch.distributed.ReduceOp.SUM,
-                group=mpu.get_tensor_model_parallel_group()
-            )
+            per_token_log_probs = mappings.reduce_from_tensor_model_parallel_region(per_token_log_probs)""",
"""             group=mpu.get_context_parallel_group()
         )
 
-        torch.distributed.all_reduce(
-            all_log_probs,
-            op=torch.distributed.ReduceOp.SUM,
-            group=mpu.get_context_parallel_group()
-            )
+        all_log_probs = mappings.reduce_from_tensor_model_parallel_region(all_log_probs)
 
         if per_token:
-            torch.distributed.all_reduce(
-                per_token_log_probs,
-                op=torch.distributed.ReduceOp.SUM,
-                group=mpu.get_context_parallel_group()
-            )
+            per_token_log_probs = mappings.reduce_from_tensor_model_parallel_region(per_token_log_probs)
 
     return all_log_probs, valid_length, per_token_log_probs"""],
    "mindspeed_llm/tasks/posttrain/rlxf/single_controller/base/scheduler.py":[
        """
import ray
 
import mindspore as ms
from mindspore import mint

from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.worker import Worker
 
 
@ray.remote
class WorkerGroupScheduler(Worker):
    def __init__(self):
        self.success = False
        if not ms.communication._comm_helper._is_initialized():
            mint.distributed.init_process_group(
                backend="hccl"
            )
            self.success = True
    
    def get_status(self):
        return self.success
 
 
def create_worker_group_scheduler(name, world_size, name_prefix):
    env_vars = {
        "MS_ROLE": "MS_SCHED",
        "MS_WORKER_NUM": str(world_size),
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
        'WORLD_SIZE': str(world_size),
        'WG_PREFIX': name_prefix,
        'WG_BACKEND': 'ray',
    }
    options = {'runtime_env': {'env_vars': env_vars}, 'name': name}
    return WorkerGroupScheduler.options(**options).remote()"""
    ],"mindspeed_llm/training/arguments.py":[
"""-    if not args.moe_tp_extend_ep and args.moe_alltoall_overlap_comm and args.tensor_model_parallel_size > 1:
-        raise AssertionError(
-            '`--moe-alltoall-overlap-comm` do not support tp for now. only support with moe_tp_extend_ep when tp > 1.')
     if args.moe_zero_memory_num_layers is not None:""","""         raise AssertionError('args.shared_expert_gate does not support gradient_accumulation_fusion.')
+    if args.moe_alltoall_overlap_comm and args.gradient_accumulation_fusion:
+        raise AssertionError('moe_alltoall_overlap_comm does not support gradient_accumulation_fusion at the same time.')
 
 
 def _validate_mla(args):""",
 """             args.first_k_dense_replace))
+    if args.num_experts is not None and args.use_mc2 and args.moe_grouped_gemm:
+        raise AssertionError('Moe Grouped Gemm is not supported with mc2 in MOE model.')
 
     if args.num_layer_list:""","""     args.adaptive_recompute_profiling_step = 10
+    # args.moe_tp_extend_ep = False
     args.recompute_in_bubble = False""",
"""+        args.use_mc2 = False
         args.use_legacy_models = not args.use_mcore_models"""],
     "mindspeed_llm/training/utils.py":["""             slice_obj[dim] = slice(i, i + window_size)
-        slices.append(tensor[tuple(slice_obj)])
+        slices.append(tensor[tuple(slice_obj)].clone())"""],
    "mindspeed_llm/core/tensor_parallel/layers.py": ["""-        weight = torch.split(weight, weight.shape[0] // args_.output_layer_slice_num, dim=0)
+        weight = torch.chunk(weight, args_.output_layer_slice_num, 0)""", """+    wrapper = staticmethod(wrapper)
     return wrapper"""],
    "mindspeed_llm/core/models/gpt/gpt_model.py":["""         if not self.share_embeddings_and_output_weights and self.share_mtp_embedding_and_output_weight:
-            output_weight = self.output_layer.weight.detach()
+            from mindspore import ops
+            output_weight = ops.stop_gradient(self.output_layer.weight)"""],
    "pretrain_gpt.py": [
        """     return batch.values()
 
 
+class ReduceFromContextParallelRegion(torch.autograd.Function):
+
+    @staticmethod
+    def symbolic(graph, input_):
+        torch.distributed.all_reduce(input_, group=mpu.get_context_parallel_group())
+        return input_
+
+    @staticmethod
+    def forward(ctx, input_):
+        torch.distributed.all_reduce(input_, group=mpu.get_context_parallel_group())
+        return input_
+
+    @staticmethod
+    def backward(ctx, grad_output):
+        return grad_output
+
+
 def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):""",
        """     if args.context_parallel_size > 1:
-        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
-        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
-        loss = loss[0] / loss[1]
+        loss1 = torch.sum(losses.view(-1) * loss_mask)
+        loss2 = loss_mask.sum()
+
+        loss1 = ReduceFromContextParallelRegion()(loss1)
+        loss2 = ReduceFromContextParallelRegion()(loss2)
+
+        loss = loss1 / loss2
     else:
         loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()"""
    ],"mindspeed_llm/tasks/posttrain/rlxf/utils/loggers.py":["""             fmt_msg += f"iteration: {iteration} / {steps} | "
             for key in msg:
-                fmt_msg += f"{key} : {format(msg[key], '.4f')} | "
+                fmt_msg += f"{key} : {format(msg[key], '.16f')} | "
             fmt_msg = fmt_msg[:-2]"""]
    },
    "megatron":{
        "core/distributed/distributed_data_parallel.py": [
"""                     param_to_buffer[param].register_grad_ready(param)
+                
+                if hasattr(param, "main_grad"):
+                    return param.main_grad
+                return param.grad 
 
         return param_hook"""],
        "core/optimizer/__init__.py": [
            """         ## apex's FusedAdam is a drop-in replacement for torch's AdamW
         ## see https://github.com/NVIDIA/apex/blob/7b73b12361068a10b0f44844534613f252a5ea75/apex/optimizers/fused_adam.py#L16
-        from torch.optim import AdamW as Adam, SGD
+        
 
 from megatron.core import mpu""","""+    # Fake params to construct optmizer
+    if len(param_groups) == 0:
+        fake_params = torch.zeros([1,], dtype=torch.float, requires_grad=True)
+        fake_params.fake = True
+        fake_params.grad = fake_params.clone()
+        fake_params.main_grad = fake_params.clone()
+        param_groups.append({'params': fake_params, 'wd_mult': 0.0, 'lr_mult': 0.0, 'is_decoupled_lr': False})
+
     # Collect grad buffers for distributed optimizer.
     per_model_buffers = {}
     per_model_ep_buffers = {}"""
        ],
        "core/tensor_parallel/mappings.py": [
""" # Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 
 import torch
+import mindspore
 
 from megatron.core.parallel_state import (
     get_expert_model_parallel_group,""",
"""         else:
             # Unequal split (all2all-v)
             output = input.new_empty(
-                size=[sum(output_split_sizes)] + list(input.size()[1:]),
+                size=[int(sum(output_split_sizes))] + list(input.size()[1:]),
                 dtype=input.dtype,
-                device=torch.cuda.current_device(),
             )
-        torch.distributed.all_to_all_single(
+        mindspore.mint.distributed.all_to_all_single(
             output,
             input,
-            output_split_sizes=output_split_sizes,
-            input_split_sizes=input_split_sizes,
-            group=group,
-        )
+            output_split_sizes=output_split_sizes.tolist() if output_split_sizes is not None else None,
+            input_split_sizes=input_split_sizes.tolist() if output_split_sizes is not None else None,
+            group=group._name,)
         return output
 
     @staticmethod"""],
        "core/distributed/param_and_grad_buffer.py":[
            """
        assert param not in self.params_with_grad, 'Cannot set grad twice'
+        if param in self.params_with_grad:
+            return
+        # assert param not in self.params_with_grad, 'Cannot set grad twice'
         assert (
             self.ddp_config.overlap_grad_reduce"""
        ],
        "core/models/common/embeddings/rotary_pos_embedding.py":[
            """             rotary_seq_len = inference_params.max_sequence_length
         else:
-            if transformer.input_tensor is not None:
+            if transformer.input_tensor is not None and len(transformer.input_tensor.shape) > 1:"""
        ],
        "core/pipeline_parallel/schedules.py":[
""" from typing import Callable, Iterator, List, Optional, Union
 
 import torch
-from torch.autograd.variable import Variable
+from mindspore.ops import composite as C
+from mindspore.common.api import _pynative_executor
 
 from megatron.core import parallel_state
 from megatron.core.enums import ModelType""",
         """             forward_data_store.append(data)
+            output_tensor = None
+    _pynative_executor.end_graph(forward_step_func, output_tensor, input_tensor[0])
 
     if config.timers is not None:""","""-def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config):
+def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model):""",
"""     if not isinstance(input_tensor, list):
         input_tensor = [input_tensor]
         unwrap_input_tensor_grad = True
-    for x in input_tensor:
-        if x is not None:
-            x.retain_grad()
+    
 
     if not isinstance(output_tensor, list):""","""     if output_tensor_grad[0] is None and config.grad_scale_func is not None:
-        output_tensor[0] = config.grad_scale_func(output_tensor[0])
+        output_tensor_grad[0] = config.grad_scale_func(torch.ones_like(output_tensor[0]))
+    if output_tensor_grad[0] is None:
+        output_tensor_grad[0] = torch.ones_like(output_tensor[0])
 
-    if config.deallocate_pipeline_outputs:
-        custom_backward(output_tensor[0], output_tensor_grad[0])
-    else:
-        torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])
+    # set input tensor for backpropagation
+    if not parallel_state.is_pipeline_first_stage():
+        model.module.set_input_tensor(input_tensor[0])
+
+    # run backward
+    grad_ = C.GradOperation(True, True, True)
+    weights = model.trainable_params()
+    _pynative_executor.check_run(grad_, config.forward_step_func, weights, None, input_tensor[0])
+    _pynative_executor.grad(config.forward_step_func, grad_, weights, None, input_tensor[0], output_tensor_grad[0])
 
     # Collect the grad of the input_tensor.
     input_tensor_grad = [None]""","""             else:
                 input_tensor_grad.append(x.grad)
 
+    if not parallel_state.is_pipeline_first_stage():
+        model.module.set_input_tensor(None)
+
     # Handle single skip connection if it exists (encoder_hidden_state in
     # model with encoder and decoder).""","""     config = get_model_config(model)
+    config.forward_step_func = forward_step_func
     if config.timers is not None:""","""     forward_data_store = []
-    input_tensor, output_tensor_grad = None, None
+    input_tensor, output_tensor_grad = [None], [None]
     total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")""","""             if not forward_only:
-                backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)
+                backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model)
 
     # Run computation for last microbatch out of context handler (want to""","""     if not forward_only:
-        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)
+        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model)
 
     if config.finalize_model_grads_func is not None and not forward_only:""","""     config = get_model_config(model[0])
+    config.forward_step_func = forward_step_func
     if config.overlap_p2p_comm and config.batch_p2p_comm:""","""                 input_tensors[model_chunk_id].append(None)
+        if input_tensors[model_chunk_id][-1] is None:
+            input_tensors[model_chunk_id][-1] = torch.tensor(0, dtype=torch.int)
         input_tensor = input_tensors[model_chunk_id][-1]""","""         output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
-        input_tensor_grad = backward_step(
-            input_tensor, output_tensor, output_tensor_grad, model_type, config
-        )
+        input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model[model_chunk_id])
 
         # launch grad synchronization (custom grad sync)""","""     config = get_model_config(model)
+    config.forward_step_func = forward_step_func
     if config.overlap_p2p_comm:""","""                 if config.grad_sync_func is None or rank == 0:
                     enable_grad_sync()
 
-            input_tensor_grad = backward_step(
-                input_tensor, output_tensor, output_tensor_grad, model_type, config
-            )
+            input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model)
 
             if last_iteration:""","""             output_tensor_grad = recv_backward(send_tensor_shapes, config)
 
-            input_tensor_grad = backward_step(
-                input_tensor, output_tensor, output_tensor_grad, model_type, config
-            )
+            input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model)
 
             send_backward(input_tensor_grad, recv_tensor_shapes, config)"""],
        "core/tensor_parallel/cross_entropy.py":[
            """                 grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output
             )
 
-        return grad_input, None, None
+        return grad_input.to(torch.bfloat16), None, None"""
        ],
           "core/tensor_parallel/data.py":[
        """     # Move back to cpu and unpack.
     sizes_cpu = sizes_cuda.cpu()
+    sizes_np = sizes_cuda.numpy()""",
        """-        while sizes_cpu[offset + i] > 0:
+        while sizes_np[offset + i] > 0:""",
    ],
        "core/tensor_parallel/layers.py":[
            """-            grad_weight = grad_output.t().matmul(total_input)
+            grad_weight = grad_output.t().contiguous().matmul(total_input)"""
        ],
        "core/tensor_parallel/random.py":[""" from torch.utils.checkpoint import detach_variable
+from torch.autograd import recompute_instance
 
 from megatron.core.parallel_state import (""","""ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()
-
+        recompute_instance.set_recompute(True)
         with torch.no_grad():
             outputs = run_function(*args)
+        recompute_instance.set_recompute(False)""",
            """
         # Compute the forward pass.
-        detached_inputs = detach_variable(inputs)
+        # detached_inputs = detach_variable(inputs)
         with torch.enable_grad():
-            outputs = ctx.run_function(*detached_inputs)
+            # outputs = ctx.run_function(*detached_inputs)
+            outputs, f_vjp = torch.autograd.vjp(ctx.run_function, *inputs)""",
"""
         # filter out non tensor outputs for backward pass
-        outputs, args = zip(*filter(lambda x: torch.is_tensor(x[0]), zip(outputs, args)))
-        torch.autograd.backward(outputs, args)
-        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
+        grads = f_vjp(*args)"""
        ],
        "core/transformer/custom_layers/transformer_engine.py":[
            """         else:
             return core_attn_out
 
-
+'''
 if _te_version >= packaging.version.Version("1.9.0.dev0"):""",
"""-    TEGroupedLinear = None
-    TEColumnParallelGroupedLinear = None
-    TERowParallelGroupedLinear = None
+'''
+TEGroupedLinear = None
+TEColumnParallelGroupedLinear = None
+TERowParallelGroupedLinear = None"""
        ],
        "training/arguments.py":[
"""     # Args from environment
-    args.rank = int(os.getenv('RANK', '0'))
+    # args.rank = int(os.getenv('RANK', '0'))
+    args.rank = int(os.getenv('MS_NODE_ID', '0'))
     args.world_size = int(os.getenv("WORLD_SIZE", '1'))
 
     return args""","""     if args.moe_grouped_gemm:
         assert args.bf16, 'Currently GroupedGEMM for MoE only supports bf16 dtype.'
-        dc = torch.cuda.get_device_capability()
-        assert dc[0] >= 8, "Unsupported compute capability for GroupedGEMM kernels."
+        # dc = torch.cuda.get_device_capability()
+        # assert dc[0] >= 8, "Unsupported compute capability for GroupedGEMM kernels."
 
     if args.weight_decay_incr_style == 'constant':
         assert args.start_weight_decay is None"""]
    },
    "mindspeed":{
        "op_builder/algorithm_builder.py": [
"""
 from mindspeed.op_builder.builder import MindSpeedOpBuilder
+from mindspore.utils import _reuse_data_ptr
 
 
 class AlgorithmOpBuilder(MindSpeedOpBuilder):
     OP_NAME = "algorithm"
+    reuse_data_ptr = _reuse_data_ptr
 
     def __init__(self):
         super(AlgorithmOpBuilder, self).__init__(self.OP_NAME)
 
     def sources(self):
-        return ['ops/csrc/algorithm/algorithm.cpp']
+        raise NotImplementedError
+    def load(self, verbose=True):
+        return self        
"""],
        "core/transformer/moe/grouped_gemm_util.py":[""" # Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
 
 import torch
-from mindspeed.ops.npu_all_to_all_all_gather_bmm import npu_alltoall_allgather_bmm
-from mindspeed.ops.npu_bmm_reduce_scatter_all_to_all import npu_bmm_reducescatter_alltoall
+#from mindspeed.ops.npu_all_to_all_all_gather_bmm import npu_alltoall_allgather_bmm
+#from mindspeed.ops.npu_bmm_reduce_scatter_all_to_all import npu_bmm_reducescatter_alltoall
 
 
 def grouped_gemm_is_available():
""",""" 
 class Ops:
     @staticmethod
-    def gmm(a, b, batch_sizes, trans_b=False, gemm_fusion=False, original_weight=None):
+    def gmm(a, b, batch_sizes, trans_b=False, gemm_fusion=False, original_weight=None, group_type=0):
         from mindspeed.ops.gmm import npu_gmm
 
         if trans_b:
             b = b.t()
         group_list = torch.cumsum(batch_sizes, dim=0)
-        return npu_gmm(a, b, bias=None, group_list=group_list, group_type=0, gemm_fusion=gemm_fusion, original_weight=original_weight)
+        return npu_gmm(a, b, bias=None, group_list=group_list, group_type=group_type, gemm_fusion=gemm_fusion, original_weight=original_weight)
 
 
 ops = Ops"""],
        "core/data_parallel/distributed_data_parallel.py":[
            """     self.grad_accs = []
     for param in self.module.parameters():
         if param.requires_grad:
-            # Expand so we get access to grad_fn.
-            param_tmp = param.expand_as(param)
-            # Get the gradient accumulator function.
-            grad_acc = param_tmp.grad_fn.next_functions[0][0]
-            grad_acc.register_hook(self._make_param_hook(param, self.param_to_buffer))
-            self.grad_accs.append(grad_acc)
+            param.register_hook(self._make_param_hook(param, self.param_to_buffer))"""],
        "core/transformer/moe/experts.py":[
        """-    return torch.zeros(zeros_shape, dtype=input_.dtype, layout=input_.layout, device=input_.device)
+    return torch.zeros(zeros_shape, dtype=input_.dtype, device=input_.device)""","""     else:
         w1 = self.weight1.view(self.config.hidden_size, -1)
         w2 = self.weight2.view(-1, self.config.hidden_size)
-    group_list = torch.cumsum(tokens_per_expert, dim=0)
+    #group_list = torch.cumsum(tokens_per_expert, dim=0)
+    group_list = tokens_per_expert
     if get_args().moe_alltoall_overlap_comm:
         return grouped_mlp_with_comp_and_comm_overlap_all2all(permuted_local_hidden_states, w1, w2,
                                                               (self.weight1, self.weight2, self.activation_func,
-                                                               group_list, ctx.layer_number),
-                                                              ctx=ctx)
+                                                               group_list, self.layer_number))
     else:
         return grouped_mlp_with_comp_and_comm_overlap_allgather(permuted_local_hidden_states, w1, w2,
                                                                 (self.weight1, self.weight2, self.activation_func,"""],
        "megatron_adaptor.py":[
"""     aspm.register_patch('transformer_engine.pytorch.DotProductAttention', torch.nn.Module, create_dummy=True)
-    aspm.register_patch('transformer_engine.pytorch.Linear', torch.nn.Module, create_dummy=True)
+
     aspm.register_patch('transformer_engine.common.recipe.DelayedScaling', torch.nn.Module, create_dummy=True)""",
     """     aspm.register_patch('megatron.core.transformer.attention.Attention.__init__', attention_init)
+    aspm.register_patch('megatron.core.transformer.attention.SelfAttention.__init__', self_attention_init_wrapper)
     aspm.register_patch('megatron.core.transformer.module.MegatronModule.__init__', megatron_module_init_wrapper)""",
     """-    from torch.utils.cpp_extension import _get_build_directory
-    build_directory = _get_build_directory("", True)
-    delete_lock = Lock()
-    delete_lock_file(build_directory, delete_lock)
+    # from torch.utils.cpp_extension import _get_build_directory
+    # build_directory = _get_build_directory("", True)
+    # delete_lock = Lock()
+    # delete_lock_file(build_directory, delete_lock)
     mindspeed_args.adaptive_recompute_enable = mindspeed_args.adaptive_recompute_device_size > 0 or mindspeed_args.adaptive_recompute_device_swap
     if (mindspeed_args.adaptive_recompute_enable and not mindspeed_args.memory_fragmentation) or mindspeed_args.swap_attention:"""],
        "ops/gmm.py":[
""" import torch
 from torch.library import impl
+from mindspore import ops
 from mindspeed.op_builder import GMMOpBuilder, GMMV2OpBuilder
 from mindspeed.op_builder.builder import AS_LIBRARY""",""" class GMMFunction(torch.autograd.Function):
-    builder = GMMOpBuilder()
-    builder2 = GMMV2OpBuilder()
+    # builder = GMMOpBuilder()
+    # builder2 = GMMV2OpBuilder()""","""     @staticmethod
-    def forward(ctx, original_weight, x, weight, bias, group_args):
-        group_list, group_type, gemm_fusion, group_list_type, group_list_data_type = group_args
+    def forward(ctx, original_weight, x, weight, bias, group_list, group_args):
+        group_type, gemm_fusion, group_list_type, group_list_data_type = group_args
         if bias is not None and bias.requires_grad:""","""         if group_list_type == 0:
-            outputs = GMMFunction.builder.load().npu_gmm([x], [weight], bias, group_list, group_type, group_list_type)
+            outputs = ops.function.math_func.gmm([x, ], [weight, ], bias=bias, group_list=group_list, group_type=group_type)
         elif group_list_type == 1:
-            outputs = GMMFunction.builder2.load().npu_gmm([x], [weight], bias, group_list, group_type, group_list_type)
+            outputs = ops.function.math_func.gmm_v2([x, ], [weight, ], bias=bias, group_list=group_list, group_type=group_type, group_list_type=group_list_type)
         if group_list_data_type == 0:""","""             return None, dx[0], grad_weight, dbias, None
         else:
             if ctx.group_list_type == 0:
-                dx, dw, dbias = GMMFunction.builder.load().npu_gmm_backward([grad_outputs], [x], [weight], group_list,
-                                                                    ctx.group_list_type)
+                dx, dw, dbias = ops.function.math_func.gmm_backward([grad_outputs, ], [x, ], [weight],
+                                                        group_list=group_list)
             elif ctx.group_list_type == 1:
-                dx, dw, dbias = GMMFunction.builder2.load().npu_gmm_backward([grad_outputs], [x], [weight], group_list,
-                                                                    ctx.group_list_type)
+                dx, dw, dbias = ops.function.math_func.gmm_v2_backward([grad_outputs, ], [x, ], [weight],
+                                                        group_list=group_list, group_list_type=ctx.group_list_type)
             dbias = None if len(dbias) == 0 else dbias[0]
 
-            return None, dx[0], dw[0], dbias, None
+            return None, dx[0], dw[0], dbias, None, None""","""     if weight.dtype not in support_dtype:
         raise TypeError(f"Only support non quant case, but got weight dtype {weight.dtype}.")
-    npu_gmm_param_verification(x, weight, bias=bias, group_list=group_list, group_type=group_type,
-                               group_list_type=group_list_type)
     if group_list_type == 0:
-        return torch.ops.mindspeed.npu_gmm(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, gemm_fusion=gemm_fusion)
+        return _npu_gmm(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, gemm_fusion=gemm_fusion)
     elif group_list_type == 1:
-        return torch.ops.mindspeed.npu_gmm_v2(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, gemm_fusion=gemm_fusion)
+        return _npu_gmm_v2(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, gemm_fusion=gemm_fusion)
     else:
         raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}.")""","""-@impl(AS_LIBRARY, "npu_gmm.List", "PrivateUse1")
-@impl(AS_LIBRARY, "npu_gmm.Tensor", "PrivateUse1")
+# @impl(AS_LIBRARY, "npu_gmm.List", "PrivateUse1")
+# @impl(AS_LIBRARY, "npu_gmm.Tensor", "PrivateUse1")
 def _npu_gmm(original_weight, x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False):
-    if isinstance(group_list, (torch.Tensor, type(None))):
-        group_list_data_type = 1
-    else:
-        group_list_data_type = 0
-    group_args = (group_list, group_type, gemm_fusion, 0, group_list_data_type)
-    return GMMFunction.apply(original_weight, x, weight, bias, group_args)
+    group_args = (group_type, gemm_fusion, 0, 0)
+    return GMMFunction.apply(original_weight, x, weight, bias, group_list, group_args)
 
 
 def npu_gmm(x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False, original_weight=None):
-    return _npu_gmm_common(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, group_list_type=0, gemm_fusion=gemm_fusion)
+    return _npu_gmm_common(original_weight, x, weight, bias=bias, group_list=group_list.tolist(), group_type=group_type, group_list_type=0, gemm_fusion=gemm_fusion)
 
 
-@impl(AS_LIBRARY, "npu_gmm_v2.Tensor", "PrivateUse1")
 def _npu_gmm_v2(original_weight, x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False):
-    group_args = (group_list, group_type, gemm_fusion, 1, 1)
-    return GMMFunction.apply(original_weight, x, weight, bias, group_args)
+    group_args = (group_type, gemm_fusion, 1, 0)
+    return GMMFunction.apply(original_weight, x, weight, bias, group_list, group_args)
 
 
 def npu_gmm_v2(x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False, original_weight=None):"""],
        "ops/npu_moe_token_permute.py":["""-from mindspeed.op_builder import MoeTokenPermuteOpBuilder
+from mindspore import ops""","""-moe_token_permute_op_builder = MoeTokenPermuteOpBuilder()""","""-    moe_token_permute_ops = moe_token_permute_op_builder.load()
-    return moe_token_permute_ops.npu_moe_token_permute(tokens, indices, num_out_tokens, padded_mode)
+    num_out_tokens = 0 if num_out_tokens is None else num_out_tokens
+    return ops.moe_token_permute(tokens, indices, num_out_tokens, padded_mode)"""],
        "ops/npu_moe_token_unpermute.py":["""-from mindspeed.op_builder import MoeTokenUnpermuteOpBuilder
+from mindspore import ops""","""-moe_token_unpermute_op_builder = MoeTokenUnpermuteOpBuilder()""","""-    moe_token_unpermute_ops = moe_token_unpermute_op_builder.load()
-    return moe_token_unpermute_ops.npu_moe_token_unpermute(
-        permuted_tokens, sorted_indices, probs, padded_mode, restore_shape)
+    return ops.moe_token_unpermute(permuted_tokens, sorted_indices, probs, padded_mode, restore_shape)"""],
"core/transformer/moe/comm_utils.py":[
""" import einops
+import mindspore
 import torch
 import torch.distributed
 import torch.distributed as dist
 ""","""         global COMM_STREAM
         if COMM_STREAM is None:
-            COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
-        with torch_npu.npu.stream(COMM_STREAM):
+            COMM_STREAM = mindspore.runtime.Stream(device=torch.cuda.current_device())
+        with mindspore.runtime.StreamCtx(COMM_STREAM):
             event.wait()
             if last_dim:
                 handle = torch.distributed.all_gather(ag_out, input_, group=group, async_op=True)""","""         # multi stream wait event
         global COMM_STREAM
         if COMM_STREAM is None:
-            COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
-        with torch_npu.npu.stream(COMM_STREAM):
+            COMM_STREAM = mindspore.runtime.Stream(device=torch.cuda.current_device())
+        with mindspore.runtime.StreamCtx(COMM_STREAM):
             if event:
                 event.wait()
             if stream:
-                torch.cuda.current_stream().wait_stream(stream)
+                mindspore.runtime.current_stream().wait_stream(stream)
             handle = torch.distributed._reduce_scatter_base(
                 rs_out, input_.contiguous(), group=group, async_op=True
             )""","""     return input_, rs_out, handle
 
 
+def transfer_tensor_last_dim_to_first(input_x):
+    num_dims = input_x.dim()
+    return einops.rearrange(input_x, "... lst -> lst ...").contiguous(), num_dims
+
+
+def transfer_tensor_first_dim_to_last(input_x, num_dims):
+    return einops.rearrange(input_x, "first ... -> ... first").contiguous()
+
+
 def async_all_to_all(input_, output_split_sizes, input_split_sizes, group, event=None):
     world_size = dist.get_world_size(group)""","""     else:
         # Unequal split (all2all-v)
         a2a_out = input_.new_empty(
-            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
+            size=[int(sum(output_split_sizes.tolist()))] + list(input_.size()[1:]),
             dtype=input_.dtype,
-            device=torch.cuda.current_device(),
         )""","""             COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
         with torch_npu.npu.stream(COMM_STREAM):
             event.wait()
-            handle = dist.all_to_all_single(
-                a2a_out,
-                input_.contiguous(),
-                output_split_sizes=output_split_sizes,
-                input_split_sizes=input_split_sizes,
-                group=group,
-                async_op=True
-            )
+            handle = mindspore.mint.distributed.all_to_all_single(
+            a2a_out,
+            input_.contiguous(),
+            output_split_sizes=output_split_sizes.tolist(),
+            input_split_sizes=input_split_sizes.tolist(),
+            group=group._name,
+            async_op=True)
     else:
-        handle = dist.all_to_all_single(
+        handle = mindspore.mint.distributed.all_to_all_single(
             a2a_out,
             input_.contiguous(),
-            output_split_sizes=output_split_sizes,
-            input_split_sizes=input_split_sizes,
-            group=group,
-            async_op=True
-        )
+            output_split_sizes=output_split_sizes.tolist(),
+            input_split_sizes=input_split_sizes.tolist(),
+            group=group._name,
+            async_op=True)
     return input_, a2a_out, handle"""],
"core/transformer/moe/grouped_mlp_with_comp_and_comm_overlap_all2all.py":["""+import mindspore
+from mindspore.common.api import _convert_python_data
 import torch
 from einops import rearrange""",""" class GroupedMlpWithCompAndCommOverlapAll2All(torch.autograd.Function):
     @staticmethod
-    def forward(ctx, inputs, weights1, weights2, args, moe_layer_ctx):
-        original_weight1, original_weight2, activation_func, group_list, layer_number = args
+    def forward(ctx, inputs, weights1, weights2, original_weight1, original_weight2, activation_func, group_list, layer_number):
         global_args = get_args()""","""         if moe_zero_memory != "disable" or moe_experts_pipeline_degree:
             inputs.untyped_storage().resize_(0)
-        act_out, detached_act_inputs = forward_func(activation_func, mm1_out)
+        act_out, detached_act_inputs, ctx.activation_func_vjp = forward_func(activation_func, mm1_out)
 
         is_only_recompute_activation = only_recompute_activation(layer_number)""","""              permute2_input_detach, permute2_graph, output_splits, input_splits,
-             input_splits_tp_ep) = get_gemm_backward_need_tensors()
-
+             input_splits_tp_ep, permutation_func1_vjp, permutation_func2_vjp) = get_gemm_backward_need_tensors()
         # grad of mm2 dx""","""             tp_group = parallel_state.get_tensor_model_parallel_group()
-            permute1_graph, scores_ep, hidden_states_ep = get_all2all_experts_output()
+            permute1_graph, scores_ep, hidden_states_ep, scores_ep_grad = get_all2all_experts_output()
             if moe_zero_memory == "disable":""","""             # grad of activation_func
-            act_graph.backward(grad_mm2_inputs)
+            act_inputs_grad = _convert_python_data(ctx.activation_func_vjp(grad_mm2_inputs)[0])
+            ctx.activation_func_vjp = None
             if moe_zero_memory == "level0" or (moe_zero_memory == "level1" and is_only_recompute_activation):""","""-                mm1_inputs_grad = gmm_op(act_inputs.grad, weights1, [], group_list, 0)[0]
+                mm1_inputs_grad = gmm_op(act_inputs_grad, weights1, [], group_list, 0)[0]""","""-                mm1_inputs_grad = torch.matmul(act_inputs.grad, weights1.t())
+                mm1_inputs_grad = torch.matmul(act_inputs_grad, weights1.t())""","""-            backward_func(permute2_graph, mm1_inputs_grad)
-            mm1_inputs_grad.untyped_storage().resize_(0)
+            permute2_input_detach_grad = _convert_python_data(permutation_func2_vjp(mm1_inputs_grad)[0])
+            permutation_func2_vjp = None
+            del mm1_inputs_grad""","""                 permute1_ep_all_to_all_handle.wait()
-                permutated_local_input_tokens.untyped_storage().resize_(0)
+                del permutated_local_input_tokens
             _, permute1_backward_input, bw_permute1_ep_all2all_handle = async_all_to_all(""","""-                permute2_input_detach.grad,
+                permute2_input_detach_grad,
                 input_splits,
                 output_splits,""",
             """         # grad of activation_func
-        grad_outs.untyped_storage().resize_(0)
-        mm2_inputs.untyped_storage().resize_(0)
+        del grad_outs
+        del mm2_inputs
         if moe_hierarchical_alltoallv:
             grad_mm2_inputs.untyped_storage().resize_(0)""","""         else:
-            act_graph.backward(grad_mm2_inputs)
-            grad_mm2_inputs.untyped_storage().resize_(0)
-            act_inputs.untyped_storage().resize_(0)
+            act_inputs_grad = _convert_python_data(ctx.activation_func_vjp(grad_mm2_inputs)[0])
+            ctx.activation_func_vjp = None
+            del grad_mm2_inputs
+            del act_inputs
             if moe_zero_memory == "level0" or (moe_zero_memory == "level1" and is_only_recompute_activation):""",
             """-                    gmm_op(act_inputs.grad, weights1, [], group_list, 0)[0]
+                    gmm_op(act_inputs_grad, weights1, [], group_list, 0)[0]""","""-                mm1_inputs_grad = torch.matmul(act_inputs.grad, weights1.t())
+                mm1_inputs_grad = torch.matmul(act_inputs_grad, weights1.t())""","""             else:
-                backward_func(permute2_graph, mm1_inputs_grad)
-                mm1_inputs_grad.untyped_storage().resize_(0)
+                permute2_input_detach_grad = _convert_python_data(permutation_func2_vjp(mm1_inputs_grad)[0])
+                permutation_func2_vjp = None
+                del mm1_inputs_grad
                 ep_group = get_expert_model_parallel_group()""","""                 permute1_ep_all_to_all_handle.wait()
-                permutated_local_input_tokens.untyped_storage().resize_(0)
+                del permutated_local_input_tokens
 
             if moe_experts_pipeline_degree:""","""             else:
                 _, permute1_backward_input, bw_permute1_ep_all2all_handle = async_all_to_all(
-                    permute2_input_detach.grad,
+                    permute2_input_detach_grad,
                     input_splits,
                     output_splits,""","""                 else:
                     mm1_weights_grad = None
             else:
-                mm1_weights_grad = gmm_op(mm1_inputs.t(), act_inputs.grad, [], group_list, 2)[0]
+                mm1_weights_grad = gmm_op(mm1_inputs.t(), act_inputs_grad, [], group_list, 2)[0]
         else:
-            mm1_weights_grad = torch.matmul(mm1_inputs.t(), act_inputs.grad)
-        act_inputs.grad.untyped_storage().resize_(0)
+            mm1_weights_grad = torch.matmul(mm1_inputs.t(), act_inputs_grad)
+        del act_inputs_grad
         if moe_experts_pipeline_degree:
             return None, mm1_weights_grad, grad_weights2, None, None
         else:
-            return mm1_inputs_grad, mm1_weights_grad, grad_weights2, None, None
+            return None, mm1_weights_grad, grad_weights2, None, None, None, None, None
 
 
-def grouped_mlp_with_comp_and_comm_overlap_all2all(inputs, weights1, weights2, args, ctx):
-    return GroupedMlpWithCompAndCommOverlapAll2All.apply(inputs, weights1, weights2, args, ctx)
+def grouped_mlp_with_comp_and_comm_overlap_all2all(inputs, weights1, weights2, args):
+    return GroupedMlpWithCompAndCommOverlapAll2All.apply(inputs, weights1, weights2, *args)"""],
"core/transformer/moe/moe_layer_overlap_all2all.py":["""+from torch.autograd import recompute_instance
+import mindspore
+from mindspore.common.api import _convert_python_data
 from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size""",
 """ from megatron.core.transformer.moe.moe_utils import permute, save_to_aux_losses_tracker
+from megatron.core.transformer.moe import grouped_gemm_util as gg
 from mindspeed.moe.utils import MoEAuxLossAutoScaler""",
 """ from mindspeed.core.transformer.moe.moe_utils import (forward_func, backward_func, permute_with_ep)
-from mindspeed.ops.gmm import GMMFunction
 from mindspeed.core.transformer.moe.moe_utils import (AG_SHARED_EXPERTS_INPUTS, only_recompute_activation,""",
 """ def gmm_op(x, weight, bias, group_list, group_type):
-    if isinstance(group_list, torch.Tensor) and group_list.device.type == 'cpu':
-        group_list = group_list.tolist()
-    return GMMFunction.builder.load().npu_gmm([x], [weight], bias, group_list, group_type, 0)
+    out = gg.ops.gmm(x, weight, group_list, trans_b=False, group_type=group_type)
+    return (out,)""","""         save_tensors = []
         ctx.input_shape = hidden_states.shape
-        hidden_states = hidden_states.detach()
+        hidden_states = mindspore.ops.stop_gradient(hidden_states)
         hidden_states.requires_grad = True
         ctx.is_only_recompute_activation = only_recompute_activation(moe_layer.layer_number)
+        def router_func_test(hidden_states):
+            scores, ctx.indices = moe_layer.router(hidden_states)
+            return scores
         ctx.layer_number = moe_layer.layer_number""","""         # router
-        with torch.enable_grad():
-            scores, indices = moe_layer.router(hidden_states)
+        if not recompute_instance.recompute:
+            router_input = mindspore.ops.stop_gradient(hidden_states)
+            router_input.requires_grad = True
+            with torch.enable_grad():
+                scores, ctx.router_func = torch.autograd.vjp(router_func_test, router_input)
+        else:
+            scores = router_func_test(hidden_states)
 
         save_tensors.append(scores)
-        scores = scores.detach()
+        scores = mindspore.ops.stop_gradient(scores)
         scores.requires_grad = True""","""             ctx.num_local_experts = moe_layer.token_dispatcher.num_local_experts
 
-        save_tensors.append(indices)
+        save_tensors.append(ctx.indices)
 
         if n_shared_experts:""","""         else:
             shared_expert_gate = None
 
-        (share_experts_output, dispatched_input, tokens_per_expert) = moe_layer.token_dispatcher.token_permutation(
-            hidden_states, scores, indices, ctx.shared_experts, save_tensors, shared_expert_gate, ctx
+        (share_experts_output, dispatched_input, tokens_per_expert, shared_experts_func, permutation_func1, permutation_func2) = moe_layer.token_dispatcher.token_permutation(
+            hidden_states, scores, ctx.indices, ctx.shared_experts, save_tensors, shared_expert_gate, ctx
         )
+
+        def experts_func_test(dispatched_input, tokens_per_expert):
+            expert_output, mlp_bias = moe_layer.experts(dispatched_input, tokens_per_expert)
+            return expert_output, mlp_bias
+
         if moe_experts_pipeline_degree:
             save_tensors.append(None)
             save_tensors.append(None)
             expert_output, mlp_bias = moe_experts_pipeline_forward_func(tokens_per_expert, moe_layer, dispatched_input, ctx, save_tensors)
-            output, mlp_bias = moe_layer.token_dispatcher.token_unpermutation(expert_output, mlp_bias, save_tensors)
-
-
+            (expert_output, mlp_bias), *_, experts_func = forward_func(experts_func_test, (dispatched_input, tokens_per_expert))
+
+            output, mlp_bias, unpermutation_func1, unpermutation_func2 = moe_layer.token_dispatcher.token_unpermutation(expert_output, mlp_bias, save_tensors)
+            ctx.permutation_func1 = permutation_func1
+            ctx.permutation_func2 = permutation_func2
+            ctx.shared_experts_func = shared_experts_func
+            ctx.experts_func = experts_func
+            ctx.unpermutation_func1 = unpermutation_func1
+            ctx.unpermutation_func2 = unpermutation_func2
+            
             if isinstance(share_experts_output, tuple):
                 share_experts_output, rs_share_experts_output, rs_shared_experts_handle = share_experts_output
             else:""","""                 rs_share_experts_output = share_experts_output
                 rs_shared_experts_handle = None
 
-            (expert_output, mlp_bias), *_ = forward_func(moe_layer.experts, (dispatched_input, tokens_per_expert, ctx))
+            (expert_output, mlp_bias), *_, experts_func = forward_func(experts_func_test, (dispatched_input, tokens_per_expert))
             save_tensors.append(expert_output)
 
-            output, mlp_bias = moe_layer.token_dispatcher.token_unpermutation(expert_output, mlp_bias, save_tensors)
-
+            output, mlp_bias, unpermutation_func1, unpermutation_func2 = moe_layer.token_dispatcher.token_unpermutation(expert_output, mlp_bias, save_tensors)
+            ctx.permutation_func1 = permutation_func1
+            ctx.permutation_func2 = permutation_func2
+            ctx.shared_experts_func = shared_experts_func
+            ctx.experts_func = experts_func
+            ctx.unpermutation_func1 = unpermutation_func1
+            ctx.unpermutation_func2 = unpermutation_func2
         if group_limited_greedy:
             save_tensors.append(moe_layer.router.l_aux)
             moe_layer.router.l_aux = moe_layer.router.l_aux.detach()""","""             if rs_shared_experts_handle is not None:
                 rs_shared_experts_handle.wait()
             output_sum = output + rs_share_experts_output
-            output.untyped_storage().resize_(0)
-            share_experts_output.untyped_storage().resize_(0)
+            share_experts_output_dtype = share_experts_output.dtype
+            del output, share_experts_output
+            share_experts_output = torch.tensor(1, dtype=share_experts_output_dtype)
         else:
-            output_sum = output.detach()
+            output_sum = mindspore.ops.stop_gradient(output)
 
         save_tensors.append(share_experts_output)""","""             set_gemm_backward_need_tensors(
                 ((hidden_states_ep, indices_ep, scores_ep, router_topk, global_input_tokens_local_experts_indices),
                  permute2_input_detach, permute2_graph,
-                 output_splits, input_splits, input_splits_tp_ep))
+                 output_splits, input_splits, input_splits_tp_ep, ctx.permutation_func1, ctx.permutation_func2))
         elif moe_experts_pipeline_degree:
             input_list = ctx.input_list
         else:
             set_gemm_backward_need_tensors(
                 ((detach_input, indices, scores_ep, router_topk, global_input_tokens_local_experts_indices),
                  permute2_input_detach, permute2_graph,
-                 output_splits, input_splits, input_splits_tp_ep))
+                 output_splits, input_splits, input_splits_tp_ep, ctx.permutation_func1, ctx.permutation_func2))
 
         if n_shared_experts:
             if get_tensor_model_parallel_world_size() > 1 and not shared_expert_gate:""","""         if moe_hierarchical_alltoallv:
             output_backward_handle.wait()
-            unpermute2_graph.backward(unpermute2_graph_backward_input)
+            unpermute2_input_grad, detach_scores_grad = _convert_python_data(ctx.unpermutation_func2(unpermute2_graph_backward_input))
         else:
-            unpermute2_graph.backward(args[0])
+            unpermute2_input_grad, detach_scores_grad = _convert_python_data(ctx.unpermutation_func2(args[0]))
+        ctx.unpermutation_func2 = None
         unpermute2_graph = None
         if moe_zero_memory == "level1" and not ctx.is_only_recompute_activation:
             if n_shared_experts:""","""         if moe_hierarchical_alltoallv:
             tp_group = parallel_state.get_tensor_model_parallel_group()
             _, unpermute1_backward_input, handle = async_all_to_all(
-                unpermute2_input_detach.grad,
+                unpermute2_input_grad,
                 output_splits,
                 input_splits,
                 tp_group,
             )
         else:
             _, unpermute1_backward_input, handle = async_all_to_all(
-                unpermute2_input_detach.grad,
+                unpermute2_input_grad,
                 output_splits,
                 input_splits,
                 ep_group,""","""         elif share_experts_graph is not None:
             if backward_ag_shared_handle is not None:
                 backward_ag_shared_handle.wait()
-            share_experts_graph.backward(backward_ag_shared)
+            detach_input_share_grad = _convert_python_data(ctx.shared_experts_func(backward_ag_shared)[0])
+            ctx.shared_experts_func = None
             share_experts_graph = None
             if backward_ag_shared_handle is not None:
-                backward_ag_shared.untyped_storage().resize_(0)
+                del backward_ag_shared
         if handle is not None:
             handle.wait()
-            unpermute2_input_detach.grad.untyped_storage().resize_(0)
+            del unpermute2_input_grad
+
+        unpermute1_input_detach_grad = _convert_python_data(ctx.unpermutation_func1(unpermute1_backward_input)[0])
+        ctx.unpermutation_func1 = None
 
-        backward_func(unpermute1_graph, unpermute1_backward_input)
+        del unpermute1_backward_input
 
-        unpermute1_backward_input.untyped_storage().resize_(0)
         if moe_hierarchical_alltoallv:
             set_all2all_experts_output((permute1_graph, scores_ep, hidden_states_ep))
             backward_func(experts_graph, unpermute1_input_detach.grad)""","""             backward_func(permute1_graph, permute2_input_detach.grad)
             permute2_input_detach.grad.untyped_storage().resize_(0)
         else:
-            backward_func(experts_graph, unpermute1_input_detach.grad)
-            unpermute1_input_detach.grad.untyped_storage().resize_(0)
+            _convert_python_data(ctx.experts_func(unpermute1_input_detach_grad))
+            ctx.experts_func = None
+            del unpermute1_input_detach_grad
             permute1_backward_input, bw_permute1_ep_all2all_handle = get_all2all_experts_output()
             bw_permute1_ep_all2all_handle.wait()
-            permute2_input_detach.grad.untyped_storage().resize_(0)
-            backward_func(permute1_graph, permute1_backward_input)
-            permute1_backward_input.untyped_storage().resize_(0)
+            del permute2_input_detach
+            hidden_states_grad = _convert_python_data(ctx.permutation_func1(permute1_backward_input)[0])
+            del permute1_backward_input
         if l_aux_graph is not None:
             l_aux_graph.backward(l_aux_detach.grad, retain_graph=True)
         if moe_zero_memory != "disable":""","""                 route_graph.backward(detach_scores_grad)
                 detach_input_handle.wait()
             else:
-                route_graph.backward(detach_scores.grad)
+                detach_input_grad1 = _convert_python_data(ctx.router_func(detach_scores_grad)[0])
+                ctx.router_func = None
         route_graph = None
         if moe_hierarchical_alltoallv:
             grad_output = detach_input.grad + detach_input_grad
         else:
-            grad_output = detach_input.grad
+            grad_output = detach_input_share_grad + hidden_states_grad + detach_input_grad1
+        ctx.saved_tensors = []
         return grad_output, None"""],
"core/transformer/moe/moe_utils.py":[""" import torch
 import torch_npu
+from torch.autograd import recompute_instance
+import mindspore
 from megatron.core.transformer.moe.moe_utils import permute_with_padded_tokens, unpermute_with_padded_tokens
 from megatron.training import get_args""",""" def get_swap_stream():
     global SWAP_STREAM2
     if SWAP_STREAM2 is None:
-        _ = torch_npu.npu.Stream(device=torch.npu.current_device())
-        SWAP_STREAM2 = torch_npu.npu.Stream(device=torch.npu.current_device())
+        _ = mindspore.runtime.Stream(device=torch.cuda.current_device())
+        SWAP_STREAM2 = mindspore.runtime.Stream(device=torch.cuda.current_device())
     stream = SWAP_STREAM2
     return stream""",""" def get_swap_status():
     global SWAP_STREAM
     if SWAP_STREAM is None:
-        SWAP_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
+        SWAP_STREAM = mindspore.runtime.Stream(device=torch.cuda.current_device())
     global SWAP_TENSOR
     stream = SWAP_STREAM
     tensor = SWAP_TENSOR""",""" def get_prob_backward_need_tensors():
     global SWAP_STREAM2
     if SWAP_STREAM2 is None:
-        _ = torch_npu.npu.Stream(device=torch.npu.current_device())
-        SWAP_STREAM2 = torch_npu.npu.Stream(device=torch.npu.current_device())
+        _ = mindspore.runtime.Stream(device=torch.cuda.current_device())
+        SWAP_STREAM2 = mindspore.runtime.Stream(device=torch.cuda.current_device())
     global MATMUL_OUTPUT_GRAD
     global UNPERMUTED_TOKENS""","""         if input_.requires_grad and input_.grad_fn is None:
             return input_
         else:
-            new_input = input_.detach()
+            new_input = mindspore.ops.stop_gradient(input_)
             new_input.requires_grad = True
         return new_input""","""     elif isinstance(inputs, torch.Tensor):
         detach_inputs.append(detach_tensor(inputs))
 
-    with torch.enable_grad():
+    if not recompute_instance.recompute:
+        with torch.enable_grad():
+            output, f_vjp = torch.autograd.vjp(func, *detach_inputs)
+    else:
         output = func(*detach_inputs)
+        f_vjp = None
 
-    return output, *detach_inputs
+    return output, *detach_inputs, f_vjp
 
 
 def backward_func(func_tensor, gradinputs):""","""         dtype=permuted_tokens.dtype,
         device=permuted_tokens.device,
     )
-    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
+    unpermuted_tokens.index_add_(0, sorted_indices, permuted_tokens)
     unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
     if probs is not None:
         unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)"""],
"core/transformer/moe/token_dispatcher.py":[""" import torch
+import mindspore
 from torch_npu.utils.collect_env import get_cann_version
 from megatron.training import get_args""","""         # permutation to get the `num_out_tokens` CPU value.
-        self.num_out_tokens = num_local_tokens_per_expert.sum().to(
-            torch.device("cpu"), non_blocking=True
-        )
+        self.num_out_tokens = num_local_tokens_per_expert.sum()
         self.cuda_sync_point = "before_permutation_1"
     elif ep_size > 1:""","""         self.input_splits = (
             num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
             .sum(axis=1)
-            .to(torch.device("cpu"), non_blocking=True)
             .numpy()
         )""","""         self.output_splits = (
-            self.num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu")).numpy()
+            self.num_global_tokens_per_local_expert.sum(axis=-1).numpy()
         )""","""     if self.num_local_experts > 1:
         if not hasattr(self, 'comm_stream'):
-            self.comm_stream = torch.cuda.Stream()
-        self.comm_stream.wait_stream(torch.cuda.current_stream())
-        with torch.cuda.stream(self.comm_stream):
+            self.comm_stream = mindspore.runtime.communication_stream()
+        self.comm_stream.wait_stream(mindspore.runtime.current_stream())
+        with mindspore.runtime.StreamCtx(self.comm_stream):
             # No further synchronization is needed because torch.repeat_interleave() calls stream""",
             """     if self.cuda_sync_point == "before_permutation_1":
-        torch.cuda.current_stream().synchronize()
+        mindspore.runtime.current_stream().synchronize()
     permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
         hidden_states,""","""     if self.cuda_sync_point == "before_ep_alltoall":
-        torch.cuda.current_stream().synchronize()
+        mindspore.runtime.current_stream().synchronize()
     global_input_tokens = tensor_parallel.all_to_all(""","""     if self.num_local_experts > 1:
         if not self.drop_and_pad:
-            torch.cuda.current_stream().wait_stream(self.comm_stream)
+            # mindspore.runtime.current_stream().wait_stream(self.comm_stream)
             global_input_tokens, self.reversed_global_input_permutation_mapping = permute(""",
             """     if self.cuda_sync_point == "before_finish":
-        torch.cuda.current_stream().synchronize()
+        mindspore.runtime.current_stream().synchronize()
 
     return global_input_tokens, tokens_per_expert""","""             self.input_splits_tp_ep = (
                 num_local_tokens_per_expert.reshape(tp_extended_ep_size, self.num_local_experts)
                 .sum(axis=1)
-                .to(torch.device("cpu"))
                 .numpy()
             )
             expert_parallel_rank = mpu.get_expert_model_parallel_rank()
""","""             self.input_splits = (
                 num_local_tokens_per_expert.reshape(tp_extended_ep_size, self.num_local_experts)
                 .sum(axis=1)
-                .to(torch.device("cpu"))
                 .numpy()
             )""","""         self.output_splits = (
-            self.num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu")).numpy()
+            self.num_global_tokens_per_local_expert.sum(axis=-1).numpy()
         )
         num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(axis=0)""",
         """     if self.num_local_experts > 1:
         if not hasattr(self, 'comm_stream'):
-            self.comm_stream = torch.cuda.Stream()
-        self.comm_stream.wait_stream(torch.cuda.current_stream())
-        with torch.cuda.stream(self.comm_stream):
+            self.comm_stream = mindspore.runtime.communication_stream()
+        self.comm_stream.wait_stream(mindspore.runtime.current_stream())
+        with mindspore.runtime.StreamCtx(self.comm_stream):
             if moe_hierarchical_alltoallv:""","""         if not self.drop_and_pad:
-            torch.cuda.current_stream().wait_stream(self.comm_stream)
+            mindspore.runtime.current_stream().wait_stream(self.comm_stream)
             global_input_tokens, self.reversed_global_input_permutation_mapping = permute(""","""         save_tensors.append(indices_ep)
 
-    def alltoall_token_permutation1(hidden_states, indices, *args):
+    if moe_hierarchical_alltoallv:
+        tokens_per_expert = self.preprocess(indices, hidden_states)
+    else:
+        tokens_per_expert = self.preprocess(indices)
+    save_tensors.append(hidden_states_ep)
+    #, indices, *args
+    def alltoall_token_permutation1(hidden_states):
         if moe_hierarchical_alltoallv:
             _, self.probs, probs_handle = async_all_gather(self.probs, group=ep_group)
-            tokens_per_expert = self.preprocess(indices, hidden_states)
-            args[1].wait()  # hidden_states_ep_handle
-            save_tensors.append(args[0])  # hidden_states_ep
+            hidden_states_ep_handle.wait()
             # hidden_states: [S/TP, B, H] -> [S*B/TP, H]
-            hidden_states = args[0].view(-1, self.hidden_shape[-1])
+            hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
             self.hidden_shape_before_permute = hidden_states.shape
             # Permutation 1: input to AlltoAll input
             if self.cuda_sync_point == "before_permutation_1":
-                torch.cuda.current_stream().synchronize()
+                mindspore.runtime.current_stream().synchronize()
             probs_handle.wait()
-            self.probs = self.probs.detach()
+            self.probs = mindspore.ops.stop_gradient(self.probs)
             self.probs.requires_grad = True
""","""         else:
-            tokens_per_expert = self.preprocess(indices)
-            save_tensors.append(args[0])
             if get_args().moe_experts_pipeline_degree:""","""             self.hiddden_shape_before_permute = hidden_states.shape
             if self.cuda_sync_point == "before_permutation_1":
-                torch.cuda.current_stream().synchronize()
+                mindspore.runtime.current_stream().synchronize()
             scores_ep = None
             save_tensors.append(scores_ep)""","""                 num_out_tokens=self.num_out_tokens,
                 padded_mode=self.drop_and_pad,
             )
-        return tokens_per_expert, permutated_local_input_tokens
+        return permutated_local_input_tokens
 
-    (tokens_per_expert, permutated_local_input_tokens), *_ = forward_func(alltoall_token_permutation1,
-                                                                          (hidden_states, indices,
-                                                                           hidden_states_ep, hidden_states_ep_handle))
+    input_hidden_states = hidden_states_ep if moe_hierarchical_alltoallv else hidden_states
+    permutated_local_input_tokens, *_, vjp_alltoall_token_permutation1 = forward_func(alltoall_token_permutation1,
+                                                                                      input_hidden_states)
 
     # permute 1
     save_tensors.append(permutated_local_input_tokens)""","""     # Perform expert parallel AlltoAll communication
     if self.cuda_sync_point == "before_ep_alltoall":
-        torch.cuda.current_stream().synchronize()
+        mindspore.runtime.current_stream().synchronize()
     if moe_hierarchical_alltoallv:
         tp_group = parallel_state.get_tensor_model_parallel_group()""","""     if shared_experts is not None:
-        (share_experts_output, _), *_ = forward_func(shared_experts, (hidden_states, moe_ctx))
+        def shared_experts_func(hidden_states):
+            output, bias = shared_experts(hidden_states)
+            return output, bias
+        (share_experts_output, _), *_, vjp_shared_experts = forward_func(shared_experts_func, hidden_states)
         if parallel_state.get_tensor_model_parallel_world_size() > 1 and shared_expert_gate is None:
             share_experts_graph, share_experts_output, rs_shared_experts_handle = async_reduce_scatter(share_experts_output, parallel_state.get_tensor_model_parallel_group(),
-                                                                                                       event=permute1_ep_all_to_all_handle, stream=torch.npu.default_stream())
+                                                                                                       event=permute1_ep_all_to_all_handle, stream=mindspore.runtime.default_stream())
             share_experts_output = (share_experts_graph, share_experts_output, rs_shared_experts_handle)
         if shared_expert_gate is not None:""","""     if permute1_ep_all_to_all_handle is not None:
         permute1_ep_all_to_all_handle.wait()
-        permutated_local_input_tokens.untyped_storage().resize_(0)
+        del permutated_local_input_tokens""","""         if self.num_local_experts > 1:
             if not self.drop_and_pad:
                 if self.comm_stream is not None:
-                    torch.cuda.current_stream().wait_stream(self.comm_stream)
+                    mindspore.runtime.current_stream().wait_stream(self.comm_stream)
                 global_input_tokens, self.reversed_global_input_permutation_mapping = permute(
                     global_input_tokens, self.global_input_tokens_local_experts_indices""","""                 global_input_tokens
             )
         if self.cuda_sync_point == "before_finish":
-            torch.cuda.current_stream().synchronize()
+            mindspore.runtime.current_stream().synchronize()
 
         return global_input_tokens""","""-    (global_input_tokens), global_input_tokens_detach = forward_func(alltoall_token_permutation2,
+    (global_input_tokens), global_input_tokens_detach, vjp_alltoall_token_permutation2 = forward_func(alltoall_token_permutation2,
                                                                      global_input_tokens)
     save_tensors.append(global_input_tokens_detach)
     save_tensors.append(global_input_tokens)""","""-    global_input_tokens_detach.untyped_storage().resize_(0)
+    del global_input_tokens_detach
 
-    return share_experts_output, global_input_tokens, tokens_per_expert
+    return share_experts_output, global_input_tokens, tokens_per_expert, vjp_shared_experts, vjp_alltoall_token_permutation1, vjp_alltoall_token_permutation2
 
 
 def alltoall_token_unpermutation_new(""","""     else:
-        hidden_states, unpermute1_input_detach = forward_func(alltoall_token_unpermutation1, hidden_states)
+        hidden_states, unpermute1_input_detach, vjp_alltoall_token_unpermutation1 = forward_func(alltoall_token_unpermutation1, hidden_states)
         save_tensors.append(unpermute1_input_detach)
         save_tensors.append(hidden_states)
-        unpermute1_input_detach.untyped_storage().resize_(0)
+        del unpermute1_input_detach
 
     ep_group = parallel_state.get_expert_model_parallel_group()""","""     if handle is not None:
         handle.wait()
-        hidden_states.untyped_storage().resize_(0)
+        del hidden_states
 
-    def alltoall_token_unpermutation2(permutated_local_input_tokens):
+    def alltoall_token_unpermutation2(permutated_local_input_tokens, probs):
         # Unpermutation 1: AlltoAll output to output
         if get_args().moe_zero_memory != "disable":
             output = UnpermuteWithoutActivation.apply(""","""-    output, unpermute2_input_detach = forward_func(alltoall_token_unpermutation2, permutated_local_input_tokens)
+    output, unpermute2_input_detach, _, vjp_alltoall_token_unpermutation2 = forward_func(alltoall_token_unpermutation2, (permutated_local_input_tokens, self.probs))""","""-        unpermute2_input_detach.untyped_storage().resize_(0)
+        del unpermute2_input_detach""","""         output_handle.wait()
         output = output.view(self.hidden_shape)
-    return output, None
+    return output, None, vjp_alltoall_token_unpermutation1, vjp_alltoall_token_unpermutation2
 
 
 def allgather_token_permutation_npu(self, hidden_states: torch.Tensor, max_prob: torch.Tensor, max_ind: torch.Tensor):""",
 """         return num_tokens_per_local_expert
     elif self.config.moe_expert_capacity_factor is not None:
         # Token drop but no pad.
-        self.num_out_tokens = num_local_tokens_per_expert.sum().to(
-            torch.device("cpu"), non_blocking=True
-        )
+        self.num_out_tokens = num_local_tokens_per_expert.sum()
         self.cuda_sync_point = "before_permutation_1"
     elif ep_size > 1:""","""         self.input_splits = (
             num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
             .sum(axis=1)
-            .to(torch.device("cpu"), non_blocking=True)
             .numpy()
         )
         num_global_tokens_per_expert = _gather_along_first_dim_expert_parallel(""","""         self.output_splits = (
-            self.num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu")).numpy()
+            self.num_global_tokens_per_local_expert.sum(axis=-1).numpy()
         )
         num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(axis=0)"""],
"optimizer/distrib_optimizer.py":[""" from megatron.training import get_args
+from megatron.core.distributed.param_and_grad_buffer import BufferType""","""     copy_group_grads(self.model_fp32_groups, self.shard_fp32_groups)
 
+from mindspore.hal import memory_allocated""",
"""                         model_param.data.detach().copy_(old_param_data)
+                        del old_param_data
                         self.shard_fp32_param_fp16_view_group.append(self.shard_main_param_res_buffers[gbuf_index][2 * data_start_index: 2 * data_end_index])
                 for i, buffer in enumerate(self.buffers):
                     buffer_numel = buffer.param_data.numel()
                     reuse_data_ptr(buffer.param_data, self.shard_main_param_res_buffers[i], buffer_numel)
+
+                    # fix mem by moving bucket.param_data manually
+                    for bucket_id, bucket in enumerate(buffer.buckets):  # fix reuse
+                        (start_index, end_index) = buffer.bucket_indices[bucket_id]
+                        bucket.param_data.assign_value(buffer._get(
+                            torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.PARAM
+                        ))
+
+                # fix mem by moving shard_model_param manually
+                for model_float16_params_this_group, shard_float16_params_this_group in zip(
+                        self.model_float16_groups, self.shard_float16_groups):
+                    for model_param, shard_model_param in zip(model_float16_params_this_group, shard_float16_params_this_group):
+                        gbuf_index, dtype, bucket_index = self.model_param_gbuf_map[model_param]
+                        gbuf_range = self.gbuf_ranges[gbuf_index][dtype][bucket_index]
+                        param_range = gbuf_range["param_map"][model_param]["param"]
+
+                        shard_model_param.assign_value(model_param.detach().view(-1)[
+                            param_range.start : param_range.end
+                        ])
+
             else:""","""-            torch_npu.npu.empty_cache()
+            # torch_npu.npu.empty_cache()"""],
    },
    "mindspeed-rl": {
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
"""         data = next(iter(data))
         for k, v in data.items():
             if v is not None:
-                data[k] = v.to(next(self.model[0].parameters()).device)
+                data[k] = v
         for model_module in self.model:
             model_module.eval()
         with torch.no_grad():""",
"""         batch = next(iter(data))
         for k, v in batch.items():
             if v is not None:
-                batch[k] = v.to(next(self.model[0].parameters()).device)
+                batch[k] = v
         mini_batches = self._split_batches(batch, batch_size=self.mini_batch_size,
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
""" from typing import List, Union
 import torch
 from torch.utils.data import DataLoader
-from torch.utils.tensorboard import SummaryWriter
+# from torch.utils.tensorboard import SummaryWriter
 
 from mindspeed_rl.workers.rule_reward import RuleReward
 from mindspeed_rl.trainer.utils.compute_utils import FixedKLController, AdaptiveKLController""",
"""         self.tensorboard = None
         if kwargs.get(\"use_wandb\", \"\") and torch.distributed.get_rank() == 0:
             self.wandb = WandbLogger(kwargs)
-        if kwargs.get(\"use_tensorboard\", \"\") and self.wandb is None and torch.distributed.get_rank() == 0:
-            self.tensorboard = SummaryWriter()
+        # if kwargs.get(\"use_tensorboard\", \"\") and self.wandb is None and torch.distributed.get_rank() == 0:
+        #     self.tensorboard = SummaryWriter()
 
     def experience_maker_init(self):
         pass"""
        ],
        "mindspeed_rl/trainer/grpo_trainer_hybrid.py": [
"""         data_iters = iter(data_loader)
 
         iteration = self.actor_worker.get_iteration()
-
+        # from mindspore import context
+        # context.set_context(pynative_synchronize=True)
         while iteration < self.train_iters:
 
             batch = next(data_iters)"""
        ],
        "mindspeed_rl/trainer/utils/transfer_dock.py": [
"""             torch.stack([self.experience_data_status[single_column] == 1 for single_column in experience_columns]),
             dim=0,
         )
+        not_consumed_indexes = not_consumed_indexes.astype(torch.int32)
+        data_ready_indexes = data_ready_indexes.astype(torch.int32)
         usable_indexes = (not_consumed_indexes & data_ready_indexes).nonzero(as_tuple=True)[0]
 
         if len(usable_indexes) < mbs:"""
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
+        return grad_output""",

"""     mpu = get_parallel_state()
     # Step 1: Compute the local max value for numerical stability
-    z_max = logits.max(dim=-1, keepdim=True).values
+    z_max = logits.max(dim=-1, keepdim=True)[0]""",
"""-    torch.distributed.all_reduce(
-        z_max,
-        op=torch.distributed.ReduceOp.MAX,
-        group=mpu.get_tensor_model_parallel_group()
-    )
+    z_max = ReduceFromContextParallelRegionDPO().apply(z_max)""",
"""-    torch.distributed.all_reduce(
-        local_sum_exp,
-        op=torch.distributed.ReduceOp.SUM,
-        group=mpu.get_tensor_model_parallel_group()
-    )
+    local_sum_exp = mappings.reduce_from_tensor_model_parallel_region(local_sum_exp)""",
"""-        torch.distributed.all_reduce(
-            all_log_probs,
-            op=torch.distributed.ReduceOp.SUM,
-            group=mpu.get_tensor_model_parallel_group()
-        )
+        all_log_probs = mappings.reduce_from_tensor_model_parallel_region(all_log_probs)""",
"""-            torch.distributed.all_reduce(
-                per_token_log_probs,
-                op=torch.distributed.ReduceOp.SUM,
-                group=mpu.get_tensor_model_parallel_group()
-            )
+            per_token_log_probs = mappings.reduce_from_tensor_model_parallel_region(per_token_log_probs)""",
"""             group=mpu.get_context_parallel_group()
         )
 
-        torch.distributed.all_reduce(
-            all_log_probs,
-            op=torch.distributed.ReduceOp.SUM,
-            group=mpu.get_context_parallel_group()
-        )
+        all_log_probs = mappings.reduce_from_tensor_model_parallel_region(all_log_probs)
 
         if per_token:
-            torch.distributed.all_reduce(
-                per_token_log_probs,
-                op=torch.distributed.ReduceOp.SUM,
-                group=mpu.get_context_parallel_group()
-            )
+            per_token_log_probs = mappings.reduce_from_tensor_model_parallel_region(per_token_log_probs)""",

        ],
        "mindspeed_rl/utils/loggers.py": [
"""             fmt_msg += f\"iteration: {iteration} / {steps} | \"
             if isinstance(msg, dict):
                 for key in msg:
-                    fmt_msg += f\"{key} : {format(msg[key], '.4f')} | \"
+                    try:
+                        fmt_msg += f\"{key} : {format(msg[key], '.16f')} | \"
+                    except:
+                        temp = float(str(msg[key]))
+                        fmt_msg += f\"{key} : {format(temp, '.16f')} | \"
+                        pass
                 fmt_msg = fmt_msg[:-2]
             else:
                 fmt_msg = f\"{fmt_msg} {str(msg)}\"""",
        ],
        "mindspeed_rl/workers/base_worker.py": [
""" 
             if get_tensor_model_parallel_rank(self.parallel_state, use_vllm) != 0 or \\
                     get_pipeline_model_parallel_rank(self.parallel_state, use_vllm) != 0:
+                batch_data_shape = batch_data_shape.numpy().tolist()
+                batch_data_dtype = batch_data_dtype.item()
                 if batch_data_dtype == 1:
                     batch_data[key] = torch.empty(batch_data_shape[0], batch_data_shape[1],
                                                   device=torch.cuda.current_device(),""",
""" 
             # 传输tensor数据
             torch.distributed.broadcast(
-                batch_data[key].cuda(), get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
+                batch_data[key], get_tensor_model_parallel_src_rank(self.parallel_state, use_vllm),
                 group=get_tensor_model_parallel_group(self.parallel_state, use_vllm)
             )
             torch.distributed.broadcast(
-                batch_data[key].cuda(), get_pipeline_model_parallel_src_rank(self.parallel_state, use_vllm),
+                batch_data[key], get_pipeline_model_parallel_src_rank(self.parallel_state, use_vllm),
                 group=get_pipeline_model_parallel_group(self.parallel_state, use_vllm)
             )
         data_loader = trans_batch_to_data_loader(batch_data,
                                                  experience_count * n_samples_per_prompt)  # experience_count
-        index = index.cpu().numpy().tolist()
+        index = index.asnumpy().tolist()
         return data_loader, index
 
     def collect_transfer_dock_data(self, output, index, n_samples_per_prompt=1, use_vllm=False):""",
"""             # 获取当前行的截断索引
             trunc_idx = index_tensor[i].item()
             # 截断当前行
-            truncated_row = tensor[i, :trunc_idx].cpu()
+            truncated_row = tensor[i, :trunc_idx]
             # 将截断后的行添加到列表中
             truncated_tensors.append(truncated_row)"""
        ],
        "mindspeed_rl/workers/resharding/megatron_sharding_manager.py": [
"""         elif isinstance(data, dict):
             return {key: self._move_to_device(value, device) for key, value in data.items()}
         elif isinstance(data, torch.Tensor):
-            return data.to(device, non_blocking=True)
+            if device == \"cpu\":
+                return data.cpu(non_blocking=True)
+            else:
+                return data.cuda(non_blocking=True)
         else:
             return data""",
"""         for train_model in self.train_model:
             for buffer in chain(train_model.buffers, train_model.expert_parallel_buffers):
                 if hasattr(buffer, 'param_data'):
-                    buffer.param_data = buffer.param_data.to(torch.cuda.current_device(), non_blocking=True)
+                    buffer.param_data = buffer.param_data
                     is_distributed_optim = True
         if not is_distributed_optim:
             for _, param in self.train_model.named_parameters():""",
"""         for train_model in self.train_model:
             for buffer in chain(train_model.buffers, train_model.expert_parallel_buffers):
                 if hasattr(buffer, 'param_data'):
-                    buffer.param_data = buffer.param_data.to('cpu', non_blocking=True)
+                    buffer.param_data = buffer.param_data
                     is_distributed_optim = True
         if not is_distributed_optim:
             for _, param in self.train_model.named_parameters():"""
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
-            memory_buffer.data = memory_buffer.data.to(\"cpu\", non_blocking=True)
+            memory_buffer.data = memory_buffer.data.cpu(non_blocking=True)
 
     def onload(self):
         for memory_buffer in self.memory_buffers.values():
-            memory_buffer.data = memory_buffer.data.to(torch.cuda.current_device(), non_blocking=True)
+            memory_buffer.data = memory_buffer.data.cuda()"""
        ],
        "mindspeed_rl/workers/resharding/vllm_weight_container.py": [
"""         name_pairs = sorted(list(set([(name, _replace_name_v2m(normal_layer_func(name), self.params_mapping))
                                       for name in weight_buffer.weight_names])))
         for hf_name, megatron_name in name_pairs:
-            megatron_param = dict(true_megatron_model.named_parameters())[megatron_name]
+            try:
+                megatron_param = dict(true_megatron_model.named_parameters())[megatron_name]
+            except KeyError:
+                # print(hf_name, megatron_name)
+                print(f\"[WARNING] megatron_name: {megatron_name} is not Found. Skip...\")
+                continue
             param = _transfer_from_megatron_division(megatron_param, megatron_name)
             weight_buffer[hf_name].copy_(param)"""
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
"""         ray.get(placement_group.ready())
         return placement_group
 
-    def create_actor_handlers(self, placement_group, world_size, rank_index, master_addr, master_port) \\
+    def create_actor_handlers(self, placement_group, world_size, rank_index, master_addr, master_port, sched_host, sched_port) \\
             -> ray.actor.ActorHandle:
         runtime_env = {
             \"env_vars\": {""",
"""                 \"MASTER_PORT\": str(master_port) if master_port else \"\",
                 \"WORLD_SIZE\": str(world_size),
                 \"RANK\": str(rank_index),
+                \"MS_ROLE\": \"MS_WORKER\",
+                \"MS_WORKER_NUM\": str(world_size),
+                \"MS_NODE_ID\": str(rank_index), 
+                \"MS_SCHED_HOST\": str(sched_host),
+                \"MS_SCHED_PORT\": str(sched_port),
+                \"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES\": \"1\",
+                \"USE_RAY\":\"true\"
+
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
+                ms_sched_host=self.ms_sched_host,
+                ms_sched_port=self.ms_sched_port,
+            )
+        scheduler_actor.get_status.remote()
+
         actor_handle = self.create_actor_handlers(
-            placement_group, world_size, 0, None, None)
+            placement_group, world_size, 0, None, None, self.ms_sched_host, self.ms_sched_port)
         self.actor_handlers.append(actor_handle)
         return actor_handle""",
"""         master_addr, master_port = ray.get(master_handler.get_master_addr_port.remote())
         for rank in range(1, world_size):
             self.actor_handlers.append(self.create_actor_handlers(
-                placement_group, world_size, rank, master_addr, master_port))
+                placement_group, world_size, rank, master_addr, master_port, self.ms_sched_host, self.ms_sched_port))
 
     def execute_async_command(self, method_name: str, *args, **kwargs):
         ray_objs = []"""
        ],
        "mindspeed_rl/workers/scheduler/scheduler.py": [
"""
import ray
 
import mindspore as ms
from mindspore import mint

from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.worker import Worker
 
 
@ray.remote
class WorkerGroupScheduler(Worker):
    def __init__(self):
        self.success = False
        if not ms.communication._comm_helper._is_initialized():
            mint.distributed.init_process_group(
                backend=\"hccl\"
            )
            self.success = True
    
    def get_status(self):
        return self.success
 
 
def create_worker_group_scheduler(name, world_size, ms_sched_host, ms_sched_port):
    env_vars: dict[str, str] = {
        \"MS_ROLE\": \"MS_SCHED\",
        \"MS_SCHED_HOST\": str(ms_sched_host),
        \"MS_SCHED_PORT\": str(ms_sched_port),
        \"MS_WORKER_NUM\": str(world_size),
        \"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES\": \"1\",
        'WORLD_SIZE': str(world_size),
        'WG_BACKEND': 'ray',
    }
    
    options = {'runtime_env': {'env_vars': env_vars}, 'name': name}
    return WorkerGroupScheduler.options(**options).remote()"""
        ]
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
"""         inv_freq = 1.0 / (base**(torch.arange(
-            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
+            0, self.rotary_dim, 2, dtype=torch.float, device=current_platform.device_type) / self.rotary_dim))"""
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
        ]
    },
    "vllm-ascend":{
        "requirements.txt": ["""-vllm==0.7.3"""],
        "setup.py": ["""-    entry_points={'vllm.platform_plugins': ["ascend = vllm_ascend:register"]})
+    entry_points={
+        'vllm.platform_plugins': ["ascend = vllm_ascend:register"],
+        'vllm.general_plugins':
+        ["ascend_enhanced_model = vllm_ascend:register_model"]
+    })"""],
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
"""             kv_heads_num = self.num_kv_heads
-            q_nope_t = torch_npu.npu_transpose(q_nope, (1, 0, 2),
-                                               require_contiguous=True)
+            # q_nope_t = torch_npu.npu_transpose(q_nope, (1, 0, 2),
+            #                                    require_contiguous=True)
+            q_nope_t = torch.transpose(q_nope, 0, 1)
             q_nope_out = torch.bmm(q_nope_t, self.w_kc)
-            q_nope = torch_npu.npu_transpose(q_nope_out, (1, 0, 2),
-                                             require_contiguous=True)
+            # q_nope = torch_npu.npu_transpose(q_nope_out, (1, 0, 2),
+            #                                  require_contiguous=True)
+            q_nope = torch.transpose(q_nope_out, 0, 1)""",
"""-                torch_npu.npu_selfattention(query=query,
-                                            key=key,
-                                            value=value,
-                                            kvcacheCfg=0,
-                                            mask=mask,
-                                            maskType=1,
-                                            isTriuMask=0,
-                                            seqLen=self.seq_lens_tensor_cpu,
-                                            scale=self.scale,
-                                            qScale=1,
-                                            scaleType=0,
-                                            headNum=self.num_heads,
-                                            kvHeadNum=self.num_heads,
-                                            mlaVHeadSize=0,
-                                            calcType=3,
-                                            kernelType=0,
-                                            clampType=0,
-                                            quantType=0,
-                                            cacheType=0,
-                                            windowSize=0,
-                                            clampMin=0,
-                                            clampMax=0,
-                                            batchRunStatusEnable=False,
-                                            inputLayout=0,
-                                            outDataType=0,
-                                            out=attn_output)
+                torch_npu.npu_selfattention(query=query.contiguous(),
+                                           key=key.contiguous(),
+                                           value=value.contiguous(),
+                                           kvcacheCfg=0,
+                                           mask=mask.contiguous(),
+                                           maskType=1,
+                                           isTriuMask=0,
+                                           seqLen=self.seq_lens_tensor_cpu.contiguous(),
+                                           scale=self.scale,
+                                           qScale=1,
+                                           scaleType=0,
+                                           headNum=self.num_heads,
+                                           kvHeadNum=self.num_heads,
+                                           mlaVHeadSize=0,
+                                           calcType=3,
+                                           kernelType=0,
+                                           clampType=0,
+                                           quantType=0,
+                                           cacheType=0,
+                                           windowSize=0,
+                                           clampMin=0,
+                                           clampMax=0,
+                                           batchRunStatusEnable=False,
+                                           inputLayout=0,
+                                           outDataType=0,
+                                           out=attn_output)""",
"""             block_tables = attn_metadata.decode_metadata.block_tables
-            torch_npu.npu_pagedattention(query=query,
-                                         keyCache=key_cache,
-                                         valueCache=None,
-                                         contextLens=self.seq_lens_tensor_cpu,
-                                         maskType=0,
-                                         kvHeadNum=self.num_kv_heads,
-                                         headNum=self.num_heads,
-                                         mlaVHeadSize=self.kv_lora_rank,
-                                         qkScale=self.scale,
-                                         blockTables=block_tables,
-                                         batchRunStatusEnable=False,
-                                         hasQuantOffset=False,
-                                         compressType=0,
-                                         calcType=0,
-                                         scaleType=0,
-                                         quantType=0,
-                                         inputLayout=0,
-                                         outDataType=-1,
-                                         attnOut=attn_output)
-            attn_output_t = torch_npu.npu_transpose(attn_output, (1, 0, 2),
-                                                    require_contiguous=True)
+            torch_npu.npu_pagedattention(query=query.contiguous(),
+                                        keyCache=key_cache.contiguous(),
+                                        valueCache=None,
+                                        contextLens=self.seq_lens_tensor_cpu,
+                                        maskType=0,
+                                        kvHeadNum=self.num_kv_heads,
+                                        headNum=self.num_heads,
+                                        mlaVHeadSize=self.kv_lora_rank,
+                                        qkScale=self.scale,
+                                        blockTables=block_tables.contiguous(),
+                                        batchRunStatusEnable=False,
+                                        hasQuantOffset=False,
+                                        compressType=0,
+                                        calcType=0,
+                                        scaleType=0,
+                                        quantType=0,
+                                        inputLayout=0,
+                                        outDataType=-1,
+                                        attnOut=attn_output)
+            # attn_output_t = torch_npu.npu_transpose(attn_output, (1, 0, 2),
+            #                                         require_contiguous=True)
+            attn_output_t = torch.transpose(attn_output, 0, 1)
             attn_output_t = torch.bmm(attn_output_t, self.w_vc)
-            attn_output = torch_npu.npu_transpose(attn_output_t, (1, 0, 2),
-                                                  require_contiguous=True)
+            # attn_output = torch_npu.npu_transpose(attn_output_t, (1, 0, 2),
+            #                                       require_contiguous=True)
+            attn_output = torch.transpose(attn_output_t, 0, 1)"""],
        "vllm_ascend/models/__init__.py": [
"""REMOVE"""
        ],
        "vllm_ascend/models/qwen2_vl.py": [
"""REMOVE"""
        ],
        "vllm_ascend/ops/fused_moe.py": [
"""+fused_experts_num = 2
 def fused_experts(hidden_states: torch.Tensor, w1: torch.Tensor,
                   w2: torch.Tensor, topk_weights: torch.Tensor,
                   topk_ids: torch.Tensor, top_k: int):""",
"""     gate_up_out_list = torch_npu.npu_grouped_matmul(x=[expanded_x],
-                                                    weight=[w1],
-                                                    split_item=2,
-                                                    group_list_type=0,
-                                                    group_type=0,
-                                                    group_list=expert_tokens)
+                                                   weight=[w1],
+                                                   split_item=3,
+                                                   group_list_type=0,
+                                                   group_type=0,
+                                                   group_list=expert_tokens)""",
"""     down_out_list = torch_npu.npu_grouped_matmul(x=[gate_up_out],
-                                                 weight=[w2],
-                                                 split_item=2,
-                                                 group_list_type=0,
-                                                 group_type=0,
-                                                 group_list=expert_tokens)
+                                                weight=[w2],
+                                                split_item=3,
+                                                group_list_type=0,
+                                                group_type=0,
+                                                group_list=expert_tokens)""",
"""     routing_weights = topk_weights.to(down_out_list.dtype)
+    _, h_skip1 = down_out_list.shape
+    N_skip1, _ = topk_ids.shape
+    skip1_val = torch.zeros(N_skip1, h_skip1, dtype=down_out_list.dtype)
+    bias_val = torch.zeros(E, h_skip1, dtype=down_out_list.dtype)
     hidden_states = torch_npu.npu_moe_finalize_routing(
-        down_out_list,
-        skip1=None,
-        skip2=None,
-        bias=None,
-        scales=routing_weights,
-        expanded_src_to_dst_row=expanded_row_idx,
-        export_for_source_row=topk_ids)
+       down_out_list,
+       skip1=skip1_val,
+       skip2=None,
+       bias=bias_val,
+       scales=routing_weights,
+       expanded_src_to_dst_row=expanded_row_idx,
+       export_for_source_row=topk_ids)""",
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
"""-        torch_npu.npu_rope(
-            positions,
-            query,
-            key,
-            self.head_size,
-            self.cos_sin_cache,
-            self.is_neox_style,
-        )
+        #torch_npu.npu_rope(
+        #    positions,
+        #    query,
+        #    key,
+        #    self.head_size,
+        #    self.cos_sin_cache,
+        #    self.is_neox_style,
+        #)""",
"""-RotaryEmbedding.forward_oot = rope_forward_oot
+# RotaryEmbedding.forward_oot = rope_forward_oot"""
        ],
        "vllm_ascend/patch/__init__.py": [
"""REMOVE"""
        ],
        "vllm_ascend/patch/patch_cache_dtype.py": [
"""REMOVE"""
        ],
        "vllm_ascend/patch/patch_metrics.py": [
"""REMOVE"""
        ],
        "vllm_ascend/patch/patch_minicpm.py": [
"""REMOVE"""
        ],
        "vllm_ascend/patch/patch_rejection_sampler.py": [
"""REMOVE"""
        ],
        "vllm_ascend/patch/ray_patch.py": [
"""REMOVE"""
        ],
        "vllm_ascend/patch_module.py": [
"""REMOVE"""
        ],
        "vllm_ascend/worker.py": [
""" 
 import torch
 import torch.distributed
+import torch_npu
 from torch import nn
 from vllm import envs
 from vllm.config import ParallelConfig, VllmConfig""",
""" from vllm.model_executor import set_random_seed
 from vllm.model_executor.layers.sampler import SamplerOutput
 from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
+from vllm.platforms import current_platform
 from vllm.prompt_adapter.request import PromptAdapterRequest
 from vllm.sequence import (ExecuteModelRequest, IntermediateTensors,
                            SequenceGroupMetadata, SequenceGroupMetadataDelta)""",
""" from vllm.worker.cache_engine import CacheEngine
 from vllm.worker.enc_dec_model_runner import EncoderDecoderModelRunner
 from vllm.worker.model_runner_base import ModelRunnerBase
+from vllm.worker.pooling_model_runner import PoolingModelRunner
 from vllm.worker.worker_base import (LocalOrDistributedWorkerBase, WorkerBase,
                                      WorkerInput)
 
-from vllm_ascend.platform import NPUPlatform
+from vllm_ascend.model_runner import NPUModelRunner
 from vllm_ascend.utils import try_register_lib
-from vllm_ascend.worker.model_runner import NPUModelRunner
-from vllm_ascend.worker.pooling_model_runner import NPUPoolingModelRunner
 
 logger = init_logger(__name__)""",
"""     \"\"\"
 
     def __init__(
-            self,
-            vllm_config: VllmConfig,
-            local_rank: int,
-            rank: int,
-            distributed_init_method: str,
-            is_driver_worker: bool = False,
-            model_runner_cls: Optional[Type[ModelRunnerBase]] = None) -> None:
-        # TODO: Remove this line after fixing the hard-coding issue in VLLM later.
-        from torch_npu.contrib import transfer_to_npu  # noqa: F401
-
-        # Register ops and patch when worker init.
+        self,
+        vllm_config: VllmConfig,
+        local_rank: int,
+        rank: int,
+        distributed_init_method: str,
+        is_driver_worker: bool = False,
+        model_runner_cls: Optional[Type[ModelRunnerBase]] = None,
+    ) -> None:
+        # Register ops when worker init.
         from vllm_ascend import ops  # noqa: F401
-        from vllm_ascend import patch  # noqa: F401
 
         WorkerBase.__init__(self, vllm_config=vllm_config)
         # Try to import mindie_turbo to accelerate vLLM inference.""",
"""         speculative_config = self.speculative_config
         model_config = self.model_config
         speculative_args = {} if speculative_config is None \\
-            or (speculative_config.draft_model_config.hf_config.model_type ==
-                model_config.hf_config.model_type) \\
+            or (speculative_config.draft_model_config.model ==
+                model_config.model) \\
             or (speculative_config.draft_model_config.hf_config.model_type
-                not in [\"medusa\", \"mlp_speculator\", \"eagle\", \"deepseek_mtp\"]) \\
+                not in [\"medusa\", \"mlp_speculator\", \"eagle\"]) \\
                     else {\"return_hidden_states\": True}
 
         ModelRunnerClass: Type[ModelRunnerBase] = NPUModelRunner
         if model_config.runner_type == \"pooling\":
-            ModelRunnerClass = NPUPoolingModelRunner
+            ModelRunnerClass = PoolingModelRunner
         elif self.model_config.is_encoder_decoder:
             ModelRunnerClass = EncoderDecoderModelRunner
         self.model_runner: ModelRunnerBase = ModelRunnerClass(""",
"""         # Torch profiler. Enabled and configured through env vars:
         # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
         if envs.VLLM_TORCH_PROFILER_DIR:
-            # lazy import so that torch_npu is not required for normal use.
-            import torch_npu
             torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
             logger.info(\"Profiling enabled. Traces will be saved to: %s\",
                         torch_profiler_trace_dir)""",
"""         else:
             self.profiler = None
 
+    def start_profile(self):
+        if self.profiler is None:
+            raise RuntimeError(\"Profiler is not enabled.\")
+        self.profiler.start()
+
+    def stop_profile(self):
+        if self.profiler is None:
+            raise RuntimeError(\"Profiler is not enabled.\")
+        self.profiler.stop()
+
     def init_device(self) -> None:
+        self.device_config.device.type = \"npu\"
         if self.device_config.device.type == \"npu\":
+            # # This env var set by Ray causes exceptions with graph building.
+            # os.environ.pop(\"NCCL_ASYNC_ERROR_HANDLING\", None)
             self.device = torch.device(f\"npu:{self.local_rank}\")
-            NPUPlatform.set_device(self.device)
-            NPUPlatform.empty_cache()
-            self.init_npu_memory = NPUPlatform.mem_get_info()[0]
+            current_platform.set_device(self.device)
+
+            current_platform.empty_cache()
+            self.init_npu_memory = current_platform.mem_get_info()[0]
         else:
             raise RuntimeError(
                 f\"Not support device type: {self.device_config.device}\")
         # Initialize the distributed environment.
-        self._init_worker_distributed_environment(self.parallel_config,
-                                                  self.rank,
-                                                  self.distributed_init_method,
-                                                  self.local_rank)
+        init_worker_distributed_environment(self.parallel_config, self.rank,
+                                            self.distributed_init_method,
+                                            self.local_rank)
         # Set random seed.
         set_random_seed(self.model_config.seed)
 
     def load_model(self):
         self.model_runner.load_model()
 
-    def start_profile(self):
-        if self.profiler is None:
-            raise RuntimeError(\"Profiler is not enabled.\")
-        self.profiler.start()
-
-    def stop_profile(self):
-        if self.profiler is None:
-            raise RuntimeError(\"Profiler is not enabled.\")
-        self.profiler.stop()
-
     def save_sharded_state(
         self,
         path: str,""",
"""         self.model_runner.save_tensorized_model(
             tensorizer_config=tensorizer_config, )
 
-    @NPUPlatform.inference_mode()
+    @current_platform.inference_mode()
     def determine_num_available_blocks(self) -> Tuple[int, int]:""",
"""         # Profile the memory usage of the model and get the maximum number of
         # cache blocks that can be allocated with the remaining free memory.
-        NPUPlatform.empty_cache()
+        current_platform.empty_cache()
 
         # Execute a forward pass with dummy inputs to profile the memory usage
         # of the model.""",
""" 
         # Calculate the number of blocks that can be allocated with the
         # profiled peak memory.
-        free_npu_memory, total_npu_memory = NPUPlatform.mem_get_info()
+        free_npu_memory, total_npu_memory = current_platform.mem_get_info()
+
         # NOTE(woosuk): Here we assume that the other processes using the same
         # GPU did not change their memory usage during the profiling.
         peak_memory = self.init_npu_memory - free_npu_memory""",
"""         gc.collect()
         # TODO: don`t need impl this func after empty_cache in
         # Worker.determine_num_available_blocks() unified`
-        NPUPlatform.empty_cache()
+        current_platform.empty_cache()
+        # num_npu_blocks //= 100
+        num_npu_blocks = 65536
         return num_npu_blocks, num_cpu_blocks
 
     def initialize_cache(self, num_gpu_blocks: int,""",
"""                         self.parallel_config, self.device_config)
             for _ in range(self.parallel_config.pipeline_parallel_size)
         ]
-        import torch_npu
-        for ve in range(self.parallel_config.pipeline_parallel_size):
-            num_layers = len(self.cache_engine[ve].gpu_cache)
-            for i in range(num_layers):
-                torch_npu.npu_format_cast(self.cache_engine[ve].gpu_cache[i],
-                                          2)
         self.gpu_cache = [
             self.cache_engine[ve].gpu_cache
             for ve in range(self.parallel_config.pipeline_parallel_size)""",
"""                                                 self.model_config,
                                                 self.parallel_config)
 
-    def _init_worker_distributed_environment(
-            self,
-            parallel_config: ParallelConfig,
-            rank: int,
-            distributed_init_method: Optional[str] = None,
-            local_rank: int = -1,
-            backend: str = \"hccl\") -> None:
-        \"\"\"Initialize the distributed environment.\"\"\"
-        set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)
-        init_distributed_environment(parallel_config.world_size, rank,
-                                     distributed_init_method, local_rank,
-                                     backend)
-        ensure_model_parallel_initialized(
-            parallel_config.tensor_parallel_size,
-            parallel_config.pipeline_parallel_size)
+
+def init_worker_distributed_environment(
+        parallel_config: ParallelConfig,
+        rank: int,
+        distributed_init_method: Optional[str] = None,
+        local_rank: int = -1,
+        backend: str = \"hccl\") -> None:
+    \"\"\"Initialize the distributed environment.\"\"\"
+    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)
+
+    init_distributed_environment(parallel_config.world_size, rank,
+                                 distributed_init_method, local_rank, backend)
+
+    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
+                                      parallel_config.pipeline_parallel_size)
 
 
 def raise_if_cache_size_invalid(num_gpu_blocks, block_size, is_attention_free,"""
        ],
        "vllm_ascend/worker.py":["""     def init_device(self) -> None:
+        self.device_config.device.type = "npu"
         if self.device_config.device.type == "npu":""",
"""         current_platform.empty_cache()
+        num_npu_blocks = 65536
         return num_npu_blocks, num_cpu_blocks"""],
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