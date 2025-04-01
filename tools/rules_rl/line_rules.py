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
         """     if args.use_fused_rotary_pos_emb:
-        t = torch_npu.npu_rotary_mul(t, cos_, sin_).to(t.dtype)
+        mode = 1 if rotary_interleaved else 0
+        t = torch_npu.npu_rotary_position_embedding(t, cos_.to(t.dtype), sin_.to(t.dtype), mode=mode).to(t.dtype)
+        # t = torch_npu.npu_rotary_mul(t, cos_, sin_).to(t.dtype)
     else:
         t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
-    
     return torch.cat((t, t_pass), dim=-1)""",
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

    "mindspeed_llm/tasks/checkpoint/models.py": [
"""-                return _get_dst_obj(self, value, **kwargs).weight.data.copy_(kwargs.get('data'))
+                set_tensor = _get_dst_obj(self, value, **kwargs)
+                set_tensor.weight.data = kwargs.get('data')
+                return set_tensor.weight.data""",
"""-                return _get_dst_obj(self, value, **kwargs).bias.data.copy_(kwargs.get('data'))
+                set_tensor = _get_dst_obj(self, value, **kwargs)
+                set_tensor.bias.data = kwargs.get('data')
+                return set_tensor.bias.data""",
"""            self.module = [AutoModelForCausalLM.from_pretrained(
-                load_dir, device_map=device_map, trust_remote_code=trust_remote_code, local_files_only=True
+                load_dir, trust_remote_code=trust_remote_code, local_files_only=True, low_cpu_mem_usage=False"""
    ],
    "mindspeed_llm/tasks/megatron_adaptor.py": [
        """         # For torch >= 2.2.0
-        torch.compile = torch.jit.script""",
        """-        megatron.core.jit.jit_fuser = dummy_jit""",
        """-        from mindspeed.core.transformer.moe.token_dispatcher import allgather_token_permutation, \\
-            allgather_token_unpermutation
-        from mindspeed.core.transformer.moe.grouped_gemm_util import Ops, grouped_gemm_is_available, \\
-            get_device_capability, assert_grouped_gemm_is_available""",
        """         from mindspeed.core.transformer.moe.moe_utils import permute, unpermute
         from mindspeed.core.transformer.moe.experts import group_mlp_forward
+        from mindspeed.core.transformer.moe.grouped_gemm_util import Ops, grouped_gemm_is_available, \\
+            assert_grouped_gemm_is_available""",
        """-        MegatronAdaptation.register('torch.cuda.get_device_capability', get_device_capability)""",
        """                 from mindspeed.core.transformer.moe.token_dispatcher import preprocess, alltoall_token_permutation
                 from mindspeed.core.transformer.moe.experts import sequential_mlp_forward
+                MegatronAdaptation.register(
+                    'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.preprocess',
+                    preprocess)""",
"""-                if args.moe_tp_extend_ep:
-                    from mindspeed.core.transformer.moe.token_dispatcher import (
-                        preprocess_tp_extend_ep, alltoall_token_unpermutation_tp_extend_ep,
-                        alltoall_token_permutation_tp_extend_ep
-                    )
+                if args.moe_alltoall_overlap_comm:
+                    from mindspeed.core.transformer.moe.token_dispatcher import alltoall_token_permutation_new, \\
+                        alltoall_token_unpermutation_new
+                    MegatronAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.forward',
+                                                group_mlp_forward)
+                    MegatronAdaptation.register(
+                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
+                        alltoall_token_permutation_new)""",
"""                     MegatronAdaptation.register(
-                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.preprocess',
-                        preprocess_tp_extend_ep)
-
-                    if args.moe_alltoall_overlap_comm:
-                        from mindspeed.core.transformer.moe.token_dispatcher import alltoall_token_permutation_new, \\
-                            alltoall_token_unpermutation_new
-                        from mindspeed.core.transformer.moe.experts import group_mlp_forward
-                        MegatronAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.forward', group_mlp_forward)
-                        MegatronAdaptation.register(
-                            'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
-                            alltoall_token_permutation_new)
-                        MegatronAdaptation.register(
-                            'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
-                            alltoall_token_unpermutation_new)
-                    else:
-                        MegatronAdaptation.register(
-                            'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
-                            alltoall_token_permutation_tp_extend_ep)
-                        MegatronAdaptation.register(
-                            'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
-                            alltoall_token_unpermutation_tp_extend_ep)
+                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
+                        alltoall_token_unpermutation_new)""",
        """-        from mindspeed.core.transformer.moe.experts import groupedmlp_init_wrapper, groupedmlp_forward_wrapper
-        MegatronAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.__init__',
-                                    groupedmlp_init_wrapper)
-        MegatronAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.forward',
-                                    groupedmlp_forward_wrapper)""",
        """-                    from mindspeed.core.transformer.moe.token_dispatcher import preprocess, alltoall_token_permutation
                     MegatronAdaptation.register(
-                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.preprocess',
-                        preprocess)
-                    if args.moe_alltoall_overlap_comm:
-                        from mindspeed.core.transformer.moe.token_dispatcher import preprocess, alltoall_token_permutation
-                        from mindspeed.core.transformer.moe.token_dispatcher import alltoall_token_permutation_new, \\
-                            alltoall_token_unpermutation_new
-                        from mindspeed.core.transformer.moe.experts import group_mlp_forward
-
-                        MegatronAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.forward', group_mlp_forward)
-                        MegatronAdaptation.register(
-                            'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
-                            alltoall_token_permutation_new)
-                        MegatronAdaptation.register(
-                            'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
-                            alltoall_token_unpermutation_new)
-                    else:
-                        MegatronAdaptation.register(
-                            'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
-                            alltoall_token_permutation)
-                            
+                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
+                        alltoall_token_permutation)""",
"""        MegatronAdaptation.register('megatron.core.transformer.moe.grouped_gemm_util.assert_grouped_gemm_is_available',
                                    assert_grouped_gemm_is_available)
+        MegatronAdaptation.register('megatron.core.transformer.moe.grouped_gemm_util.ops', Ops)
+        MegatronAdaptation.register('megatron.core.transformer.moe.grouped_gemm_util.grouped_gemm_is_available',
+                                    grouped_gemm_is_available)""",
"""             'megatron.core.transformer.transformer_layer.TransformerLayer._get_layer_offset',
             get_layer_offset_wrapper)
-        MegatronAdaptation.register(
-            'megatron.core.parallel_state.get_nccl_options', get_nccl_options_wrapper)
+        # MegatronAdaptation.register(
+        #     'megatron.core.parallel_state.get_nccl_options', get_nccl_options_wrapper)""",
        """-        MegatronAdaptation.register('megatron.core.tensor_parallel.random.CheckpointFunction.forward',
-                                    checkpoint_forward_wrapper)
-        MegatronAdaptation.register('megatron.core.tensor_parallel.random.CheckpointFunction.backward',
-                                    checkpoint_backward_wrapper)
+        # MegatronAdaptation.register('megatron.core.tensor_parallel.random.CheckpointFunction.forward',
+        #                             checkpoint_forward_wrapper)
+        # MegatronAdaptation.register('megatron.core.tensor_parallel.random.CheckpointFunction.backward',
+        #                             checkpoint_backward_wrapper)"""
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
+        rank = os.environ.get("MS_ROLE", None)""",
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
        """ from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base import WorkerGroup, ResourcePool, ClassWithInitArgs, Worker
+from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.scheduler import create_worker_group_scheduler""",
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
    "mindspeed_llm/tasks/posttrain/rlxf/workers/actor_train_infer.py":[
"""-                    additional_val = batch.get(additional_key).view(-1).cpu().numpy().tolist()
+                    additional_val = batch.get(additional_key).view(-1).asnumpy().tolist()""",
"""                tokens = batch["input_ids"]
-                tokens_list = tokens.view(-1).cpu().numpy().tolist()
+                tokens_list = tokens.view(-1).asnumpy().tolist()""",
        """     def update_actor(self, data):
         device = next(self.node.actor.model[0].parameters()).device
-        data = data.to(device)""",
"""         output = DataProto(meta_info={'metrics': metrics})
-        output = output.to('cpu')""",
"""         if old_log_probs is not None:  # pp last stage
             data.batch['old_log_probs'] = old_log_probs
-            data = data.to('cpu')""",
"""     max_length = max_length if max_length % pad_multi_of == 0 else (max_length // pad_multi_of + 1) * pad_multi_of
     torch.distributed.all_reduce(max_length, op=torch.distributed.ReduceOp.MAX)
+    max_length = max_length.item()""",
"""             return {'log_probs': log_probs}
 
         # We make recompute_old_log_prob by default here.
-        data = data.to(next(self.model[0].parameters()).device)""",
"""         # broadcast from last pp rank to all other pp ranks
         # TODO: actually, we just need to control the sampling order.
 
-        data.batch['attention_mask'] = data.batch['attention_mask'].to(bool)
+        data.batch['attention_mask'] = data.batch['attention_mask'].to(torch.bool)""",
"""             and users have to combine the output in each dp rank manually.
 
         \"\"\"
+        # TODO check
+        self.args.use_kv_cache = False""",
"""         self.args.consumed_train_samples += self.args.global_batch_size
         self.num_floating_point_operations_so_far += num_floating_point_operations(self.args, self.args.global_batch_size)
 
         # add empty cache after each compute
         torch.cuda.empty_cache()
-
+        self.args.use_kv_cache = True
         return metrics""","""         # We make recompute_old_log_prob by default here.
-        data = data.to(next(self.model[0].parameters()).device)
+        # data = data.to(next(self.model[0].parameters()).device)""","""-        data.batch['attention_mask'] = data.batch['attention_mask'].to(bool)
+        data.batch['attention_mask'] = data.batch['attention_mask'].to(torch.bool)"""
    ],
    "mindspeed_llm/tasks/posttrain/rlxf/workers/reference.py":[
        """     @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
     def compute_ref_log_prob(self, data: DataProto):
-        data = data.to('cuda')
         output = self.reference.compute_log_prob(data=data)
         if output is not None:
             output = DataProto.from_dict(tensors={'ref_log_prob': output})
-            output = output.to('cpu')""",
"""         for model_module in self.model:
             model_module.eval()
 
-        data = data.to(next(self.model[0].parameters()).device)""",
"""         - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
         \"\"\"
         args = get_args()
-        data.batch['attention_mask'] = data.batch['attention_mask'].to(bool)""",
    ],
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
+    z_max = ReduceFromContextParallelRegionDPO()(z_max)""",
"""     # Step 5: Perform all-reduce to get the global sum of exp(x - z_max) across all processes
-    torch.distributed.all_reduce(
-        local_sum_exp,
-        op=torch.distributed.ReduceOp.SUM,
-        group=mpu.get_tensor_model_parallel_group()
-    )
+    local_sum_exp = mappings.reduce_from_tensor_model_parallel_region(local_sum_exp)""",
"""         all_log_probs = per_token_log_probs.sum(-1)
         valid_length = loss_mask.sum(-1)
 
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
+            per_token_log_probs = mappings.reduce_from_tensor_model_parallel_region(per_token_log_probs)"""

    ],
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
+    if args.moe_alltoall_overlap_comm and args.tensor_model_parallel_size > 1:
+        raise AssertionError('`--moe-alltoall-overlap-comm` do not support tp for now.')""",
"""     args.adaptive_recompute_profiling_step = 10
+    args.moe_tp_extend_ep = False
     args.recompute_in_bubble = False"""],
     "mindspeed_llm/training/utils.py":["""             slice_obj[dim] = slice(i, i + window_size)
-        slices.append(tensor[tuple(slice_obj)])
+        slices.append(tensor[tuple(slice_obj)].clone())"""],
    "mindspeed_llm/core/tensor_parallel/layers.py":["""-        weight = torch.split(weight, weight.shape[0] // args_.output_layer_slice_num, dim=0)
+        weight = torch.chunk(weight, args_.output_layer_slice_num, 0)"""],
    "mindspeed_llm/core/models/gpt/gpt_model.py":[
"""         if not self.share_embeddings_and_output_weights and self.share_mtp_embedding_and_output_weight:
-            output_weight = self.output_layer.weight.detach()
+            # output_weight = self.output_layer.weight.detach()
+            from mindspore import ops
+            output_weight = ops.stop_gradient(self.output_layer.weight)""",
"""             loss += args.mtp_loss_scale / args.num_nextn_predict_layers * mtp_loss
+    
+    logits, _ = self.output_layer(hidden_states, weight=self.output_layer.weight)
+    # new add to scale logits
+    if args.output_multiplier_scale:
+        logits = logits * args.output_multiplier_scale
+
+    if args.output_logit_softcapping:
+        logits = logits / args.output_logit_softcapping
+        logits = torch.tanh(logits)
+        logits = logits * args.output_logit_softcapping
+
+    if labels[0] is None:
+        # [s b h] => [b s h]
+        return logits.transpose(0, 1).contiguous()"""],
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
"""         for param in self.module.parameters():
             if param.requires_grad:
                 # Expand so we get access to grad_fn.
-                param_tmp = param.expand_as(param)
                 # Get the gradient accumulator function.
-                grad_acc = param_tmp.grad_fn.next_functions[0][0]
-                grad_acc.register_hook(self._make_param_hook(param, self.param_to_buffer))
-                self.grad_accs.append(grad_acc)
+                param.register_hook(self._make_param_hook(param, self.param_to_buffer))
 
     def forward(self, *inputs, **kwargs):
         \"\"\"""",
"""                 if param.grad is not None and (
                     not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
                 ):
-                    param.main_grad.add_(param.grad.data)
+                    param.main_grad.add_(*unused)
                 param.grad = None
 
                 if self.ddp_config.overlap_grad_reduce:
                     param_to_buffer[param].register_grad_ready(param)
 
+                if hasattr(param, \"main_grad\"):
+                    return param.main_grad
+                return param.grad
+
         return param_hook
 
     @contextmanager"""
        ],
        "core/tensor_parallel/mappings.py": ["""-            output_split_sizes=output_split_sizes.tolist(),
-            input_split_sizes=input_split_sizes.tolist(),
+            output_split_sizes=output_split_sizes.tolist() if output_split_sizes else None,
+            input_split_sizes=input_split_sizes.tolist() if output_split_sizes else None,"""],
        "core/distributed/param_and_grad_buffer.py": [
"""         # gradient_scaling_factor already takes into account whether we are computing
         # an average or sum in the data-parallel collective.
         if self.gradient_scaling_factor != 1.0:
-            self.grad_data *= self.gradient_scaling_factor
+            self.grad_data.mul_(self.gradient_scaling_factor)
 
         # Decide reduce_op.
         reduce_op = torch.distributed.ReduceOp.SUM""",
"""         \"\"\"
         assert param in self.params, 'Param is not in the bucket'
         assert param not in self.params_with_grad, 'Cannot set grad twice'
+        if param in self.params_with_grad:
+            return
+        # assert param in self.params, 'Param is not in the bucket'
+        if param in self.params_with_grad:
+            return
+        # assert param not in self.params_with_grad, 'Cannot set grad twice'
         assert (
             self.ddp_config.overlap_grad_reduce
         ), 'register_grad_ready() should be called only when overlapping grad reduce'""",
"""         When the number of microbatches is greater than 1, we only want to register
         grads as ready when processing the last microbatch and overlap_grad_reduce is True.
         \"\"\"
+        if param in self.params_with_grad:
+            return
+        # assert param in self.params, 'Param is not in the bucket'
+        if param in self.params_with_grad:
+            return
+        # assert param not in self.params_with_grad, 'Cannot set grad twice'
         assert (
             self.ddp_config.overlap_grad_reduce
         ), 'register_grad_ready() should only be called when overlap_grad_reduce is True'"""
        ],
        "core/models/common/embeddings/rotary_pos_embedding.py":[
            """             rotary_seq_len = inference_params.max_sequence_length
         else:
-            if transformer.input_tensor is not None:
+            if transformer.input_tensor is not None and len(transformer.input_tensor.shape) > 1:"""
        ],"core/pipeline_parallel/schedules.py":["""     config = get_model_config(model[0])
+    config.forward_step_func = forward_step_func
     if config.overlap_p2p_comm and config.batch_p2p_comm:""","""         if parallel_state.is_pipeline_first_stage():
             if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                 input_tensors[model_chunk_id].append(None)
+        if input_tensors[model_chunk_id][-1] is None:
+            input_tensors[model_chunk_id][-1] = torch.tensor(0, dtype=torch.int)""","""         else:
             data = loss_func(output_tensor, non_loss_data=True)
             forward_data_store.append(data)
+            output_tensor = None
+    _pynative_executor.end_graph(forward_step_func, output_tensor, input_tensor[0])
 
     if config.timers is not None:
         config.timers('forward-compute').stop()""",],
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
        "training/arguments.py":["""     # Args from environment
-    args.rank = int(os.getenv('RANK', '0'))
+    # args.rank = int(os.getenv('RANK', '0'))
+    args.rank = int(os.getenv('MS_NODE_ID', '0'))
     args.world_size = int(os.getenv("WORLD_SIZE", '1'))
 
     return args""",
        """     if args.moe_grouped_gemm:
         assert args.bf16, 'Currently GroupedGEMM for MoE only supports bf16 dtype.'
-        dc = torch.cuda.get_device_capability()
-        assert dc[0] >= 8, "Unsupported compute capability for GroupedGEMM kernels."
+        # dc = torch.cuda.get_device_capability()
+        # assert dc[0] >= 8, "Unsupported compute capability for GroupedGEMM kernels.\""""
        ],
    },
    "mindspeed":{
        "core/transformer/moe/grouped_gemm_util.py":[""" # Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
 
 import torch
-from mindspeed.ops.npu_all_to_all_all_gather_bmm import npu_alltoall_allgather_bmm
-from mindspeed.ops.npu_bmm_reduce_scatter_all_to_all import npu_bmm_reducescatter_alltoall
+#from mindspeed.ops.npu_all_to_all_all_gather_bmm import npu_alltoall_allgather_bmm
+#from mindspeed.ops.npu_bmm_reduce_scatter_all_to_all import npu_bmm_reducescatter_alltoall
 
 
 def grouped_gemm_is_available():
"""],
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
+            param.register_hook(self._make_param_hook(param, self.param_to_buffer))""",
        """-        def param_hook(*unused):
+        def param_hook(grad):""",
        """-                    param.main_grad.add_(*unused)
+                    param.main_grad.add_(grad)""",
        """                 if self.ddp_config.overlap_grad_reduce:
                     param_to_buffer[param].register_grad_ready(param)
+                if hasattr(param, "main_grad"):
+                    return param.main_grad
+                return param.grad"""
        ],
        "core/transformer/moe/experts.py":[
        """-    return torch.zeros(zeros_shape, dtype=input_.dtype, layout=input_.layout, device=input_.device)
+    return torch.zeros(zeros_shape, dtype=input_.dtype, device=input_.device)""",
    ],
        "megatron_adaptor.py":["""-    aspm.register_patch('transformer_engine.pytorch.Linear', torch.nn.Module, create_dummy=True)""",
            """-    from torch.utils.cpp_extension import _get_build_directory
-    build_directory = _get_build_directory("", True)
-    delete_lock = Lock()
-    delete_lock_file(build_directory, delete_lock)
+    # from torch.utils.cpp_extension import _get_build_directory
+    # build_directory = _get_build_directory("", True)
+    # delete_lock = Lock()
+    # delete_lock_file(build_directory, delete_lock)"""
        ],
        "ops/gmm.py":[""" from torch.library import impl
+from mindspore import ops""",
            """ class GMMFunction(torch.autograd.Function):
-    builder = GMMOpBuilder()
-    builder2 = GMMV2OpBuilder()
+    # builder = GMMOpBuilder()
+    # builder2 = GMMV2OpBuilder()""","""-    def forward(ctx, original_weight, x, weight, bias, group_args):
-        group_list, group_type, gemm_fusion, group_list_type, group_list_data_type = group_args
+    def forward(ctx, original_weight, x, weight, bias, group_list, group_args):
+        group_type, gemm_fusion, group_list_type, group_list_data_type = group_args""","""-            outputs = GMMFunction.builder.load().npu_gmm([x], [weight], bias, group_list, group_type, group_list_type)
+            outputs = ops.function.math_func.gmm([x, ], [weight, ], bias=bias, group_list=group_list, group_type=group_type)
         elif group_list_type == 1:
-            outputs = GMMFunction.builder2.load().npu_gmm([x], [weight], bias, group_list, group_type, group_list_type)
+            outputs = ops.function.math_func.gmm_v2([x, ], [weight, ], bias=bias, group_list=group_list, group_type=group_type, group_list_type=group_list_type)""",
"""-                dx, dw, dbias = GMMFunction.builder.load().npu_gmm_backward([grad_outputs], [x], [weight], group_list,
-                                                                    ctx.group_list_type)
+                dx, dw, dbias = ops.function.math_func.gmm_backward([grad_outputs, ], [x, ], [weight],
+                                                        group_list=group_list)""","""-                dx, dw, dbias = GMMFunction.builder2.load().npu_gmm_backward([grad_outputs], [x], [weight], group_list,
-                                                                    ctx.group_list_type)
+                dx, dw, dbias = ops.function.math_func.gmm_v2_backward([grad_outputs, ], [x, ], [weight],
+                                                        group_list=group_list, group_list_type=ctx.group_list_type)""","""-            return None, dx[0], dw[0], dbias, None
+            return None, dx[0], dw[0], dbias, None, None""","""-        return torch.ops.mindspeed.npu_gmm(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, gemm_fusion=gemm_fusion)
+        return _npu_gmm(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, gemm_fusion=gemm_fusion)
     elif group_list_type == 1:
-        return torch.ops.mindspeed.npu_gmm_v2(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, gemm_fusion=gemm_fusion)
+        return _npu_gmm_v2(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, gemm_fusion=gemm_fusion)""",
"""-@impl(AS_LIBRARY, "npu_gmm.List", "PrivateUse1")
-@impl(AS_LIBRARY, "npu_gmm.Tensor", "PrivateUse1")
+# @impl(AS_LIBRARY, "npu_gmm.List", "PrivateUse1")
+# @impl(AS_LIBRARY, "npu_gmm.Tensor", "PrivateUse1")""",
"""     return _npu_gmm_common(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, group_list_type=0, gemm_fusion=gemm_fusion)
 
 
-@impl(AS_LIBRARY, "npu_gmm_v2.Tensor", "PrivateUse1")
+# @impl(AS_LIBRARY, "npu_gmm_v2.Tensor", "PrivateUse1")""","""-    if isinstance(group_list, (torch.Tensor, type(None))):
-        group_list_data_type = 1
-    else:
-        group_list_data_type = 0
-    group_args = (group_list, group_type, gemm_fusion, 0, group_list_data_type)
-    return GMMFunction.apply(original_weight, x, weight, bias, group_args)
+    group_args = (group_type, gemm_fusion, 0, 0)
+    return GMMFunction.apply(original_weight, x, weight, bias, group_list, group_args)""",""" def npu_gmm(x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False, original_weight=None):
-    return _npu_gmm_common(original_weight, x, weight, bias=bias, group_list=group_list, group_type=group_type, group_list_type=0, gemm_fusion=gemm_fusion)
+    return _npu_gmm_common(original_weight, x, weight, bias=bias, group_list=group_list.tolist(), group_type=group_type, group_list_type=0, gemm_fusion=gemm_fusion)""","""-@impl(AS_LIBRARY, "npu_gmm_v2.Tensor", "PrivateUse1")
+# @impl(AS_LIBRARY, "npu_gmm_v2.Tensor", "PrivateUse1")
 def _npu_gmm_v2(original_weight, x, weight, *, bias=None, group_list=None, group_type=0, gemm_fusion=False):
-    group_args = (group_list, group_type, gemm_fusion, 1, 1)
-    return GMMFunction.apply(original_weight, x, weight, bias, group_args)
+    group_args = (group_type, gemm_fusion, 1, 0)
+    return GMMFunction.apply(original_weight, x, weight, bias, group_list, group_args)""",
        ],
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
        "core/tensor_parallel/cross_entropy.py":["""     ) -> Tuple[torch.Tensor, torch.Tensor]:
 
         vocab_parallel_logits_fp32 = vocab_parallel_logits.float()
-        vocab_parallel_logits.untyped_storage().resize_(0)
+        del vocab_parallel_logits
+        # vocab_parallel_logits.untyped_storage().resize_(0)"""],
"core/transformer/moe/comm_utils.py":["""+import mindspore
 import torch
 import torch.distributed
 import torch.distributed as dist""","""     else:
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
     return input_, a2a_out, handle""","""-def group_mlp_forward(self, permuted_local_hidden_states, tokens_per_expert, ctx=None):
+def group_mlp_forward(self, permuted_local_hidden_states, tokens_per_expert):
     if permuted_local_hidden_states.nelement() != 0:
         w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
         w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)""","""         return grouped_mlp_with_comp_and_comm_overlap_all2all(permuted_local_hidden_states, w1, w2,
-                                                              (self.weight1, self.weight2, self.activation_func, group_list, self.layer_number),
-                                                              ctx=ctx)
+                                                              self.weight1, self.weight2, self.activation_func, group_list, self.layer_number,
+                                                              )
     else: 
         return grouped_mlp_with_comp_and_comm_overlap_allgather(permuted_local_hidden_states, w1, w2,""",],
"core/transformer/moe/grouped_mlp_with_comp_and_comm_overlap_all2all.py":[""" # See the License for the specific language governing permissions and
 # limitations under the License.
-
+import mindspore
+from mindspore.common.api import _convert_python_data
 import torch
 from einops import rearrange""","""     @staticmethod
-    def forward(ctx, inputs, weights1, weights2, args, moe_layer_ctx):
-        original_weight1, original_weight2, activation_func, group_list, layer_number = args
+    def forward(ctx, inputs, weights1, weights2, original_weight1, original_weight2, activation_func, group_list, layer_number):
         global_args = get_args()""","""         use_gmm = (inputs.nelement() != 0)
         ctx.use_gmm = use_gmm
         if use_gmm:
-            mm1_out = gmm_op(inputs, weights1, [], group_list, 0)[0]
+            mm1_out = mindspore.ops.function.math_func.gmm([inputs, ], [weights1, ], bias=[], group_list=group_list.tolist(), group_type=0)[0]
         else:
             mm1_out = torch.matmul(inputs, weights1)
         if moe_zero_memory != "disable":
-            inputs.untyped_storage().resize_(0)
-        act_out, detached_act_inputs = forward_func(activation_func, mm1_out)
+            del inputs
+        act_out, detached_act_inputs, ctx.activation_func_vjp = forward_func(activation_func, mm1_out)
 
         is_only_recompute_activation = only_recompute_activation(layer_number)
         if moe_zero_memory == "level1" and not is_only_recompute_activation:
-            mm1_out.untyped_storage().resize_(0)
+            del mm1_out
         if use_gmm:
-            mm2_out = gmm_op(act_out, weights2, [], group_list, 0)[0]
+            mm2_out = mindspore.ops.function.math_func.gmm([act_out, ], [weights2, ], bias=[], group_list=group_list.tolist(), group_type=0)[0]
         else:
             mm2_out = torch.matmul(act_out, weights2)
 
-        if moe_zero_memory == "level1" and not is_only_recompute_activation:
-            act_out.untyped_storage().resize_(0)
-            moe_layer_ctx.recompute_tensors = (inputs, mm1_out, act_out)
+        # if moe_zero_memory == "level1" and not is_only_recompute_activation:
+        #     act_out.untyped_storage().resize_(0)
+        #     moe_layer_ctx.recompute_tensors = (inputs, mm1_out, act_out)
         is_recompute_activation = moe_zero_memory == "level0" or should_recompute_activation(layer_number) or (
                     moe_zero_memory == "level1" and is_only_recompute_activation)
         if is_recompute_activation:
-            act_out.untyped_storage().resize_(0)
+            del act_out
             ctx.activation_func = activation_func
         if moe_zero_memory != "level0" and not (moe_zero_memory == "level1" and is_only_recompute_activation):
             ctx.save_for_backward(inputs, detached_act_inputs, act_out, weights1, weights2, original_weight1,""","""     def backward(ctx, *grad_outs):
-        grad_outs = grad_outs[0]
+        grad_outs = grad_outs[0][0]
         global_args = get_args()
         layer_number = ctx.layer_number""","""         ((detach_input, indices, router_topk, global_input_tokens_local_experts_indices),
-         permute2_input_detach, permute2_graph, output_splits, input_splits) = get_gemm_backward_need_tensors()
+         permute2_input_detach, permute2_graph, output_splits, input_splits, permutation_func2_vjp) = get_gemm_backward_need_tensors()
 
         # grad of mm2
         if ctx.use_gmm:
             weights2 = rearrange(weights2, 'n h f -> n f h')
-            grad_mm2_inputs = gmm_op(grad_outs, weights2, [], group_list, 0)[0]
+            grad_mm2_inputs = mindspore.ops.function.math_func.gmm([grad_outs, ], [weights2, ], bias=[], group_list=group_list.tolist(), group_type=0)[0]  
         else:
             grad_mm2_inputs = torch.matmul(grad_outs, weights2.t())
         act_graph = mm2_inputs""","""-                grad_weights2 = gmm_op(mm2_inputs.t(), grad_outs, [], group_list, 2)[0]
+                grad_weights2 = mindspore.ops.function.math_func.gmm([mm2_inputs.t(), ], [grad_outs, ], bias=[], group_list=group_list.tolist(), group_type=2)[0] ""","""-        grad_outs.untyped_storage().resize_(0)
-        mm2_inputs.untyped_storage().resize_(0)
-        act_graph.backward(grad_mm2_inputs)
-        grad_mm2_inputs.untyped_storage().resize_(0)
-        act_inputs.untyped_storage().resize_(0)
+        del grad_outs, mm2_inputs
+        act_inputs_grad = _convert_python_data(ctx.activation_func_vjp(grad_mm2_inputs)[0])
+        ctx.activation_func_vjp = None
+        del grad_mm2_inputs, act_inputs""","""             weights1 = rearrange(weights1, 'n h f -> n f h')
-            mm1_inputs_grad = gmm_op(act_inputs.grad, weights1, [], group_list, 0)[0]
+            mm1_inputs_grad = mindspore.ops.function.math_func.gmm([act_inputs_grad, ], [weights1, ], bias=[], group_list=group_list.tolist(), group_type=0)[0]""","""-        backward_func(permute2_graph, mm1_inputs_grad)
-        mm1_inputs_grad.untyped_storage().resize_(0)
+        permute2_input_detach_grad = _convert_python_data(permutation_func2_vjp(mm1_inputs_grad)[0])
+        permutation_func2_vjp = None
+        del mm1_inputs_grad""","""-            permutated_local_input_tokens.untyped_storage().resize_(0)
+            del permutated_local_input_tokens""","""         _, permute1_backward_input, bw_permute1_ep_all2all_handle = async_all_to_all(
-            permute2_input_detach.grad,
+            permute2_input_detach_grad,
             input_splits,""","""-            global_input_tokens.untyped_storage().resize_(0)
+            del global_input_tokens
 
         if ctx.use_gmm:
             if get_args().gemm_gradient_accumulation_fusion:
 
-                npu_groupmatmul_add_fp32(mm1_inputs, act_inputs.grad, group_list, original_weight1.main_grad)
+                npu_groupmatmul_add_fp32(mm1_inputs, act_inputs_grad, group_list, original_weight1.main_grad)
 
                 if hasattr(original_weight1, 'grad_added_to_main_grad'):""","""                 else:
                     mm1_weights_grad = None
             else:
-                mm1_weights_grad = gmm_op(mm1_inputs.t(), act_inputs.grad, [], group_list, 2)[0]
+                mm1_weights_grad = mindspore.ops.function.math_func.gmm([mm1_inputs.t(), ], [act_inputs_grad, ], bias=[], group_list=group_list.tolist(), group_type=2)[0]
         else:
-            mm1_weights_grad = torch.matmul(mm1_inputs.t(), act_inputs.grad)
-        act_inputs.grad.untyped_storage().resize_(0)
-        return mm1_inputs_grad, mm1_weights_grad, grad_weights2, None, None
+            mm1_weights_grad = torch.matmul(mm1_inputs.t(), act_inputs_grad)
+        ctx.saved_tensors = []
+        return None, mm1_weights_grad, grad_weights2, None, None, None, None, None
+
 
+def grouped_mlp_with_comp_and_comm_overlap_all2all(inputs, weights1, weights2, original_weight1, original_weight2, activation_func, group_list, layer_number):
+    return GroupedMlpWithCompAndCommOverlapAll2All.apply(inputs, weights1, weights2, original_weight1, original_weight2, activation_func, group_list, layer_number)
 
-def grouped_mlp_with_comp_and_comm_overlap_all2all(inputs, weights1, weights2, args, ctx):
-    return GroupedMlpWithCompAndCommOverlapAll2All.apply(inputs, weights1, weights2, args, ctx)"""],
"core/transformer/moe/moe_layer_overlap_all2all.py":["""+from torch.autograd import recompute_instance
+import mindspore
+from mindspore.common.api import _convert_python_data
 from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size
 from megatron.core import tensor_parallel, parallel_state
 from megatron.core.transformer.moe.moe_layer import MoELayer""","""-        hidden_states = hidden_states.detach()
+        hidden_states = mindspore.ops.stop_gradient(hidden_states)""","""         ctx.is_only_recompute_activation = only_recompute_activation(moe_layer.layer_number)
+        def router_func_test(hidden_states):
+            scores, ctx.indices = moe_layer.router(hidden_states)
+            return scores
""","""-        with torch.enable_grad():
-            scores, indices = moe_layer.router(hidden_states)
+        if not recompute_instance.recompute:
+            router_input = mindspore.ops.stop_gradient(hidden_states)
+            router_input.requires_grad = True
+            with torch.enable_grad():
+                scores, ctx.router_func = torch.autograd.vjp(router_func_test, router_input)
+        else:
+            scores = router_func_test(hidden_states)""","""-        scores = scores.detach()
+        scores = mindspore.ops.stop_gradient(scores)""","""-        save_tensors.append(indices)
+        save_tensors.append(ctx.indices)""","""-        (share_experts_output, dispatched_input, tokens_per_expert) = moe_layer.token_dispatcher.token_permutation(
-            hidden_states, scores, indices, ctx.shared_experts, save_tensors, shared_expert_gate, ctx
+        (share_experts_output, dispatched_input, tokens_per_expert, shared_experts_func, permutation_func1, permutation_func2) = moe_layer.token_dispatcher.token_permutation(
+            hidden_states, scores, ctx.indices, ctx.shared_experts, save_tensors, shared_expert_gate, ctx""","""         )
         if moe_experts_pipeline_degree:
             save_tensors.append(None)
             save_tensors.append(None)
             expert_output, mlp_bias = moe_experts_pipeline_forward_func(tokens_per_expert, moe_layer, dispatched_input, ctx, save_tensors)
-            output, mlp_bias = moe_layer.token_dispatcher.token_unpermutation(expert_output, mlp_bias, save_tensors)
-
+            (expert_output, mlp_bias), *_, experts_func = forward_func(experts_func_test, (dispatched_input, tokens_per_expert))
+            save_tensors.append(expert_output)
 
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
             else:""","""             (expert_output, mlp_bias), *_ = forward_func(moe_layer.experts, (dispatched_input, tokens_per_expert, ctx))
             save_tensors.append(expert_output)
 
-            output, mlp_bias = moe_layer.token_dispatcher.token_unpermutation(expert_output, mlp_bias, save_tensors)
+            (expert_output, mlp_bias), *_, experts_func = forward_func(experts_func_test, (dispatched_input, tokens_per_expert))
+            save_tensors.append(expert_output)
 
+            output, mlp_bias, unpermutation_func1, unpermutation_func2 = moe_layer.token_dispatcher.token_unpermutation(expert_output, mlp_bias, save_tensors)
+            ctx.permutation_func1 = permutation_func1
+            ctx.permutation_func2 = permutation_func2
+            ctx.shared_experts_func = shared_experts_func
+            ctx.experts_func = experts_func
+            ctx.unpermutation_func1 = unpermutation_func1
+            ctx.unpermutation_func2 = unpermutation_func2""",
"""-            output.untyped_storage().resize_(0)
-            share_experts_output.untyped_storage().resize_(0)
+            del output, share_experts_output
+            share_experts_output = 1""","""-            output_sum = output.detach()
+            output_sum = mindspore.ops.stop_gradient(output)""","""              permute2_input_detach, permute2_graph,
-             output_splits, input_splits))
+             output_splits, input_splits, ctx.permutation_func2))""","""-        unpermute2_graph.backward(args[0])
+        unpermute2_input_grad, detach_scores_grad = _convert_python_data(ctx.unpermutation_func2(args[0][0]))
+        ctx.unpermutation_func2 = None""","""         _, unpermute1_backward_input, handle = async_all_to_all(
-            unpermute2_input_detach.grad,
+            unpermute2_input_grad,""","""         elif share_experts_graph is not None:
             if backward_ag_shared_handle is not None:
                 backward_ag_shared_handle.wait()
-            share_experts_graph.backward(backward_ag_shared)
+            detach_input_grad = _convert_python_data(ctx.shared_experts_func(*backward_ag_shared)[0])
+            ctx.shared_experts_func = None
             share_experts_graph = None
             if backward_ag_shared_handle is not None:
-                backward_ag_shared.untyped_storage().resize_(0)
+                del backward_ag_shared_handle
         handle.wait()
-        unpermute2_input_detach.grad.untyped_storage().resize_(0)
-
-        backward_func(unpermute1_graph, unpermute1_backward_input)
+        del unpermute2_input_grad
 
-        unpermute1_backward_input.untyped_storage().resize_(0)
+        unpermute1_input_detach_grad = _convert_python_data(ctx.unpermutation_func1(unpermute1_backward_input)[0])
+        ctx.unpermutation_func1 = None
+        del unpermute1_backward_input
 
-        backward_func(experts_graph, unpermute1_input_detach.grad)
-        unpermute1_input_detach.grad.untyped_storage().resize_(0)
+        _convert_python_data(ctx.experts_func(unpermute1_input_detach_grad))
+        ctx.experts_func = None
 
         permute1_backward_input, bw_permute1_ep_all2all_handle = get_all2all_experts_output()
         bw_permute1_ep_all2all_handle.wait()
-        permute2_input_detach.grad.untyped_storage().resize_(0)
-        backward_func(permute1_graph, permute1_backward_input)
-        permute1_backward_input.untyped_storage().resize_(0)
+        del permute2_input_detach
+        hidden_states_grad = _convert_python_data(ctx.permutation_func1(permute1_backward_input)[0])
+        del permute1_backward_input
+        # TODO
         if l_aux_graph is not None:
             l_aux_graph.backward(l_aux_detach.grad, retain_graph=True)
         if moe_zero_memory != "disable":
             if ctx.router_topk > 1:
+                from mindspeed.core.transformer.moe.moe_utils import get_prob_backward_need_tensors
                 stream, matmul_output_grad, unpermuted_tokens = get_prob_backward_need_tensors()
                 torch.npu.current_stream().wait_stream(stream)
                 probs_grad = (matmul_output_grad * unpermuted_tokens).sum(-1).squeeze(-1)
                 route_graph.backward(probs_grad)
             ctx.router_topk = None
         else:
-            route_graph.backward(detach_scores.grad)
+            detach_input_grad1 = _convert_python_data(ctx.router_func(detach_scores_grad)[0])
+            ctx.router_func = None
         route_graph = None
-        grad_output = detach_input.grad
+        grad_output = detach_input_grad + hidden_states_grad + detach_input_grad1
+        ctx.saved_tensors = []
         return grad_output, None"""],
"core/transformer/moe/moe_utils.py":[""" import torch
 import torch_npu
+from torch.autograd import recompute_instance
+import mindspore""","""-            new_input = input_.detach()
+            new_input = mindspore.ops.stop_gradient(input_)""","""-    with torch.enable_grad():
+    if not recompute_instance.recompute:
+        with torch.enable_grad():
+            output, f_vjp = torch.autograd.vjp(func, *detach_inputs)
+    else:
         output = func(*detach_inputs)
+        f_vjp = None
 
-    return output, *detach_inputs
+    return output, *detach_inputs, f_vjp""","""-    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
+    unpermuted_tokens.index_add_(0, sorted_indices, permuted_tokens)""",
"""-    sorted_indices = torch.argsort(flatten_indices, stable=True)
+    sorted_indices = torch.argsort(flatten_indices)""",],
"core/transformer/moe/token_dispatcher.py":[""" import torch
+import mindspore""",
"""-    def alltoall_token_permutation1(hidden_states, indices):
-        tokens_per_expert = self.preprocess(indices)
+    tokens_per_expert = self.preprocess(indices)
+    def alltoall_token_permutation1(hidden_states):""",
"""-        self.hiddden_shape_before_permute = hidden_states.shape
         if self.cuda_sync_point == "before_permutation_1":
             torch.cuda.current_stream().synchronize()
+        self.hiddden_shape_before_permute = hidden_states.shape""",
"""-        return tokens_per_expert, permutated_local_input_tokens
+        return permutated_local_input_tokens
 
-    (tokens_per_expert, permutated_local_input_tokens), *_ = forward_func(alltoall_token_permutation1,
-                                                                          (hidden_states, indices))
+    permutated_local_input_tokens, *_, vjp_alltoall_token_permutation1 = forward_func(alltoall_token_permutation1,
+                                                                          hidden_states)""",
"""     if shared_experts is not None:
-        (share_experts_output, _), *_ = forward_func(shared_experts, (hidden_states, moe_ctx))
+        def shared_experts_func(hidden_states):
+            output, bias = shared_experts(hidden_states)
+            return output, bias
+        (share_experts_output, _), *_, vjp_shared_experts = forward_func(shared_experts_func, hidden_states)""",
"""     permute1_ep_all_to_all_handle.wait()
-    permutated_local_input_tokens.untyped_storage().resize_(0)
+    del permutated_local_input_tokens""",
"""-    (global_input_tokens), global_input_tokens_detach = forward_func(alltoall_token_permutation2,
+    (global_input_tokens), global_input_tokens_detach, vjp_alltoall_token_permutation2 = forward_func(alltoall_token_permutation2,
                                                                      global_input_tokens)
""","""-    global_input_tokens_detach.untyped_storage().resize_(0)
+    del global_input_tokens_detach
""","""-    return share_experts_output, global_input_tokens, tokens_per_expert
+    return share_experts_output, global_input_tokens, tokens_per_expert, vjp_shared_experts, vjp_alltoall_token_permutation1, vjp_alltoall_token_permutation2""",
"""-    hidden_states, unpermute1_input_detach = forward_func(alltoall_token_unpermutation1, hidden_states)
+    hidden_states, unpermute1_input_detach, vjp_alltoall_token_unpermutation1 = forward_func(alltoall_token_unpermutation1, hidden_states)""",
"""-    unpermute1_input_detach.untyped_storage().resize_(0)
+    del unpermute1_input_detach""",
"""     handle.wait()
-    hidden_states.untyped_storage().resize_(0)
+    del hidden_states""",
"""-    def alltoall_token_unpermutation2(permutated_local_input_tokens):
+    def alltoall_token_unpermutation2(permutated_local_input_tokens, probs):
         # Unpermutation 1: AlltoAll output to output
         if get_args().moe_zero_memory != "disable":
             output = UnpermuteWithoutActivation.apply(
                 permutated_local_input_tokens,
                 self.reversed_local_input_permutation_mapping,
-                self.probs
+                probs
             )
         else:
             output = unpermute(
                 permutated_local_input_tokens,
                 self.reversed_local_input_permutation_mapping,
-                probs=self.probs,
+                probs=probs,
                 padded_mode=self.drop_and_pad,""",
"""-    output, unpermute2_input_detach = forward_func(alltoall_token_unpermutation2, permutated_local_input_tokens)
+    output, unpermute2_input_detach, _, vjp_alltoall_token_unpermutation2 = forward_func(alltoall_token_unpermutation2, (permutated_local_input_tokens, self.probs))""",
"""-        unpermute2_input_detach.untyped_storage().resize_(0)
-    return output, None
+        del unpermute2_input_detach
+    return output, None, vjp_alltoall_token_unpermutation1, vjp_alltoall_token_unpermutation2"""],
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
         for model_module in self.model:""",
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
         pass""",
""" 
 from typing import List, Union
 from torch.utils.data import DataLoader
-from torch.utils.tensorboard import SummaryWriter
+# from torch.utils.tensorboard import SummaryWriter
 
 from mindspeed_rl.utils.tokenizer import BaseTokenizer
 from mindspeed_rl.workers.rule_reward import RuleReward"""
       
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
"""-        index = index.cpu().numpy().tolist()
+        index = index.asnumpy().tolist()""",
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
             for _, param in self.train_model.named_parameters():""",
             
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
"""     def swap_tensors_to_host(self, tensor):
         if tensor not in self.tensor_to_cpu_states_map:
             self.tensor_to_cpu_states_map[tensor] = torch.empty_like(tensor, device='cpu')
-        if tensor.storage().size() != 0:
+        # if tensor.storage().size() != 0:
+        if tensor.size() != 0:
             cpu_state = self.tensor_to_cpu_states_map[tensor]
             cpu_state.copy_(tensor, non_blocking=False)
-            tensor.storage().resize_(0)
+            # tensor.storage().resize_(0)
+            # tensor.assign_value(torch.tensor([]))
 
     def swap_tensors_to_device(self, tensor):
-        if tensor.storage().size() == 0:
+        # if tensor.storage().size() == 0:
+        if tensor.size() == 0:
             cpu_state = self.tensor_to_cpu_states_map[tensor]
-            tensor.storage().resize_(cpu_state.storage().size())
+            # tensor.storage().resize_(cpu_state.storage().size())
+            # tensor.assign_value(torch.empty(cpu_state.size()))
             tensor.copy_(cpu_state, non_blocking=False)"""
        
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
+            memory_buffer.data = memory_buffer.data.cuda()""",
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
         ray_objs = []""",
         
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
+                ms_sched_host=self.ms_sched_host,
+                ms_sched_port=self.ms_sched_port,
+            )
+        scheduler_actor.get_status.remote()
+
         actor_handle = self.create_actor_handlers(
-            ActorHandlerParams(placement_group[0], world_size, 0, 0, None, None))
+            ActorHandlerParams(placement_group[0], world_size, 0, 0, self.ms_sched_host, self.ms_sched_port))
         self.actor_handlers.append(actor_handle)
         return actor_handle"""
        
        ],
        "mindspeed_rl/workers/scheduler/scheduler.py": [
"""
import ray
 
import mindspore as ms
from mindspore import mint

from .worker import Worker
 
 
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
        ],
        "mindspeed/core/transformer/moe/token_dispatcher.py": [
"""     # Permutation 2: AlltoAll output to expert input if num_local_experts > 1
     if self.num_local_experts > 1:
         if not self.drop_and_pad:
-            torch.cuda.current_stream().wait_stream(self.comm_stream)
+            # mindspore.runtime.current_stream().wait_stream(self.comm_stream)
             global_input_tokens, self.reversed_global_input_permutation_mapping = permute(
                 global_input_tokens, self.global_input_tokens_local_experts_indices
             )"""
        ],
        "mindspeed_llm/tasks/megatron_adaptor.py": [
"""     def patch_core_transformers(self):
         import megatron.core
         from mindspeed.core.transformer.transformer_config import transformer_config_post_init_wrapper
+        from mindspeed.core.transformer.moe.grouped_gemm_util import Ops, grouped_gemm_is_available, \\
+            get_device_capability, assert_grouped_gemm_is_available"""
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
        "vllm/model_executor/layers/linear.py": [
"""     def apply(self,
               layer: torch.nn.Module,
               x: torch.Tensor,
               bias: Optional[torch.Tensor] = None) -> torch.Tensor:
+        x.data_sync(True)
+        layer.weight.data_sync(True)
 
         return F.linear(x, layer.weight, bias)"""
        ],
        
    },
    "vllm-ascend":{
        "vllm_ascend/attention.py": [
"""if TYPE_CHECKING:
-    from vllm_ascend.worker.model_runner import ModelInputForNPUBuilder
+    from vllm_ascend.model_runner import ModelInputForNPUBuilder""",
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
"""         num_kv_heads: int,
         head_size: int,
     ) -> Tuple[int, ...]:
-        return (2, num_blocks, block_size, num_kv_heads, head_size)
+        return (2, num_blocks, block_size, num_kv_heads * head_size)
 
     @staticmethod
     def swap_blocks(""",
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
+            attn_output = torch.transpose(attn_output_t, 0, 1)""",
"""                                       self.v_head_dim,
                                       dtype=query.dtype,
                                       device=query.device)
+            attn_output = torch.ones_like(attn_output, dtype=query.dtype)
             if (attn_metadata.block_tables is None
                     or attn_metadata.block_tables.numel() == 0):
                 assert attn_metadata.attn_mask is not None""",
"""                     np.array(attn_metadata.prefill_metadata.seq_lens).astype(
                         np.int32))
                 torch_npu._npu_flash_attention(
-                    query=query,
-                    key=key,
-                    value=value,
+                    query=query.contiguous(),
+                    key=key.contiguous(),
+                    value=value.contiguous(),
                     mask=mask,
-                    seq_len=self.seq_lens_tensor_cpu,
+                    seq_len=self.seq_lens_tensor_cpu.contiguous(),
                     scale_value=self.scale,
                     num_heads=self.num_heads,
                     num_kv_heads=self.num_heads,""",
"""                     np.int32))
             block_tables = attn_metadata.decode_metadata.block_tables
             torch_npu._npu_paged_attention_mla(
-                query=query,
-                key_cache=key_cache,
+                query=query.contiguous(),
+                key_cache=key_cache.contiguous(),
                 num_kv_heads=self.num_kv_heads,
                 num_heads=self.num_heads,
                 scale_value=self.scale,
                 block_table=block_tables,
-                context_lens=self.seq_lens_tensor_cpu,
+                context_lens=self.seq_lens_tensor_cpu.contiguous(),
                 mla_vheadsize=self.kv_lora_rank,
                 out=attn_output)
             attn_output_t = torch.transpose(attn_output, 0, 1)"""

],
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
"""-        torch_npu._npu_rotary_embedding(
-            positions,
-            query,
-            key,
-            self.head_size,
-            self.cos_sin_cache,
-            self.is_neox_style,
-        )
+        #torch_npu._npu_rotary_embedding(
+        #    positions,
+        #    query,
+        #    key,
+        #    self.head_size,
+        #    self.cos_sin_cache,
+        #    self.is_neox_style,
+        #)""",
"""-RotaryEmbedding.forward_oot = rope_forward_oot
+# RotaryEmbedding.forward_oot = rope_forward_oot""",
"""-DeepseekScalingRotaryEmbedding.forward = rope_deepseek_forward_oot
+# DeepseekScalingRotaryEmbedding.forward = rope_deepseek_forward_oot"""
        ],

        "vllm_ascend/worker/worker.py": [
"""                              cache_block_size)
         num_npu_blocks = max(num_npu_blocks, 0)
         num_cpu_blocks = max(num_cpu_blocks, 0)
+        num_npu_blocks = 65536
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
