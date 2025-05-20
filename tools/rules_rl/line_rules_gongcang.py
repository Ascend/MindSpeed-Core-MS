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
