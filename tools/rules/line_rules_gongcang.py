LINE_RULES = {
    "MindSpeed-LLM": {
    "convert_ckpt.py":["""if __name__ == '__main__':
+    import mindspore as ms
+    ms.set_context(device_target = "CPU", pynative_synchronize=True)
+    import torch
+    torch.configs.set_pyboost(False)"""],
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
+                load_dir, trust_remote_code=trust_remote_code, local_files_only=True, low_cpu_mem_usage=False
             )]
         if hasattr(self.args, "torch_dtype") and self.args.torch_dtype in ["float16", "bfloat16"]:"""],
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
    }
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