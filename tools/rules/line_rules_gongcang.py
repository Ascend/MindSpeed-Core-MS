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
+                data = kwargs.get('data')
+                if data.dtype != set_tensor.weight.dtype:
+                   data = data.to(dtype = set_tensor.weight.dtype)
+                set_tensor.weight.data = data


+                return set_tensor.weight.data
             return func""",
             """         def _func_generator_set_bias(value):
             def func(self, **kwargs):
-                return _get_dst_obj(self, value, **kwargs).bias.data.copy_(kwargs.get('data'))
+                set_tensor = _get_dst_obj(self, value, **kwargs)
+                data = kwargs.get('data')
+                if data.dtype != set_tensor.weight.dtype:
+                   data = data.to(dtype = set_tensor.weight.dtype)
+                set_tensor.bias.data = data
+                return set_tensor.bias.data
             return func""",
     """             self.module = [AutoModelForCausalLM.from_pretrained(
-                load_dir, device_map=device_map, trust_remote_code=trust_remote_code, local_files_only=True
+                load_dir, trust_remote_code=trust_remote_code, local_files_only=True, low_cpu_mem_usage=False
             )]
"""],
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
        "core/pipeline_parallel/schedules.py":[
""" from typing import Callable, Iterator, List, Optional, Union
 
 import torch
-from torch.autograd.variable import Variable
+from mindspore.ops import composite as C
+from mindspore.common.api import _pynative_executor
 
 from megatron.core import parallel_state
 from megatron.core.enums import ModelType""","""     set_input_tensor(input_tensor)
 
+    if not parallel_state.is_pipeline_first_stage() and input_tensor is not None:
+        input_tensor[0].retain_grad()
+
+    # run forward
+    num_tokens = torch.tensor(0, dtype=torch.int)
+    if input_tensor[0] is None:
+        input_tensor[0] = num_tokens
+
     if config.enable_autocast:
         context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
     else:
         context_manager = contextlib.nullcontext()
+    _pynative_executor.set_grad_flag(True)
+    _pynative_executor.new_graph(forward_step_func, input_tensor[0])
     with context_manager:
         if checkpoint_activations_microbatch is None:""","""             forward_data_store.append(data)
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
    },
    "mindspeed":{
    "core/auto_parallel/auto_parallel_apply.py":[""" from mindspeed.core.auto_parallel import set_kv_store
-from mindspeed.core.auto_parallel.auto_parallel_optimizer import SearchByGreyBox
 from mindspeed.core.auto_parallel.auto_parallel_memory import MemoryCostModel
"""],
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