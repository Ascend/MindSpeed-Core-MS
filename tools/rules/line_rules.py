LINE_RULES_MindSpeed_LLM = {
"mindspeed_llm/core/datasets/blended_megatron_dataset_builder.py": [""" 
 import torch
 
+from mindspore.communication import get_local_rank
+
 from megatron.training import get_args
 from megatron.core.datasets.blended_dataset import BlendedDataset
 from megatron.core.datasets.megatron_dataset import MegatronDataset, LowLevelDataset, MockDataset""","""     if share_save:
         return rank == 0
     gpus_per_node = torch.cuda.device_count()
-    current_rank = torch.cuda.current_device()
+    current_rank = get_local_rank()
     if args.tensor_model_parallel_size > gpus_per_node:
         return mpu.get_tensor_model_parallel_rank() == 0
     return mpu.get_tensor_model_parallel_rank() == 0 and current_rank % gpus_per_node == 0
"""
],
"mindspeed_llm/core/models/common/embeddings/rotary_pos_embedding.py": ["""     high_freq_wavelen = original_length / args.high_freq_factor
     new_freqs = []
     for freq in freqs:
-        wavelen = 2 * math.pi / freq
+        wavelen = 2 / freq * math.pi
         if wavelen < high_freq_wavelen:
             new_freqs.append(freq)
         elif wavelen > low_freq_wavelen:""","""     )
 
     inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
-        device=freqs.device, dtype=torch.float32
+        dtype=torch.float32
     )
 
     inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask""","""     if args.use_fused_rotary_pos_emb:
         cos = torch.cos(freqs)
         sin = torch.sin(freqs)
-        return torch_npu.npu_rotary_mul(t, cos, sin).to(t.dtype)
+        mode = 1 if rotary_interleaved else 0
+        return torch_npu.npu_rotary_position_embedding(t, cos.to(t.dtype), sin.to(t.dtype), mode=mode).to(t.dtype)
+        # return torch_npu.npu_rotary_mul(t, cos, sin).to(t.dtype)
 
     rot_dim = freqs.shape[-1]
     t, t_pass = t[..., :rot_dim], t[..., rot_dim:]""","""     sin_ = (torch.sin(freqs) * _mscale).to(t.dtype)
     
     if args.use_fused_rotary_pos_emb:
-        t = torch_npu.npu_rotary_mul(t, cos_, sin_).to(t.dtype)
+        mode = 1 if rotary_interleaved else 0
+        t = torch_npu.npu_rotary_position_embedding(t, cos_.to(t.dtype), sin_.to(t.dtype), mode=mode).to(t.dtype)
+        # t = torch_npu.npu_rotary_mul(t, cos_, sin_).to(t.dtype)
     else:
         t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
-    
     return torch.cat((t, t_pass), dim=-1)
 
 
"""
],
"mindspeed_llm/core/transformer/moe/router.py": ["""     
     scores = F.softmax(logits, dim=1)
     group_scores = (
-        scores.view(args.micro_batch_size * seq_length, args.expert_model_parallel_size, -1).max(dim=-1).values
+        scores.view(args.micro_batch_size * seq_length, args.expert_model_parallel_size, -1).max(dim=-1)[0]
     )  # [n, EP]
 
     group_idx = torch.topk(group_scores, k=args.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
"""
],
"mindspeed_llm/legacy/data/data_samplers.py": ["""         )
     else:
         collator = None
-
     # Torch dataloader.
     return torch.utils.data.DataLoader(dataset,
                                        batch_sampler=batch_sampler,
                                        num_workers=args.num_workers,
-                                       generator=torch.Generator().manual_seed(args.seed),
                                        collate_fn=collator,
-                                       pin_memory=True)
+                                       pin_memory=False)
"""
],
"mindspeed_llm/tasks/checkpoint/loader_hf.py": [""" 
     def queue_put(name, msg):
         logger.info(f"sending {name}")
+        for k, v in msg.items():
+            if isinstance(v, torch.Tensor):
+                msg[k] = v.asnumpy()
         msg["name"] = name
         queue.put(msg)
 
"""
],
"mindspeed_llm/tasks/checkpoint/models.py": ["""     return f"shape: {shape} mean_val: {mean_val} min_val: {min_val} max_val: {max_val}"
 
 
+class FakesubModule():
+    def __init__(self, module_name, weight_dict):
+        self.weight = weight_dict.get(f"{module_name}.weight", )
+        self.bias = weight_dict.get(f"{module_name}.bias")
+
+
+class FakeModule():
+    def __init__(self, weight_dicts, module_mapping):
+        self.module_keys = set(map(lambda x: ".".join(x.split(".")[:-1]),weight_dicts.keys()))
+        for module_name in self.module_keys:
+            weight_dict = dict(filter(lambda x : module_name in x[0], weight_dicts.items()))
+            setattr(self, module_name, self.assemodule(module_name, weight_dict))
+
+    def assemodule(self, module_name, weight_dict):
+        return FakesubModule(module_name, weight_dict)
+    
+    def to(self, model_type):
+        return self
+
+
 class ModelBase(abc.ABC):
     def __init__(self, args_cmd=None):
         self.args_cmd = args_cmd""","""         super(HuggingfaceModel, self).__init__(args_cmd)
         self.initialize_args()
         self.layers_self_attention_linear_qkv_caches = {"layer_idx": -1, "weight": None, "bias": None}
+        self.__register_functions()
 
     def initialize_args(self):
         # Read huggingface args.""","""         self.args.add_dense_bias = self.args_cmd.add_dense_bias
         self.args.post_norm = self.args_cmd.post_norm
 
+    def __register_functions(self):
+        self.get_module_mapping()
+
+        def _get_obj(self, value, **kwargs):
+            self.update_kwargs_idx(**kwargs)
+            obj = self.get_model_item(**kwargs)
+            if "layer_idx" in value:
+                attr_idx = self.kwargs_idx["layer_idx"]
+                value = value.replace("[layer_idx]", f".{attr_idx}")
+            return getattr(obj, value, None)
+
+        def _func_generator_get_module(value):
+            def func(self, **kwargs):
+                return _get_obj(self, value, **kwargs)
+            return func
+
+        def _func_generator_get_weight(value):
+            def func(self, **kwargs):
+                return _get_obj(self, value, **kwargs).weight.data
+            return func
+
+        def _func_generator_get_bias(value):
+            def func(self, **kwargs):
+                return _get_obj(self, value, **kwargs).bias.data
+            return func
+
+        def _func_generator_set_weight(value):
+            def func(self, **kwargs):
+                return _get_obj(self, value, **kwargs).weight.data.copy_(kwargs.get('data'))
+            return func
+
+        def _func_generator_set_module(value):
+            def func(self, **kwargs):
+                return _get_obj(self, value, **kwargs).data.copy_(kwargs.get('data'))
+            return func
+
+        def _func_generator_set_bias(value):
+            def func(self, **kwargs):
+                return _get_obj(self, value, **kwargs).bias.data.copy_(kwargs.get('data'))
+            return func
+
+        def _func_generator_has_module(value):
+            def func(self, **kwargs):
+                # print("self", self)
+                obj = _get_obj(self, value, **kwargs)
+                return True if obj else False
+            return func
+        
+        def _func_generator_has_bias(value):
+            def func(self, **kwargs):
+                bias = getattr(_get_obj(self, value, **kwargs), 'bias', None)
+                return bias is not None
+            return func
+
+        if self.module_mapping:
+            for key, value in self.module_mapping.items():
+                setattr(self, "get_" + key + "_module", _func_generator_get_module(value).__get__(self, ModelBase))
+                setattr(self, "set_" + key + "_module", _func_generator_set_module(value).__get__(self, ModelBase))
+                setattr(self, "get_" + key + "_weight", _func_generator_get_weight(value).__get__(self, ModelBase))
+                setattr(self, "get_" + key + "_bias", _func_generator_get_bias(value).__get__(self, ModelBase))
+                setattr(self, "set_" + key + "_weight", _func_generator_set_weight(value).__get__(self, ModelBase))
+                setattr(self, "set_" + key + "_bias", _func_generator_set_bias(value).__get__(self, ModelBase))
+                setattr(self, "has_" + key + "_module", _func_generator_has_module(value).__get__(self, ModelBase))
+                setattr(self, "has_" + key + "_bias", _func_generator_has_bias(value).__get__(self, ModelBase))
+
     def get_modules_from_pretrained(self, device_map="cpu", trust_remote_code=True):
         # Load Huggingface model.
         if self.args_cmd.save_model_type == "hf":
             load_dir = self.args_cmd.save_dir
         else:
             load_dir = self.args_cmd.load_dir
-        self.module = [AutoModelForCausalLM.from_pretrained(load_dir, device_map=device_map, trust_remote_code=trust_remote_code, local_files_only=True)]
-        if hasattr(self.args, "torch_dtype") and self.args.torch_dtype in ["float16", "bfloat16"]:
+        import glob
+        from torch.serialization import safe_load_file
+        hf_model_dict = {}
+        checkpoint_files_path = load_dir + "*.safetensors"
+        checkpoint_files = glob.glob(checkpoint_files_path)
+        for checkpoint_file in checkpoint_files:
+            checkpoint = safe_load_file(checkpoint_file)
+            hf_model_dict.update(checkpoint)
+        self.module = [FakeModule(hf_model_dict, self.module_mapping)]
+        if hasattr(self.args, "torch_dtype") and self.args.torch_dtype in ["float16", "bfloat16"]: #不一样
             self.module[0] = self.module[0].to(eval(f'torch.{self.args.torch_dtype}'))
 
     def get_module_mapping(self):
"""
],
"mindspeed_llm/tasks/checkpoint/saver.py": [""" import os
 import sys
 import logging as logger
+import numpy as np
 import torch
 from megatron.training.checkpointing import save_checkpoint
 from megatron.core import mpu""",""" 
     def queue_get(name=None):
         val = queue.get()
+        if isinstance(val, dict):
+            for k, v in val.items():
+                if isinstance(v, np.ndarray):
+                    val[k] = torch.Tensor(v)
         if val == "exit":
             logger.error("Loader exited, exiting saver")
             exit(1)
"""
],
"mindspeed_llm/tasks/evaluation/eval_impl/ceval_exam.py": [""" 
         for file in os.listdir(self.test_dir):
             file_path = os.path.join(self.test_dir, file)
-            
-            if os.path.exists(file_path):
-                data_df = pd.read_csv(file_path)
-            else:
-                raise FileNotFoundError(f"Error: {file_path} does not exist.")
-            
+            data_df = pd.read_csv(file_path)
             subject_name = re.sub(r'(?:_test|_val|_dev)?\\.\\w+$', "", file)
             subject_result = {}
             sample_n += len(data_df)
"""
],
"mindspeed_llm/tasks/evaluation/eval_impl/gsm8k_eval.py": [""" 
         for file in os.listdir(self.test_dir):
             file_path = os.path.join(self.test_dir, file)
-            
-            if not os.path.exists(file_path):
-                raise FileNotFoundError(f"Error: {file_path} does not exist.")
-            
             with open(file_path, encoding='utf-8') as f:
                 gsm8k_list = []
                 for line in f.readlines():
"""
],
"mindspeed_llm/tasks/evaluation/eval_impl/human_eval.py": [""" 
         for file in os.listdir(test_dir):
             test_code_path = os.path.join(self.test_dir, file)
-            
-            if not os.path.exists(test_code_path):
-                raise FileNotFoundError(f"Error: {test_code_path} does not exist.")
-
             with open(test_code_path, 'r') as fp:
                 for line in fp:
                     if any(not x.isspace() for x in line):
"""
],
"mindspeed_llm/tasks/evaluation/eval_impl/mmlu_eval.py": [""" 
         for file in os.listdir(self.test_dir):
             file_path = os.path.join(self.test_dir, file)
-            
-            if os.path.exists(file_path):
-                data_df = pd.read_csv(file_path, names=['question', 'A', 'B', 'C', 'D', 'answer'])
-            else:
-                raise FileNotFoundError(f"Error: {file_path} does not exist.")
-            
+            data_df = pd.read_csv(file_path, names=['question', 'A', 'B', 'C', 'D', 'answer'])
             subject_name = re.sub(r'(?:_test|_val|_dev)?\\.\\w+$', "", file)  # 文件命名规则类似  {subject}_test.csv
             subject = subject_name.replace("_", " ")
             subject_result = {}
"""
],
"mindspeed_llm/tasks/finetune/lora/cc_lora_forward.py": [""" 
 
 def _reduce_async(input_):
-    \"\"\"ALL-Reduce the input tensor across model parallel group async.\"\"\"
+    if get_tensor_model_parallel_world_size() == 1:
+        return input_, None
+
+    # All-reduce.
     handle = torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group(), async_op=True)
     return input_, handle
 
 
-def lora_backward(grad_output_, input_b, grad_ax, input_, scaling):
-    grad_weight_b = grad_output_.t().matmul(input_b)
-    grad_weight_a = grad_ax.t().matmul(input_) * scaling
-    return grad_weight_a, grad_weight_b
-
-
 class _FusedColumnSeqParallelLoRAFunction(torch.autograd.Function):
     \"\"\"Accelerate ColumnParallelLoRA with TP and SP.\"\"\"
 ""","""     def forward(ctx, input_, weight, weight_a, weight_b, scaling):
         \"\"\"
         1. gx = gather(x)
-              a_scale = a * scaling
-              ax = a_scale * x
-              W_combine = w + b @ a_scale
-        2. output = W_combine * gx
+            a_scale = a * scaling
+            ax = a_scale * x
+        2. gax = gather(ax)
+            output = w * gx
+        3. bx = b * gax
+        4. output += bx
         \"\"\"
         total_input, handle = _gather_along_first_dim_async(input_)
         weight_a_scale = weight_a * scaling
         ax = torch.matmul(input_, weight_a_scale.t())
-        weight_combine = weight + weight_b @ weight_a_scale
         handle.wait()
-        output = torch.matmul(total_input, weight_combine.t())
-        ctx.save_for_backward(input_, ax, weight, weight_a_scale, weight_b)
+        total_ax, handle = _gather_along_first_dim_async(ax)
+        output = torch.matmul(total_input, weight.t())
+        handle.wait()
+        bx = torch.matmul(total_ax, weight_b.t())
+        output += bx
+        ctx.save_for_backward(input_, ax, weight, weight_b)
         ctx.scaling = scaling
         return output
 
     @staticmethod
     def backward(ctx, grad_output):
-        input_, input_b, weight, weight_a_scale, weight_b = ctx.saved_tensors
+        input_, input_b, weight, weight_b = ctx.saved_tensors
         is_dense = len(grad_output.shape) == 3
         total_a, handle = _gather_along_first_dim_async(input_b)
         if is_dense:
-            grad_output_ = grad_output.reshape(-1, grad_output.shape[-1])
+            grad_output_ = grad_output.view(grad_output.shape[0] * grad_output.shape[1],
+                                            grad_output.shape[2])
         else:
             grad_output_ = grad_output
         grad_gax = grad_output_.matmul(weight_b)
-        delta_weight = weight_b @ weight_a_scale
         handle.wait()
         grad_ax, handle = _reduce_scatter_along_first_dim_async(grad_gax)
-        grad_input = grad_output.matmul(weight + delta_weight)
+        grad_input = grad_output.matmul(weight)
         handle.wait()
         grad_sub_input, handle = _reduce_scatter_along_first_dim_async(grad_input)
         if is_dense:
-            input_ = input_.reshape(-1, input_.shape[-1])
-            total_a = total_a.reshape(-1, total_a.shape[-1])
-        grad_weight_a, grad_weight_b = lora_backward(grad_output_, total_a, grad_ax, input_, ctx.scaling)
+            x_ = input_.view(input_.shape[0] * input_.shape[1], input_.shape[2])
+            total_a = total_a.view(total_a.shape[0] * total_a.shape[1], total_a.shape[2])
+        else:
+            x_ = input_
+        grad_weight_b = grad_output_.t().matmul(total_a)
+        grad_weight_a = grad_ax.t().matmul(x_) * ctx.scaling
         handle.wait()
         return grad_sub_input, None, grad_weight_a, grad_weight_b, None
 ""","""         1. a_scale = a * scaling
         2. ax = a_scale * x
         3. rax = reduce_scatter(ax)
-              W_combine = w + b @ a_scale
-        4. output = reduce_scatter(W_combine * x)
+              output = w * x
+        4. output = reduce_scatter(output)
+              bx = b * rax
+        5. output += bx
         \"\"\"
-
         weight_a_scale = weight_a * scaling
         ax = torch.matmul(input_, weight_a_scale.t())
         rax, handle = _reduce_scatter_along_first_dim_async(ax)
-        weight_combine = weight + weight_b @ weight_a_scale
-        if input_.dim() == 3:
-            reshape = True
-            seq_len, batch, d = input_.shape[:]
-            input_ = input_.view(seq_len * batch, d)
+        output = torch.matmul(input_, weight.t())
+        handle.wait()
+        output_parallel, handle = _reduce_scatter_along_first_dim_async(output)
+        bx = torch.matmul(rax, weight_b.t())
         group = get_tensor_model_parallel_group()
         rank = torch.distributed.get_rank(group)
         if torch.__version__ > "2.0":""","""         ctx.hcomm_info = hcomm_info
         ctx.world_size = world_size
         handle.wait()
-        output_parallel = torch_npu.npu_mm_reduce_scatter_base(
-            input_, weight_combine.t(), hcomm_info, world_size, reduce_op="sum", bias=None
-        )
-        ctx.save_for_backward(input_, rax, weight, weight_a_scale, weight_b)
+        output_parallel += bx
+        ctx.save_for_backward(input_, rax, weight, weight_b)
         ctx.scaling = scaling
-        return output_parallel.view(seq_len // world_size, batch, -1) if reshape else output_parallel
+        return output_parallel
 
     @staticmethod
     def backward(ctx, grad_output):
-        \"\"\"
-        grad_weight_b = grad_out * scaling * reduce_scatter(a * x)
-                      = grad_out * (scaling * reduce_scatter(a * x))
-                      = grad_out * input_b
-        grad_weight_a = gather(grad_out * scaling * b) * x
-                      = gather(grad_out) * b * x * scaling
-        grad_input = gather(grad_out) * w_combine
-        \"\"\"
-
-        input_, input_b, weight, weight_a_scale, weight_b = ctx.saved_tensors
+        input_, input_b, weight, weight_b = ctx.saved_tensors
         is_dense = len(grad_output.shape) == 3
         if is_dense:
-            grad_output_ = grad_output.reshape(-1, grad_output.shape[-1])
-            input_b = input_b.reshape(-1, input_b.shape[-1])
-            input_ = input_.reshape(-1, input_.shape[-1])
+            grad_output_ = grad_output.reshape(
+                grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
+            )
+            input_b = input_b.view(input_b.shape[0] * input_b.shape[1], input_b.shape[2])
+            x = input_.view(input_.shape[0] * input_.shape[1], input_.shape[2])
         else:
             grad_output_ = grad_output
+            x = input_
         grad_input, grad_total_output = torch_npu.npu_all_gather_base_mm(
             grad_output_, weight, ctx.hcomm_info, ctx.world_size, bias=None, gather_index=0, gather_output=True
         )
+        grad_weight_b = grad_output_.t().matmul(input_b)
         grad_ax = grad_total_output.matmul(weight_b)
-        grad_weight_a, grad_weight_b = lora_backward(grad_output_, input_b, grad_ax, input_, ctx.scaling)
-        grad_input += grad_ax.matmul(weight_a_scale)
-        grad_input = grad_input.view(-1, grad_output.shape[1], input_.shape[-1])
+        grad_weight_a = grad_ax.t().matmul(x) * ctx.scaling
+        grad_input = grad_input.view_as(input_)
         return grad_input, None, grad_weight_a, grad_weight_b, None
 
 ""","""         1. a_scale = a * scaling
         2. ax = a_scale * x
         3. rax = _reduce_async(ax)
-              output = w * x
+            output = w * x
         4. output = _reduce_async(output)
-              bx = b * rax
+            bx = b * rax
         5. output += bx
+        reduce_from_tensor_model_parallel_region
         \"\"\"
         weight_a_scale = weight_a * scaling
         ax = torch.matmul(input_, weight_a_scale.t())
         rax, handle = _reduce_async(ax)
         output = torch.matmul(input_, weight.t())
-        handle.wait()
+        if handle is not None:
+            handle.wait()
         output_parallel, handle = _reduce_async(output)
         bx = torch.matmul(rax, weight_b.t())
-        handle.wait()
+        if handle is not None:
+            handle.wait()
         output_parallel += bx
-        ctx.save_for_backward(input_, rax, weight, weight_a_scale, weight_b)
+        ctx.save_for_backward(input_, rax, weight, weight_b)
         ctx.scaling = scaling
         return output_parallel
 
     @staticmethod
     def backward(ctx, grad_output):
-        input_, input_b, weight, weight_a_scale, weight_b = ctx.saved_tensors
-        if grad_output.dim() == 3:
-            grad_output_ = grad_output.reshape(-1, grad_output.shape[-1])
-            input_b = input_b.reshape(-1, input_b.shape[-1])
-            input_ = input_.reshape(-1, input_.shape[-1])
+        input_, input_b, weight, weight_b = ctx.saved_tensors
+        is_dense = len(grad_output.shape) == 3
+        grad_output_ = grad_output.reshape(
+            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
+        )
+        if is_dense:
+            input_b = input_b.view(input_b.shape[0] * input_b.shape[1], input_b.shape[2])
+            x = input_.view(input_.shape[0] * input_.shape[1], input_.shape[2])
         else:
-            grad_output_ = grad_output
+            x = input_
+        grad_weight_b = grad_output_.t().matmul(input_b)
         grad_ax = grad_output_.matmul(weight_b)
-        grad_weight_a, grad_weight_b = lora_backward(grad_output_, input_b, grad_ax, input_, ctx.scaling)
+        grad_weight_a = grad_ax.t().matmul(x) * ctx.scaling
         grad_input = grad_output.matmul(weight)
-        grad_input += grad_ax.matmul(weight_a_scale).view_as(grad_input)
         return grad_input, None, grad_weight_a, grad_weight_b, None
 
 ""","""         ax = torch.matmul(input_, weight_a_scale.t())
         bx = torch.matmul(ax, weight_b.t())
         output += bx
-        ctx.save_for_backward(input_, ax, weight, weight_a_scale, weight_b)
+        ctx.save_for_backward(input_, ax, weight, weight_b)
         ctx.scaling = scaling
         return output
 
     @staticmethod
     def backward(ctx, grad_output):
-        input_, input_b, weight, weight_a_scale, weight_b = ctx.saved_tensors
-        if grad_output.dim() == 3:
-            grad_output_ = grad_output.reshape(-1, grad_output.shape[-1])
-            input_b = input_b.reshape(-1, input_b.shape[-1])
-            input_ = input_.reshape(-1, input_.shape[-1])
+        input_, input_b, weight, weight_b = ctx.saved_tensors
+        is_dense = len(grad_output.shape) == 3
+        if is_dense:
+            grad_output_ = grad_output.reshape(
+                grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
+            )
+            input_b = input_b.view(input_b.shape[0] * input_b.shape[1], input_b.shape[2])
+            x = input_.view(input_.shape[0] * input_.shape[1], input_.shape[2])
         else:
             grad_output_ = grad_output
+            x = input_
         grad_ax = grad_output_.matmul(weight_b)
         grad_ax, handle = _reduce_async(grad_ax)
-        grad_input = grad_output.matmul(weight + weight_b @ weight_a_scale)
-        handle.wait()
+        grad_input = grad_output.matmul(weight)
+        grad_weight_b = grad_output_.t().matmul(input_b)
+        if handle is not None:
+            handle.wait()
         grad_input, handle = _reduce_async(grad_input)
-        grad_weight_a, grad_weight_b = lora_backward(grad_output_, input_b, grad_ax, input_, ctx.scaling)
-        handle.wait()
-        return grad_input, None, grad_weight_a, grad_weight_b, None
-
-
-class _FusedBaseParallelLoRAFunction(torch.autograd.Function):
-    \"\"\"Accelerate ParallelLoRA.\"\"\"
-
-    @staticmethod
-    def forward(ctx, input_, weight, weight_a, weight_b, scaling):
-        if input_.dim() == 3:
-            seq_len, batch, d = input_.shape[:]
-            seq_size = seq_len * batch
-        else:
-            seq_size, d = input_.shape[:]
-        weight_a_scale = weight_a * scaling
-        ax = torch.matmul(input_, weight_a_scale.t())
-        if seq_size < d:
-            ctx.combine = False
-            output = torch.matmul(input_, weight.t())
-            bx = torch.matmul(ax, weight_b.t())
-            output += bx
-        else:
-            ctx.combine = True
-            weight_combine = weight + weight_b @ weight_a_scale
-            output = torch.matmul(input_, weight_combine.t())
-        ctx.save_for_backward(input_, ax, weight_a_scale, weight_b, weight)
-        ctx.scaling = scaling
-        return output
-
-    @staticmethod
-    def backward(ctx, grad_output):
-        input_, input_b, weight_a_scale, weight_b, weight = ctx.saved_tensors
-        if grad_output.dim() == 3:
-            grad_output_ = grad_output.reshape(-1, grad_output.shape[-1])
-            input_b = input_b.reshape(-1, input_b.shape[-1])
-            input_ = input_.reshape(-1, input_.shape[-1])
-        else:
-            grad_output_ = grad_output
-        grad_ax = grad_output_.matmul(weight_b)
-        grad_weight_a, grad_weight_b = lora_backward(grad_output_, input_b, grad_ax, input_, ctx.scaling)
-        if ctx.combine:
-            grad_input = grad_output.matmul((weight + weight_b @ weight_a_scale))
-        else:
-            grad_input = grad_output.matmul(weight)
-            grad_input += grad_ax.matmul(weight_a_scale).view_as(grad_input)
+        grad_weight_a = grad_ax.t().matmul(x) * ctx.scaling
+        if handle is not None:
+            handle.wait()
         return grad_input, None, grad_weight_a, grad_weight_b, None
 
 ""","""     Forward of ColumnParallelLinear with CCLora
     \"\"\"
     weight = base_layer.weight
-    bias = base_layer.bias if not base_layer.skip_bias_add else None
-    lora_params = [input_, weight, weight_a, weight_b, scaling]
-    if base_layer.explicit_expert_comm or get_tensor_model_parallel_world_size() == 1:
-        output_parallel = _FusedBaseParallelLoRAFunction.apply(*lora_params)
-    elif base_layer.sequence_parallel:
-        output_parallel = _FusedColumnSeqParallelLoRAFunction.apply(*lora_params)
+    skip_bias_add, bias = base_layer.skip_bias_add, base_layer.bias
+    sequence_parallel = base_layer.sequence_parallel
+    bias = bias if not skip_bias_add else None
+    if sequence_parallel:
+        output_parallel = _FusedColumnSeqParallelLoRAFunction.apply(input_, weight, weight_a, weight_b, scaling)
     else:
-        output_parallel = _FusedColumnNoSeqParallelLoRAFunction.apply(*lora_params)
+        output_parallel = _FusedColumnNoSeqParallelLoRAFunction.apply(input_, weight, weight_a, weight_b, scaling)
     if bias is not None:
         output_parallel = output_parallel + bias
-    output_bias = base_layer.bias if base_layer.skip_bias_add else None
-    return output_parallel, output_bias
+    output = output_parallel
+    output_bias = bias if skip_bias_add else None
+    return output, output_bias
 
 
 def row_cc_lora_parallel_linear_forward(input_, base_layer, weight_a, weight_b, scaling):""","""     \"\"\"
     weight = base_layer.weight
     skip_bias_add, bias = base_layer.skip_bias_add, base_layer.bias
-    lora_params = [input_, weight, weight_a, weight_b, scaling]
-    if base_layer.explicit_expert_comm or get_tensor_model_parallel_world_size() == 1:
-        output_ = _FusedBaseParallelLoRAFunction.apply(*lora_params)
-    elif base_layer.sequence_parallel:
-        output_ = _FusedRowSeqParallelLoRAFunction.apply(*lora_params)
+    sequence_parallel = base_layer.sequence_parallel
+    if sequence_parallel:
+        output_ = _FusedRowSeqParallelLoRAFunction.apply(input_, weight, weight_a, weight_b, scaling)
     else:
-        output_ = _FusedRowNoSeqParallelLoRAFunction.apply(*lora_params)
+        output_ = _FusedRowNoSeqParallelLoRAFunction.apply(input_, weight, weight_a, weight_b, scaling)
     if not skip_bias_add:
         output = (output_ + bias) if bias is not None else output_
         output_bias = None
"""
],
"mindspeed_llm/tasks/finetune/lora/lora_moe.py": ["""         weight = getattr(self.get_base_layer(), "weight", None)
         if weight is not None:
             # the layer is already completely initialized, this is an update
-            if weight.dtype.is_floating_point or weight.dtype.is_complex:
-                self.to(weight.device, dtype=weight.dtype)
+            floating_point_list = [torch.float64, torch.float32, torch.float16, torch.bfloat16]
+            # if weight.dtype.is_floating_point or weight.dtype.is_complex:
+            if weight.dtype in floating_point_list or weight.dtype.is_complex:
+                # self.to(weight.device, dtype=weight.dtype)
+                self.to(dtype=weight.dtype)
             else:
                 self.to(weight.device)
         self.set_adapter(self.active_adapters)
"""
],
"mindspeed_llm/tasks/megatron_adaptor.py": [""" import torch
 
 
-def dummy_jit(fn):
-    def wrapper(*args, **kwargs):
-        return fn(*args, **kwargs)
-    return wrapper
-
-
 class MegatronAdaptation:
     \"\"\"
         A module manager supports adaptation registration, application and execution.""","""         from mindspeed.megatron_adaptor import te_adaptation, apex_adaptation, torch_adaptation
 
         # For torch >= 2.2.0
-        torch.compile = torch.jit.script
+        # torch.compile = torch.jit.script
 
         te_adaptation(MindSpeedPatchesManager)
         apex_adaptation(MindSpeedPatchesManager)""","""         self.patch_utils()
 
     def patch_fusions(self):
-        import megatron.core
         from mindspeed.core.fusions.fused_layer_norm import (FusedLayerNormAffineFunction, FastLayerNormFN)
         from mindspeed.core.fusions.fused_softmax import (is_kernel_available, ScaledUpperTriangMaskedSoftmax,
                                                           ScaledMaskedSoftmax, ScaledSoftmax, forward_fused_softmax)""","""         MegatronAdaptation.register('megatron.core.fusions.fused_bias_swiglu.BiasSwiGLUFunction',
                                     BiasSwiGLUFunction)
 
-        megatron.core.jit.jit_fuser = dummy_jit
-
     def patch_core_models(self):
         from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
         from mindspeed.core.models.common.embeddings.rotary_pos_embedding import get_pos_emb_on_this_cp_rank""","""     def patch_core_transformers(self):
         from mindspeed.core.transformer.moe.token_dispatcher import allgather_token_permutation, \\
             allgather_token_unpermutation
-        from mindspeed.core.transformer.moe.grouped_gemm_util import Ops, grouped_gemm_is_available, \\
-            get_device_capability
+        # from mindspeed.core.transformer.moe.grouped_gemm_util import Ops, grouped_gemm_is_available, \\
+        #     get_device_capability
+        from mindspeed.core.transformer.moe.grouped_gemm_util import get_device_capability
         from mindspeed.core.transformer.transformer import core_mlp_forward_wrapper
 
         from ..core.transformer.moe.moe_layer import moe_layer_init_wrapper, moe_layer_forward""","""                             TransformerLayerSubmodules,
                             transformer_layer_forward, get_num_layers_to_build_wrapper,
                             transformer_block_init_wrapper, transformer_block_forward, core_mlp_init)
-        MegatronAdaptation.register('torch.cuda.get_device_capability', get_device_capability)
+        # MegatronAdaptation.register('torch.cuda.get_device_capability', get_device_capability)
         MegatronAdaptation.register('megatron.core.transformer.transformer_block.TENorm', PTNorm)
         MegatronAdaptation.register('megatron.core.transformer.moe.router.TopKRouter.routing', topk_router_routing)
         MegatronAdaptation.register('megatron.core.transformer.moe.router.TopKRouter.forward', topk_router_forward)""","""         MegatronAdaptation.register('megatron.core.transformer.moe.router.z_loss_func', z_loss_func)
         MegatronAdaptation.register('megatron.core.transformer.transformer_block.get_num_layers_to_build',
                                     get_num_layers_to_build_wrapper)
-        MegatronAdaptation.register('megatron.core.transformer.moe.grouped_gemm_util.ops', Ops)
-        MegatronAdaptation.register('megatron.core.transformer.moe.grouped_gemm_util.grouped_gemm_is_available',
-                                    grouped_gemm_is_available)
+        # MegatronAdaptation.register('megatron.core.transformer.moe.grouped_gemm_util.ops', Ops)
+        # MegatronAdaptation.register('megatron.core.transformer.moe.grouped_gemm_util.grouped_gemm_is_available',
+        #                             grouped_gemm_is_available)
 
         # Transformer block
         MegatronAdaptation.register('megatron.core.transformer.transformer_block.TransformerBlock.__init__',""","""                     MoEAlltoAllTokenDispatcher)
 
         # For groupMLP especially deepseek
-        from mindspeed.core.transformer.moe.experts import groupedmlp_init_wrapper, groupedmlp_forward_wrapper
-        MegatronAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.__init__',
-                                    groupedmlp_init_wrapper)
-        MegatronAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.forward',
-                                    groupedmlp_forward_wrapper)
+        # from mindspeed.core.transformer.moe.experts import groupedmlp_init_wrapper, groupedmlp_forward_wrapper
+        # MegatronAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.__init__',
+        #                             groupedmlp_init_wrapper)
+        # MegatronAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.forward',
+                                    # groupedmlp_forward_wrapper)
 
     def patch_pipeline_parallel(self):
         from ..core.pipeline_parallel.p2p_communication import _batched_p2p_ops""","""     def patch_tensor_parallel(self):
         from mindspeed.core.tensor_parallel.layers import vocab_parallel_embedding_forward
         from mindspeed.core.tensor_parallel.random import _set_cuda_rng_state
-        from mindspeed.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy_forward
+        # from mindspeed.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy_forward
         from ..core import vocab_embedding_forward_wrapper, vocab_embedding_init_wrapper, checkpoint_forward_wrapper, checkpoint_backward_wrapper
 
         # default_generators need replace after set_device
         MegatronAdaptation.register('megatron.core.tensor_parallel.random._set_cuda_rng_state', _set_cuda_rng_state)
         # change masked_target for better performance
-        MegatronAdaptation.register(
-            'megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward',
-            vocab_parallel_cross_entropy_forward)
+        # MegatronAdaptation.register(
+        #     'megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward',
+        #     vocab_parallel_cross_entropy_forward)
         MegatronAdaptation.register('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward',
                                     vocab_parallel_embedding_forward)
         MegatronAdaptation.register('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward',
                                     vocab_embedding_forward_wrapper)
         MegatronAdaptation.register('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.__init__',
                                     vocab_embedding_init_wrapper)
-        MegatronAdaptation.register('megatron.core.tensor_parallel.random.CheckpointFunction.forward',
-                                    checkpoint_forward_wrapper)
-        MegatronAdaptation.register('megatron.core.tensor_parallel.random.CheckpointFunction.backward',
-                                    checkpoint_backward_wrapper)
+        # MegatronAdaptation.register('megatron.core.tensor_parallel.random.CheckpointFunction.forward',
+        #                             checkpoint_forward_wrapper)
+        # MegatronAdaptation.register('megatron.core.tensor_parallel.random.CheckpointFunction.backward',
+        #                             checkpoint_backward_wrapper)
         # For recompute-in-advance
         from mindspeed.core.tensor_parallel.random import checkpoint_wrapper
         MegatronAdaptation.register('megatron.core.tensor_parallel.random.checkpoint', checkpoint_wrapper)
"""
],
"mindspeed_llm/tasks/post_train/dpo/dpo_trainer.py": [""" from mindspeed_llm.tasks.post_train.base import BaseTrainer
 from mindspeed_llm.tasks.post_train.dpo.dpo_model import DPOModel
 from mindspeed_llm.training.utils import get_tune_attention_mask, get_finetune_data_on_this_tp_rank
+from megatron.core.tensor_parallel import mappings
 
 
+class ReduceFromContextParallelRegionDPO(torch.autograd.Function):
+    \"\"\"All-reduce the input from the model parallel region.\"\"\"
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
+        return grad_output
+
 class DPOTrainer(BaseTrainer):
     \"\"\"
     A trainer class for Direct Preference Optimization (DPO).""","""             Tensor: Log softmax values.
         \"\"\"
         # Step 1: Compute the local max value for numerical stability
-        z_max = logits.max(dim=-1, keepdim=True).values
+        z_max = logits.max(dim=-1, keepdim=True)[0]
 
         # Step 2: Perform all-reduce to get the global max across all processes
-        torch.distributed.all_reduce(
-            z_max,
-            op=torch.distributed.ReduceOp.MAX,
-            group=mpu.get_tensor_model_parallel_group()
-        )
+        # torch.distributed.all_reduce(
+        #     z_max,
+        #     op=torch.distributed.ReduceOp.MAX,
+        #     group=mpu.get_tensor_model_parallel_group()
+        # )
+        z_max = ReduceFromContextParallelRegionDPO()(z_max)
 
         # Step 3: Compute the log(exp(x - z_max)) for local logits
         local_exp = torch.exp(logits - z_max)""","""         local_sum_exp = local_exp.sum(dim=-1, keepdim=True)
 
         # Step 5: Perform all-reduce to get the global sum of exp(x - z_max) across all processes
-        torch.distributed.all_reduce(
-            local_sum_exp,
-            op=torch.distributed.ReduceOp.SUM,
-            group=mpu.get_tensor_model_parallel_group()
-        )
+        # torch.distributed.all_reduce(
+        #     local_sum_exp,
+        #     op=torch.distributed.ReduceOp.SUM,
+        #     group=mpu.get_tensor_model_parallel_group()
+        # )
+        local_sum_exp = mappings.reduce_from_tensor_model_parallel_region(local_sum_exp)
 
         # Step 6: Compute the log of the global sum of exp(x - z_max)
         log_sum_exp = local_sum_exp.log()""","""             all_log_probs = (per_token_log_probs * loss_mask).sum(-1)
             valid_length = loss_mask.sum(-1)
 
-            torch.distributed.all_reduce(
-                all_log_probs,
-                op=torch.distributed.ReduceOp.SUM,
-                group=mpu.get_tensor_model_parallel_group()
-            )
+            # torch.distributed.all_reduce(
+            #     all_log_probs,
+            #     op=torch.distributed.ReduceOp.SUM,
+            #     group=mpu.get_tensor_model_parallel_group()
+            # )
+            all_log_probs = mappings.reduce_from_tensor_model_parallel_region(all_log_probs)
 
             torch.distributed.all_reduce(
                 valid_length,
"""
],
"mindspeed_llm/tasks/post_train/launcher.py": [""" from mindspeed_llm.tasks.post_train.sft import SFTTrainer
 from mindspeed_llm.tasks.post_train.dpo import DPOTrainer
 from mindspeed_llm.tasks.post_train.rm import RMTrainer
+from mindspeed_llm.tasks.post_train.prm import PRMTrainer
 
 logger = logging.getLogger(__name__)
 ""","""         return DPOTrainer()
     elif stage == "rm":
         return RMTrainer()
+    elif stage == "prm":
+        return PRMTrainer()
     else:
         logger.info(f'Unknown Stage: {stage}')
         return None
"""
],
"mindspeed_llm/tasks/post_train/prm/__init__.py": ["""
+# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
+
+__all__ = ["PRMTrainer"]
+
+
+from .prm_trainer import PRMTrainer
"""
],

"mindspeed_llm/tasks/post_train/prm/prm_trainer.py": ["""
+# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
+from typing import Union
+from functools import partial
+import torch
+import torch.nn as nn
+from megatron.core import mpu, tensor_parallel
+from megatron.training import get_args, get_tokenizer
+from megatron.training.utils import average_losses_across_data_parallel_group
+from mindspeed_llm.tasks.post_train.base import BaseTrainer
+from mindspeed_llm.tasks.post_train.utils import convert_token_to_id
+from mindspeed_llm.training.utils import get_tune_attention_mask, get_finetune_data_on_this_tp_rank
+
+
+class PRMTrainer(BaseTrainer):
+    \"\"\"
+    A trainer class for Process Reward Model (PRM).
+
+    This class provides methods for model initialize, computing losses and metrics, and training.
+    \"\"\"
+
+    def __init__(self):
+        \"\"\"
+        Initializes the PRMTrainer instance.
+
+        Sets up the instance variables for the model provider, actual micro batch size,
+        and initializes the PRM model.
+        \"\"\"
+        super().__init__(
+            model_provider=self.model_provider,
+            get_batch_func=self.get_batch,
+            loss_func=self.loss_func,
+            forward_step_func=self.forward_step)
+        
+        args = get_args()
+        self.cross_entropy_loss = nn.CrossEntropyLoss()
+        self.tokenizer = get_tokenizer().tokenizer
+        # set placeholder token
+        self.placeholder_token_id = convert_token_to_id(args.placeholder_token, self.tokenizer)
+        self.reward_token_ids = args.reward_tokens
+        if self.reward_token_ids is not None:
+            self.reward_token_ids = sorted(
+                [convert_token_to_id(token, self.tokenizer) for token in self.reward_token_ids]
+            )
+
+    @staticmethod
+    def get_batch(data_iterator):
+        \"\"\"Generate a batch.\"\"\"
+
+        args = get_args()
+
+        if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
+            if args.variable_seq_lengths and args.pipeline_model_parallel_size > 2:
+                tokens, attention_mask = get_finetune_data_on_this_tp_rank(data_iterator)
+
+                return tokens, None, None, attention_mask, None
+            else:
+                return None, None, None, None, None
+        # Items and their type.
+        keys = ['input_ids', 'attention_mask', 'labels']
+        data_type = torch.int64
+
+        # Broadcast data.
+        data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)
+
+        # Unpack
+        labels = data_b.get('labels').long()
+        tokens = data_b.get('input_ids').long()
+        attention_mask_1d = data_b.get('attention_mask').long()
+        loss_mask = attention_mask_1d
+
+        attention_mask = get_tune_attention_mask(attention_mask_1d)
+
+        return tokens, labels, loss_mask, attention_mask, None
+
+
+    def loss_func(self, input_ids: torch.Tensor, labels: torch.Tensor, output_tensor: torch.Tensor):
+        \"\"\"PRM Loss function.
+        \"\"\"
+        placeholder_mask = input_ids == self.placeholder_token_id
+
+        output_tensor = tensor_parallel.mappings.gather_from_tensor_model_parallel_region(output_tensor)
+        logits = output_tensor[placeholder_mask]
+        labels = labels[placeholder_mask]
+
+        if self.reward_token_ids is not None:
+            # hard label with reward_token_ids set. (otherwise the whole vocab will be trained together.)
+            logits = logits[..., self.reward_token_ids]
+            # this is slow....
+            for i, token in enumerate(self.reward_token_ids):
+                labels = torch.where(labels == token, i, labels)
+
+        loss = self.cross_entropy_loss(logits, labels)
+        averaged_loss = average_losses_across_data_parallel_group([loss])
+        
+        with torch.no_grad():
+            acc = (logits.argmax(dim=-1) == labels).float().mean()
+
+        return loss * self.args.context_parallel_size, {'lm loss': averaged_loss[0], 'acc': acc}
+
+
+    def forward_step(self, data_iterator, model):
+        \"\"\"PRM Forward training step.
+
+        Args:
+            data_iterator : Input data iterator
+            model (GPTModel): The GPT Model
+        \"\"\"
+        # Get the batch.
+        self.timers('batch-generator', log_level=2).start()
+        tokens, labels, _, attention_mask, position_ids = self.get_batch(
+            data_iterator)
+        self.timers('batch-generator').stop()
+
+        output_tensor = model(tokens, position_ids, attention_mask)
+
+        return output_tensor, partial(self.loss_func, tokens, labels)
\\ No newline at end of file
"""
],
"mindspeed_llm/tasks/post_train/utils.py": [""" 
 def load_checkpoint_loosely():
     args = get_args()
-    return args.load_checkpoint_loosely
\\ No newline at end of file
+    return args.load_checkpoint_loosely
+
+def convert_token_to_id(token, tokenizer):
+    if isinstance(token, str):
+        token = tokenizer.encode(token, add_special_tokens=False)
+        assert len(token) == 1
+        return token[0]
+    else:
+        raise ValueError("token should be int or str")
\\ No newline at end of file
"""
],
"mindspeed_llm/tasks/preprocess/data_handler.py": [""" from megatron.core.datasets import indexed_dataset
 
 from mindspeed_llm.tasks.preprocess.templates import Prompter, AlpacaTemplate, get_model_template
+from mindspeed_llm.tasks.post_train.utils import convert_token_to_id
 from .utils import get_dataset_list, get_handler_dataset_attr, load_single_dataset, merge_dataset, align_dataset
 from .utils import greedy_knapsack
 ""","""             return align_dataset(raw_datasets, handler_dataset_attr, args)
 
     return raw_datasets
+
+class AlpacaStyleProcessRewardHandler(BaseDatasetHandler):
+    \"\"\"
+    Handle alpaca style dataset format in process reward dataset used in PRM training
+    \"\"\"
+
+    def __init__(self, args, raw_datasets, tokenizer, splitter):
+        super().__init__(args, raw_datasets, tokenizer, splitter)
+        self.train_on_inputs = False
+        self.args.json_keys = ["input_ids", "labels", 'attention_mask']
+        self.args.output_prefix = self.args.output_prefix + "_packed"
+
+        # set placeholder token
+        self.placeholder_token_id = convert_token_to_id(args.placeholder_token, \\
+                                                        self._unwrapped_tokenizer)
+        self.reward_tokens = args.reward_tokens
+
+    def _filter(self, sample):
+        inputs = self._unwrapped_tokenizer(sample["input"], padding=False, add_special_tokens=False)
+
+        input_ids = inputs["input_ids"]
+        label_values = sample["value"]
+
+        assert isinstance(label_values, list), "labels should be a list of strings or numbers"
+        label_tokens = []
+        for label in label_values:
+            assert (
+                self.reward_tokens is None or label in self.reward_tokens
+            ), f"label should be in reward tokens {self.reward_tokens}, got {label}"
+            label_tokens.append(convert_token_to_id(label, self._unwrapped_tokenizer))
+
+        labels = [-100] * len(input_ids)
+        indices = [index for index, item in enumerate(input_ids) if item == self.placeholder_token_id]
+        for index, indice in enumerate(indices):
+            labels[indice] = label_tokens[index]
+
+        input_token = inputs['input_ids']
+        attention_mask = inputs['attention_mask']
+        label_token = labels
+
+        concatenated_ids = {
+            "input_ids": [input_token],
+            "attention_mask":[attention_mask],
+            "labels": [label_token]
+        }
+
+        assert len(input_token) == len(label_token)
+
+        return concatenated_ids
"""
],
"mindspeed_llm/tasks/preprocess/decoder_packed_mtf_dataset.py": ["""                 "labels": self._cut_token(item["labels"], np.int64),
                 "position_ids": self._cut_token(position_ids.numpy(), np.int64)
             }
+        elif self.args.stage == "prm":
+            return {
+                "input_ids": self._cut_token(item['input_ids'], np.int64),
+                "attention_mask": self._cut_token(item["attention_mask"], np.int64),
+                "labels": self._cut_token(item["labels"], np.int64)
+            }
         else:
             return self._cut_instruction_token(item, np.int64)
 
"""
],
"mindspeed_llm/training/arguments.py": ["""     group.add_argument(
         '--stage',
         default=None,
-        choices=["sft", "dpo", "rm"],
+        choices=["sft", "dpo", "rm", "prm"],
         help='Determine training mode'
     )
 ""","""         "--is-pairwise-dataset", action='store_true',
         help="Whether the dataset is pairwise format that has a chosen sequence and rejected "
              "sequence, which usually used in reinforce learning.")
+    group.add_argument(
+        '--placeholder-token',
+        default='ки',
+        help="A special placeholder token marking the end of each step where the PRM can make predictions.",
+    )
+    group.add_argument(
+        '--reward-tokens',
+        nargs='+',
+        type=str,
+        default=[],
+        help="The labels represent the correctness of each reasoning step in the entire reasoning process.",
+    )
     return parser
 
 
"""
],
"mindspeed_llm/training/utils.py": ["""     else:
         via_length = torch.empty((1), dtype=torch.int64, device=torch.cuda.current_device())
         _broadcast(via_length)
+        via_length = via_length.item()
         tokens = torch.empty((micro_batch_size, via_length), dtype=torch.int64, device=torch.cuda.current_device())
         _broadcast(tokens)
         attention_mask_1d = torch.empty((micro_batch_size, via_length), dtype=torch.int64,
"""
],
"preprocess_data.py": ["""                             'This value must be greater than the initial size of the tokenizer.'
                             ' If this argument is used the value of `make-vocab-size-divisible-by` '
                             'will be ignored.')
+    group.add_argument(
+        '--placeholder-token',
+        default='ки',
+        help="A special placeholder token marking the end of each step where the PRM can make predictions.",
+    )
+    group.add_argument(
+        '--reward-tokens',
+        nargs='+',
+        type=str,
+        default=[],
+        help="The labels represent the correctness of each reasoning step in the entire reasoning process.",
+    )
 
 
 def add_output_args(parser):""","""         "AlpacaStyleInstructionHandler",
         "SharegptStyleInstructionHandler",
         "AlpacaStylePairwiseHandler",
-        "SharegptStylePairwiseHandler"
+        "SharegptStylePairwiseHandler",
+        "AlpacaStyleProcessRewardHandler"
     ]
     if args.prompt_type is not None and args.handler_name not in support_prompt_type_handler:
         raise AssertionError(f'If specify prompt_type , handler name must be in:\\n{support_prompt_type_handler}.')
"""
],
"pretrain_gpt.py": ["""     return batch.values()
 
 
+class ReduceFromContextParallelRegion(torch.autograd.Function):
+    \"\"\"All-reduce the input from the model parallel region.\"\"\"
+
+    @staticmethod
+    def symbolic(graph, input_):
+        torch.distributed.all_reduce(input_, group=mpu.get_context_parallel_group())
+        return input_
+        # return _reduce(input_)
+
+    @staticmethod
+    def forward(ctx, input_):
+        torch.distributed.all_reduce(input_, group=mpu.get_context_parallel_group())
+        return input_
+        # torch.distributed.all_reduce(_input, group=mpu.get_context_parallel_group())
+        # return _reduce(input_)
+
+    @staticmethod
+    def backward(ctx, grad_output):
+        return grad_output
+
+
 def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
     \"\"\"Loss function.
 ""","""     losses = output_tensor.float()
     loss_mask = loss_mask.view(-1).float()
     if args.context_parallel_size > 1:
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
         loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
 ""","""         if loss.isnan():
             raise ValueError(f'Rank {global_rank}: found NaN in local forward loss calculation. '
                              f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')
-
     # Reduce loss for logging.
     averaged_loss = average_losses_across_data_parallel_group([loss])
 
     return loss * args.context_parallel_size, {'lm loss': averaged_loss[0]}
 
 
+# def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
+#     \"\"\"Loss function.
+
+#     Args:
+#         loss_mask (torch.Tensor): Used to mask out some portions of the loss
+#         output_tensor (torch.Tensor): The tensor with the losses
+#     \"\"\"    
+#     args = get_args()
+
+#     losses = output_tensor.float()
+#     loss_mask = loss_mask.view(-1).float()
+#     if args.context_parallel_size > 1:
+#         loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
+#         torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
+#         loss = loss[0] / loss[1]
+#     else:
+#         loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
+
+#     # Check individual rank losses are not NaN prior to DP all-reduce.
+#     if args.check_for_nan_in_loss_and_grad:
+#         global_rank = torch.distributed.get_rank()
+#         if loss.isnan():
+#             raise ValueError(f'Rank {global_rank}: found NaN in local forward loss calculation. '
+#                              f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')
+
+#     # Reduce loss for logging.
+#     averaged_loss = average_losses_across_data_parallel_group([loss])
+
+#     return loss * args.context_parallel_size, {'lm loss': averaged_loss[0]}
+
+
 def forward_step(data_iterator, model: GPTModel):
     \"\"\"Forward training step.
"""
],

}


LINE_RULES_MindSpeed_MM = {
"examples/qwen2vl/dot_product_attention.py": [""" from megatron.training import get_args
 from megatron.core import mpu
 from mindspeed.core.models.common.embeddings.rotary_pos_embedding import yarn_get_mscale
-from mindspeed.core.context_parallel.ring_context_parallel import ringattn_context_parallel
-from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
+#from mindspeed.core.context_parallel.ring_context_parallel import ringattn_context_parallel
+#from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
 from mindspeed.model.alibi_mask import AlibiForFusionAttnSingleton
 from mindspeed.core.parallel_state import (get_context_parallel_group_for_hybrid_ring,
                                            get_context_parallel_for_hybrid_ring_world_size,""","""             padding_mask=None,
             atten_mask=None,
             scale=1.0 / math.sqrt(query.shape[-1]),
-            keep_prob=1,
+            keep_prob=1.0,
             input_layout='TND',
             actual_seq_qlen=actual_seq_len,
             actual_seq_kvlen=actual_seq_len,""","""             padding_mask=None,
             atten_mask=attention_mask_npu,
             scale=1.0 / math.sqrt(query.shape[-1]),
-            keep_prob=1,
+            keep_prob=1.0,
             input_layout='TND',
             actual_seq_qlen=tuple(cu_seq_lens[1:].cpu().numpy().tolist()),
             actual_seq_kvlen=tuple(cu_seq_lens[1:].cpu().numpy().tolist()),""","""     def forward(ctx, input_tensor, indices):
         ctx.save_for_backward(indices)
         ctx.first_axis_dim, other_shape = input_tensor.shape[0], input_tensor.shape[1:]
+        other_shape = torch.Size(other_shape)
         second_dim = other_shape.numel()
         return torch.gather(
-            rearrange(input_tensor, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)
+            rearrange(input_tensor, "b ... -> b (...)").contiguous(), 0, repeat(indices, "z -> z d", d=second_dim).contiguous()
         ).reshape(-1, *other_shape)
 
     @staticmethod
"""
],
"examples/qwen2vl/qwen2vl_convert_to_mm_ckpt.py": [""" from copy import deepcopy
 
 import torch
-from safetensors.torch import load_file
+from torch.serialization import safe_load_file
 
 
 def load_from_hf(_load_dir):""","""     load_dir = Path(_load_dir)
     state_dict = {}
     for safe_path in load_dir.glob("*.safetensors"):
-        state_dict.update(load_file(str(safe_path), device='cpu'))
+        state_dict.update(safe_load_file(str(safe_path), device='cpu'))
     return state_dict
 
 
"""
],
"mindspeed_mm/__init__.py": ["""     VLModel
 )
 from mindspeed_mm.patchs import PatchesManager
-from mindspeed_mm.tasks import sora_pipeline_dict, vlm_pipeline_dict
+# from mindspeed_mm.tasks import sora_pipeline_dict, vlm_pipeline_dict
 from mindspeed_mm.utils.utils import (
     is_npu_available,
     get_device,
"""
],
"mindspeed_mm/data/data_utils/func_utils/mm_plugin.py": ["""                 content = content.replace(
                     IMAGE_PLACEHOLDER,
                     "<|vision_start|>{}<|vision_end|>".format(
-                        self.image_token * (image_grid_thw[num_image_tokens].prod() // merge_length)
+                        self.image_token * int(image_grid_thw[num_image_tokens].prod() // merge_length)
                     ),
                     1,
                 )
"""
],
"mindspeed_mm/data/data_utils/utils.py": [""" from torchvision.io.video import (
     _align_audio_frames,
     _check_av_available,
-    _log_api_usage_once,
     _read_from_stream,
     _video_opt,
 )
"""
],
"mindspeed_mm/data/datasets/qwen2vl_dataset.py": ["""-import os
 from functools import partial
 
 from datasets import load_dataset
-from torch.utils.data import Dataset
-from transformers.training_args import TrainingArguments
+from torch.utils.data import Dataset, ConcatDataset
+from transformers import AutoProcessor
 
 from mindspeed_mm.data.data_utils.func_utils.convert import DataArguments, DatasetAttr, load_tokenizer, \\
     convert_sharegpt, preprocess_supervised_dataset
-from mindspeed_mm.data.data_utils.func_utils.log import get_logger
 from mindspeed_mm.data.data_utils.func_utils.model_args import ProcessorArguments
 from mindspeed_mm.data.data_utils.func_utils.template import get_template_and_fix_tokenizer
 
-logger = get_logger(__name__)
-
 
 def get_qwen2vl_dataset(basic_param, preprocess_param, dataset_param):
     data_args = DataArguments(**basic_param)
     process_args = ProcessorArguments(**preprocess_param)
     dataset_attr = DatasetAttr(**dataset_param["attr"])
-
     tokenizer_module = load_tokenizer(process_args)
-    tokenizer, processor = tokenizer_module['tokenizer'], tokenizer_module['processor']
+    tokenizer = tokenizer_module['tokenizer']
+    processor = AutoProcessor.from_pretrained(process_args.model_name_or_path, local_files_only=True)
     template = get_template_and_fix_tokenizer(tokenizer, data_args.template)
-    # 确保主进程进行数据处理，其他进程复用缓存避免重复计算，该策略和llamafactory对数据处理策略一致
-    with TrainingArguments(output_dir='./').main_process_first(desc="pre-process dataset"):
-        # -----------------load dataset from file-------------------------------------------------------------------------
-        dataset = load_dataset(path="json", data_files=data_args.dataset, split="train", cache_dir=data_args.cache_dir,
-                               streaming=data_args.streaming)
-        if data_args.max_samples:
-            dataset = dataset.select(range(data_args.max_samples))
-        local_process_index = int(os.getenv("LOCAL_RANK", -1))
-        if data_args.streaming:
-            kwargs = {}
-        else:
-            kwargs = {
-                "num_proc": data_args.preprocessing_num_workers,
-                # 配置了overwrite_cache为false（默认为false)时，非rank0节点读取cache不再进行map处理
-                # 配置了overwrite_cache为true（默认为false)时，所有节点都读取cache不再进行map处理
-                "load_from_cache_file": (not data_args.overwrite_cache) or (local_process_index != 0)
-            }
-        logger.debug(f'Rank: %s, kwargs: %s', local_process_index, kwargs)
-        # -----------------convert to sharegpt ---------------------------------------------------------------------------
-        convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr, dataset_dir=data_args.dataset_dir)
-        dataset = dataset.map(
-            convert_func,
-            batched=False,
-            remove_columns=(list(next(iter(dataset)).keys())),
-            desc=f"Rank {local_process_index}, Converting format of dataset",
-            **kwargs,
-        )
-        # -----------------convert text to token id ----------------------------------------------------------------------
-        preprocess_func = partial(
-            preprocess_supervised_dataset,
-            template=template,
-            tokenizer=tokenizer,
-            processor=processor,
-            data_args=data_args,
+    # -----------------load dataset from file-------------------------------------------------------------------------
+    dataset = load_dataset(
+        path="json",
+        name=None,
+        data_dir=None,
+        data_files=data_args.dataset,
+        split="train",
+        cache_dir=data_args.cache_dir,
+        token=None,
+        streaming=data_args.streaming,
+        trust_remote_code=False,
+    )
+    # -----------------convert to sharegpt ---------------------------------------------------------------------------
+    convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr, dataset_dir=data_args.dataset_dir)
+    column_names = list(next(iter(dataset)).keys())
+    kwargs = {}
+    if not data_args.streaming:
+        local_process_index = 0
+        kwargs = dict(
+            num_proc=data_args.preprocessing_num_workers,
+            load_from_cache_file=(not data_args.overwrite_cache) or (local_process_index != 0),
+            desc="Converting format of dataset",
         )
-        dataset = dataset.map(
-            preprocess_func,
-            batched=True,
-            batch_size=data_args.preprocessing_batch_size,
-            remove_columns=(list(next(iter(dataset)).keys())),
-            desc=f"Rank {local_process_index}, Running tokenizer on dataset",
-            **kwargs,
+    if data_args.max_samples:
+        dataset = dataset.select(range(data_args.max_samples))
+    dataset = dataset.map(
+        convert_func,
+        batched=False,
+        remove_columns=column_names,
+        **kwargs,
+    )
+    # -----------------convert text to token id ----------------------------------------------------------------------
+    preprocess_func = partial(
+        preprocess_supervised_dataset,
+        template=template,
+        tokenizer=tokenizer,
+        processor=processor,
+        data_args=data_args,
+    )
+    column_names = list(next(iter(dataset)).keys())
+    kwargs = {}
+    if not data_args.streaming:
+        kwargs = dict(
+            num_proc=data_args.preprocessing_num_workers,
+            load_from_cache_file=(not data_args.overwrite_cache) or (local_process_index != 0),
+            desc="Running tokenizer on dataset",
         )
-        return dataset
+    dataset = dataset.map(
+        preprocess_func,
+        batched=True,
+        batch_size=data_args.preprocessing_batch_size,
+        remove_columns=column_names,
+        **kwargs,
+    )
+    return dataset
 
 
 class Qwen2vlDataset(Dataset):
"""
],
"mindspeed_mm/models/ae/vae.py": [""" import torch
 import torch.nn as nn
-from diffusers.models import AutoencoderKL
+# from diffusers.models import AutoencoderKL
 from einops import rearrange
 from megatron.core import mpu
 
"""
],
"mindspeed_mm/models/common/blocks.py": [""" import torch
 import torch.nn as nn
 
-from diffusers.models.activations import GELU, GEGLU, ApproximateGELU
+# from diffusers.models.activations import GELU, GEGLU, ApproximateGELU
 from mindspeed_mm.models.common.linear import MatmulAddLinear
 
 
"""
],
"mindspeed_mm/models/common/checkpoint.py": [""" 
 import torch
 import torch.nn as nn
-from torch.utils.checkpoint import checkpoint, checkpoint_sequential
+# from torch.utils.checkpoint import checkpoint, checkpoint_sequential
 
 import safetensors
 
"""
],
"mindspeed_mm/models/common/embeddings/common_embeddings.py": [""" 
 import torch
 from torch import nn
-from timm.models.vision_transformer import Mlp
+# from timm.models.vision_transformer import Mlp
 
 
 class TimestepEmbedder(nn.Module):
"""
],
"mindspeed_mm/models/common/embeddings/pos_embeddings.py": ["""         freqs = broad_cat((freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1)
 
         freqs = freqs.contiguous()
-        self.freqs_sin = freqs.sin().npu()
-        self.freqs_cos = freqs.cos().npu()
+        self.freqs_sin = freqs.sin()
+        self.freqs_cos = freqs.cos()
 
         self.text_length = text_length
         if learnable_pos_embed:
"""
],
"mindspeed_mm/models/common/embeddings/time_embeddings.py": [""" import math
-
+import numpy as np
 import torch
 import torch.nn as nn
 from einops import rearrange, repeat""","""     \"\"\"
     if not repeat_only:
         half = dim // 2
-        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
-            device=timesteps.device
-        )
+
+        tmp = -math.log(max_period) * np.arange(0, half) / half
+        freqs = torch.exp(torch.Tensor(tmp.astype(np.float32)))
         args = timesteps[:, None].float() * freqs[None]
         embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
         if dim % 2:
"""
],
"mindspeed_mm/models/common/ffn.py": [""" import torch
 from torch import nn
 import torch.nn.functional as F
-from diffusers.models.activations import GEGLU, ApproximateGELU
+# from diffusers.models.activations import GEGLU, ApproximateGELU
 from megatron.core import mpu, tensor_parallel
 from megatron.training import get_args
 from megatron.training.arguments import core_transformer_config_from_args
"""
],
"mindspeed_mm/models/common/regularizer.py": ["""         self.std = torch.exp(0.5 * self.logvar)
         self.var = torch.exp(self.logvar)
         if self.deterministic:
-            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)
+            self.var = self.std = torch.zeros_like(self.mean)
 
     def sample(self):
-        x = self.mean + self.std * torch.randn_like(self.mean).to(device=self.parameters.device)
+        x = self.mean + self.std * torch.randn_like(self.mean)
         return x
 
     def kl(self, other=None):
"""
],
"mindspeed_mm/models/diffusion/cogvideo_diffusion.py": ["""     linear_end=2e-2,
 ):
     if schedule == "linear":
-        betas = torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64) ** 2
+        betas = np.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64) ** 2
     else:
         raise NotImplementedError("Only support linear schedule")
-    return betas.numpy()
+    return betas
 
 
 class Discretization:""","""         return c_in, c_noise, c_out, c_skip
 
     def sigma_to_idx(self, sigma):
-        dists = sigma - self.sigmas.to(sigma.device)[:, None]
+        dists = sigma - self.sigmas[:, None]
         return dists.abs().argmin(dim=0).view(sigma.shape)
 
     def idx_to_sigma(self, idx):
-        return self.sigmas.to(idx.device)[idx]
+        return self.sigmas[idx]
 
     def possibly_quantize_sigma(self, sigma):
         return self.idx_to_sigma(self.sigma_to_idx(sigma))""",""" 
         additional_model_inputs = dict()
         alphas_cumprod_sqrt, idx = self.sigma_sampler(latents.shape[0], return_idx=True)
-        self.alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(latents.device)
-        idx = idx.to(latents.device)
+        self.alphas_cumprod_sqrt = alphas_cumprod_sqrt
+        idx = idx
 
         # broadcast noise here
 """,""" 
         if self.offset_noise_level > 0.0:
             noise = (
-                    noise + append_dims(torch.randn(latents.shape[0]).to(latents.device),
+                    noise + append_dims(torch.randn(latents.shape[0]),
                                         latents.ndim) * self.offset_noise_level
             )
 
"""
],
"mindspeed_mm/models/diffusion/diffusers_scheduler.py": [""" from torch import Tensor
 from tqdm.auto import tqdm
 import torch.nn.functional as F
-from diffusers.schedulers import (
-    DDIMScheduler,
-    DDPMScheduler,
-    PNDMScheduler,
-    EulerDiscreteScheduler,
-    DPMSolverMultistepScheduler,
-    HeunDiscreteScheduler,
-    EulerAncestralDiscreteScheduler,
-    DEISMultistepScheduler,
-    KDPM2AncestralDiscreteScheduler,
-    CogVideoXDPMScheduler,
-    CogVideoXDDIMScheduler,
-    FlowMatchEulerDiscreteScheduler
-)
-from diffusers.training_utils import compute_snr
+# from diffusers.schedulers import (
+#     DDIMScheduler,
+#     DDPMScheduler,
+#     PNDMScheduler,
+#     EulerDiscreteScheduler,
+#     DPMSolverMultistepScheduler,
+#     HeunDiscreteScheduler,
+#     EulerAncestralDiscreteScheduler,
+#     DEISMultistepScheduler,
+#     KDPM2AncestralDiscreteScheduler,
+#     CogVideoXDPMScheduler,
+#     CogVideoXDDIMScheduler,
+#     FlowMatchEulerDiscreteScheduler
+# )
+# from diffusers.training_utils import compute_snr
 from megatron.core import mpu
 
 from mindspeed_mm.models.diffusion.diffusion_utils import explicit_uniform_sampling
 from mindspeed_mm.utils.utils import get_device
 
 DIFFUSERS_SCHEDULE_MAPPINGS = {
-    "DDIM": DDIMScheduler,
-    "EulerDiscrete": EulerDiscreteScheduler,
-    "DDPM": DDPMScheduler,
-    "DPMSolverMultistep": DPMSolverMultistepScheduler,
-    "PNDM": PNDMScheduler,
-    "HeunDiscrete": HeunDiscreteScheduler,
-    "EulerAncestralDiscrete": EulerAncestralDiscreteScheduler,
-    "DEISMultistep": DEISMultistepScheduler,
-    "KDPM2AncestralDiscrete": KDPM2AncestralDiscreteScheduler,
-    "cogvideox_5b": CogVideoXDPMScheduler,
-    "cogvideox_2b": CogVideoXDDIMScheduler
+    # "DDIM": DDIMScheduler,
+    # "EulerDiscrete": EulerDiscreteScheduler,
+    # "DDPM": DDPMScheduler,
+    # "DPMSolverMultistep": DPMSolverMultistepScheduler,
+    # "PNDM": PNDMScheduler,
+    # "HeunDiscrete": HeunDiscreteScheduler,
+    # "EulerAncestralDiscrete": EulerAncestralDiscreteScheduler,
+    # "DEISMultistep": DEISMultistepScheduler,
+    # "KDPM2AncestralDiscrete": KDPM2AncestralDiscreteScheduler,
+    # "cogvideox_5b": CogVideoXDPMScheduler,
+    # "cogvideox_2b": CogVideoXDDIMScheduler
 }
 
 
"""
],
"mindspeed_mm/models/diffusion/rflow.py": [""" from tqdm.auto import tqdm
 import torch
 from torch import Tensor
-from torch.distributions import LogisticNormal
+# from torch.distributions import LogisticNormal
 
 from .diffusion_utils import extract_into_tensor, mean_flat
 
"""
],
"mindspeed_mm/models/internvl_model.py": ["""             Make causal mask used for bi-directional self-attention.
             \"\"\"
             bsz, tgt_len = input_ids_shape
-            mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
+            type_min = -3.4028234663852886e+38
+            mask = torch.full((tgt_len, tgt_len), torch.tensor(type_min))
             mask_cond = torch.arange(mask.size(-1), device=device)
             mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
             mask = mask.to(dtype)""","""             expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
 
             inverted_mask = 1.0 - expanded_mask
-
-            return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
+            type_min = -3.4028234663852886e+38
+            return inverted_mask.masked_fill(inverted_mask.to(torch.bool), type_min)
 
         input_shape = attention_mask.shape
         # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]""",""" 
         if attention_mask is not None:
             # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
-            expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_len=input_shape[-1]).to(device)
+            expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_len=input_shape[-1])
             combined_attention_mask = (
                 expanded_attn_mask if combined_attention_mask is None
                 else expanded_attn_mask + combined_attention_mask""","""         shift_logits = shift_logits.view(-1, self.vocab_size)
         shift_labels = shift_labels.view(-1)
 
-        shift_labels = shift_labels.to(shift_logits.device)
+        shift_labels = shift_labels
         loss = loss_fct(shift_logits, shift_labels)
         if ignore_flag:
             loss = loss * 0.0
"""
],
"mindspeed_mm/models/predictor/dits/__init__.py": [""" from .stdit import STDiT
 from .stdit3 import STDiT3
 from .sat_dit import SatDiT
-from .pt_dit_diffusers import PTDiTDiffuser as PTDiT
+# from .pt_dit_diffusers import PTDiTDiffuser as PTDiT
 
-__all__ = ["VideoDiT", "VideoDitSparse", "Latte", "STDiT", "STDiT3", "SatDiT", "VideoDitSparseI2V", "PTDiT"]
+__all__ = ["VideoDiT", "VideoDitSparse", "Latte", "STDiT", "STDiT3", "SatDiT", "VideoDitSparseI2V"]
"""
],
"mindspeed_mm/models/predictor/dits/pt_dit_diffusers.py": [""" import torch
 import torch.nn.functional as F
 import torch.nn.init as init
-from diffusers.configuration_utils import ConfigMixin, register_to_config
-from diffusers.models.embeddings import Timesteps, TimestepEmbedding
-from diffusers.models.modeling_utils import ModelMixin
-from diffusers.models.normalization import AdaLayerNormSingle
-from diffusers.models.attention import AdaLayerNorm, FeedForward
-from diffusers.models.attention_processor import Attention
-
-from diffusers.utils import BaseOutput, is_torch_version
+# from diffusers.configuration_utils import ConfigMixin, register_to_config
+# from diffusers.models.embeddings import Timesteps, TimestepEmbedding
+# from diffusers.models.modeling_utils import ModelMixin
+# from diffusers.models.normalization import AdaLayerNormSingle
+# from diffusers.models.attention import AdaLayerNorm, FeedForward
+# from diffusers.models.attention_processor import Attention
+
+# from diffusers.utils import BaseOutput, is_torch_version
 from einops import rearrange, repeat
 from torch import nn
-from diffusers.utils.torch_utils import maybe_allow_in_graph
+# from diffusers.utils.torch_utils import maybe_allow_in_graph
 
 from mindspeed_mm.models.common.embeddings import PatchEmbed2D_3DsincosPE
 
 
-try:
-    from diffusers.models.embeddings import PixArtAlphaTextProjection
-except ImportError:
-    from diffusers.models.embeddings import \\
-        CaptionProjection as PixArtAlphaTextProjection
+# try:
+#     from diffusers.models.embeddings import PixArtAlphaTextProjection
+# except ImportError:
+#     from diffusers.models.embeddings import \\
+#         CaptionProjection as PixArtAlphaTextProjection
 
 
 def zero_module(module):""","""     return module
 
 
-@maybe_allow_in_graph
+# @maybe_allow_in_graph
 class ProxyTokensTransformerBlock(nn.Module):
     r\"\"\"
     Parameters:
"""
],
"mindspeed_mm/models/predictor/dits/sat_dit.py": [""" from contextlib import nullcontext
 
 import torch
-from diffusers.models.embeddings import SinusoidalPositionalEmbedding
+# from diffusers.models.embeddings import SinusoidalPositionalEmbedding
 from einops import rearrange
 from megatron.core import mpu, tensor_parallel
 from megatron.training import get_args""","""         else:
             self.patch_embed = VideoPatch2D(in_channels, inner_dim, self.patch_size_h)
 
+        # self.pos_embed = Rotary3DPositionEmbedding(
+        #     hidden_size_head=head_dim,
+        #     text_length=text_length,
+        #     height=input_size[1] // self.patch_size_h,
+        #     width=input_size[2] // self.patch_size_w,
+        #     compressed_num_frames=(input_size[0] - 1) // interpolation_scale[0] + 1,
+        #     hidden_size=inner_dim,
+        #     learnable_pos_embed=learnable_pos_embed
+        # )
+
         self.pos_embed = Rotary3DPositionEmbedding(
             hidden_size_head=head_dim,
             text_length=text_length,""","""             width=input_size[2] // self.patch_size_w,
             compressed_num_frames=(input_size[0] - 1) // interpolation_scale[0] + 1,
             hidden_size=inner_dim,
-            learnable_pos_embed=learnable_pos_embed
+            learnable_pos_embed=False#learnable_pos_embed
         )
+
+        self.learnable_pos_embed=learnable_pos_embed
+        if  self.learnable_pos_embed:
+
+            height=input_size[1] // self.patch_size_h
+            width=input_size[2] // self.patch_size_w
+            compressed_num_frames=(input_size[0] - 1) // interpolation_scale[0] + 1
+            hidden_size=inner_dim
+            num_patches = int(height * width * compressed_num_frames + text_length)
+            self.text_length_tmp = text_length
+            self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, int(hidden_size)), requires_grad=True)
+        else:
+            self.pos_embedding = None
+
         # Init VideoDiTBlock
         self.videodit_blocks = nn.ModuleList(
             [""","""         height, width = latents.shape[-2] // self.patch_size_h, latents.shape[-1] // self.patch_size_w
 
         if "masked_video" in kwargs.keys() and kwargs["masked_video"] is not None:
-            latents = torch.cat([latents, kwargs["masked_video"]], dim=1)
+            latents = torch.cat([latents, kwargs["masked_video"].to(latents.dtype)], dim=1)
 
         added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
         latents_vid, latents_img, prompt_vid, prompt_img, timestep_vid, timestep_img, \\""","""                                                         rope_H=h // self.patch_size[1],
                                                         rope_W=w // self.patch_size[2])
             _, seq_len, _ = latents_vid.shape
-            pos_emb = self.pos_embed.position_embedding_forward(latents.to(self.dtype),
-                                                                seq_length=seq_len - self.text_length)
+            #megatron sync parameter in forward only
+
+            if self.learnable_pos_embed:
+                seq_length=seq_len - self.text_length_tmp
+                pos_emb = self.pos_embedding[:, :self.text_length_tmp + seq_length]
+            else:
+                pos_emb = self.pos_embed.position_embedding_forward(latents.to(self.dtype),
+                        seq_length=seq_len - self.text_length)
+
             if pos_emb is not None:
                 latents_vid = latents_vid + pos_emb
         else:""",""" 
         # unpatchify
         output = rearrange(latents, "b (t h w) (c o p q) -> b (t o) c (h p) (w q)",
-                           b=latents.shape[0], h=height, w=width,
+                           b=latents.shape[0], h=height.item(), w=width.item(),
                            o=self.patch_size_t, p=self.patch_size_h, q=self.patch_size_w,
                            c=self.out_channels).transpose(1, 2)
         return output
"""
],
"mindspeed_mm/models/predictor/dits/stdit.py": [""" import torch
 import torch.nn as nn
 from einops import rearrange
-from timm.models.layers import DropPath
-from timm.models.vision_transformer import Mlp
+# from timm.models.layers import DropPath
+# from timm.models.vision_transformer import Mlp
 from megatron.core import mpu, tensor_parallel
 from megatron.training import get_args
 
"""
],
"mindspeed_mm/models/predictor/dits/stdit3.py": [""" import torch.distributed as dist
 import torch.nn as nn
 from einops import rearrange
-from timm.models.layers import DropPath
-from timm.models.vision_transformer import Mlp
+# from timm.models.layers import DropPath
+# from timm.models.vision_transformer import Mlp
 
 from megatron.core import mpu
 from mindspeed_mm.models.common.module import MultiModalModule
"""
],
"mindspeed_mm/models/predictor/dits/video_dit.py": [""" import torch
 from torch import nn
 import torch.nn.functional as F
-from diffusers.models.embeddings import SinusoidalPositionalEmbedding, PixArtAlphaTextProjection
-from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormSingle
-from diffusers.models.attention import FeedForward
+# from diffusers.models.embeddings import SinusoidalPositionalEmbedding, PixArtAlphaTextProjection
+# from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormSingle
+# from diffusers.models.attention import FeedForward
 from megatron.core import mpu, tensor_parallel
 from megatron.training import get_args
 
"""
],
"mindspeed_mm/models/predictor/dits/video_dit_sparse.py": [""" import torch
 from torch import nn
 import torch.nn.functional as F
-from diffusers.models.embeddings import PixArtAlphaTextProjection
-from diffusers.models.normalization import AdaLayerNormSingle
+# from diffusers.models.embeddings import PixArtAlphaTextProjection
+# from diffusers.models.normalization import AdaLayerNormSingle
 from megatron.core import mpu, tensor_parallel
 from megatron.training import get_args
 
"""
],
"mindspeed_mm/models/predictor/predict_model.py": [""" from megatron.training.utils import print_rank_0
 
 from mindspeed_mm.models.common.checkpoint import load_checkpoint
-from .dits import VideoDiT, Latte, STDiT, STDiT3, VideoDitSparse, SatDiT, VideoDitSparseI2V, PTDiT
+from .dits import VideoDiT, Latte, STDiT, STDiT3, VideoDitSparse, SatDiT, VideoDitSparseI2V
 
 PREDICTOR_MODEL_MAPPINGS = {
     "videodit": VideoDiT,""","""     "stdit": STDiT,
     "stdit3": STDiT3,
     "satdit": SatDiT,
-    "ptdit": PTDiT,
 }
 
 
"""
],
"mindspeed_mm/models/qwen2vl_model.py": ["""                 input_embeds = input_embeds.transpose(0, 1)
                 image_mask = torch.eq(input_ids, self.img_context_token_id).unsqueeze(-1).expand_as(input_embeds)
                 vit_embeds = vit_embeds[:, 0, :]
+                orig_dtype = vit_embeds.dtype
+                input_embeds = input_embeds.to(torch.float32)
+                vit_embeds = vit_embeds.to(torch.float32)
                 input_embeds = input_embeds.masked_scatter(image_mask, vit_embeds)
+                input_embeds = input_embeds.to(orig_dtype)
                 input_embeds = input_embeds.transpose(0, 1).clone()
 
             past_seen_tokens = 0
"""
],
"mindspeed_mm/models/text_encoder/text_encoder.py": ["""         # Only huggingface backend is supported, OpenMind backend will be supported soon.
         module = importlib.import_module("transformers")
         automodel = getattr(module, self.automodel_name)
+        config["low_cpu_mem_usage"] = False
         self.model = automodel.from_pretrained(**config)
 
     def get_model(self):
"""
],
"mindspeed_mm/models/vision/vision_encoders/clip_vit_model.py": ["""         **kwargs,
     ) -> None:
         super().__init__(config=config)
-        self.device = get_device(config.device)
+        self.device = config.device
         self.class_token_len = config.class_token_len
         self.visual_hidden_size = config.hidden_size
         self.patch_size = config.patch_size""","""         if attention_mask is None:
             attention_mask = torch.ones(
                 1, 1, self.seq_length, self.seq_length
-            ).to(self.device)  # [1, 1, s, s]
+            )  # [1, 1, s, s]
             attention_mask = attention_mask < 0.5  # to bool
 
         x = self.decoder(x, attention_mask)
"""
],
"mindspeed_mm/models/vision/vision_encoders/internvit_model.py": [""" import torch.nn.functional as F
 import torch.utils.checkpoint
 import torch_npu
-from timm.models.layers import DropPath
+# from timm.models.layers import DropPath
 
 from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
 from megatron.core.transformer.enums import AttnMaskType
"""
],
"mindspeed_mm/models/vision/vision_encoders/qwen2vl_vit_model.py": ["""     if use_fused_rope:
         import torch_npu
         cos, sin = cos[:1], sin[:1]
-        q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
-        k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
+        #q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
+        #k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
+        q_embed = torch_npu.npu_rotary_position_embedding(q, cos, sin)
+        k_embed = torch_npu.npu_rotary_position_embedding(k, cos, sin)
     else:
         q_embed = (q * cos) + (rotate_half(q) * sin)
         k_embed = (k * cos) + (rotate_half(k) * sin)""","""     sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
     if use_fused_rope:
         import torch_npu
-        output = torch_npu.npu_rotary_mul(tensor, cos, sin).to(orig_dtype)
+        #output = torch_npu.npu_rotary_mul(tensor, cos, sin).to(orig_dtype)
+        output = torch_npu.npu_rotary_position_embedding(t, cos, sin).to(orig_dtype)
     else:
         output = ((tensor * cos) + (rotate_half(tensor) * sin)).to(orig_dtype)
     return output""",""" 
         inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
         position_ids_expanded = position_ids[:, :, None, :].float()
-        device_type = x_device.type
+        device_type = x_device#.type
         device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
         with torch.autocast(device_type=device_type, enabled=False):
             freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)""","""         self.theta = theta
 
     def forward(self, seqlen: int) -> torch.Tensor:
-        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.bfloat16) / self.dim)).to(
-            self.inv_freq.device)
+        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.bfloat16) / self.dim))
+        #.to(self.inv_freq.device)
         self.register_buffer("inv_freq", inv_freq, persistent=False)
         seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=torch.bfloat16)
         freqs = torch.outer(seq, self.inv_freq)""",""" 
         seq_len = images.shape[0]
         attention_mask = torch.full(
-            [1, seq_len, seq_len], torch.finfo(images.dtype).min, device=images.device,
+            [1, seq_len, seq_len], -3.3895313892515355e+38, device=images.device,
             dtype=torch.bool
         )
         for i in range(1, len(cu_seqlens)):
"""
],
"mindspeed_mm/models/vl_model.py": ["""             shift_logits = shift_logits.view(-1, self.text_decoder.vocab_size)
             shift_labels = shift_labels.view(-1)
             # Enable model parallelism
-            shift_labels = shift_labels.to(shift_logits.device)
+            shift_labels = shift_labels
             loss = loss_fct(shift_logits, shift_labels)
 
         return loss""","""         load_params = torch.load(ckpt_path, map_location="cpu")
         print(model.load_state_dict(load_params, strict=False))
     else:
-        print("Warning: ckpt path is None or empty, skipping loading ckpt.")
\\ No newline at end of file
+        print("Warning: ckpt path is None or empty, skipping loading ckpt.")
"""
],
"mindspeed_mm/patchs/diffusers_patches.py": [""" # limitations under the License.
 
 import torch_npu
-from diffusers.utils.deprecation_utils import deprecate
-from diffusers.utils.import_utils import is_torch_npu_available
+# from diffusers.utils.deprecation_utils import deprecate
+# from diffusers.utils.import_utils import is_torch_npu_available
 
 
 def geglu_forward(self, hidden_states, *args, **kwargs):
"""
],
"mindspeed_mm/tasks/__init__.py": ["""-from mindspeed_mm.tasks.inference import sora_pipeline_dict, vlm_pipeline_dict
+# from mindspeed_mm.tasks.inference import sora_pipeline_dict, vlm_pipeline_dict
 
-__all__ = ["sora_pipeline_dict", "vlm_pipeline_dict"]
+# __all__ = ["sora_pipeline_dict", "vlm_pipeline_dict"]
"""
],
"mindspeed_mm/training.py": ["""     if args.log_progress:
         append_to_progress_log("Starting job")
 
-    torch.backends.cuda.matmul.allow_tf32 = getattr(args.mm.model, "allow_tf32", False)
-    torch.npu.config.allow_internal_format = getattr(args.mm.model, "allow_internal_format", False)
+    # torch.backends.cuda.matmul.allow_tf32 = getattr(args.mm.model, "allow_tf32", False)
+    # torch.npu.config.allow_internal_format = getattr(args.mm.model, "allow_internal_format", False)
 
     # Set pytorch JIT layer fusion options and warmup JIT functions.
     set_jit_fusion_options()
"""
],
"mindspeed_mm/utils/extra_processor/cogvideox_i2v_processor.py": ["""         self.noised_image_input = config.get("noised_image_input", True)
 
     def add_noise_to_image(self, image):
-        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(image.device)
+        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],))
         sigma = torch.exp(sigma).to(image.dtype)
         image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
         image = image + image_noise
"""
],
"mindspeed_mm/utils/utils.py": [""" 
 def get_dtype(dtype):
     \"\"\"return torch type according to the string\"\"\"
-    if isinstance(dtype, torch.dtype):
+    if dtype == torch.bfloat16 or dtype == torch.float32:
         return dtype
     dtype_mapping = {
         "int32": torch.int32,""",""" 
 
 def quick_gelu(x: torch.Tensor) -> torch.Tensor:
-    return x * torch.sigmoid(1.702 * x)
+    return x * torch.sigmoid(torch.Tensor([1.702], dtype=x.dtype) * x)
 
 
 _CONTEXT_PARALLEL_GROUP = None
"""
],
"pretrain_internvl.py": [""" import torch.distributed
 import torch.nn.functional as F
 import mindspeed.megatron_adaptor
-from mindspeed.utils import get_batch_on_this_cp_rank
+# from mindspeed.utils import get_batch_on_this_cp_rank
 
 from megatron.core import mpu
 from megatron.core.enums import ModelType""","""         else:
             batch = None
 
-        input_ids = batch['input_ids'].to(torch.cuda.current_device())
-        labels = batch['labels'].to(torch.cuda.current_device())
-        attention_mask = batch['attention_mask'].to(torch.cuda.current_device())
-        image = batch['pixel_values'].to(torch.cuda.current_device())
-        image_flags = batch['image_flags'].to(torch.cuda.current_device())
+        input_ids = batch['input_ids']
+        labels = batch['labels']
+        attention_mask = batch['attention_mask']
+        image = batch['pixel_values']
+        image_flags = batch['image_flags']
         _broadcast(input_ids)
         _broadcast(labels)
         _broadcast(attention_mask)""","""         batch = next(data_iterator)
     else:
         raise ValueError("Data iterator is None. Unable to retrieve batch.")
-    input_ids = batch['input_ids'].to(torch.cuda.current_device())
-    labels = batch['labels'].to(torch.cuda.current_device())
-    attention_mask = batch['attention_mask'].to(torch.cuda.current_device())
-    image = batch['pixel_values'].to(torch.cuda.current_device())
-    image_flags = batch['image_flags'].to(torch.cuda.current_device())
+    input_ids = batch['input_ids']
+    labels = batch['labels']
+    attention_mask = batch['attention_mask']
+    image = batch['pixel_values']
+    image_flags = batch['image_flags']
     batch = {
         'input_ids': input_ids,
         'labels': labels,
"""
],
"pretrain_llava.py": ["""         data = next(data_iterator)
     else:
         data = None
-    images = data["pixel_values"].to(dtype=torch.bfloat16, device=torch.cuda.current_device())
-    input_ids = data["input_ids"].to(device=torch.cuda.current_device())
-    labels = data["labels"].to(device=torch.cuda.current_device())
-    attention_mask = data["attention_mask"].to(device=torch.cuda.current_device())
+    images = data["pixel_values"]
+    input_ids = data["input_ids"]
+    labels = data["labels"]
+    attention_mask = data["attention_mask"]
 
     return images, input_ids, labels, attention_mask
 
"""
],
"pretrain_qwen2vl.py": ["""         batch = next(data_iterator)
     else:
         raise ValueError("Data iterator is None. Unable to retrieve batch.")
-    input_ids = batch['input_ids'].to(torch.cuda.current_device())
-    labels = batch['labels'].to(torch.cuda.current_device())
-    attention_mask = batch['attention_mask'].to(torch.cuda.current_device())
-    pixel_values = batch['pixel_values'].to(torch.cuda.current_device())
-    image_grid_thw = batch['image_grid_thw'].to(torch.cuda.current_device())
+    input_ids = batch['input_ids']#.to(torch.cuda.current_device())
+    labels = batch['labels']#.to(torch.cuda.current_device())
+    attention_mask = batch['attention_mask']#.to(torch.cuda.current_device())
+    pixel_values = batch['pixel_values']#.to(torch.cuda.current_device())
+    image_grid_thw = batch['image_grid_thw']#.to(torch.cuda.current_device())
     batch = {
         'input_ids': input_ids,
         'labels': labels,
"""
],
"pretrain_sora.py": ["""         batch = next(data_iterator)
     else:
         return None
-    for k, v in batch.items():
-        if isinstance(v, torch.Tensor):
-            batch[k] = v.to(torch.cuda.current_device())
+    # for k, v in batch.items():
+    #     if isinstance(v, torch.Tensor):
+    #         batch[k] = v.to(torch.cuda.current_device())
     return batch
"""
],

}


LINE_RULES_transformers = {
"src/transformers/cache_utils.py": ["""         axis_value: Optional[int] = 0,
         q_group_size: Optional[int] = 64,
         residual_length: Optional[int] = 128,
-        compute_dtype: Optional[torch.dtype] = torch.float16,
+        compute_dtype: Optional[int] = torch.float16,
         device: Optional[str] = "cpu",
     ):
         self.backend = backend""","""         max_batch_size: int,
         max_cache_len: Optional[int],
         device: Union[str, torch.device],
-        dtype: Optional[torch.dtype] = None,
+        dtype: Optional[int] = None,
         offload_device: Union[str, torch.device] = torch.device("cpu"),
         layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
     ) -> None:
"""
],
"src/transformers/configuration_utils.py": ["""         converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
         string, which can then be stored in the json format.
         \"\"\"
+        # if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
+        #     d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
         if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
-            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
+            d["torch_dtype"] = str(d["torch_dtype"])
         for value in d.values():
             if isinstance(value, dict):
                 self.dict_torch_dtype_to_str(value)
"""
],
"src/transformers/modeling_utils.py": [""" from packaging import version
 from torch import Tensor, nn
 from torch.nn import CrossEntropyLoss, Identity
-from torch.utils.checkpoint import checkpoint
+# from torch.utils.checkpoint import checkpoint
 
 from .activations import get_activation
 from .configuration_utils import PretrainedConfig
 from .dynamic_module_utils import custom_object_save
 from .generation import CompileConfig, GenerationConfig, GenerationMixin
 from .integrations import PeftAdapterMixin, deepspeed_config, is_deepspeed_zero3_enabled
-from .loss.loss_utils import LOSS_MAPPING
+# from .loss.loss_utils import LOSS_MAPPING
 from .pytorch_utils import (  # noqa: F401
     Conv1D,
     apply_chunking_to_forward,""",""" 
     Note: We fully disable this if we are using `deepspeed`
     \"\"\"
-    if model_to_load.device.type == "meta":
-        return False
+    # if model_to_load.device.type == "meta":
+    #     return False
 
-    if len([key for key in state_dict if key.startswith(start_prefix)]) == 0:
-        return False
+    # if len([key for key in state_dict if key.startswith(start_prefix)]) == 0:
+    #     return False
 
-    if is_deepspeed_zero3_enabled():
-        return False
+    # if is_deepspeed_zero3_enabled():
+    #     return False
 
-    # Some models explicitly do not support param buffer assignment
-    if not getattr(model_to_load, "_supports_param_buffer_assignment", True):
-        logger.debug(
-            f"{model_to_load.__class__.__name__} does not support param buffer assignment, loading will be slower"
-        )
-        return False
+    # # Some models explicitly do not support param buffer assignment
+    # if not getattr(model_to_load, "_supports_param_buffer_assignment", True):
+    #     logger.debug(
+    #         f"{model_to_load.__class__.__name__} does not support param buffer assignment, loading will be slower"
+    #     )
+    #     return False
 
-    # If the model does, the incoming `state_dict` and the `model_to_load` must be the same dtype
-    first_key = next(iter(model_to_load.state_dict().keys()))
-    if start_prefix + first_key in state_dict:
-        return state_dict[start_prefix + first_key].dtype == model_to_load.state_dict()[first_key].dtype
+    # # If the model does, the incoming `state_dict` and the `model_to_load` must be the same dtype
+    # first_key = next(iter(model_to_load.state_dict().keys()))
+    # if start_prefix + first_key in state_dict:
+    #     return state_dict[start_prefix + first_key].dtype == model_to_load.state_dict()[first_key].dtype
 
     # For cases when the `state_dict` doesn't contain real weights to the model (`test_model_weights_reload_no_missing_tied_weights`)
     return False""","""     \"\"\"
     if checkpoint_file.endswith(".safetensors") and is_safetensors_available():
         # Check format of the archive
-        with safe_open(checkpoint_file, framework="pt") as f:
-            metadata = f.metadata()
-        if metadata.get("format") not in ["pt", "tf", "flax", "mlx"]:
-            raise OSError(
-                f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
-                "you save your model with the `save_pretrained` method."
-            )
+
+        # with safe_open(checkpoint_file, framework="pt") as f:
+        #     metadata = f.metadata()
+        # if metadata.get("format") not in ["pt", "tf", "flax", "mlx"]:
+        #     raise OSError(
+        #         f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
+        #         "you save your model with the `save_pretrained` method."
+        #     )
+        from torch.serialization import safe_load_file
         return safe_load_file(checkpoint_file)
     try:
         if map_location is None:""","""         cls,
         config,
         use_flash_attention_2: bool = False,
-        torch_dtype: Optional[torch.dtype] = None,
+        torch_dtype: Optional[int] = None,
         device_map: Optional[Union[str, Dict[str, int]]] = None,
         check_device_map: bool = True,
     ):""","""                 hard_check_only=False if requested_attn_implementation is None else True,
             )
 
-            if (
-                torch.version.hip is not None
-                and config._attn_implementation == "sdpa"
-                and torch.cuda.device_count() > 1
-            ):
-                logger.warning_once(
-                    "Using the `SDPA` attention implementation on multi-gpu setup with ROCM may lead to performance issues due to the FA backend. Disabling it to use alternative backends."
-                )
-                torch.backends.cuda.enable_flash_sdp(False)
+            # if (
+            #     torch.version.hip is not None
+            #     and config._attn_implementation == "sdpa"
+            #     and torch.cuda.device_count() > 1
+            # ):
+            #     logger.warning_once(
+            #         "Using the `SDPA` attention implementation on multi-gpu setup with ROCM may lead to performance issues due to the FA backend. Disabling it to use alternative backends."
+            #     )
+            #     torch.backends.cuda.enable_flash_sdp(False)
         elif isinstance(requested_attn_implementation, dict):
             config._attn_implementation = None
         else:""","""         Note `set_default_dtype` currently only works with floating-point types and asserts if for example,
         `torch.int64` is passed. So if a non-float `dtype` is passed this functions will throw an exception.
         \"\"\"
-        if not dtype.is_floating_point:
-            raise ValueError(
-                f"Can't instantiate {cls.__name__} model under dtype={dtype} since it is not a floating point dtype"
-            )
+        # if not dtype.is_floating_point:
+        #     raise ValueError(
+        #         f"Can't instantiate {cls.__name__} model under dtype={dtype} since it is not a floating point dtype"
+        #     )
 
         logger.info(f"Instantiating {cls.__name__} model under default dtype {dtype}.")
         dtype_orig = torch.get_default_dtype()""","""     def _check_and_enable_flash_attn_2(
         cls,
         config,
-        torch_dtype: Optional[torch.dtype] = None,
+        torch_dtype: Optional[int] = None,
         device_map: Optional[Union[str, Dict[str, int]]] = None,
         check_device_map: bool = True,
         hard_check_only: bool = False,""","""             # the gradients to make sure the gradient flows.
             self.enable_input_require_grads()
 
-    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func: Callable = checkpoint):
+    def _set_gradient_checkpointing(self, enable, gradient_checkpointing_func): # Callable = checkpoint):
         is_gradient_checkpointing_set = False
 
         # Apply it on the top-level module in case the top-level modules supports it""","""         if device_map is None and not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():
             ptrs = collections.defaultdict(list)
             for name, tensor in model.state_dict().items():
-                id_tensor = id_tensor_storage(tensor)
+                # id_tensor = id_tensor_storage(tensor)
+                id_tensor = id(tensor)
                 ptrs[id_tensor].append(name)
 
             # These are all the pointers of shared tensors.""","""                         hf_quantizer.create_quantized_param(model, value, key, "cpu", state_dict, unexpected_keys)
 
         # retrieve uninitialized modules and initialize before maybe overriding that with the pretrained weights.
+        _fast_init = False
         if _fast_init:
             if not ignore_mismatched_sizes:
                 if remove_prefix_from_model:
"""
],
"src/transformers/models/t5/modeling_t5.py": ["""         # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
         # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
         # half-precision inputs is done in fp32
-
         variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
 
         # convert into half-precision if necessary
         if self.weight.dtype in [torch.float16, torch.bfloat16]:
             hidden_states = hidden_states.to(self.weight.dtype)
-
         return self.weight * hidden_states
 
 ""","""             / math.log(max_distance / max_exact)
             * (num_buckets - max_exact)
         ).to(torch.long)
-        relative_position_if_large = torch.min(
-            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
-        )
+        relative_position_buckets = torch.full_like(relative_position_if_large, num_buckets - 1)
+        mask = (relative_position_if_large - relative_position_buckets) < 0
+        mask = mask.astype(relative_position_if_large.dtype)
+        relative_position_if_large = relative_position_if_large * mask + relative_position_buckets * (1 - mask)
+        # relative_position_if_large = torch.min(
+        #     relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
+        # )
 
         relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
         return relative_buckets""","""         if cache_position is None:
             context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
         else:
-            context_position = cache_position[:, None].to(device)
-        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
+            context_position = cache_position[:, None]
+        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
         relative_position = memory_position - context_position  # shape (query_length, key_length)
         relative_position_bucket = self._relative_position_bucket(
             relative_position,  # shape (query_length, key_length)""","""         )
         values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
         values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
+        
         return values
 
     def forward(""","""         )
         hidden_states, past_key_value = self_attention_outputs[:2]
         attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights
-
         # clamp inf values to enable fp16 training
         if hidden_states.dtype == torch.float16:
             clamp_value = torch.where(""",""" 
         # Apply Feed Forward layer
         hidden_states = self.layer[-1](hidden_states)
-
         # clamp inf values to enable fp16 training
         if hidden_states.dtype == torch.float16:
             clamp_value = torch.where(""",""" class T5Stack(T5PreTrainedModel):
     def __init__(self, config, embed_tokens=None):
         super().__init__(config)
-
+        # config.num_layers = 1
         self.embed_tokens = embed_tokens
         self.is_decoder = config.is_decoder
 ""","""         elif attention_mask is not None:
             causal_mask = attention_mask[:, None, None, :]
             causal_mask = causal_mask.to(dtype=inputs_embeds.dtype)
-            causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min
+            if inputs_embeds.dtype == torch.bfloat16:
+                finfo = -3.38953e+38
+            else:
+                finfo = torch.finfo(inputs_embeds.dtype).min
+            causal_mask = (1.0 - causal_mask) * finfo
+            # causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min
         else:
             causal_mask = None
 ""","""         encoder_decoder_position_bias = None
 
         hidden_states = self.dropout(inputs_embeds)
-
         for i, layer_module in enumerate(self.block):
             layer_head_mask = head_mask[i]
             cross_attn_layer_head_mask = cross_attn_head_mask[i]
"""
],
"src/transformers/pytorch_utils.py": ["""     guaranteed to be unique and constant for this tensor's storage during its lifetime. Two tensor storages with
     non-overlapping lifetimes may have the same id.
     \"\"\"
-    if tensor.device.type == "xla" and is_torch_xla_available():
-        # NOTE: xla tensors dont have storage
-        # use some other unique id to distinguish.
-        # this is a XLA tensor, it must be created using torch_xla's
-        # device. So the following import is safe:
-        import torch_xla
-
-        unique_id = torch_xla._XLAC._xla_get_tensor_id(tensor)
-    else:
-        unique_id = storage_ptr(tensor)
+    # if tensor.device.type == "xla" and is_torch_xla_available():
+    #     # NOTE: xla tensors dont have storage
+    #     # use some other unique id to distinguish.
+    #     # this is a XLA tensor, it must be created using torch_xla's
+    #     # device. So the following import is safe:
+    #     import torch_xla
+
+    #     unique_id = torch_xla._XLAC._xla_get_tensor_id(tensor)
+    # else:
+    unique_id = storage_ptr(tensor)
 
     return tensor.device, unique_id, storage_size(tensor)
 
"""
],
"src/transformers/utils/import_utils.py": [""" _liger_kernel_available = _is_package_available("liger_kernel")
 
 
-_torch_version = "N/A"
-_torch_available = False
-if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
-    _torch_available, _torch_version = _is_package_available("torch", return_version=True)
-else:
-    logger.info("Disabling PyTorch because USE_TF is set")
-    _torch_available = False
+_torch_version = "2.1.0"
+_torch_available = True
+# _torch_available = False
+# if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
+#     _torch_available, _torch_version = _is_package_available("torch", return_version=True)
+# else:
+#     logger.info("Disabling PyTorch because USE_TF is set")
+#     _torch_available = False
 
 
 _tf_version = "N/A"
"""
],

}


LINE_RULES_megatron = {
"megatron/core/datasets/blended_megatron_dataset_builder.py": ["""                 prefix_per_dataset,
                 weight_per_dataset,
                 sizes_per_dataset,
-            ) = _get_prefixes_weights_and_sizes_for_blend(blend, self.sizes)
+            ) = _get_prefixes_weights_and_sizes_for_blend(blend, self.sizes, self.config.dataset_margin)
 
             megatron_datasets = [[] for _ in range(len(Split))]
 ""","""                         prefix_per_dataset,
                         weight_per_dataset,
                         sizes_per_dataset,
-                    ) = _get_prefixes_weights_and_sizes_for_blend(blend, sizes_spoof)
+                    ) = _get_prefixes_weights_and_sizes_for_blend(blend, sizes_spoof, self.config.dataset_margin)
 
                     megatron_datasets = []
                     for j in range(len(prefix_per_dataset)):""",""" 
 
 def _get_prefixes_weights_and_sizes_for_blend(
-    blend: List[str], target_num_samples_per_split: List[int]
+    blend: List[str], target_num_samples_per_split: List[int], dataset_margin: float
 ) -> Tuple[List[str], List[float], List[List[int]]]:
     \"\"\"Determine the contribution of the MegatronDataset splits to the BlendedDataset splits
     ""","""     # Use 0.5% target margin to ensure we satiate the network
     sizes_per_dataset = [
         [
-            int(math.ceil(target_num_samples * weight * 1.005))
+            int(math.ceil(target_num_samples * weight * dataset_margin))
             for target_num_samples in target_num_samples_per_split
         ]
         for weight in weights
"""
],
"megatron/core/datasets/gpt_dataset.py": ["""        generates masks by itself.
     \"\"\"
 
+    dataset_margin: float = 1.005
+    \"\"\"Option to use 0.5% target margin to ensure we satiate the network \"\"\"
+
     def __post_init__(self) -> None:
         \"\"\"Do asserts and set fields post init
         \"\"\"
"""
],
"megatron/core/dist_checkpointing/serialization.py": ["""         )
         merge(common_state_dict, sharded_objects)
     sharded_state_dict, _ = extract_sharded_base(sharded_state_dict)
-
     if validate_access_integrity:
         validate_sharding_integrity(nested_values(sharded_state_dict))
 
"""
],
"megatron/core/dist_checkpointing/strategies/state_dict_saver.py": ["""     Returns: None
     \"\"\"
     write_results = storage_writer.retrieve_write_results()
-
     # Gather the write results that will be saved to the metadata file.
     gather_start = time()
     all_results = dist_wrapper.gather_object(write_results)
"""
],
"megatron/core/distributed/distributed_data_parallel.py": [""" from contextlib import contextmanager
 from typing import Dict, Optional
 
+import os
 import torch
 
 from .. import parallel_state""","""             data_parallel_group,
             gradient_scaling_factor=1.0 / data_parallel_world_size,
         )
-
         # Allocate separate param+grad buffers for expert parallel params' grads.
         self.expert_parallel_buffers = allocate_buffers_for_parameters(
             expert_parallel_params,""","""         # Register backward hook.
         # Accumulation function for the gradients need to be stored so they
         # don't go out of scope.
-        self.grad_accs = []
+        # self.grad_accs = []
         for param in self.module.parameters():
             if param.requires_grad:
                 # Expand so we get access to grad_fn.
-                param_tmp = param.expand_as(param)
+                # param_tmp = param.expand_as(param)
                 # Get the gradient accumulator function.
-                grad_acc = param_tmp.grad_fn.next_functions[0][0]
-                grad_acc.register_hook(self._make_param_hook(param, self.param_to_buffer))
-                self.grad_accs.append(grad_acc)
+                # grad_acc = param_tmp.grad_fn.next_functions[0][0]
+                # grad_acc.register_hook(self._make_param_hook(param, self.param_to_buffer))
+                # self.grad_accs.append(grad_acc)
+                param.register_hook(self._make_param_hook(param, self.param_to_buffer))
 
     def forward(self, *inputs, **kwargs):
         \"\"\"""","""         Creates the all-reduce / reduce-scatter hook for backprop.
         \"\"\"

-        def param_hook(*unused):
+        def param_hook(grad):
             if param.requires_grad:
-                if self.overlap_grad_reduce:
-                    assert (
-                        param.grad is not None
-                    ), 'param.grad being None is not safe when overlap_grad_reduce is True'
+                # if self.overlap_grad_reduce:
+                #     assert (
+                #         param.grad is not None
+                #     ), 'param.grad being None is not safe when overlap_grad_reduce is True'
                 if param.grad is not None and (
                     not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
                 ):
-                    param.main_grad.add_(param.grad.data)
+                    param.main_grad.add_(grad)
+
                 param.grad = None

                 if self.overlap_grad_reduce:
                     param_to_buffer[param].register_grad_ready(param)
-
+                if hasattr(param, "main_grad"):
+                    return param.main_grad
+                return param.grad
         return param_hook

     @contextmanager
"""
],
"megatron/core/distributed/finalize_model_grads.py": [""" # Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 
-from typing import List
-
 import torch
 from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
+from typing import List
 
 from .. import parallel_state
 from ..transformer.transformer_config import TransformerConfig""","""     \"\"\"
 
     if (
-        parallel_state.is_rank_in_embedding_group(ignore_virtual=True)
-        and parallel_state.get_pipeline_model_parallel_world_size() > 1
+            parallel_state.is_rank_in_embedding_group(ignore_virtual=True)
+            and parallel_state.get_pipeline_model_parallel_world_size() > 1
     ):
         if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
             model_module = model[0]""","""         # other wrapper classes inherit from non-core MegatronModule that has
         # 'share_embeddings_and_output_weights' and 'shared_embedding_or_output_weight'
         # attributes already, causing get_attr_wrapped_model() to not unwrap anything here.
-        # TODO: Clean this up once the wrapper classes inherit from core MegatronModule.
         model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
         if model_module.share_embeddings_and_output_weights:
             weight = model_module.shared_embedding_or_output_weight()""","""     with pipeline parallelism.
     \"\"\"
     if (
-        parallel_state.is_rank_in_position_embedding_group()
-        and parallel_state.get_pipeline_model_parallel_world_size() > 1
-        and config.pipeline_model_parallel_split_rank is not None
+            parallel_state.is_rank_in_position_embedding_group()
+            and parallel_state.get_pipeline_model_parallel_world_size() > 1
+            and config.pipeline_model_parallel_split_rank is not None
+    ):
+        model_module = model[0]
+        grad = get_attr_wrapped_model(
+            model_module, 'language_model.embedding.position_embeddings.weight.main_grad'
+        )
+        torch.distributed.all_reduce(grad, group=parallel_state.get_position_embedding_group())
+
+
+def _allreduce_word_embedding_grads_mm(model: List[torch.nn.Module], config: TransformerConfig):
+    \"\"\"
+    All-reduce word embedding grads.
+
+    Reduce grads across first and last stages to ensure that word_embeddings parameters stay in
+    sync. This should only run for models that support pipelined model parallelism (BERT and GPT).
+    \"\"\"
+
+    if (
+            parallel_state._EMBEDDING_GLOBAL_RANKS
+            and parallel_state.is_rank_in_embedding_group(ignore_virtual=True)
+            and parallel_state.get_pipeline_model_parallel_world_size() > 1
+    ):
+        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
+            model_module = model[0]
+        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
+            model_module = model[-1]
+        else:  # We do not support the interleaved schedule for T5 yet.
+            model_module = model[0]
+
+        # Look for module with 'pre_process' attribute to get around the fact that DDP and
+        # other wrapper classes inherit from non-core MegatronModule that has
+        # 'share_embeddings_and_output_weights' and 'shared_embedding_or_output_weight'
+        # attributes already, causing get_attr_wrapped_model() to not unwrap anything here.
+        if hasattr(model_module.module, "language_model"):
+            model_module = model_module.module.language_model
+            if (model_module.pre_process or model_module.post_process) \\
+                    and model_module.share_embeddings_and_output_weights:
+                weight = model_module.shared_embedding_or_output_weight()
+                grad = weight.main_grad
+                torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())
+
+
+def _allreduce_position_embedding_grads_mm(model: List[torch.nn.Module], config: TransformerConfig):
+    \"\"\"
+    All-reduce position_embeddings grad across first (encoder) and split (decoder) stages to
+    ensure that position embeddings parameters stay in sync. This should only run for T5 models
+    with pipeline parallelism.
+    \"\"\"
+    is_parallel_state = parallel_state._POSITION_EMBEDDING_GLOBAL_RANKS \\
+                        and parallel_state.is_rank_in_position_embedding_group() \\
+                        and parallel_state.get_pipeline_model_parallel_world_size() > 1
+    is_split_rank = config.pipeline_model_parallel_split_rank is not None
+    if (
+         is_parallel_state and is_split_rank
     ):
         model_module = model[0]
         grad = get_attr_wrapped_model(""","""     \"\"\"
     All-reduce both word and position embeddings.
     \"\"\"
-    _allreduce_word_embedding_grads(model, config)
-    _allreduce_position_embedding_grads(model, config)
+    if not config.multimodal:
+        _allreduce_word_embedding_grads(model, config)
+        _allreduce_position_embedding_grads(model, config)
+    else:
+        _allreduce_word_embedding_grads_mm(model, config)
+        _allreduce_position_embedding_grads_mm(model, config)
 
 
 def _allreduce_layernorm_grads(model: List[torch.nn.Module], config: TransformerConfig):""","""     # All-reduce layernorm parameters across model parallel nodes
     # when sequence parallelism is used
     if parallel_state.get_tensor_model_parallel_world_size() > 1 and (
-        config.sequence_parallel or config.qk_layernorm
+            config.sequence_parallel or config.qk_layernorm
     ):
         grads = []
         for model_chunk in model:
             for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
                 if (
-                    getattr(param, 'sequence_parallel', False)
-                    or 'q_layernorm' in name
-                    or 'k_layernorm' in name
+                        param.requires_grad
+                        and getattr(param, 'sequence_parallel', False)
+                        or 'q_layernorm' in name
+                        or 'k_layernorm' in name
                 ):
                     grad = param.main_grad
                     grads.append(grad.data)""","""                 buf.copy_(synced)
 
 
+def _allreduce_duplicate_grads(model: List[torch.nn.Module], config: TransformerConfig):
+    \"\"\"
+    All-reduce duplicate param grads .
+    \"\"\"
+
+    # All-reduce duplicate parameters across model parallel nodes
+    grads = []
+    for model_chunk in model:
+        for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
+            if "vision_model.Qformer" in name or \\
+                    "vision_model.query_tokens" in name or \\
+                    "vision_model.norm" in name or \\
+                    "projection" in name or \\
+                    "vision_model.c_abstractor" in name:
+                grad = param.main_grad
+                grads.append(grad.data)
+    if grads:
+        data_tensor_parallel_world_size = parallel_state.get_tensor_model_parallel_world_size()
+        for item in grads:
+            item /= data_tensor_parallel_world_size
+            torch.distributed.all_reduce(item, group=parallel_state.get_tensor_model_parallel_group())
+
+
 def finalize_model_grads(model: List[torch.nn.Module]):
     \"\"\"
     All-reduce all model grads across DP replicas, layernorm grads for sequence parallelism,""","""     _allreduce_embedding_grads(model, config)
     if config.timers is not None:
         config.timers('embedding-grads-all-reduce').stop()
+
+    # For Multimodal: all-reduce duplicate grads if needed.
+    if config.timers is not None:
+        config.timers('duplicate-grads-all-reduce', log_level=1).start(
+            barrier=config.barrier_with_L1_time)
+    _allreduce_duplicate_grads(model, config)
+    if config.timers is not None:
+        config.timers('duplicate-grads-all-reduce').stop()
"""
],
"megatron/core/distributed/param_and_grad_buffer.py": ["""                 f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
             )

-        self.grad_data *= self.gradient_scaling_factor
+        # self.grad_data *= self.gradient_scaling_factor
+        # self.grad_data.copy_(self.grad_data * self.gradient_scaling_factor)
+        self.grad_data.div_(1 / self.gradient_scaling_factor)
         # Use async_op only when overlap_grad_reduce is True.
         if self.use_distributed_optimizer:
             local_data_view = shard_buffer(self.grad_data, self.data_parallel_world_size)[""","""         gradient_scaling_factor: float,
         check_for_nan_in_grad: bool,
     ):
-
+        self.param_index_map_full = {}
         # Check that params are unique.
         unique_params = set()
         for param in params:""","""                     # data_start_index should already be padded.
                     assert data_start_index % self.data_parallel_world_size == 0
                 _create_new_bucket(data_start_index)
-
             self.param_index_map[param] = (
                 data_start_index,
                 data_end_index,
"""
],
"megatron/core/fusions/fused_bias_dropout.py": [""" import torch
 
 from megatron.core.jit import jit_fuser
+from megatron.training import get_args
 
 
 def _bias_dropout_add_func(x_with_bias, residual, prob, training):""","""     # in fp32, and it will up-cast the result to fp32, causing pipeline parallel
     # GPU communication to hang. Therefore, we need to cast residual to the same
     # dtype as x.
-    residual = residual if residual.dtype == x.dtype else residual.to(x.dtype)
+
+    args = get_args()
+    if not args.fp32_residual_connection:
+        residual = residual if residual.dtype == x.dtype else residual.to(x.dtype)
 
     # The Dropout operation, Residual Addition and the tensor returning can be
     # done generically outside the if statement, but that stops fusing of Bias
"""
],
"megatron/core/model_parallel_config.py": ["""     params_dtype: torch.dtype = torch.float32
     \"\"\"dtype used when intializing the weights.\"\"\"
 
+    embedding_dtype: torch.dtype = torch.float32
+    \"\"\"dtype used when intializing the embedding weights.\"\"\"
+ 
     timers: Callable = None
     \"\"\"Timers object to call for various timing functions. See megatron.core.timers.Timers\"\"\"
 
"""
],
"megatron/core/models/common/embeddings/language_model_embedding.py": [""" from megatron.core import tensor_parallel
 from megatron.core.transformer.module import MegatronModule
 from megatron.core.transformer.transformer_config import TransformerConfig
-
+from megatron.training.global_vars import get_args
 
 class LanguageModelEmbedding(MegatronModule):
     \"\"\"Language model embeddings.""","""         num_tokentypes: int = 0,
     ):
         super().__init__(config=config)
-
+        self.args = get_args()
         self.config: TransformerConfig = config
         self.vocab_size: int = vocab_size
         self.max_sequence_length: int = max_sequence_length""","""             embeddings = embeddings.float()
 
         # Dropout.
-        if self.config.sequence_parallel:
+        if self.config.sequence_parallel and not self.args.multimodal:
             embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
             # `scatter_to_sequence_parallel_region` returns a view, which prevents
             # the original tensor from being garbage collected. Clone to facilitate GC.
"""
],
"megatron/core/models/common/embeddings/rotary_pos_embedding.py": [""" from torch import Tensor, nn
 
 from megatron.core import parallel_state
+import numpy as np
 
 logger = logging.getLogger(__name__)
 ""","""     HAVE_APPLY_ROPE_FUSION = False
 
 
-__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']
-
+__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb', 'RoPEClassic']
+
+_ROTATION_MATRIX = None
+def get_rotation_matrix(x):
+    global _ROTATION_MATRIX
+    if _ROTATION_MATRIX is None:
+        import numpy as np
+        dim = x.shape[-1]
+        index1 = np.ones(dim)
+        index1[::2] = 0
+        index2 = np.zeros(dim)
+        index2[::2] = -1
+        rotation_matrix = np.eye(dim, k=1) * index1 + np.eye(dim, k=-1) * index2
+        _ROTATION_MATRIX = (
+            torch.from_numpy(rotation_matrix[None, None, :, :]).to(x.dtype).to(x.device)
+        )
+    return _ROTATION_MATRIX
 
 def get_pos_emb_on_this_cp_rank(pos_emb, seq_dim):
     cp_size = parallel_state.get_context_parallel_world_size()""","""     pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :])
     return pos_emb
 
+class RoPEClassic(nn.Module):
+    __cos_encoding = None
+    __sin_encoding = None
+    __rotation_matrix = None
+
+    def __init__(self, kv_channels, max_seq_len, dtype, base=10000.0):
+        super().__init__()
+        self.dim = kv_channels
+        self.max_seq_len = max_seq_len
+        exponent = torch.floor(
+            torch.arange(0, self.dim, dtype=torch.float32).to(torch.npu.current_device()) / 2.
+        ) * 2. / self.dim
+        self.theta = 1.0 / (base ** exponent)#.float()
+        if self.__cos_encoding is None:
+            self._set_cos_sin_encoding(max_seq_len, dtype)
+        if self.__rotation_matrix is None:
+            self._set_rotation_matrix(dtype)
+
+    def _set_cos_sin_encoding(self, max_seq_len, dtype):
+        self.max_seq_len = max_seq_len
+        position_idx = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.theta.device)
+        _encoding = torch.outer(position_idx, self.theta)
+        RoPEClassic.__cos_encoding = _encoding.cos()[:, None, None, :].to(dtype)
+        RoPEClassic.__sin_encoding = _encoding.sin()[:, None, None, :].to(dtype)
+
+    def _set_rotation_matrix(self, dtype):
+        index1 = np.ones(self.dim)
+        index1[::2] = 0
+        index2 = np.zeros(self.dim)
+        index2[::2] = -1
+        rotation_matrix = np.eye(self.dim, k=1) * index1 + np.eye(self.dim, k=-1) * index2
+        RoPEClassic.__rotation_matrix = (
+            torch.from_numpy(rotation_matrix[None, None, :, :]).to(torch.float32).to(self.theta.device)
+        )
+    # @staticmethod
+    # def rotate_half(x):
+    #     x1 = x[..., ::2]
+    #     x2 = x[..., 1::2]
+    #     return torch.cat((-x2[..., None], x1[..., None]), dim=-1).reshape(x.shape)
+
+    def rotate_half(self, x):
+        # [s, b, n_attn_head/tp, dim]
+        return torch.matmul(x, self.__rotation_matrix.to(x.dtype))
+
+    def forward(self, x, offset=0):
+        seq_len = x.size(0) + offset
+        if seq_len > self.max_seq_len:
+            # [hmhm] add warning of over max_seq_len here
+            self._set_cos_sin_encoding(seq_len, x.dtype)
+        _cos_encoding = self.__cos_encoding[offset: seq_len, ...].to(x.dtype)
+        _sin_encoding = self.__sin_encoding[offset: seq_len, ...].to(x.dtype)
+
+        return x * _cos_encoding + self.rotate_half(x) * _sin_encoding
 
 class RotaryEmbedding(nn.Module):
     \"\"\"Rotary Embedding for language model.""","""         if inference_params is not None:
             rotary_seq_len = inference_params.max_sequence_length
         else:
-            if transformer.input_tensor is not None:
+            if transformer.input_tensor is not None and len(transformer.input_tensor.size()) > 0:
                 rotary_seq_len = transformer.input_tensor.size(0)
             else:
                 rotary_seq_len = transformer_input.size(0)""","""         x1, x2 = torch.chunk(x, 2, dim=-1)
         return torch.cat((-x2, x1), dim=-1)
     else:
-        x1 = x[:, :, :, ::2]
-        x2 = x[:, :, :, 1::2]
-        x_new = torch.stack((-x2, x1), dim=-1)
-        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)
+        return torch.matmul(x, get_rotation_matrix(x))
 
 
 def apply_rotary_pos_emb_bshd(t: Tensor, freqs: Tensor, rotary_interleaved: bool = False) -> Tensor:""","""     if config.apply_rope_fusion and not HAVE_APPLY_ROPE_FUSION:
         # setting apply_rope_fusion in config to False so that subsequent queries to this config also return False
         config.apply_rope_fusion = False
-        if not getattr(apply_rotary_pos_emb, "printed_fused_warning", False):
-            logger.warning(
-                "Setting apply_rope_fusion to false because its implementation"
-                " is not included in Apex. Try upgrading to the latest version"
-            )
-            apply_rotary_pos_emb.printed_fused_warning = True
+        # if not getattr(apply_rotary_pos_emb, "printed_fused_warning", False):
+        #     logger.warning(
+        #         "Setting apply_rope_fusion to false because its implementation"
+        #         " is not included in Apex. Try upgrading to the latest version"
+        #     )
+        #     apply_rotary_pos_emb.printed_fused_warning = True
     if config.apply_rope_fusion:
         if cu_seqlens is None:
             return fused_apply_rotary_pos_emb(t, freqs, transpose_output_memory=True)
"""
],
"megatron/core/models/common/language_module/language_module.py": [""" from megatron.core.transformer.module import MegatronModule
 from megatron.core.transformer.transformer_config import TransformerConfig
 from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint
+from megatron.training import get_args
 
 
 class LanguageModule(MegatronModule):""","""             return
 
         if self.pre_process and not self.post_process:
-            assert parallel_state.is_pipeline_first_stage()
             self.shared_embedding_or_output_weight().shared_embedding = True
 
         if self.post_process and not self.pre_process:
-            assert not parallel_state.is_pipeline_first_stage()
             # set word_embeddings weights to 0 here, then copy first
             # stage's weights using all_reduce below.
             self.output_layer.weight.data.fill_(0)""","""         if torch.distributed.is_initialized():
             if parallel_state.is_rank_in_embedding_group():
                 weight = self.shared_embedding_or_output_weight()
-                weight.data = weight.data.cuda()
+                args = get_args()
+                weight.data = weight.data.cuda().to(args.embedding_dtype)
                 torch.distributed.all_reduce(
                     weight.data, group=parallel_state.get_embedding_group()
                 )
"""
],
"megatron/core/models/gpt/gpt_layer_specs.py": ["""             submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear,)
             if not moe_grouped_gemm
             else None,
-        )
+        )
\\ No newline at end of file
"""
],
"megatron/core/optimizer/__init__.py": [""" from apex.optimizers import FusedAdam as Adam
 from apex.optimizers import FusedSGD as SGD
 
+# from torch.optim import AdamW
 from megatron.core import mpu
 
 from ..distributed import ParamAndGradBuffer""","""                 scale_lr = False
 
             if not no_wd and not scale_lr:
-                wd_mult, lr_mult = 1.0, 1.0
+                wd_mult, _lr_mult = 1.0, 1.0
             elif not no_wd and scale_lr:
-                wd_mult, lr_mult = 1.0, lr_mult
+                wd_mult, _lr_mult = 1.0, lr_mult
             elif no_wd and not scale_lr:
-                wd_mult, lr_mult = 0.0, 1.0
+                wd_mult, _lr_mult = 0.0, 1.0
             else:
-                wd_mult, lr_mult = 0.0, lr_mult
+                wd_mult, _lr_mult = 0.0, lr_mult
 
             is_decoupled_lr = False
             # For input/embedding and output layer: embedding.word_embeddings.weight / output_layer.weight.""","""             ):
                 is_decoupled_lr = True
 
-            key = (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr)
+            key = (wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr)
             if key not in params_map:
                 params_map[key] = []
             params_map[key].append(param)
 
     param_groups = []
-    for (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr), params in params_map.items():
+    for (wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr), params in params_map.items():
         assert len(params) > 0
         param_groups.append(
             {
                 'params': params,
                 'wd_mult': wd_mult,
-                'lr_mult': lr_mult,
+                'lr_mult': _lr_mult,
                 'is_expert_parallel': is_expert_parallel,
                 'is_decoupled_lr': is_decoupled_lr,
             }""","""         decoupled_min_lr=config.decoupled_min_lr,
     )
 
+    # Fake params to construct optmizer
+    if len(param_groups) == 0:
+        # device = next(model_chunks[0].parameters()).device
+        fake_params = torch.zeros([1,], dtype=torch.float, requires_grad=True)
+        fake_params.fake = True
+        fake_params.grad = fake_params.clone()
+        fake_params.main_grad = fake_params.clone()
+        param_groups.append({'params': fake_params, 'wd_mult': 0.0, 'lr_mult': 0.0, 'is_decoupled_lr': False})
+
+
     # Collect grad buffers for distributed optimizer.
     per_model_buffers = {}
     per_model_ep_buffers = {}""",""" 
     # Split param groups into dense and MoE params (since data-parallel groups for MoE
     # parameters can be different with expert parallelism).
-    dense_param_groups = list(filter(lambda g: not g['is_expert_parallel'], param_groups))
-    moe_param_groups = list(filter(lambda g: g['is_expert_parallel'], param_groups))
+    try:
+        dense_param_groups = list(filter(lambda g: not g['is_expert_parallel'], param_groups))
+        moe_param_groups = list(filter(lambda g: g['is_expert_parallel'], param_groups))
+    except Exception as e:
+        print(f"An error occurred in get_megatron_optimizer: {e}")
+        dense_param_groups = param_groups
+        moe_param_groups = []
 
     # Create optimizers.
     model_parallel_rank = torch.distributed.get_rank(mpu.get_model_parallel_group())
"""
],
"megatron/core/optimizer/distrib_optimizer.py": [""" 
 \"\"\"Megatron distributed optimizer.\"\"\"
 
-
 import itertools
 from logging import getLogger
 from typing import Callable, Dict, List, Optional, Tuple""","""         world_param_group_map = {}
         for group_index, group in enumerate(param_groups):
             for param in group["params"]:
-                assert param.requires_grad
+                # assert param.requires_grad
                 world_param_group_map[param] = group_index
 
         # Optimizer group ranges & param-group mapping.
"""
],
"megatron/core/optimizer/optimizer.py": ["""         \"\"\"Input optimizer is the base optimizer (e.g., Adam).\"\"\"
         self.optimizer = optimizer
         assert self.optimizer, 'no optimizer is provided.'
+        self.empty_optmizer = False
+        # Fake optimizer params list
+        if getattr(self.optimizer.param_groups[0]['params'][0], 'fake', False):
+            self.empty_optmizer = True
+        print(f'[DEBUG] rank {torch.distributed.get_rank()} empty_optmizer:{self.empty_optmizer}')
+
         self.config = config
         self.init_state_fn = init_state_fn
 
"""
],
"megatron/core/parallel_state.py": ["""     if ignore_virtual:
         return rank in _EMBEDDING_GLOBAL_RANKS
     if rank in _EMBEDDING_GLOBAL_RANKS:
-        if rank == _EMBEDDING_GLOBAL_RANKS[0]:
-            return is_pipeline_first_stage(ignore_virtual=False)
-        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
-            return is_pipeline_last_stage(ignore_virtual=False)
+        from megatron.training import get_args
+        if get_args().multimodal:
+            if rank == _EMBEDDING_GLOBAL_RANKS[-1]:
+                return is_pipeline_last_stage(ignore_virtual=False)
+            else:
+                return True
         else:
-            return True
+            if rank == _EMBEDDING_GLOBAL_RANKS[0]:
+                return is_pipeline_first_stage(ignore_virtual=False)
+            elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
+                return is_pipeline_last_stage(ignore_virtual=False)
+            else:
+                return True
     return False
 
 
"""
],
"megatron/core/pipeline_parallel/schedules.py": [""" from typing import Callable, Iterator, List, Optional, Union
 
 import torch
-from torch.autograd.variable import Variable
+from mindspore.ops import composite as C
+from mindspore.common.api import _pynative_executor
 
 from megatron.core import parallel_state
 from megatron.core.enums import ModelType""","""     set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
     set_input_tensor(input_tensor)
 
+    if not parallel_state.is_pipeline_first_stage() and input_tensor is not None:
+        input_tensor[0].retain_grad()
+
+    # run forward
+    num_tokens = torch.tensor(0, dtype=torch.int)
+    if input_tensor[0] is None:
+        input_tensor[0] = num_tokens
     if config.enable_autocast:
         context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
     else:
         context_manager = contextlib.nullcontext()
+    _pynative_executor.set_grad_flag(True)
+    _pynative_executor.new_graph(forward_step_func, input_tensor[0])
     with context_manager:
         if checkpoint_activations_microbatch is None:
             output_tensor, loss_func = forward_step_func(data_iterator, model)""","""         else:
             data = loss_func(output_tensor, non_loss_data=True)
             forward_data_store.append(data)
+    _pynative_executor.end_graph(forward_step_func, output_tensor, input_tensor[0])
 
     if config.timers is not None:
         config.timers('forward-compute').stop()""","""     return [output_tensor]
 
 
-def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config):
+def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model):
     \"\"\"Backward step through passed-in output tensor.
 
     If last stage, output_tensor_grad is None, otherwise gradient of loss""","""     if not isinstance(input_tensor, list):
         input_tensor = [input_tensor]
         unwrap_input_tensor_grad = True
-    for x in input_tensor:
-        if x is not None:
-            x.retain_grad()
 
     if not isinstance(output_tensor, list):
         output_tensor = [output_tensor]
     if not isinstance(output_tensor_grad, list):
         output_tensor_grad = [output_tensor_grad]
 
-    # Backward pass.
+    # init dout if in last stage
     if output_tensor_grad[0] is None and config.grad_scale_func is not None:
-        output_tensor[0] = config.grad_scale_func(output_tensor[0])
+        output_tensor_grad[0] = config.grad_scale_func(torch.ones_like(output_tensor[0]))
+    if output_tensor_grad[0] is None:
+        output_tensor_grad[0] = torch.ones_like(output_tensor[0])
+
+    # set input tensor for backpropagation
+    if not parallel_state.is_pipeline_first_stage():
+        model.module.set_input_tensor(input_tensor[0])
+
+    # run backward
+    grad_ = C.GradOperation(True, True, True)
+    weights = model.trainable_params()
+    _pynative_executor.check_run(grad_, config.forward_step_func, weights, None, input_tensor[0])
+    _pynative_executor.grad(config.forward_step_func, grad_, weights, None, input_tensor[0], output_tensor_grad[0])
 
-    if config.deallocate_pipeline_outputs:
-        custom_backward(output_tensor[0], output_tensor_grad[0])
-    else:
-        torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])
 
     # Collect the grad of the input_tensor.
     input_tensor_grad = [None]""","""             else:
                 input_tensor_grad.append(x.grad)
 
+    if not parallel_state.is_pipeline_first_stage():
+        model.module.set_input_tensor(None)
+
     # Handle single skip connection if it exists (encoder_hidden_state in
     # model with encoder and decoder).
     if (""","""         data_iterator = data_iterator[0]
 
     config = get_model_config(model)
+    config.forward_step_func = forward_step_func
     if config.timers is not None:
         config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)
 ""","""     model_type = get_model_type(model)
 
     forward_data_store = []
-    input_tensor, output_tensor_grad = None, None
+    input_tensor, output_tensor_grad = [None], [None]
     with no_sync_func():
         for i in range(num_microbatches - 1):
             output_tensor = forward_step(""","""                 is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
             )
             if not forward_only:
-                backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)
+                backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model)
 
     # Run computation for last microbatch out of context handler (want to
     # synchronize gradients).""","""     )
 
     if not forward_only:
-        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)
+        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model)
 
     if config.timers is not None:
         config.timers('forward-backward').stop()""","""     ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"
 
     config = get_model_config(model[0])
+    config.forward_step_func = forward_step_func
     if config.overlap_p2p_comm and config.batch_p2p_comm:
         raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")
 ""","""         if parallel_state.is_pipeline_first_stage():
             if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                 input_tensors[model_chunk_id].append(None)
+
+        # for the first stage, ms need a fake tensor as flag to discriminate different micro
+        if input_tensors[model_chunk_id][-1] is None:
+            input_tensors[model_chunk_id][-1] = torch.tensor(0, dtype=torch.int)
+
         input_tensor = input_tensors[model_chunk_id][-1]
 
         output_tensor = forward_step(""","""         output_tensor = output_tensors[model_chunk_id].pop(0)
         output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
         input_tensor_grad = backward_step(
-            input_tensor, output_tensor, output_tensor_grad, model_type, config
+            input_tensor, output_tensor, output_tensor_grad, model_type, config, model[model_chunk_id]
         )
 
         # launch grad synchronization (custom grad sync)""","""         data_iterator = data_iterator[0]
 
     config = get_model_config(model)
+    config.forward_step_func = forward_step_func
     if config.overlap_p2p_comm:
         raise ValueError(
             "Non-interleaved pipeline parallelism does not support overlapping p2p communication\"""","""                     enable_grad_sync()
 
             input_tensor_grad = backward_step(
-                input_tensor, output_tensor, output_tensor_grad, model_type, config
+                input_tensor, output_tensor, output_tensor_grad, model_type, config, model
             )
 
             if last_iteration:""","""             output_tensor_grad = recv_backward(send_tensor_shapes, config)
 
             input_tensor_grad = backward_step(
-                input_tensor, output_tensor, output_tensor_grad, model_type, config
+                input_tensor, output_tensor, output_tensor_grad, model_type, config, model
             )
 
             send_backward(input_tensor_grad, recv_tensor_shapes, config)""","""         # embedding all-reduce for pipeline parallelism).
         config.finalize_model_grads_func([model])
 
-    return forward_data_store
+    return forward_data_store
\\ No newline at end of file
"""
],
"megatron/core/tensor_parallel/__init__.py": ["""     copy_tensor_model_parallel_attributes,
     linear_with_grad_accumulation_and_async_allreduce,
     param_is_not_tensor_parallel_duplicate,
+    linear_with_frozen_weight,
     set_defaults_if_not_set_tensor_model_parallel_attributes,
     set_tensor_model_parallel_attributes,
 )""","""     reduce_scatter_to_sequence_parallel_region_from_moe,
     scatter_to_sequence_parallel_region,
     scatter_to_tensor_model_parallel_region,
+    reduce_from_tensor_model_parallel_region,
+    reduce_scatter_to_sequence_parallel_region,
 )
 from .random import (
     checkpoint,""","""     "set_defaults_if_not_set_tensor_model_parallel_attributes",
     "copy_tensor_model_parallel_attributes",
     "param_is_not_tensor_parallel_duplicate",
+    "linear_with_frozen_weight",
     "linear_with_grad_accumulation_and_async_allreduce",
     # mappings.py
     "copy_to_tensor_model_parallel_region",
     "gather_from_tensor_model_parallel_region",
     "gather_from_sequence_parallel_region",
-    #    "reduce_from_tensor_model_parallel_region",
+    "reduce_scatter_to_sequence_parallel_region",
+    "reduce_from_tensor_model_parallel_region",
     "scatter_to_tensor_model_parallel_region",
     "scatter_to_sequence_parallel_region",
     # random.py
"""
],
"megatron/core/tensor_parallel/cross_entropy.py": ["""         # Create a mask of valid vocab ids (1 means it needs to be masked).
         target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
         masked_target = target.clone() - vocab_start_index
-        masked_target[target_mask] = 0
+        masked_target = masked_target * (1-target_mask)
+        # masked_target[target_mask] = 0
 
         # Get predicted-logits = logits[target].
         # For Simplicity, we convert logits to a 2-D tensor with size""","""         predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
         predicted_logits_1d = predicted_logits_1d.clone().contiguous()
         predicted_logits = predicted_logits_1d.view_as(target)
-        predicted_logits[target_mask] = 0.0
+        predicted_logits = predicted_logits * (1-target_mask)
+        # predicted_logits[target_mask] = 0.0
         # All reduce is needed to get the chunks from other GPUs.
         torch.distributed.all_reduce(
             predicted_logits,
"""
],
"megatron/core/tensor_parallel/layers.py": ["""         if config.use_cpu_initialization:
             self.weight = Parameter(
                 torch.empty(
-                    self.num_embeddings_per_partition, self.embedding_dim, dtype=config.params_dtype
+                    self.num_embeddings_per_partition, self.embedding_dim, dtype=config.embedding_dtype
                 )
             )
             if config.perform_initialization:""","""                     self.num_embeddings_per_partition,
                     0,
                     init_method,
-                    params_dtype=config.params_dtype,
+                    params_dtype=config.embedding_dtype,
                 )
         else:
             self.weight = Parameter(""","""                     self.num_embeddings_per_partition,
                     self.embedding_dim,
                     device=torch.cuda.current_device(),
-                    dtype=config.params_dtype,
+                    dtype=config.embedding_dtype,
                 )
             )
             if config.perform_initialization:""","""         ctx.async_grad_allreduce = async_grad_allreduce
         ctx.sequence_parallel = sequence_parallel
         ctx.grad_output_buffer = grad_output_buffer
-
         if sequence_parallel:
             world_size = get_tensor_model_parallel_world_size()
             dim_size = list(input.size())""","""         bias = self.bias if not self.skip_bias_add else None
 
         if (
-            self.async_tensor_model_parallel_allreduce
+            (self.async_tensor_model_parallel_allreduce and weight.requires_grad)
             or self.sequence_parallel
             or self.explicit_expert_comm
         ):
"""
],
"megatron/core/tensor_parallel/mappings.py": [""" # Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 
 import torch
-
+import mindspore
 from megatron.core.parallel_state import (
     get_expert_model_parallel_group,
     get_tensor_and_expert_parallel_group,""","""         else:
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
+            output_split_sizes=output_split_sizes.tolist(),
+            input_split_sizes=input_split_sizes.tolist(),
+            group=group._name,
         )
         return output
 
"""
],
"megatron/core/tensor_parallel/random.py": [""" import contextlib
 
 import torch
-from torch import _C
-from torch.cuda import _lazy_call
-from torch.cuda import device as device_ctx_manager
-from torch.utils.checkpoint import detach_variable
 
 from megatron.core.parallel_state import (
     get_data_parallel_rank,""","""     with a single change: the input state is not cloned. Cloning caused
     major performance issues for +4 GPU cases.
     \"\"\"
-    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
-        # older PyTorch
-        def cb():
-            with device_ctx_manager(device):
-                _C._cuda_setRNGState(new_state)
-
-    else:
-        # newer PyTorch
-        if device == -1:
-            device = torch.device('cuda')
-        elif isinstance(device, str):
-            device = torch.device(device)
-        elif isinstance(device, int):
-            device = torch.device('cuda', device)
-
-        def cb():
-            idx = device.index
-            if idx is None:
-                idx = torch.cuda.current_device()
-            default_generator = torch.cuda.default_generators[idx]
-            default_generator.set_state(new_state)
-
-    _lazy_call(cb)
-
+    # if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
+    #     # older PyTorch
+    #     def cb():
+    #         with device_ctx_manager(device):
+    #             _C._cuda_setRNGState(new_state)
+
+    # else:
+    #     # newer PyTorch
+    #     if device == -1:
+    #         device = torch.device('cuda')
+    #     elif isinstance(device, str):
+    #         device = torch.device(device)
+    #     elif isinstance(device, int):
+    #         device = torch.device('cuda', device)
+
+    #     def cb():
+    #         idx = device.index
+    #         if idx is None:
+    #             idx = torch.cuda.current_device()
+    #         default_generator = torch.cuda.default_generators[idx]
+    #         default_generator.set_state(new_state)
+
+    # _lazy_call(cb)
+    torch.cuda.set_rng_state(new_state)
 
 def get_expert_parallel_rng_tracker_name():
     global _EXPERT_PARALLEL_RNG_TRACKER_NAME""","""             raise Exception('cuda rng state {} already exists'.format(name))
         # Get the current rng state.
         orig_rng_state = torch.cuda.get_rng_state()
+        
         # Set the new state and store it.
         torch.cuda.manual_seed(seed)
+
         self.states_[name] = torch.cuda.get_rng_state()
         # Reset rng state to what it was.
         _set_cuda_rng_state(orig_rng_state)""","""     # and model parallel state.
     _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, tensor_model_parallel_seed)
 
+
     expert_parallel_seed = (
         seed + 1024 + 100 * get_expert_model_parallel_rank() + get_tensor_model_parallel_rank()
     )
     _CUDA_RNG_STATE_TRACKER.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME, expert_parallel_seed)
 
 
+
 class CheckpointFunction(torch.autograd.Function):
     \"\"\"Checkpoint Function 
 ""","""         get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)
 
         # Compute the forward pass.
-        detached_inputs = detach_variable(inputs)
+        # detached_inputs = detach_variable(inputs)
         with torch.enable_grad():
-            outputs = ctx.run_function(*detached_inputs)
+            outputs, f_vjp = torch.autograd.vjp(ctx.run_function, *inputs)
 
         # Set the states back to what it was at the start of this function.
         torch.set_rng_state(bwd_cpu_rng_state)""",""" 
         # filter out non tensor outputs for backward pass
         outputs, args = zip(*filter(lambda x: torch.is_tensor(x[0]), zip(outputs, args)))
-        torch.autograd.backward(outputs, args)
-        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
+        grads = f_vjp(*args)
+
         return (None, None) + grads
 
 
"""
],
"megatron/core/transformer/attention.py": [""" from pkg_resources import packaging
 
 from megatron.core import parallel_state, tensor_parallel
-from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
+from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb, RoPEClassic
 from megatron.core.parallel_state import (
     get_data_parallel_group,
     get_data_parallel_rank,""",""" 
 from .enums import AttnMaskType
 from .transformer_config import TransformerConfig
-
+from megatron.training import get_args
 
 @dataclass
 class SelfAttentionSubmodules:""","""         attention_type: str,
     ):
         super().__init__(config=config)
-
+        args = get_args()
         self.config = config
         self.layer_number = layer_number
         self.attn_mask_type = attn_mask_type""","""             tp_comm_buffer_name='proj',
         )
 
+        if self.config.use_rope:
+            self.use_classic = True
+            self.apply_optimized_rotary_pos_emb = RoPEClassic(self.config.kv_channels, args.max_position_embeddings,
+                                                                  config.params_dtype)
+        else:
+            self.use_classic = False
+
     def _checkpointed_attention_forward(
         self,
         query,""","""             # absolute positional embedding.
             # otherwise, only relative positional embedding takes effect
             # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)
+        elif self.use_classic:
+            query = self.apply_optimized_rotary_pos_emb(query)
+            key = self.apply_optimized_rotary_pos_emb(key)
 
         # ==================================
         # core attention computation
"""
],
"megatron/core/transformer/custom_layers/transformer_engine.py": [""" from megatron.core.transformer.transformer_config import TransformerConfig
 from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
 
-_te_version = packaging.version.Version(version("transformer-engine"))
-
+# _te_version = packaging.version.Version(version("transformer-engine"))
+_te_version = packaging.version.parse(te.__version__)
 
 def _get_extra_te_kwargs(config: TransformerConfig):
     extra_transformer_engine_kwargs = {
"""
],
"megatron/core/transformer/moe/moe_utils.py": [""" import torch
 
 from megatron.core import parallel_state
-
+from mindspore import mint
 
 def switch_load_balancing_loss_func(gates, mask, moe_aux_loss_coeff):
     \"\"\"Calculate the auxiliary loss for better load balacing. ""","""     if topk > 1:
         assert indices.size(1) == topk
     flatten_indices = indices.view(-1)
-    sorted_indices = torch.argsort(flatten_indices, stable=True)
+    sorted_indices = mint.argsort(flatten_indices)
     permuted_tokens = tokens.index_select(0, sorted_indices // topk)
     return permuted_tokens, sorted_indices
 ""","""         assert probs.size(1) == topk, f"probs size {probs.size()} merge_factor {topk}"
 
     unpermuted_tokens = torch.zeros_like(permuted_tokens)
-    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
+    unpermuted_tokens.index_add_(0, sorted_indices, permuted_tokens)
 
     unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
 
"""
],
"megatron/core/transformer/moe/token_dispatcher.py": [""" from typing import List, Optional, Tuple
 
 import torch
-
+import mindspore
 from megatron.core import parallel_state, tensor_parallel
 from megatron.core.tensor_parallel.mappings import _gather_along_first_dim_expert_parallel
 from megatron.core.transformer.moe.moe_utils import permute, unpermute""","""         with torch.no_grad():
             # The indices of local_indices that give its sorted order along dim 0.
             self.indices = torch.argsort(local_indices, dim=0)
-            tokens_per_expert = torch.histc(
+            tokens_per_expert = mindspore.mint.histc(
                 local_indices,
                 bins=self.num_local_experts,
                 min=self.local_expert_indices[0],""","""         Returns:
             torch.Tensor: Tensor containing the number of tokens assigned to local expert.
         \"\"\"
-        num_local_tokens_per_expert = torch.histc(
+        num_local_tokens_per_expert = mindspore.mint.histc(
             indices, bins=self.num_experts, min=0, max=self.num_experts
         )
         # num_local_tokens_per_expert: [num_experts]""","""             self.input_splits = (
                 num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
                 .sum(axis=1)
-                .to(torch.device("cpu"))
                 .numpy()
             )
             num_global_tokens_per_expert = _gather_along_first_dim_expert_parallel(""","""                 :, self.local_expert_indices
             ]
             self.output_splits = (
-                self.num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu")).numpy()
-            )
-            num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(axis=0).to(
-                torch.device("cpu"), non_blocking=True
+                self.num_global_tokens_per_local_expert.sum(axis=-1).numpy()
             )
+            num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(axis=0)
+
             # ===================================================
             # num_global_tokens_per_expert: [ep_size, num_experts]
             # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
"""
],
"megatron/core/transformer/transformer_block.py": ["""         if issubclass(spec.module, TransformerBlock):
             return spec.submodules
         elif issubclass(spec.module, BaseTransformerLayer):
-            num_layers = get_num_layers_to_build(config)
+            from megatron.training import get_args 
+            if "pipeline_layer_index" in vars(get_args()):
+                num_layers = config.num_layers
+            else:
+                num_layers = get_num_layers_to_build(config)
             return TransformerBlockSubmodules(layer_specs=[spec] * num_layers)
         else:
             raise Exception(f"specialize for {spec.module.__name__}.")""","""         self.input_tensor = None
 
         self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'
+        self.fp32_residual_connection = self.config.fp32_residual_connection
 
         if get_cpu_offload_context is not None:
             (""","""             # See set_input_tensor()
             hidden_states = self.input_tensor
 
+        if self.fp32_residual_connection:
+            hidden_states = hidden_states.contiguous().float()
         # Viewless tensor.
         # - We only need to create a viewless tensor in the case of micro batch
         #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
"""
],
"megatron/core/transformer/transformer_config.py": ["""     test_mode: bool = False
     \"\"\"Whether to run real-time tests.\"\"\"
 
+    multimodal: bool = False
+    \"\"\"Whether mm model\"\"\"
+
     ####################
     # initialization
     ####################""","""     apply_rope_fusion: bool = False
     \"\"\"If True, use fused RoPE kernel.\"\"\"
 
+    use_rope: bool = False  # TODO: xywu, delete it later
+
     ####################
     # activation recomputation
     ####################""","""                     "When bias_activation_fusion is True, gated_linear_unit is False, "
                     "and activation function is gelu, add_bias_linear must also be True."
                 )
-        if self.apply_rope_fusion and self.rotary_interleaved:
-            raise ValueError(f'rotary_interleaved does not work with apply_rope_fusion.')
 
         if self.init_method is None:
             self.init_method = init_method_normal(self.init_method_std)""","""         if self.output_layer_init_method is None:
             self.output_layer_init_method = scaled_init_method_normal(
                 self.init_method_std, self.num_layers
-            )
+            )
\\ No newline at end of file
"""
],
"megatron/core/transformer/utils.py": ["""     return attention_scores
 
 
-@jit_fuser
+# @jit_fuser
 def gelu_impl(x):
     \"\"\"OpenAI's gelu implementation.\"\"\"
     return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))""",""" 
 
 # This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
-@jit_fuser
+# @jit_fuser
 def erf_gelu(x):
     return (
         x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype) + torch.ones_like(x).to(dtype=x.dtype))
"""
],
"megatron/legacy/fused_kernels/__init__.py": [""" import pathlib
 import subprocess
 
-from torch.utils import cpp_extension
+# from torch.utils import cpp_extension
 
 # Setting this param to a list has a problem of generating different
 # compilation commands (with diferent order of architectures) and
"""
],
"megatron/legacy/model/gpt_model.py": [""" def post_language_model_processing(lm_output, labels, logit_weights,
                                    parallel_output,
                                    fp16_lm_cross_entropy):
-
+    args = get_args()
+    if args.final_logit_softcapping is not None:
+        lm_output = lm_output / args.final_logit_softcapping
+        lm_output = torch.tanh(lm_output)
+        lm_output = lm_output * args.final_logit_softcapping
     # Output. Format [s b h]
     output = parallel_lm_logits(
         lm_output,
"""
],
"megatron/legacy/model/language_model.py": [""" from .module import MegatronModule
 from .transformer import ParallelTransformer
 from .utils import get_linear_layer
-from .utils import init_method_normal, scaled_init_method_normal
+from .utils import init_method_normal, scaled_init_method_normal, get_norm
 
 
 def parallel_lm_logits(input_, word_embeddings_weight, parallel_output,""",""" 
         # Word embeddings (parallel).
         self.params_dtype = args.params_dtype
+        emb_init_method = init_method_normal(args.emb_init_method_std)
+
         self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
-            vocab_size, self.hidden_size, config=config, init_method=config.init_method)
+            vocab_size, self.hidden_size, config=config, init_method=emb_init_method)
         self._word_embeddings_key = 'word_embeddings'
 
         # Position embedding (serial).""","""                                        self.num_tokentypes)
             self._embedding_key = 'embedding'
 
+        self.embedding_scaling = args.embedding_scaling
+
         # Rotary positional embeddings
         self.use_rotary_position_embeddings = \\
             args.position_embedding_type == 'rope'""","""             self.rotary_pos_emb = RotaryEmbedding(
                 kv_channels=rotary_dim,
                 rotary_percent=args.rotary_percent,
+                rotary_interleaved=args.rotary_interleaved,
                 seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
             )
 ""","""         if self.pre_process:
             encoder_input = self.embedding(enc_input_ids, enc_position_ids,
                                            tokentype_ids=tokentype_ids)
+            if self.embedding_scaling:
+                embedding_scaling = torch.tensor(self.embedding_scaling, dtype=encoder_input.dtype)
+                encoder_input = encoder_input * embedding_scaling
         else:
             encoder_input = None
-
         # Retriever embedding.
         if self.add_retriever and self.pre_process:
             retriever_input = self.embedding(retriever_input_ids,
"""
],
"megatron/legacy/model/module.py": ["""         if isinstance(val_typecheck, (Parameter, Variable)):
             val_typecheck = val.data
         if isinstance(val_typecheck, _FLOAT_TYPES):
-            val = float16_convertor(val)
+            if val_typecheck.dtype == torch.float32:
+                val = float16_convertor(val)
         return val
     return conversion_helper(val, half_conversion)
 ""","""         if isinstance(val_typecheck, (Parameter, Variable)):
             val_typecheck = val.data
         if isinstance(val_typecheck, (_BF16_TYPES, _HALF_TYPES)):
-            val = val.float()
+            if val_typecheck.dtype in (torch.float16, torch.bfloat16):
+                val = val.float()
         return val
     return conversion_helper(val, float_conversion)
 
"""
],
"megatron/legacy/model/rms_norm.py": ["""     def __init__(self,
                  dim: int,
                  eps: float = 1e-6,
-                 sequence_parallel: bool = False):
+                 sequence_parallel: bool = False,
+                 scale=1.0):
         \"\"\"RMS Normaliation module
 
         Args:""","""         \"\"\"
         super().__init__()
         self.eps = eps
-        self.weight = nn.Parameter(torch.ones(dim))
+        self.weight = nn.Parameter(torch.ones(dim) * scale)
 
         setattr(self.weight, 'sequence_parallel', sequence_parallel)
 
"""
],
"megatron/legacy/model/transformer.py": ["""         else:
             if bias_parallel is not None:
                 intermediate_parallel = intermediate_parallel + bias_parallel
-            intermediate_parallel = self.activation_func(intermediate_parallel)
+            if self.use_vanilla_activation:
+                alpha = self.fastgelu(self.dense_h_to_4h_2(hidden_states)[0])
+                intermediate_parallel = self.activation_func(intermediate_parallel, alpha)
+            else:
+                intermediate_parallel = self.activation_func(intermediate_parallel)
 
         # [s, b, h]
         output, output_bias = self.dense_4h_to_h(intermediate_parallel)""","""         # Query, Key, and Value
         # =====================
         if self.attention_type == AttnType.self_attn:
-
             # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
             mixed_x_layer, _ = self.query_key_value(hidden_states)
 ""","""         return output, bias
 
 
-def bias_dropout_add(x, bias, residual, prob, training):
+def bias_dropout_add(x, bias, residual, prob, training, add_residual=True):
     # type: (Tensor, Optional[Tensor], Tensor, float, bool) -> Tensor
     if bias is not None:
         x = x + bias
     out = torch.nn.functional.dropout(x, p=prob, training=training)
-    out = residual + out
+    if add_residual:
+        out = residual + out
     return out
 
 
 def get_bias_dropout_add(training):
-    def _bias_dropout_add(x, bias, residual, prob):
-        return bias_dropout_add(x, bias, residual, prob, training)
+    def _bias_dropout_add(x, bias, residual, prob, add_residual=True):
+        return bias_dropout_add(x, bias, residual, prob, training, add_residual=add_residual)
     return _bias_dropout_add
 
 """,""" def bias_dropout_add_fused_train(x: torch.Tensor,
                                  bias: Optional[torch.Tensor],
                                  residual: torch.Tensor,
-                                 prob: float) -> torch.Tensor:
-    return bias_dropout_add(x, bias, residual, prob, True)
+                                 prob: float,
+                                 add_residual=True) -> torch.Tensor:
+    return bias_dropout_add(x, bias, residual, prob, True,  add_residual=add_residual)
 
 
 @jit_fuser
 def bias_dropout_add_fused_inference(x: torch.Tensor,
                                      bias: Optional[torch.Tensor],
                                      residual: torch.Tensor,
-                                     prob: float) -> torch.Tensor:
-    return bias_dropout_add(x, bias, residual, prob, False)
+                                     prob: float,
+                                     add_residual=True) -> torch.Tensor:
+    return bias_dropout_add(x, bias, residual, prob, False, add_residual=add_residual)
 
 
 class ParallelTransformerLayer(MegatronModule):""","""         self.layer_number = layer_number
         self.layer_type = layer_type
 
+        self.use_sandwich_norm = args.use_sandwich_norm
+        self.attn_post_norm_scale = args.attn_post_norm_scale
+        self.ffn_post_norm_scale = args.ffn_post_norm_scale
+        
         self.apply_residual_connection_post_norm \\
             = config.apply_residual_connection_post_layernorm
 """,""" 
         # Normalize the input data.
         self.input_norm = get_norm(config)
+        if self.use_sandwich_norm:
+            self.attn_post_norm = get_norm(config, self.attn_post_norm_scale)
 
         # Self attention.
         self.self_attention = ParallelAttention(""",""" 
         # Normalize the attention output
         self.post_attention_norm = get_norm(config)
+        if self.use_sandwich_norm:
+            self.ffn_post_norm = get_norm(config, self.ffn_post_norm_scale)
 
         # Cross attention.
         if self.layer_type in (LayerType.decoder,""","""         if args.num_experts is not None:
             self.mlp = SwitchMLP(config)
         else:
-            self.mlp = ParallelMLP(config)
+            self.mlp = ParallelMLP(config, layer_number)
+
+        self.use_augs_attention = False
+        if args.augs_attention:
+            if layer_number >= args.augs_attention_start_layer and (
+                    args.augs_attention_end_layer == -1 or layer_number <= args.augs_attention_end_layer):
+                self.use_augs_attention = True
+
+        if self.use_augs_attention:
+            # Augs should be 16 downsample
+            self.attn_linear = torch.nn.Sequential(
+                torch.nn.Linear(config.hidden_size, config.hidden_size // args.augs_ratio,
+                                bias=False),
+                FastGELU(),
+                torch.nn.Linear(config.hidden_size // args.augs_ratio, config.hidden_size,
+                                bias=False)
+            )
+            for param in self.attn_linear.parameters():
+                setattr(param, 'sequence_parallel', config.sequence_parallel)
 
         # Set bias+dropout+add fusion grad_enable execution handler.
         TORCH_MAJOR = int(torch.__version__.split('.')[0])""","""                 args.retro_num_retrieved_chunks * args.retro_chunk_length
 
         # hidden_states: [s, b, h]
+        add_residual = False if self.use_sandwich_norm else True
 
         # Layer norm at the beginning of the transformer layer.
         norm_output = self.input_norm(hidden_states)""","""                     attention_output,
                     attention_bias,
                     residual,
-                    self.hidden_dropout)
+                    self.hidden_dropout,
+                    add_residual=add_residual)
+                if self.use_sandwich_norm:
+                    norm_input = self.attn_post_norm(norm_input)
+                    norm_input += residual
         else:
-            out = torch.nn.functional.dropout(attention_output + attention_bias,
+            out = attention_output + attention_bias
+            out = torch.nn.functional.dropout(out,
                                               p=self.hidden_dropout,
                                               training=self.training)
+            if self.use_sandwich_norm:
+                out = self.attn_post_norm(out)
+            
             norm_input = residual + self.drop_path(out)
 
+        if self.use_augs_attention:
+            norm_input = norm_input + self.attn_linear(attention_output + attention_bias)  # todo hewei
+
         # Layer norm post the self attention.
         norm_output = self.post_attention_norm(norm_input)
 
+        if self.use_sandwich_norm and encoder_output is not None:
+                raise Exception("Sandwich normalization does not support cross attention now.")
         # Cross attention.
         if self.layer_type == LayerType.encoder:
             pass""","""                     mlp_output,
                     mlp_bias,
                     residual,
-                    self.hidden_dropout)
+                    self.hidden_dropout,
+                    add_residual=add_residual)
+                if self.use_sandwich_norm:
+                    output = self.ffn_post_norm(output)
+                    output += residual
+                
 
             # Jit compiled function creates 'view' tensor. This tensor
             # potentially gets saved in the MPU checkpoint function context,""","""         else:
             if mlp_bias is not None:
                 mlp_output = mlp_output + mlp_bias
-            out = torch.nn.functional.dropout(mlp_output,
+            mlp_output = torch.nn.functional.dropout(mlp_output,
                                               p=self.hidden_dropout,
                                               training=self.training)
-            output = residual + self.drop_path(out)
+            if self.use_sandwich_norm:
+                mlp_output = self.ffn_post_norm(mlp_output)
+
+            output = residual + self.drop_path(mlp_output)
 
         if self.layer_type == LayerType.retro_decoder_with_retriever:
             return output, retriever_output""","""             state_dict_[newkey] = state_dict[key]
 
         super().load_state_dict(state_dict_, strict)
+
+
+class FastGELU(torch.nn.Module):
+    \"\"\"
+    Applies GELU approximation that is the same as mindspore
+    \"\"\"
+
+    def forward(self, input: torch.Tensor) -> torch.Tensor:
+        abs_value = torch.abs(input)
+        return input * torch.sigmoid(1.702 * abs_value) * torch.exp(
+            0.851 * (input - abs_value))
"""
],
"megatron/legacy/model/utils.py": ["""     return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype)+torch.ones_like(x).to(dtype=x.dtype))
 
 
-def get_norm(config):
+def get_norm(config, scale=1.0):
     args = get_args()
     if args.normalization == "LayerNorm":
+        if scale != 1.0:
+            raise Exception("Layernorm only supports the scale value of 1.")
         return LayerNorm(
             config.hidden_size,
             eps=config.layernorm_epsilon,""",""" 
         return RMSNorm(dim=config.hidden_size,
                        eps=config.layernorm_epsilon,
-                       sequence_parallel=config.sequence_parallel)
+                       sequence_parallel=config.sequence_parallel,
+                       scale=scale)
     else:
         raise Exception(f"unsupported norm type '{args.normalization}'.")
"""
],
"megatron/training/arguments.py": [""" import types
 
 import torch.nn.functional as F
-from megatron.core.models.retro.utils import (
-    get_config_path as get_retro_config_path,
-    get_gpt_data_dir as get_retro_data_dir,
-)
+# from megatron.core.models.retro.utils import (
+#     get_config_path as get_retro_config_path,
+#     get_gpt_data_dir as get_retro_data_dir,
+# )
 from megatron.core.transformer import TransformerConfig
 
 ""","""         assert args.pipeline_model_parallel_size > 2, \\
             'pipeline-model-parallel size should be greater than 2 with ' \\
             'interleaved schedule'
-        assert args.num_layers % args.transformer_pipeline_model_parallel_size == 0, \\
-            'number of layers should be divisible by the pipeline parallel size'
         num_layers_per_pipeline_stage = args.num_layers // args.transformer_pipeline_model_parallel_size
-        assert num_layers_per_pipeline_stage % args.num_layers_per_virtual_pipeline_stage == 0, \\
-            'number of layers per pipeline stage must be divisible number of layers per virtual pipeline stage'
         args.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \\
             args.num_layers_per_virtual_pipeline_stage
     else:""","""                 print('accumulate and all-reduce gradients in fp32 for '
                       'bfloat16 data type.', flush=True)
 
+    # Embedding dtype.
+    args.embedding_dtype = torch.float
+    if args.embedding_dtype == 'fp16':
+        args.embedding_dtype = torch.half
+    elif args.embedding_dtype == 'bf16':
+        args.embedding_dtype = torch.bfloat16
+    elif args.embedding_dtype == 'fp32':
+        args.embedding_dtype = torch.float
+    else:
+        args.embedding_dtype = args.params_dtype
+
     if args.rank == 0:
         print('using {} for parameters ...'.format(args.params_dtype),
               flush=True)""","""     # Legacy RoPE arguments
     if args.use_rotary_position_embeddings:
         args.position_embedding_type = 'rope'
-    if args.rotary_interleaved and args.apply_rope_fusion:
-        raise RuntimeError('--rotary-interleaved does not work with rope_fusion.')
-    if args.rotary_interleaved and not args.use_mcore_models:
-        raise RuntimeError('--rotary-interleaved only support Megatron Core, please add --use-mcore-models.')
+
+    # if args.rotary_interleaved and not args.use_mcore_models:
+    #     raise RuntimeError('--rotary-interleaved only support Megatron Core, please add --use-mcore-models.')
 
     # Would just need to add 'NoPE' as a position_embedding_type to support this, but for now
     # don't allow it to keep things simple""","""         kw_args['bias_activation_fusion'] = args.bias_swiglu_fusion
     else:
         kw_args['bias_activation_fusion'] = args.bias_gelu_fusion
+    
+    args.fast_gelu = False
+    if args.fast_gelu:
+        import torch_npu
+        def fast_gelu(x):
+            return torch_npu.fast_gelu(x)
+        kw_args['activation_func'] = fast_gelu
     if args.squared_relu:
         assert not args.swiglu
         def squared_relu(x):""","""         kw_args['num_query_groups'] = args.num_query_groups
     else:
         kw_args['num_query_groups'] = None
+    args.multimodal = False
+    
+    
+    kw_args['multimodal'] = args.multimodal
 
     # Return config.
     return config_class(**kw_args)""","""                        help='Maximum number of position embeddings to use. '
                        'This is the size of position embedding.')
     group.add_argument('--position-embedding-type', type=str, default='learned_absolute',
-                       choices=['learned_absolute', 'rope'],
+                       choices=['learned_absolute', 'rope', 'classic'],
                        help='Position embedding type.')
     group.add_argument('--use-rotary-position-embeddings', action='store_true',
                        help='Use rotary positional embeddings or not. '""","""                        help='Probability of producing a short sequence.')
     group.add_argument('--num-workers', type=int, default=2,
                        help="Dataloader number of workers.")
-    group.add_argument('--tokenizer-type', type=str,
-                       default=None,
-                       choices=['BertWordPieceLowerCase',
-                                'BertWordPieceCase',
-                                'GPT2BPETokenizer',
-                                'SentencePieceTokenizer',
-                                'GPTSentencePieceTokenizer',
-                                'Llama2Tokenizer',
-                                'NullTokenizer'],
-                       help='What type of tokenizer to use.')
     group.add_argument('--tokenizer-model', type=str, default=None,
                        help='Sentencepiece tokenizer model.')
     group.add_argument('--reset-position-ids', action='store_true',
"""
],
"megatron/training/checkpointing.py": [""" from .global_vars import get_args
 from .utils import (unwrap_model,
                     print_rank_0)
-
+# from pangu.tasks.finetune.lora.utils import is_enable_lora
 
 _CHECKPOINT_VERSION = None
 ""","""     # Only rank zero of the data parallel writes to the disk.
     model = unwrap_model(model)
 
+    # Automatically update word embeddings and merge LoRA weights.
+    if isinstance(model, list):
+        [_.eval() for _ in model]
+    else:
+        model.eval()
+
     ckpt_format = args.dist_ckpt_format if args.use_dist_ckpt else 'torch'
     print_rank_0('saving checkpoint at iteration {:7d} to {} in {} format'.format(
         iteration, args.save, ckpt_format))""","""     print_rank_0('  successfully saved checkpoint at iteration {:7d} to {}' \\
                  .format(iteration, args.save))
 
+    # Automatically unmerge LoRA weights.
+    if isinstance(model, list):
+        [_.train() for _ in model]
+    else:
+        model.train()
+
     # And update the latest iteration
     if not torch.distributed.is_initialized() \\
        or torch.distributed.get_rank() == 0:""","""     # Model.
     strict = False if args.retro_add_retriever else strict
     if len(model) == 1:
-        model[0].load_state_dict(state_dict['model'], strict=strict)
+        if not hasattr(args, 'multimodal'):
+            args.multimodal = False
+        if args.multimodal:
+            if args.LoRA:
+                model[0].load_state_dict(state_dict['model'], strict=False)
+            else:
+                model[0].load_state_dict(state_dict['model'], strict=strict)
+                # If you used the old ckpt from multimodal_input_ap, you can try using load_state_dict_own
+                # to load:# model[0].load_state_dict_own(state_dict['model'], strict=strict)
+        else:
+            model[0].load_state_dict(state_dict['model'], strict=strict)
+
     else:
         for i in range(len(model)):
             mpu.set_virtual_pipeline_model_parallel_rank(i)
"""
],
"megatron/training/initialize.py": ["""             seed = seed + (10 * mpu.get_data_parallel_rank())
         random.seed(seed)
         np.random.seed(seed)
+
         torch.manual_seed(seed)
+
         if torch.cuda.device_count() > 0:
             tensor_parallel.model_parallel_cuda_manual_seed(seed)
+
     else:
         raise ValueError("Seed ({}) should be a positive integer.".format(seed))
 
"""
],
"megatron/training/tokenizer/tokenizer.py": ["""         return None
 
 
-class _NullTokenizer:
+class _NullTokenizer(MegatronTokenizer):
     def __init__(self, vocab_size):
-        vocab_size = int(vocab_size)
-        self._eos_id = vocab_size
-        self.vocab_size = vocab_size+1
+        super().__init__(None, vocab_size=vocab_size)
+        self._vocab_size_without_eod = int(vocab_size)
+        self._eod_id = self._vocab_size_without_eod
 
     def tokenize(self, text):
         return [int(x) for x in text.split(' ')]""","""         text = [str(x) for x in ids]
         return ' '.join(text)
 
+    @property
+    def vocab_size(self):
+        return self._vocab_size_without_eod + 1
+
+    @property
+    def vocab(self):
+        raise NotImplementedError
+
+    @property
+    def inv_vocab(self):
+        raise NotImplementedError
+
     @property
     def cls(self):
         return -1""",""" 
     @property
     def eod(self):
-        return self._eos_id
+        return self._eod_id
 
     @property
     def additional_special_tokens_ids(self):
"""
],
"megatron/training/training.py": [""" from megatron.legacy.data.data_samplers import build_pretraining_data_loader
 from megatron.core.transformer.moe.moe_utils import track_moe_metrics
 from megatron.core.pipeline_parallel import get_forward_backward_func
+# from pangu.training.utils import freeze_module #
+
 
 from .utils import (
     calc_params_l2_norm,""","""     )
 
 
+def mm_num_floating_point_operations(args, batch_size):
+    # vit FLOPS
+    resolution, patch_size, seq_len = 448, 14, args.encoder_seq_length
+    vit_seq_len = (resolution // patch_size) ** 2
+    vit_img_nums = seq_len // args.image_token_length
+    vit_bsz = vit_img_nums * batch_size
+    vit_hidden_dim = args.visual_hidden_size
+    vit_num_layers = args.visual_num_layers
+    vit_img_feature_dim = (patch_size ** 2) * 3
+
+    vit_patch_linear_forward_flops = batch_size * vit_seq_len * vit_img_feature_dim * vit_hidden_dim * 2
+    vit_transformer_forward_flops = 24 * vit_bsz * vit_seq_len * (vit_hidden_dim ** 2) + \\
+                                    4 * vit_bsz * (vit_seq_len ** 2) * vit_hidden_dim
+
+    vit_forward_flops = vit_num_layers * vit_transformer_forward_flops + vit_patch_linear_forward_flops
+    vit_backward_flops = 2 * vit_forward_flops
+
+    vit_total_flops = vit_forward_flops + vit_backward_flops if args.Unfreeze_ViT else vit_forward_flops
+
+    # c_abs FLOPS
+    ori_h_ori_w = 32
+    new_h_ori_w = 16
+    se_module_intermediate_hidden_dim = 320
+    conv2_group = vit_hidden_dim
+
+    c_abs_forward_flops = (vit_hidden_dim ** 2) * (ori_h_ori_w ** 2) * 4 \\
+                          + 3 * 3 * (vit_hidden_dim ** 2) / conv2_group * (ori_h_ori_w ** 2) * 2 \\
+                          + vit_hidden_dim * se_module_intermediate_hidden_dim * (ori_h_ori_w ** 2) * 4 \\
+                          + (vit_hidden_dim ** 2) * (new_h_ori_w ** 2) * 4 \\
+                          + 3 * 3 * (vit_hidden_dim ** 2) / conv2_group * (new_h_ori_w ** 2) * 2 \\
+                          + vit_hidden_dim * se_module_intermediate_hidden_dim * (new_h_ori_w ** 2) * 4
+
+    c_abs_forward_flops = c_abs_forward_flops * 2
+    c_abs_backward_flops = 2 * c_abs_forward_flops
+
+    c_abs_total_flops = c_abs_forward_flops + c_abs_backward_flops
+
+    # MLP projection FLOPS
+    mlp_output_seq_len = 256  # args.image_token_length
+    mlp_intermediate_hidden_dim = 2048
+    mlp_final_hidden_dim = args.hidden_size
+
+    mlp_forward_flops = 2 * (vit_bsz * mlp_output_seq_len * vit_hidden_dim * mlp_intermediate_hidden_dim) + \\
+                        2 * (vit_bsz * mlp_output_seq_len * mlp_intermediate_hidden_dim * mlp_final_hidden_dim)
+    mlp_backward_flops = 2 * mlp_forward_flops
+
+    mlp_total_flops = mlp_forward_flops + mlp_backward_flops
+
+    # LLM FLOPS
+    word_embedding_vocab_size = args.vocab_size
+    llm_seq_length = args.encoder_seq_length
+    llm_hidden_dim = args.hidden_size
+    word_embedding_flops = 2 * batch_size * llm_seq_length * llm_hidden_dim * word_embedding_vocab_size
+    llm_transformer_flops = 24 * batch_size * llm_seq_length * (llm_hidden_dim ** 2) + \\
+                            4 * batch_size * (llm_seq_length ** 2) * llm_hidden_dim
+    llm_transformer_flops = llm_transformer_flops * args.num_layers
+    llm_forward_flops = word_embedding_flops + llm_transformer_flops
+    if args.Unfreeze_LLM:
+        llm_backward_flops = 2 * llm_forward_flops
+    else:
+        llm_backward_flops = llm_forward_flops + \\
+                             4 * batch_size * (llm_seq_length ** 2) * llm_hidden_dim * args.num_layers
+
+    llm_total_flops = llm_forward_flops + llm_backward_flops
+
+    # final flops
+    total_flops = vit_total_flops + c_abs_total_flops + mlp_total_flops + llm_total_flops
+
+    return total_flops
+
+
 def append_to_progress_log(string):
     args = get_args()
     if args.save is None:""","""         for param in model_module.parameters():
             tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
 
-    # Print number of parameters.
-    if mpu.get_data_parallel_rank() == 0:
-        print(' > number of parameters on (tensor, pipeline) '
-              'model parallel rank ({}, {}): {}'.format(
-            mpu.get_tensor_model_parallel_rank(),
-            mpu.get_pipeline_model_parallel_rank(),
-            sum([sum([p.nelement() for p in model_module.parameters()])
-                 for model_module in model])), flush=True)
 
     # GPU allocation.
     for model_module in model:
         model_module.cuda(torch.cuda.current_device())
-
+    
     # Fp16 conversion.
-    if args.fp16 or args.bf16:
+    args.preserve_orig_param_dtype = False
+    if args.preserve_orig_param_dtype:
+        model = [model_module for model_module in model]
+    elif args.fp16 or args.bf16:
         model = [Float16Module(model_module, args) for model_module in model]
 
+    # model = freeze_module(model)
+
+    if mpu.get_data_parallel_rank() == 0:
+        print_rank_0('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
+        for model_module in model:
+            for name, parameters in model_module.named_parameters():
+                print_rank_0('{} : {} : {} : {}'.format(name, parameters.dtype, parameters.size(), parameters.requires_grad))
+
     if wrap_with_ddp:
         config = get_model_config(model[0])
         model = [DDP(config,""","""     total_iterations = total_loss_dict[advanced_iters_key] + \\
                        total_loss_dict[skipped_iters_key]
 
+    # Calculate batch token
+    batch_token = batch_size * args.seq_length
+    if hasattr(args, 'seq_len_in_single_batch') and args.seq_len_in_single_batch is not None:
+        batch_token = batch_size * args.seq_len_in_single_batch
+
     # Tensorboard values.
     # Timer requires all the ranks to call.
     if args.log_timers_to_tensorboard and \\""","""     if iteration % args.log_interval == 0:
         elapsed_time = timers('interval-time').elapsed(barrier=True)
         elapsed_time_per_iteration = elapsed_time / total_iterations
-
-        throughput = num_floating_point_operations(args, batch_size) / (
-            elapsed_time_per_iteration * 10**12 * args.world_size)
+        args.multimodal = False
+        if args.multimodal:
+            total_flops = mm_num_floating_point_operations(args, batch_size)
+            throughput = total_flops / (
+                    elapsed_time_per_iteration * 10 ** 12 * args.world_size)
+        else:
+            throughput = num_floating_point_operations(args, batch_size) / (
+                    elapsed_time_per_iteration * 10 ** 12 * args.world_size)
+        throughput_per_day = batch_token / elapsed_time_per_iteration * 3600 * 24
         if args.log_timers_to_tensorboard:
             if writer:
                 writer.add_scalar('iteration-time',""","""             iteration, args.train_iters)
         log_string += ' consumed samples: {:12d} |'.format(
             args.consumed_train_samples)
+        if hasattr(args, 'seq_len_in_single_batch') and args.seq_len_in_single_batch is not None:
+            log_string += ' consumed tokens per iteration: {:5d} |'.format(batch_token)
         log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
             elapsed_time_per_iteration * 1000.0)
         if args.log_throughput:""","""                     writer.add_scalar('throughput', throughput, iteration)
                 if wandb_writer:
                     wandb_writer.log({'throughput': throughput}, iteration)
+            log_string += f' throughput: {throughput_per_day/1e9:.1f}B tokens/day |'
+        # log_string += f' baseline(ms):{args.baseline_time} |'
         assert learning_rate is not None
         # Decoupled_learning_rate should be not None only on first and last pipeline stage.
-        log_string += ' learning rate: {:.6E} |'.format(learning_rate)
+        log_string += ' learning rate: {:.16f} |'.format(learning_rate)
         if args.decoupled_lr is not None and (mpu.is_pipeline_first_stage(ignore_virtual=True) or
                                               mpu.is_pipeline_last_stage(ignore_virtual=True)):
             assert decoupled_learning_rate is not None""","""                            nan_iters_key]:
                 avg = total_loss_dict[key].item() / \\
                       float(max(1, total_loss_dict[advanced_iters_key]))
-                if avg > 0.0:
-                    log_string += ' {}: {:.6E} |'.format(key, avg)
+                if avg > 0.0 or len(total_loss_dict) > 2:
+                    log_string += ' {}: {:.16f} |'.format(key, avg)
                 total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device='cuda')
         log_string += ' loss scale: {:.1f} |'.format(loss_scale)
         if grad_norm is not None:
-            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
+            log_string += ' grad norm: {:.16f} |'.format(grad_norm)
         if num_zeros_in_grad is not None:
             log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
         if params_norm is not None:""","""         if args.train_samples is None:
             args.consumed_valid_samples = (args.iteration // args.eval_interval) * \\
                 args.eval_iters * args.global_batch_size
+    
+    args.new_dataset = False
+    if args.new_dataset:
+        print_rank_0('> Iteration from checkpoint is {}. Use --new_dataset to reset '
+                     'args.consumed_train_samples and args.iteration as 0...'.format(args.iteration))
+        args.consumed_train_samples = 0
+        args.print_iteration += args.iteration
+        args.iteration = 0
+        print_rank_0('> Use args.print_iteration {} for printing...'.format(args.print_iteration))
 
     # Rely on distributed-aware core datasets, temporary
     is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)
"""
],
"megatron/training/yaml_arguments.py": ["""     # Load retro args (used by both Retro & GPT).
     if getattr(args, 'retro_project_dir', None) is not None:
         raise Exception("Retro untested for yaml args. See arguments.py.")
-
-    if args.language_model.rotary_interleaved and args.language_model.apply_rope_fusion:
-        raise RuntimeError('--rotary-interleaved does not work with rope_fusion.')
     
     # MoE Spec check
     if args.language_model.num_moe_experts is not None:
"""
],

}


LINE_RULES_acclerate_mindspeed_llm = {
"mindspeed/arguments.py": ["""     parser = _add_profile_args(parser)
     parser = _add_deepseek_args(parser)
     parser = _add_high_availability_args(parser)
+    parser = _add_self_defined_pipeline_args(parser)
 
     return parser
 ""","""                        help="high availability feature, enable hbmfault repair")
     return parser
 
+def _add_self_defined_pipeline_args(parser):
+    group = parser.add_argument_group(title='self_defined_pipeline')
+    group.add_argument('--pipeline-layer-index',
+                       type=str, default=None,
+                       help='self defined pipeline layer nums'
+                      )
+    group.add_argument('--pipeline-shapes',
+                       type=str, default=None,
+                       help='self defined pipeline shapes'
+                       )
+    return parser
+
 
 def _add_profile_args(parser):
     group = parser.add_argument_group(title='profile')""","""                                          f'to 0 and smaller than args.num_layers({args.num_layers})')
                 noop_layers.add(int(x))
             args.noop_layers = noop_layers
+        if args.pipeline_layer_index:
+            if args.num_layer_list is not None:
+                raise AssertionError('--num-layer-list switch can not open with pipeline-layer-index')
+
+            import ast
+            args.pipeline_layer_index = ast.literal_eval(args.pipeline_layer_index)
+            args.pipeline_shapes = ast.literal_eval(args.pipeline_shapes)
 
         from megatron.training.arguments import _print_args
         _print_args('arguments', args, True)
"""
],
"mindspeed/core/context_parallel/adaptive_context_parallel.py": [""" # Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
 import torch
 import torch_npu
-from mindspeed.ops.npu_ring_attention_update import npu_ring_attention_update
+# from mindspeed.ops.npu_ring_attention_update import npu_ring_attention_update
 
 
 def flash_attn_p2p_communicate(scheduling_info, send_q_dst, recv_q_src, send_kv_dst, recv_kv_src, cp_group, rank):
"""
],
"mindspeed/core/context_parallel/ring_context_parallel.py": [""" # Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
 import torch
 import torch_npu
-from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention, npu_fusion_attention_grad
+# from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention, npu_fusion_attention_grad
 from .utils import RingP2P, causal_out_update, general_out_update, forward_update
 
 
-
 def causal_forward_fetch(q_block_id, kv_block_id, q, cur_k, cur_v, attn_mask=None):
     cur_attn_mask = None
     if q_block_id == kv_block_id:""","""         cur_q = q[1]
         # [2, s, b, h] -> [2s, b, h]
         cur_k, cur_v = [x.view(-1, *x.shape[2:]) for x in [cur_k, cur_v]]
-    
+
     return cur_q, cur_k, cur_v, cur_attn_mask
 
 
-def causal_backward_fetch(q_block_id, kv_block_id, q, cur_k, cur_v, attn_out, dout, 
+def causal_backward_fetch(q_block_id, kv_block_id, q, cur_k, cur_v, attn_out, dout,
                           softmax_max, softmax_sum, attn_mask=None):
     cur_attn_mask = None
     if q_block_id >= kv_block_id:
         # [b, n, 2, s, 8] -> [b, n, 2s, 8]
         cur_softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1], -1,
-                                            softmax_max.shape[-1])
+                                           softmax_max.shape[-1])
         cur_softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1], -1,
-                                            softmax_sum.shape[-1])
+                                           softmax_sum.shape[-1])
         # [2, s, b, h] -> [2s, b, h]
         cur_q, cur_attn_out, cur_dout = [x.view(-1, *x.shape[2:]) for x in [q, attn_out, dout]]
         if q_block_id == kv_block_id:""","""         # only q[1] attn_out[1] and dout[1] need to be calculated
         cur_q, cur_attn_out, cur_dout = [x[1] for x in [q, attn_out, dout]]
         cur_softmax_max, cur_softmax_sum = [x[:, :, 1, :, :] for x in [softmax_max, softmax_sum]]
-    
+
     return cur_q, cur_k, cur_v, cur_attn_out, cur_dout, cur_softmax_max, cur_softmax_sum, cur_attn_mask
 
 ""","""         dv[0].add_(cur_dv)
     else:
         dq[1].add_(cur_dq)
-        cur_dk = cur_dk.view(dk.shape) # [2s, b, h] -> [2, s, b, h]
+        cur_dk = cur_dk.view(dk.shape)  # [2s, b, h] -> [2, s, b, h]
         cur_dv = cur_dv.view(dv.shape)
         dk.add_(cur_dk)
         dv.add_(cur_dv)
-    
+
     return dq, dk, dv
 
 
 def cal_row(cur_q, cur_k, cur_v, s, attn_info):
     # q: [s, b, h], kv: [2s, b, h]
     n, pse, pse_type, attn_mask, softmax_scale, keep_prob, \\
-    q_index_list, kv_index_list = attn_info
+        q_index_list, kv_index_list = attn_info
 
     # r1c0
     cur_attn_mask = None""",""" 
 def flash_attention_with_alibi_pse(q_block_id, kv_block_id, cur_qkv, attn_info, s):
     n, pse, pse_type, cur_attn_mask, softmax_scale, keep_prob, \\
-    q_index_list, kv_index_list = attn_info
+        q_index_list, kv_index_list = attn_info
     cur_q, cur_k, cur_v = cur_qkv
     if q_block_id == kv_block_id:
         attn_outs_r0c0 = npu_fusion_attention(""",""" def cal_row_grad(cur_q, cur_k, cur_v, cur_dout, cur_softmax_max, cur_softmax_sum, cur_attn_out,
                  attn_grad_info, s, kv_block_id):
     n, pse, pse_type, attn_mask, softmax_scale, keep_prob, rng_states, \\
-    q_index_list, kv_index_list = attn_grad_info
+        q_index_list, kv_index_list = attn_grad_info
 
     cur_attn_mask = None
     attn_grad_outs_r1c0 = npu_fusion_attention_grad(""",""" def flash_attention_with_alibi_pse_grad(q_block_id, kv_block_id, cur_qkv, cur_dout, cur_attn_out,
                                         cur_softmax_max, cur_softmax_sum, attn_grad_info, s):
     n, pse, pse_type, cur_attn_mask, softmax_scale, keep_prob, rng_states, \\
-    q_index_list, kv_index_list = attn_grad_info
+        q_index_list, kv_index_list = attn_grad_info
     cur_q, cur_k, cur_v = cur_qkv
 
     if q_block_id == kv_block_id:""","""         attn_grad_outs.append(torch.cat([attn_grad_outs_r1c0[1], attn_grad_outs_r1c1[1]]))
         attn_grad_outs.append(torch.cat([attn_grad_outs_r1c0[2], attn_grad_outs_r1c1[2]]))
 
-
     return attn_grad_outs
 
 
+from mindspore import nn
 
 
-class AttentionWithCp(torch.autograd.Function):
+class AttentionWithCp(nn.Cell):
     \"\"\"Attention implementation with context parallelism\"\"\"
 
-    
-    @staticmethod
-    def forward(ctx, q, k, v, n, cp_para, softmax_scale=None, attn_mask=None, dropout_p=0.,
-                actual_seq_qlen=None, actual_seq_kvlen=None):    
+    def __init__(self, cp_para):
+        super().__init__()
+        self.block_size = None
+        self.batch_size = None
+        self.cp_para = cp_para
+
+    def construct(self, q, k, v, n, softmax_scale=None, attn_mask=None, dropout_p=0.,
+                  actual_seq_qlen=None, actual_seq_kvlen=None):
         keep_prob = 1. - dropout_p
-        causal = cp_para['causal']
-        cp_group = cp_para.get("cp_group")
-        cp_size = cp_para.get("cp_size")
-        rank = cp_para.get("rank")
-        cp_global_ranks = cp_para.get("cp_global_ranks")
-        cp_group_for_send_recv_overlap = cp_para.get("cp_group_for_send_recv_overlap")
+        causal = self.cp_para['causal']
+        cp_group = self.cp_para.get("cp_group")
+        cp_size = self.cp_para.get("cp_size")
+        rank = self.cp_para.get("rank")
+        cp_global_ranks = self.cp_para.get("cp_global_ranks")
+        cp_group_for_send_recv_overlap = self.cp_para.get("cp_group_for_send_recv_overlap")
         # WARNING: Degrade to original ring attention, if ranks and comm groups for double ring are not provided
-        cp_inner_ranks = cp_para.get("cp_inner_ranks", [torch.distributed.get_rank()])
-        cp_outer_ranks = cp_para.get("cp_outer_ranks", cp_global_ranks)
-        cp_group_for_intra_window = cp_para.get('cp_group_for_intra_window')
-        cp_group_for_intra_window_send_recv_overlap = cp_para.get('cp_group_for_intra_window_send_recv_overlap')
+        cp_inner_ranks = self.cp_para.get("cp_inner_ranks", [torch.distributed.get_rank()])
+        cp_outer_ranks = self.cp_para.get("cp_outer_ranks", cp_global_ranks)
+        cp_group_for_intra_window = self.cp_para.get('cp_group_for_intra_window')
+        cp_group_for_intra_window_send_recv_overlap = self.cp_para.get('cp_group_for_intra_window_send_recv_overlap')
 
-        pse = cp_para.get("pse")
-        pse_type = cp_para.get("pse_type")
+        pse = self.cp_para.get("pse")
+        pse_type = self.cp_para.get("pse_type")
 
         inner_ring = RingP2P(cp_inner_ranks, cp_group_for_intra_window, cp_group_for_intra_window_send_recv_overlap)
         outer_ring = RingP2P(cp_outer_ranks, cp_group, cp_group_for_send_recv_overlap)""","""             # split chunk[i]~chunk[cp_size-i-1] into chunk[i] and chunk[cp_size-i-1],, [2s, b, h] -> [2, s, b, h]
             q, k, v = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v]]
 
-        cur_kv = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0) # [2, 2, s, b, h]
+        cur_kv = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)  # [2, 2, s, b, h]
         next_kv = torch.empty_like(cur_kv)
         next_round_kv = torch.empty_like(cur_kv)
         attn_out, softmax_max, softmax_sum = None, None, None""","""                 if i < inner_size - 1:
                     inner_ring.async_send_recv(send_tensor=cur_kv, recv_tensor=next_kv)
 
-                cur_k, cur_v = cur_kv[0], cur_kv[1] # [2, s, b, h]
+                cur_k, cur_v = cur_kv[0], cur_kv[1]  # [2, s, b, h]
                 if causal:
                     cur_q, cur_k, cur_v, cur_attn_mask = causal_forward_fetch(q_block_id, kv_block_id,
-                                                                            q, cur_k, cur_v, attn_mask)
+                                                                              q, cur_k, cur_v, attn_mask)
 
                     # flash attention forward
                     if pse is None:""","""                     # [2s, b, h], [b, n, 2s, 8], [b, n, 2s, 8]
                     this_mask = AttentionWithCp.compute_mask(
                         actual_seq_qlen, actual_seq_kvlen,
-                        q_block_id, kv_block_id, 
+                        q_block_id, kv_block_id,
                         attn_mask
                     )
 ""","""                     )
 
                     global_attn_outs = general_out_update(q_block_id, kv_block_id, attn_outs, global_attn_outs)
-                
+
                 if inner_ring.wait():
-                    cur_kv, next_kv = next_kv, cur_kv # double buffer
+                    cur_kv, next_kv = next_kv, cur_kv  # double buffer
                     kv_block_id = (kv_block_id + inner_size - 1) % inner_size + kv_block_offset
 
             if outer_ring.wait():
-                cur_kv, next_round_kv = next_round_kv, cur_kv # double buffer
+                cur_kv, next_round_kv = next_round_kv, cur_kv  # double buffer
                 kv_block_id_outer = (kv_block_id_outer + cp_size - inner_size) % cp_size
 
-
-
         k, v = cur_kv[0], cur_kv[1]
         attn_out, softmax_max, softmax_sum, rng_states = global_attn_outs
         if causal:
             q, k, v = [x.view(-1, *x.shape[2:]) for x in [q, k, v]]
-        
+
         attn_mask = attn_mask if isinstance(attn_mask, list) else [attn_mask]
-        
-        ctx.save_for_backward(q, k, v, *attn_mask, attn_out, softmax_max, softmax_sum)
-        ctx.n = n
-        ctx.causal = causal
-        ctx.softmax_scale = softmax_scale
-        ctx.cp_group = cp_group
-        ctx.cp_size = cp_size
-        ctx.cp_rank = rank
-        ctx.cp_global_ranks = cp_global_ranks
-        ctx.cp_inner_ranks = cp_inner_ranks
-        ctx.cp_outer_ranks = cp_outer_ranks
-        ctx.cp_dkv_outer_ranks = cp_para.get('cp_dkv_outer_ranks', cp_global_ranks)
-        ctx.kv_block_id = kv_block_id
-        ctx.keep_prob = keep_prob
-        ctx.rng_states = rng_states
-        ctx.pse = pse
-        ctx.pse_type = pse_type
-        ctx.cp_group_for_send_recv_overlap = cp_group_for_send_recv_overlap
-        ctx.cp_group_for_intra_window = cp_group_for_intra_window
-        ctx.cp_group_for_intra_window_send_recv_overlap = cp_group_for_intra_window_send_recv_overlap
-        ctx.actual_seq_qlen = actual_seq_qlen
-        ctx.actual_seq_kvlen = actual_seq_kvlen
+
+        self.k = k
+        self.v = v
+        self.attn_mask = attn_mask
+        # save forward outputs
+        self.softmax_max = softmax_max
+        self.softmax_sum = softmax_sum
+        self.causal = causal
+        self.softmax_scale = softmax_scale
+        self.cp_group = cp_group
+        self.cp_size = cp_size
+        self.cp_rank = rank
+        self.cp_global_ranks = cp_global_ranks
+        self.cp_inner_ranks = cp_inner_ranks
+        self.cp_outer_ranks = cp_outer_ranks
+        self.cp_dkv_outer_ranks = self.cp_para.get('cp_dkv_outer_ranks', cp_global_ranks)
+        self.kv_block_id = kv_block_id
+        self.keep_prob = keep_prob
+        self.rng_states = rng_states
+        self.pse = pse
+        self.cp_group_for_send_recv_overlap = cp_group_for_send_recv_overlap
+        self.cp_group_for_intra_window = cp_group_for_intra_window
+        self.cp_group_for_intra_window_send_recv_overlap = cp_group_for_intra_window_send_recv_overlap
 
         return attn_out
 
-    @staticmethod
-    def backward(ctx, dout):
-        q, k, v, *attn_mask, attn_out, softmax_max, softmax_sum = ctx.saved_tensors
+    def bprop(self, q, k, v, n, softmax_scale, attn_mask, dropout_p, actual_seq_qlen, actual_seq_kvlen, attn_out, dout):
+        k = self.k
+        v = self.v
+        cp_para = self.cp_para
+        softmax_max = self.softmax_max
+        softmax_sum = self.softmax_sum
+        attn_mask = self.attn_mask
+
         if len(attn_mask) == 1:
             attn_mask = attn_mask[0]
 
-        n = ctx.n
-        causal = ctx.causal
-        softmax_scale = ctx.softmax_scale
-        cp_group = ctx.cp_group
-        cp_size = ctx.cp_size
-        rank = ctx.cp_rank
-        keep_prob = ctx.keep_prob
-        rng_states = ctx.rng_states
-        pse = ctx.pse
-        pse_type = ctx.pse_type
-        cp_group_for_send_recv_overlap = ctx.cp_group_for_send_recv_overlap
-        cp_group_for_intra_window = ctx.cp_group_for_intra_window
-        cp_group_for_intra_window_send_recv_overlap = ctx.cp_group_for_intra_window_send_recv_overlap
+        causal = self.causal
+        cp_group = self.cp_group
+        cp_size = self.cp_size
+        rank = self.cp_rank
+        keep_prob = self.keep_prob
+        rng_states = self.rng_states
+        pse = self.pse
+        cp_group_for_send_recv_overlap = self.cp_group_for_send_recv_overlap
+        cp_group_for_intra_window = self.cp_group_for_intra_window
+        cp_group_for_intra_window_send_recv_overlap = self.cp_group_for_intra_window_send_recv_overlap
+        cp_shape_order = cp_para.get("cp_shape_order", "SBH")
+
         # Reversed order of forward
-        inner_size = len(ctx.cp_inner_ranks)
-        outer_size = len(ctx.cp_outer_ranks)
-        
-        intra_kv_comm = RingP2P(ctx.cp_inner_ranks, cp_group_for_intra_window, cp_group_for_intra_window_send_recv_overlap, is_backward=True)
-        intra_dkv_comm = RingP2P(ctx.cp_inner_ranks, cp_group_for_intra_window, cp_group_for_intra_window_send_recv_overlap, is_backward=True)
-        inter_kv_comm = RingP2P(ctx.cp_outer_ranks, cp_group, cp_group_for_send_recv_overlap, is_backward=True)
-        inter_dkv_comm = RingP2P(ctx.cp_dkv_outer_ranks, cp_group, cp_group_for_send_recv_overlap, is_backward=True)
+        inner_size = len(self.cp_inner_ranks)
+        outer_size = len(self.cp_outer_ranks)
 
+        intra_kv_comm = RingP2P(self.cp_inner_ranks, cp_group_for_intra_window,
+                                cp_group_for_intra_window_send_recv_overlap, is_backward=True)
+        intra_dkv_comm = RingP2P(self.cp_inner_ranks, cp_group_for_intra_window,
+                                 cp_group_for_intra_window_send_recv_overlap, is_backward=True)
+        inter_kv_comm = RingP2P(self.cp_outer_ranks, cp_group, cp_group_for_send_recv_overlap, is_backward=True)
+        inter_dkv_comm = RingP2P(self.cp_dkv_outer_ranks, cp_group, cp_group_for_send_recv_overlap, is_backward=True)
 
         if causal:
             # split chunk[i]~chunk[cp_size-i-1] into chunk[i] and chunk[cp_size-i-1], [2s, b, h] -> [2, s, b, h]""","""             else:
                 this_mask = AttentionWithCp.compute_mask(
                     ctx.actual_seq_qlen, ctx.actual_seq_kvlen,
-                    q_block_id, kv_block_id, 
+                    q_block_id, kv_block_id,
                     attn_mask
-                )                
+                )
                 attn_grad_outs = torch_npu.npu_fusion_attention_grad(
                     q, cur_k, cur_v, dout, n,
                     "SBH",""","""                     numels=rng_states[kv_block_id][2],
                 )
                 cur_dq, cur_dk, cur_dv = attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]
-            
-            return cur_dq, cur_dk, cur_dv
 
+            return cur_dq, cur_dk, cur_dv
 
-        cur_kv_dkv = torch.zeros((2, 2, *k.shape), dtype=k.dtype, device=k.device) # [2, 2, 2, s, b, h]
+        cur_kv_dkv = torch.zeros((2, 2, *k.shape), dtype=k.dtype, device=k.device)  # [2, 2, 2, s, b, h]
         cur_kv_dkv[0].copy_(torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0))
         next_kv_dkv = cur_kv_dkv.clone()
         next_round_kv_dkv = cur_kv_dkv.clone()""","""         next_kv, next_dkv = next_kv_dkv[0], next_kv_dkv[1]
         next_round_kv, next_round_dkv = next_round_kv_dkv[0], next_round_kv_dkv[1]
 
-        q_block_id, kv_block_id, kv_block_id_outer = rank, ctx.kv_block_id, ctx.kv_block_id
+        q_block_id, kv_block_id, kv_block_id_outer = rank, self.kv_block_id, self.kv_block_id
 
-
-        dq = torch.zeros_like(q)# [2, s, b, h]
+        dq = torch.zeros_like(q)  # [2, s, b, h]
         for j in range(outer_size):
             kv_block_id = kv_block_id_outer
             kv_block_offset = (kv_block_id // inner_size) * inner_size""","""             if j + 1 != outer_size:
                 inter_kv_comm.async_send_recv(send_tensor=cur_kv, recv_tensor=next_round_kv)
 
-
             for i in range(inner_size):
                 if i > 0:
                     intra_kv_comm.wait()""",""" 
                 if i + 1 != inner_size:
                     intra_kv_comm.async_send_recv(send_tensor=cur_kv, recv_tensor=next_kv)
-                
+
                 cur_k, cur_v = cur_kv[0], cur_kv[1]
 
                 dq_step, dk_step, dv_step = backward_step_helper(q_block_id, kv_block_id, q, cur_k, cur_v)
 
-                if i == 0 and j > 0: # receive dk dv from last window
+                if i == 0 and j > 0:  # receive dk dv from last window
                     inter_dkv_comm.wait()
                     cur_dkv, next_round_dkv = next_round_dkv, cur_dkv
-                elif i > 0: # receive dk dv from last step
+                elif i > 0:  # receive dk dv from last step
                     intra_dkv_comm.wait()
                     cur_dkv, next_dkv = next_dkv, cur_dkv
-                
+
                 dk, dv = cur_dkv[0], cur_dkv[1]
                 # update qkv grades
                 if causal:""",""" 
         dk, dv = cur_dkv[0], cur_dkv[1]
 
-
         # [2, s, b, h] -> [2s, b, h]
         if causal:
             dq, dk, dv = [x.view(-1, *x.shape[2:]) for x in [dq, dk, dv]]""","""             seq_batch = [seq1d[indexes[i]:indexes[i + 1]] for i in range(len(indexes) - 1)]
             return [[elem - i * seq_len for elem in seq] for i, seq in enumerate(seq_batch)]
 
-        if actual_seq_qlen:  
+        if actual_seq_qlen:
             actual_seq_qlen = batch_index(actual_seq_qlen)
             actual_seq_kvlen = batch_index(actual_seq_kvlen)
             block_size = cls.block_size
             actual_seq_qlen = [[0] + lst for lst in actual_seq_qlen]
             sub_seq_qlen = [torch.tensor(x[1:]) - torch.tensor(x[:-1]) for x in actual_seq_qlen]
-            sub_seq_qid = torch.stack([torch.arange(len(lst)).repeat_interleave(lst) for lst in sub_seq_qlen]).npu() # B S
+            sub_seq_qid = torch.stack(
+                [torch.arange(len(lst)).repeat_interleave(lst) for lst in sub_seq_qlen]).npu()  # B S
 
             this_ids = sub_seq_qid[:, q_block_id * block_size:(q_block_id + 1) * block_size].npu()
-            this_tile = this_ids.unsqueeze(dim=2) # B S 1
+            this_tile = this_ids.unsqueeze(dim=2)  # B S 1
 
             actual_seq_kvlen = [[0] + lst for lst in actual_seq_kvlen]
             sub_seq_kvlen = [torch.tensor(x[1:]) - torch.tensor(x[:-1]) for x in actual_seq_kvlen]
-            sub_seq_kvid = torch.stack([torch.arange(len(lst)).repeat_interleave(lst) for lst in sub_seq_kvlen]).npu() # B S
+            sub_seq_kvid = torch.stack(
+                [torch.arange(len(lst)).repeat_interleave(lst) for lst in sub_seq_kvlen]).npu()  # B S
             other_ids = sub_seq_kvid[:, kv_block_id * block_size:(kv_block_id + 1) * block_size].npu()
-            other_tile = other_ids.unsqueeze(dim=1) # B 1 S
+            other_tile = other_ids.unsqueeze(dim=1)  # B 1 S
 
-            mask = this_tile == other_tile # B S S
+            mask = this_tile == other_tile  # B S S
             if kv_block_id > q_block_id:
                 mask = torch.zeros_like(mask)
             elif kv_block_id == q_block_id:
                 mask = torch.tril(mask)
-            
+
             return torch.logical_not(mask).unsqueeze(dim=1).npu()  # B 1 S S
         else:
-            return attn_mask[kv_block_id] if isinstance(attn_mask, list) else None  
-            
+            return attn_mask[kv_block_id] if isinstance(attn_mask, list) else None
 
 
 def ringattn_context_parallel(q, k, v, n, cp_para, softmax_scale=None, attn_mask=None, dropout_p=0.,
                               actual_seq_qlen=None, actual_seq_kvlen=None):
-    AttentionWithCp.block_size = q.shape[0]
-    AttentionWithCp.batch_size = q.shape[1]
-    out = AttentionWithCp.apply(
-        q, k, v, n, cp_para, softmax_scale, attn_mask, dropout_p,
+    attn_with_cp = AttentionWithCp(cp_para)
+    attn_with_cp.block_size = q.shape[0]
+    attn_with_cp.batch_size = q.shape[1]
+    out = attn_with_cp(
+        q, k, v, n, softmax_scale, attn_mask, dropout_p,
         actual_seq_qlen, actual_seq_kvlen
     )
     return out
"""
],
"mindspeed/core/context_parallel/utils.py": [""" 
 from megatron.training import get_args
 from megatron.core.parallel_state import get_context_parallel_global_ranks
-from mindspeed.ops.npu_ring_attention_update import npu_ring_attention_update
+# from mindspeed.ops.npu_ring_attention_update import npu_ring_attention_update
 from mindspeed.core.parallel_state import get_context_parallel_for_hybrid_ring_global_ranks
-from mindspeed.op_builder import AdaptiveCpOpBuilder
+# from mindspeed.op_builder import AdaptiveCpOpBuilder
 
 
 ADAPTIVE_CP_SCHEDULING_INFO = None""","""         return mask_list
 
 
-adaptive_cp_ops = AdaptiveCpOps()
+# adaptive_cp_ops = AdaptiveCpOps()
 
"""
],
"mindspeed/core/models/common/embeddings/rotary_pos_embedding.py": [""" from megatron.training import get_args
 from megatron.core import parallel_state
 from mindspeed.utils import get_position_ids
-from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding
+# from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding
 
 from mindspeed.core.parallel_state import (get_context_parallel_for_hybrid_ulysses_world_size,
                                              get_context_parallel_for_hybrid_ulysses_rank,""","""         t = npu_rotary_position_embedding(t.contiguous(), cos_, sin_, mode).to(t.dtype)
     else:
         t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
-
     return torch.cat((t, t_pass), dim=-1)
 
 
"""
],
"mindspeed/core/tensor_parallel/random.py": [""" import torch
 from torch import _C
 from torch_npu.npu import _lazy_call, device as device_ctx_manager
-from torch.utils.checkpoint import _get_autocast_kwargs
 from megatron.training import get_args
 from megatron.core.tensor_parallel.utils import gather_split_1d_tensor
 from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
"""
],
"mindspeed/core/transformer/moe/moe_layer_overlap_all2all.py": [""" from megatron.training import get_args
 from mindspeed.core.transformer.moe.comm_utils import async_all_to_all, async_all_gather
 from mindspeed.core.transformer.moe.moe_utils import forward_func, backward_func
-from mindspeed.ops.gmm import GMMFunction
+# from mindspeed.ops.gmm import GMMFunction
 from mindspeed.core.transformer.moe.moe_utils import AG_SHARED_EXPERTS_INPUTS
 
 
"""
],
"mindspeed/core/transformer/moe/router.py": ["""         return _gather_along_first_dim_moe_async(input_, async_op=True)
 
     @staticmethod
-    def backward(ctx, grad_output, grad_handle):
+    def backward(ctx, grad_output, grad_handle=None):
+        if isinstance(grad_output, tuple) and len(grad_output) > 1 and grad_output[1] is None:
+            grad_output = grad_output[0]
         return _reduce_scatter_along_first_dim_moe(grad_output)
 
 
"""
],
"mindspeed/core/transformer/moe/token_dispatcher.py": [""" # Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
 # Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
 import torch
-from torch_npu.utils.collect_env import get_cann_version
+import mindspore
+# from torch_npu.utils.collect_env import get_cann_version
 from megatron.training import get_args
 from megatron.core import parallel_state, tensor_parallel
 from megatron.core.transformer.moe.moe_utils import permute, unpermute""",""" 
 
 def is_less_or_equal_rc2_cann_version():
-    cann_starts_with = ('8.0.RC1', '8.0.RC2')
-    cann_all = ('not known', '8.0.T1', '8.0.T2', '8.0.T3', '8.0.T37', '8.0.T5', '8.0.T6', '8.0.T7',
-                '8.0.T8', '8.0.T10', '8.0.T13', '8.0.T16', '8.0.T50', '8.0.T51', '8.0.T52')
-    cann_version = get_cann_version()
-    return cann_version in cann_all or cann_version.startswith(cann_starts_with)
+    # cann_starts_with = ('8.0.RC1', '8.0.RC2')
+    # cann_all = ('not known', '8.0.T1', '8.0.T2', '8.0.T3', '8.0.T37', '8.0.T5', '8.0.T6', '8.0.T7',
+    #             '8.0.T8', '8.0.T10', '8.0.T13', '8.0.T16', '8.0.T50', '8.0.T51', '8.0.T52')
+    # cann_version = get_cann_version()
+    # return cann_version in cann_all or cann_version.startswith(cann_starts_with)
+    return True
 
 
 cann_version_check = is_less_or_equal_rc2_cann_version()""",""" 
         with torch.no_grad():
             gi_handle.wait()
-            global_local_mask = (global_indices >= self.local_expert_indices[0]) & \\
-                                (global_indices <= self.local_expert_indices[-1])
+            tmp_a = global_indices >= self.local_expert_indices[0]
+            tmp_b = global_indices <= self.local_expert_indices[-1]
+            global_local_mask = mindspore.mint.bitwise_and(tmp_a, tmp_b)
             local_indices = global_indices.masked_select(global_local_mask)
             self.indices = torch.argsort(local_indices.float(), dim=0)
             num_global_experts = self.num_local_experts * parallel_state.get_expert_model_parallel_world_size()
             if args.moe_tp_extend_ep:
                 num_global_experts *= parallel_state.get_tensor_model_parallel_world_size()
-            all_tokens_per_expert = torch.histc(
-                global_indices,
+            all_tokens_per_expert = mindspore.ops.histc(
+                global_indices.to(torch.int32),
                 bins=num_global_experts,
                 min=0,
                 max=num_global_experts - 1,""","""             return output.to(ori_dtype)
         return output
 
-    def backward(self, grad_input):
+    def backward(self, ctx, grad_input):
         map_ = self.map_
         grad_output = grad_input.index_select(0, map_[0])
         return None, None, grad_output
"""
],
"mindspeed/megatron_adaptor.py": ["""     aspm.register_patch('importlib.metadata.version', version_wrapper)
     aspm.register_patch('transformer_engine.pytorch.LayerNormLinear', torch.nn.Module, create_dummy=True)
     aspm.register_patch('transformer_engine.pytorch.DotProductAttention', torch.nn.Module, create_dummy=True)
-    aspm.register_patch('transformer_engine.pytorch.Linear', torch.nn.Module, create_dummy=True)
+    # aspm.register_patch('transformer_engine.pytorch.Linear', torch.nn.Module, create_dummy=True)
     aspm.register_patch('flash_attn.flash_attn_interface.flash_attn_unpadded_func', create_dummy=True)
 
 
 def apex_adaptation(aspm):
     from .optimizer.adamw import AdamW
     from .core.fusions.fused_layer_norm import fused_layer_norm_affine
-    from .ops.npu_matmul_add import npu_matmul_add_fp32, npu_matmul_add_fp16
+    # from .ops.npu_matmul_add import npu_matmul_add_fp32, npu_matmul_add_fp16
     aspm.register_patch('apex.optimizers.FusedAdam', AdamW, create_dummy=True)
     aspm.register_patch('amp_C.multi_tensor_l2norm', multi_tensor_l2norm, create_dummy=True)
     aspm.register_patch('amp_C.multi_tensor_scale', multi_tensor_scale, create_dummy=True)""","""     aspm.register_patch('apex.multi_tensor_apply.multi_tensor_applier', multi_tensor_applier, create_dummy=True)
     aspm.register_patch('apex.normalization.fused_layer_norm.fused_layer_norm_affine', fused_layer_norm_affine,
                         create_dummy=True)
-    aspm.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32', npu_matmul_add_fp32, create_dummy=True)
-    aspm.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16', npu_matmul_add_fp16, create_dummy=True)
+    # aspm.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32', npu_matmul_add_fp32, create_dummy=True)
+    # aspm.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16', npu_matmul_add_fp16, create_dummy=True)
 
 
 def torch_adaptation(aspm):""",""" 
 
 def mcore_tensor_parallel_adaptation_l1(aspm):
-    from .core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy_forward
+    # from .core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy_forward
     # use logical negation followed by multiplication to achieve the same effect as setting selected elements to zero
-    aspm.register_patch('megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward',
-                        vocab_parallel_cross_entropy_forward)
+    # aspm.register_patch('megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward',
+    #                     vocab_parallel_cross_entropy_forward)
+    pass
 
 
 def mcore_tensor_parallel_adaptation(aspm):
"""
],
"mindspeed/model/transformer.py": ["""                                            get_ring_group_for_intra_window_send_recv_overlap)
 from mindspeed.core.fusions.fused_bias_swiglu import fused_swiglu
 from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
-from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
+# from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
 from mindspeed.core.tensor_parallel.layers import Nd_ParallelLinear
 from mindspeed.core.tensor_parallel.checkpoint_manager import get_pipeline_checkpoint_manager
 from mindspeed.model.alibi_mask import AlibiForFusionAttnSingleton, get_slopes""","""     args = get_args()
     if args.use_flash_attn and (args.seq_length > 2048 or args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']):
         args.sparse_mode = 2
-        _GLOBAL_ATTN_MASK = torch.triu(torch.ones([2048, 2048], dtype=bool, device=torch.cuda.current_device()), diagonal=1)
+        _GLOBAL_ATTN_MASK = torch.triu(torch.ones([2048, 2048], dtype=torch.bool, device=torch.cuda.current_device()), diagonal=1)
     else:
         args.sparse_mode = 0
         if args.attention_mask_on_cpu:
"""
],
"mindspeed/op_builder/algorithm_builder.py": [""" from mindspeed.op_builder.builder import MindSpeedOpBuilder
-
+from mindspore.utils import _reuse_data_ptr
 
 class AlgorithmOpBuilder(MindSpeedOpBuilder):
     OP_NAME = "algorithm"
+    reuse_data_ptr = _reuse_data_ptr
 
     def __init__(self):
         super(AlgorithmOpBuilder, self).__init__(self.OP_NAME)
 
     def sources(self):
-        return ['ops/csrc/algorithm/algorithm.cpp']
+        raise NotImplementedError
+
+    def load(self, verbose=True):
+        return self
"""
],

}


LINE_RULES_acclerate_mindspeed_mm = {
"mindspeed/core/context_parallel/utils.py": [""" 
 from megatron.training import get_args
 from megatron.core.parallel_state import get_context_parallel_global_ranks
-from mindspeed.ops.npu_ring_attention_update import npu_ring_attention_update
+#from mindspeed.ops.npu_ring_attention_update import npu_ring_attention_update
 from mindspeed.core.parallel_state import get_context_parallel_for_hybrid_ring_global_ranks
 from mindspeed.op_builder import AdaptiveCpOpBuilder
 ""","""     - updated_softmax_sum (Tensor): The updated sum of the softmax distribution.
     \"\"\"
     _args = get_args()
-    if hasattr(_args, 'use_fused_ring_attention_update') and _args.use_fused_ring_attention_update:
-        return npu_ring_attention_update(prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out,
-                                         cur_softmax_max, cur_softmax_sum, actual_seq_qlen, layout)
+    #if hasattr(_args, 'use_fused_ring_attention_update') and _args.use_fused_ring_attention_update:
+    #    return npu_ring_attention_update(prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out,
+    #                                     cur_softmax_max, cur_softmax_sum, actual_seq_qlen, layout)
     return forward_update_without_fused(prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out,
                                     cur_softmax_max, cur_softmax_sum, actual_seq_qlen, layout)
 
"""
],
"mindspeed/core/models/common/embeddings/rotary_pos_embedding.py": [""" from megatron.training import get_args
 from megatron.core import parallel_state
 from mindspeed.utils import get_position_ids
-from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding
+#from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding
 
 from mindspeed.core.parallel_state import (get_context_parallel_for_hybrid_ulysses_world_size,
                                              get_context_parallel_for_hybrid_ulysses_rank,
                                              get_context_parallel_for_hybrid_ring_world_size,
                                              get_context_parallel_for_hybrid_ring_rank)
-from mindspeed.core.context_parallel.utils import get_remapped_seq_order
+#from mindspeed.core.context_parallel.utils import get_remapped_seq_order
 from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP
 
 ""","""     cos_ = (torch.cos(freqs) * _mscale).to(t.dtype)
     sin_ = (torch.sin(freqs) * _mscale).to(t.dtype)
 
-    if args.use_fused_rotary_pos_emb:
-        mode = 1 if rotary_interleaved else 0
-        t = npu_rotary_position_embedding(t.contiguous(), cos_, sin_, mode).to(t.dtype)
-    else:
-        t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
+    #if args.use_fused_rotary_pos_emb:
+    #    mode = 1 if rotary_interleaved else 0
+    #    t = npu_rotary_position_embedding(t.contiguous(), cos_, sin_, mode).to(t.dtype)
+    #else:
+    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
 
     return torch.cat((t, t_pass), dim=-1)
 ""","""     ulys_rank = get_context_parallel_for_hybrid_ulysses_rank()
     adap_rank = get_context_parallel_for_hybrid_ring_rank()
 
-    remapped_seq_order = get_remapped_seq_order()
+    remapped_seq_order = None #get_remapped_seq_order()
     if remapped_seq_order is not None:
         per = pos_emd.shape[seq_dim] // adap_size // ulys_size
         which_per = adap_rank * ulys_size + ulys_rank
"""
],
"mindspeed/core/models/gpt/gpt_layer_specs.py": [""" from megatron.training import get_args
 from megatron.core.transformer.moe.moe_layer import MoELayer
 from megatron.core.transformer.custom_layers.transformer_engine import TENorm
-from mindspeed.core.transformer.transformer import norm_recompute_forward
+#from mindspeed.core.transformer.transformer import norm_recompute_forward
 from mindspeed.core.transformer.transformer_block import NoopTransformerLayer
-from mindspeed.model.transformer import should_recompute_norm
+#from mindspeed.model.transformer import should_recompute_norm
 import types
 
 ""","""         for layer in self.layers:
             if isinstance(layer, NoopTransformerLayer):
                 continue
-            if should_recompute_norm(layer.layer_number):
-                layer.forward = types.MethodType(norm_recompute_forward, layer)
-    return wrapper
\\ No newline at end of file
+            #if should_recompute_norm(layer.layer_number):
+                #layer.forward = types.MethodType(norm_recompute_forward, layer)
+            #    pass
+    return wrapper
"""
],
"mindspeed/core/tensor_parallel/random.py": [""" import torch
 from torch import _C
 from torch_npu.npu import _lazy_call, device as device_ctx_manager
-from torch.utils.checkpoint import _get_autocast_kwargs
+#from torch.utils.checkpoint import _get_autocast_kwargs
 from megatron.training import get_args
 from megatron.core.tensor_parallel.utils import gather_split_1d_tensor
 from megatron.core.tensor_parallel.random import get_cuda_rng_tracker""",""" 
         return CheckpointFunctionRipipe.apply(function, distribute_saved_activations, *args)
 
-    return wrapper
\\ No newline at end of file
+    return wrapper
"""
],
"mindspeed/megatron_adaptor.py": ["""     aspm.register_patch('importlib.metadata.version', version_wrapper)
     aspm.register_patch('transformer_engine.pytorch.LayerNormLinear', torch.nn.Module, create_dummy=True)
     aspm.register_patch('transformer_engine.pytorch.DotProductAttention', torch.nn.Module, create_dummy=True)
-    aspm.register_patch('transformer_engine.pytorch.Linear', torch.nn.Module, create_dummy=True)
+    # aspm.register_patch('transformer_engine.pytorch.Linear', torch.nn.Module, create_dummy=True)
     aspm.register_patch('flash_attn.flash_attn_interface.flash_attn_unpadded_func', create_dummy=True)
 
 
 def apex_adaptation(aspm):
     from .core.fusions.fused_layer_norm import fused_layer_norm_affine
-    from .ops.npu_matmul_add import npu_matmul_add_fp32, npu_matmul_add_fp16
+    #from .ops.npu_matmul_add import npu_matmul_add_fp32, npu_matmul_add_fp16
     aspm.register_patch('amp_C.multi_tensor_l2norm', multi_tensor_l2norm, create_dummy=True)
     aspm.register_patch('amp_C.multi_tensor_scale', multi_tensor_scale, create_dummy=True)
     aspm.register_patch('fused_layer_norm_cuda', create_dummy=True)
     aspm.register_patch('apex.multi_tensor_apply.multi_tensor_applier', multi_tensor_applier, create_dummy=True)
     aspm.register_patch('apex.normalization.fused_layer_norm.fused_layer_norm_affine', fused_layer_norm_affine,
                         create_dummy=True)
-    aspm.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32', npu_matmul_add_fp32, create_dummy=True)
-    aspm.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16', npu_matmul_add_fp16, create_dummy=True)
+    #aspm.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32', npu_matmul_add_fp32, create_dummy=True)
+    #aspm.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16', npu_matmul_add_fp16, create_dummy=True)
 
 
 def torch_adaptation(aspm):""","""
 def mcore_tensor_parallel_adaptation_l0(aspm):
     from .core.tensor_parallel.random import _set_cuda_rng_state
-    aspm.register_patch('megatron.core.tensor_parallel.random._set_cuda_rng_state', _set_cuda_rng_state)
+    # aspm.register_patch('megatron.core.tensor_parallel.random._set_cuda_rng_state', _set_cuda_rng_state)


 def mcore_tensor_parallel_adaptation_l1(aspm):
     from .core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy_forward
     from .utils import checkpoint_forward_wrapper, checkpoint_backward_wrapper
     # use logical negation followed by multiplication to achieve the same effect as setting selected elements to zero
-    aspm.register_patch('megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward',
-                        vocab_parallel_cross_entropy_forward)
+    #aspm.register_patch('megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward',
+    #                    vocab_parallel_cross_entropy_forward)
     aspm.register_patch('megatron.core.tensor_parallel.random.CheckpointFunction.forward',
                         checkpoint_forward_wrapper)
     aspm.register_patch('megatron.core.tensor_parallel.random.CheckpointFunction.backward',""",""" 
 def megatron_training_adaptation_l0(aspm, args):
     from .initialize import _compile_dependencies, set_jit_fusion_options_wrapper
-    from .utils import get_batch_on_this_cp_rank
+    #from .utils import get_batch_on_this_cp_rank
     from .training import pretrain
     from .arguments import parse_args_wrapper, validate_args_wrapper, core_transformer_config_from_args_wrapper
     from .yaml_arguments import core_transformer_config_from_yaml_wrapper, print_args_wrapper""","""     aspm.register_patch('megatron.training.yaml_arguments.core_transformer_config_from_yaml',
                         core_transformer_config_from_yaml_wrapper)
     aspm.register_patch('megatron.training.initialize._compile_dependencies', _compile_dependencies)
-    aspm.register_patch('megatron.training.utils.get_batch_on_this_cp_rank', get_batch_on_this_cp_rank)
+    #aspm.register_patch('megatron.training.utils.get_batch_on_this_cp_rank', get_batch_on_this_cp_rank)
     aspm.register_patch('megatron.training.arguments.parse_args', parse_args_wrapper)
     aspm.register_patch('megatron.training.arguments.validate_args', validate_args_wrapper)
     aspm.register_patch('megatron.training.arguments._print_args', print_args_wrapper)""","""         aspm.register_patch('megatron.training.initialize_megatron', megatron.training.initialize.initialize_megatron)
 
 
-def mcore_moe_adaptation_l0(pm):
-    from .core.transformer.moe.grouped_gemm_util import Ops, grouped_gemm_is_available, get_device_capability
-    pm.register_patch('megatron.core.transformer.moe.grouped_gemm_util.ops', Ops)
-    pm.register_patch('megatron.core.transformer.moe.grouped_gemm_util.grouped_gemm_is_available',
-                      grouped_gemm_is_available)
-    pm.register_patch('torch.cuda.get_device_capability', get_device_capability)
-
-
-def mcore_moe_adaptation(pm, args):
-    from .core.pipeline_parallel.schedules import forward_step
-    pm.register_patch('megatron.core.pipeline_parallel.schedules.forward_step',
-                        forward_step)
-    if args.moe_permutation_async_comm:
-        if hasattr(args, 'moe_token_dispatcher_type') and args.moe_token_dispatcher_type == 'alltoall':
-            from .core.transformer.moe.experts import sequential_mlp_forward
-            from .core.transformer.moe.moe_utils import permute, unpermute
-            if args.moe_tp_extend_ep:
-                from .core.transformer.moe.token_dispatcher import (
-                    preprocess_tp_extend_ep, alltoall_token_unpermutation_tp_extend_ep,
-                    alltoall_token_permutation_tp_extend_ep
-                )
-                from .core.transformer.moe.router import routing_tp_extend_ep
-                from .core.transformer.moe.moe_layer import base_moe_init_wrapper
-                pm.register_patch('megatron.core.transformer.moe.moe_layer.BaseMoELayer.__init__',
-                                  base_moe_init_wrapper)
-                pm.register_patch(
-                    'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.preprocess',
-                    preprocess_tp_extend_ep)
-                pm.register_patch('megatron.core.transformer.moe.router.TopKRouter.routing', routing_tp_extend_ep)
-
-                if args.moe_alltoall_overlap_comm:
-                    from .core.transformer.moe.token_dispatcher import alltoall_token_permutation_new, \\
-                        alltoall_token_unpermutation_new
-                    from .core.transformer.moe.experts import group_mlp_forward
-                    from .core.transformer.mlp import mlp_init
-                    pm.register_patch('megatron.core.transformer.mlp.MLP.__init__', mlp_init)
-                    pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.forward', group_mlp_forward)
-                    pm.register_patch(
-                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
-                        alltoall_token_permutation_new)
-                    pm.register_patch(
-                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
-                        alltoall_token_unpermutation_new)
-                else:
-                    pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
-                                      alltoall_token_permutation_tp_extend_ep)
-                    pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
-                                      alltoall_token_unpermutation_tp_extend_ep)
-            else:
-                from .core.transformer.moe.token_dispatcher import preprocess, alltoall_token_permutation
-                pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.preprocess', preprocess)
-                if args.moe_alltoall_overlap_comm:
-                    from .core.transformer.moe.token_dispatcher import preprocess, alltoall_token_permutation
-                    from .core.transformer.moe.token_dispatcher import alltoall_token_permutation_new, \\
-                        alltoall_token_unpermutation_new
-                    from .core.transformer.moe.experts import group_mlp_forward
-                    from .core.transformer.mlp import mlp_init
-                    pm.register_patch('megatron.core.transformer.mlp.MLP.__init__', mlp_init)
-                    pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.forward', group_mlp_forward)
-                    pm.register_patch(
-                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
-                        alltoall_token_permutation_new)
-                    pm.register_patch(
-                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
-                        alltoall_token_unpermutation_new)
-                else:
-                    pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
-                                      alltoall_token_permutation)
-            pm.register_patch('megatron.core.transformer.moe.experts.SequentialMLP.forward', sequential_mlp_forward)
-            pm.register_patch('megatron.core.transformer.moe.moe_utils.permute', permute)
-            pm.register_patch('megatron.core.transformer.moe.moe_utils.unpermute', unpermute)
-        else:
-            from .core.transformer.moe.router import aux_loss_load_balancing
-            pm.register_patch('megatron.core.transformer.moe.router.TopKRouter.aux_loss_load_balancing', aux_loss_load_balancing)
-
-            if args.moe_tp_extend_ep:
-                from .core.transformer.moe.moe_layer import base_moe_init_wrapper
-                pm.register_patch('megatron.core.transformer.moe.moe_layer.BaseMoELayer.__init__', base_moe_init_wrapper)
-
-            if args.moe_allgather_overlap_comm:
-                from .core.transformer.moe.token_dispatcher import (allgather_token_permutation_new,
-                                                                    allgather_token_unpermutation_new)
-                from .core.transformer.moe.experts import group_mlp_forward
-                from .core.transformer.mlp import mlp_init
-                pm.register_patch('megatron.core.transformer.mlp.MLP.__init__', mlp_init)
-                pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.forward', group_mlp_forward)
-                pm.register_patch(
-                    'megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_permutation',
-                    allgather_token_permutation_new)
-                pm.register_patch(
-                    'megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_unpermutation',
-                    allgather_token_unpermutation_new)
-            else:
-                from .core.transformer.moe.token_dispatcher import (allgather_token_permutation,
-                                                                    allgather_token_unpermutation)
-                pm.register_patch(
-                    'megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_permutation',
-                    allgather_token_permutation)
-                pm.register_patch(
-                    'megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_unpermutation',
-                    allgather_token_unpermutation)
-
-        from .core.transformer.moe.moe_layer import moe_layer_init_wrapper
-        pm.register_patch('megatron.core.transformer.moe.moe_layer.MoELayer.__init__', moe_layer_init_wrapper)
-
-    from .core.transformer.moe.experts import groupedmlp_init_wrapper, groupedmlp_forward
-    pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.__init__', groupedmlp_init_wrapper)
-    if not args.moe_alltoall_overlap_comm and not args.moe_allgather_overlap_comm:
-        pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.forward', groupedmlp_forward)
-
-
-    if args.use_ascend_mc2 and not hasattr(args, 'moe_grouped_gemm'):
-        # MoE MLP not use mc2 linear
-        from .core.models.gpt.gpt_layer_specs import build_layers_wrapper
-        from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
-        from megatron.core.transformer.transformer_block import TransformerBlock
-        TransformerBlock._build_layers = build_layers_wrapper(TransformerBlock._build_layers, ColumnParallelLinear.forward,
-            RowParallelLinear.forward)
-
-
-def deepspeed_moe_adaptation(pm, args):
-    if args.use_pipe_experts or args.use_nanopipe:
-        from .core.tensor_parallel.layers import (row_parallel_moe, column_parallel_moe,
-                                                  linear_with_grad_accumulation_and_async_allreduce_moe)
-        pm.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear.forward', row_parallel_moe)
-        pm.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear.forward', column_parallel_moe)
-        pm.register_patch('megatron.core.tensor_parallel.layers.linear_with_grad_accumulation_and_async_allreduce',
-                          linear_with_grad_accumulation_and_async_allreduce_moe)
-    if args.use_pipe_experts:
-        from .core.distributed.param_and_grad_buffer import pipe_register_grad_ready
-        pm.register_patch('megatron.core.distributed.ParamAndGradBuffer.register_grad_ready', pipe_register_grad_ready)
+#def mcore_moe_adaptation_l0(pm):
+#    from .core.transformer.moe.grouped_gemm_util import Ops, grouped_gemm_is_available, get_device_capability
+#    pm.register_patch('megatron.core.transformer.moe.grouped_gemm_util.ops', Ops)
+#    pm.register_patch('megatron.core.transformer.moe.grouped_gemm_util.grouped_gemm_is_available',
+#                      grouped_gemm_is_available)
+#    pm.register_patch('torch.cuda.get_device_capability', get_device_capability)
+#
+#
+#def mcore_moe_adaptation(pm, args):
+#    from .core.pipeline_parallel.schedules import forward_step
+#    pm.register_patch('megatron.core.pipeline_parallel.schedules.forward_step',
+#                        forward_step)
+#    if args.moe_permutation_async_comm:
+#        if hasattr(args, 'moe_token_dispatcher_type') and args.moe_token_dispatcher_type == 'alltoall':
+#            from .core.transformer.moe.experts import sequential_mlp_forward
+#            from .core.transformer.moe.moe_utils import permute, unpermute
+#            if args.moe_tp_extend_ep:
+#                from .core.transformer.moe.token_dispatcher import (
+#                    preprocess_tp_extend_ep, alltoall_token_unpermutation_tp_extend_ep,
+#                    alltoall_token_permutation_tp_extend_ep
+#                )
+#                from .core.transformer.moe.router import routing_tp_extend_ep
+#                from .core.transformer.moe.moe_layer import base_moe_init_wrapper
+#                pm.register_patch('megatron.core.transformer.moe.moe_layer.BaseMoELayer.__init__',
+#                                  base_moe_init_wrapper)
+#                pm.register_patch(
+#                    'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.preprocess',
+#                    preprocess_tp_extend_ep)
+#                pm.register_patch('megatron.core.transformer.moe.router.TopKRouter.routing', routing_tp_extend_ep)
+#
+#                if args.moe_alltoall_overlap_comm:
+#                    from .core.transformer.moe.token_dispatcher import alltoall_token_permutation_new, \\
+#                        alltoall_token_unpermutation_new
+#                    from .core.transformer.moe.experts import group_mlp_forward
+#                    from .core.transformer.mlp import mlp_init
+#                    pm.register_patch('megatron.core.transformer.mlp.MLP.__init__', mlp_init)
+#                    pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.forward', group_mlp_forward)
+#                    pm.register_patch(
+#                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
+#                        alltoall_token_permutation_new)
+#                    pm.register_patch(
+#                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
+#                        alltoall_token_unpermutation_new)
+#                else:
+#                    pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
+#                                      alltoall_token_permutation_tp_extend_ep)
+#                    pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
+#                                      alltoall_token_unpermutation_tp_extend_ep)
+#            else:
+#                from .core.transformer.moe.token_dispatcher import preprocess, alltoall_token_permutation
+#                pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.preprocess', preprocess)
+#                if args.moe_alltoall_overlap_comm:
+#                    from .core.transformer.moe.token_dispatcher import preprocess, alltoall_token_permutation
+#                    from .core.transformer.moe.token_dispatcher import alltoall_token_permutation_new, \\
+#                        alltoall_token_unpermutation_new
+#                    from .core.transformer.moe.experts import group_mlp_forward
+#                    from .core.transformer.mlp import mlp_init
+#                    pm.register_patch('megatron.core.transformer.mlp.MLP.__init__', mlp_init)
+#                    pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.forward', group_mlp_forward)
+#                    pm.register_patch(
+#                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
+#                        alltoall_token_permutation_new)
+#                    pm.register_patch(
+#                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
+#                        alltoall_token_unpermutation_new)
+#                else:
+#                    pm.register_patch('megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
+#                                      alltoall_token_permutation)
+#            pm.register_patch('megatron.core.transformer.moe.experts.SequentialMLP.forward', sequential_mlp_forward)
+#            pm.register_patch('megatron.core.transformer.moe.moe_utils.permute', permute)
+#            pm.register_patch('megatron.core.transformer.moe.moe_utils.unpermute', unpermute)
+#        else:
+#            from .core.transformer.moe.router import aux_loss_load_balancing
+#            pm.register_patch('megatron.core.transformer.moe.router.TopKRouter.aux_loss_load_balancing', aux_loss_load_balancing)
+#
+#            if args.moe_tp_extend_ep:
+#                from .core.transformer.moe.moe_layer import base_moe_init_wrapper
+#                pm.register_patch('megatron.core.transformer.moe.moe_layer.BaseMoELayer.__init__', base_moe_init_wrapper)
+#
+#            if args.moe_allgather_overlap_comm:
+#                from .core.transformer.moe.token_dispatcher import (allgather_token_permutation_new,
+#                                                                    allgather_token_unpermutation_new)
+#                from .core.transformer.moe.experts import group_mlp_forward
+#                from .core.transformer.mlp import mlp_init
+#                pm.register_patch('megatron.core.transformer.mlp.MLP.__init__', mlp_init)
+#                pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.forward', group_mlp_forward)
+#                pm.register_patch(
+#                    'megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_permutation',
+#                    allgather_token_permutation_new)
+#                pm.register_patch(
+#                    'megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_unpermutation',
+#                    allgather_token_unpermutation_new)
+#            else:
+#                from .core.transformer.moe.token_dispatcher import (allgather_token_permutation,
+#                                                                    allgather_token_unpermutation)
+#                pm.register_patch(
+#                    'megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_permutation',
+#                    allgather_token_permutation)
+#                pm.register_patch(
+#                    'megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_unpermutation',
+#                    allgather_token_unpermutation)
+#
+#        from .core.transformer.moe.moe_layer import moe_layer_init_wrapper
+#        pm.register_patch('megatron.core.transformer.moe.moe_layer.MoELayer.__init__', moe_layer_init_wrapper)
+#
+#    from .core.transformer.moe.experts import groupedmlp_init_wrapper, groupedmlp_forward
+#    pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.__init__', groupedmlp_init_wrapper)
+#    if not args.moe_alltoall_overlap_comm and not args.moe_allgather_overlap_comm:
+#        pm.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.forward', groupedmlp_forward)
+#
+#
+#    if args.use_ascend_mc2 and not hasattr(args, 'moe_grouped_gemm'):
+#        # MoE MLP not use mc2 linear
+#        from .core.models.gpt.gpt_layer_specs import build_layers_wrapper
+#        from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
+#        from megatron.core.transformer.transformer_block import TransformerBlock
+#        TransformerBlock._build_layers = build_layers_wrapper(TransformerBlock._build_layers, ColumnParallelLinear.forward,
+#            RowParallelLinear.forward)
+#
+#
+#def deepspeed_moe_adaptation(pm, args):
+#    if args.use_pipe_experts or args.use_nanopipe:
+#        from .core.tensor_parallel.layers import (row_parallel_moe, column_parallel_moe,
+#                                                  linear_with_grad_accumulation_and_async_allreduce_moe)
+#        pm.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear.forward', row_parallel_moe)
+#        pm.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear.forward', column_parallel_moe)
+#        pm.register_patch('megatron.core.tensor_parallel.layers.linear_with_grad_accumulation_and_async_allreduce',
+#                          linear_with_grad_accumulation_and_async_allreduce_moe)
+#    if args.use_pipe_experts:
+#        from .core.distributed.param_and_grad_buffer import pipe_register_grad_ready
+#        pm.register_patch('megatron.core.distributed.ParamAndGradBuffer.register_grad_ready', pipe_register_grad_ready)
 
 
 def coc_adaptation(aspm, args):""","""     mcore_models_adaptation_l0(aspm)
     mcore_tensor_parallel_adaptation_l0(aspm)
     mcore_transformer_adaptation_l0(aspm)
-    mcore_moe_adaptation_l0(aspm)
+    #mcore_moe_adaptation_l0(aspm)
     legacy_model_transformer_l0(aspm)
     megatron_training_adaptation_l0(aspm, args)
     # context parallel(ring attention) requires mcore parallel state patch""","""     megatron_training_adaptation(aspm, mindspeed_args)
     ascend_adaptation(aspm, mindspeed_args)
     coc_adaptation(aspm, mindspeed_args)
-    mcore_moe_adaptation(aspm, mindspeed_args)
-    deepspeed_moe_adaptation(aspm, mindspeed_args)
+    #mcore_moe_adaptation(aspm, mindspeed_args)
+    #deepspeed_moe_adaptation(aspm, mindspeed_args)
     zero3_adaptation(aspm, mindspeed_args)
     high_availability_adaptation(aspm, mindspeed_args)
     tensor_2d_adaptation(aspm, mindspeed_args)""","""         adaptation_l2(aspm, mindspeed_args)
 
     aspm.apply_patches()
-
     # accelerate package will check TE on sys.modules，so we need remove this patch
     del sys.modules['transformer_engine']
 
 
-exe_adaptation()
\\ No newline at end of file
+exe_adaptation()
"""
],
"mindspeed/model/transformer.py": [""" from megatron.core.transformer.module import MegatronModule
 
 from mindspeed.core.context_parallel.ulysses_context_parallel import UlyssesContextAttention
-from mindspeed.core.context_parallel.ring_context_parallel import ringattn_context_parallel
+#from mindspeed.core.context_parallel.ring_context_parallel import ringattn_context_parallel
 from mindspeed.core.parallel_state import (get_context_parallel_group_for_hybrid_ulysses,
                                            get_context_parallel_group_for_hybrid_ring,
                                            get_context_parallel_for_hybrid_ring_world_size,""",""" from mindspeed.core.tensor_parallel.tp_2d.parallel_linear_2d import ParallelLinear2D
 from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
 from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP
-from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
+#from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
 from mindspeed.core.tensor_parallel.layers import Nd_ParallelLinear
 from mindspeed.core.tensor_parallel.checkpoint_manager import get_pipeline_checkpoint_manager
 from mindspeed.model.alibi_mask import AlibiForFusionAttnSingleton, get_slopes
-from mindspeed.core.context_parallel.adaptive_context_parallel import adaptive_attn_context_parallel
-from mindspeed.core.context_parallel.utils import get_scheduling_info
+#from mindspeed.core.context_parallel.adaptive_context_parallel import adaptive_attn_context_parallel
+#from mindspeed.core.context_parallel.utils import get_scheduling_info
 
 try:
     from einops import rearrange""","""                 cp_para['cp_group_for_intra_window'] = get_ring_group_for_intra_window()
                 cp_para['cp_group_for_intra_window_send_recv_overlap'] = get_ring_group_for_intra_window_send_recv_overlap()
 
-            output = ringattn_context_parallel(q, k, v, head_num, cp_para, scale, attention_mask, self.dropout_p)
+            #output = ringattn_context_parallel(q, k, v, head_num, cp_para, scale, attention_mask, self.dropout_p)
         else:
-            cp_para['scheduling_info'] = get_scheduling_info()
-            output = adaptive_attn_context_parallel(q, k, v, head_num, cp_para, scale, attention_mask, self.dropout_p)
+            pass
+            #cp_para['scheduling_info'] = get_scheduling_info()
+            #output = adaptive_attn_context_parallel(q, k, v, head_num, cp_para, scale, attention_mask, self.dropout_p)
     else:
         if args.use_fusion_attn_v2:
-            output = npu_fusion_attention(
+            output = torch_npu.npu_fusion_attention(
                 q, k, v, head_num, args.shape_order,
                 pse=self.pse,
                 padding_mask=None,
"""
],
"mindspeed/ops/npu_rotary_position_embedding.py": [""" import torch
-from mindspeed.op_builder import RotaryPositionEmbeddingOpBuilder
+#from mindspeed.op_builder import RotaryPositionEmbeddingOpBuilder
 
 __all__ = ["npu_rotary_position_embedding"]
 
 
-rope_op_builder = RotaryPositionEmbeddingOpBuilder()
+#rope_op_builder = RotaryPositionEmbeddingOpBuilder()
 
 
+# def npu_rotary_position_embedding(x, cos, sin, mode=0):
+#     rope_ops = rope_op_builder.load()
+#     return rope_ops.npu_rotary_position_embedding(x, cos, sin, mode)
+import mindspore
+
 def npu_rotary_position_embedding(x, cos, sin, mode=0):
-    rope_ops = rope_op_builder.load()
-    return rope_ops.npu_rotary_position_embedding(x, cos, sin, mode)
+    if not isinstance(x, torch.Tensor):
+        raise TypeError("Input x must be a torch.Tensor, got {}".format(type(x)))
+    if not isinstance(cos, torch.Tensor):
+        raise TypeError("Input cos must be a torch.Tensor, got {}".format(type(cos)))
+    if not isinstance(sin, torch.Tensor):
+        raise TypeError("Input sin must be a torch.Tensor, got {}".format(type(sin)))
+    if not isinstance(mode, (int, type(None))):
+        raise TypeError("Input mode must be an int or None, got {}".format(type(mode)))
+    # return RotaryPositionEmbeddingFunction.apply(x, cos, sin, mode)
+    return mindspore.ops.rotary_position_embedding(x, cos, sin, mode)
"""
],
"mindspeed/utils.py": ["""                                              get_context_parallel_for_hybrid_ulysses_rank,
                                              get_context_parallel_for_hybrid_ring_world_size,
                                              get_context_parallel_for_hybrid_ring_rank)
-from mindspeed.core.context_parallel.utils import (set_scheduling_info,
-                                                   set_remapped_seq_order,
-                                                   adaptive_reschedule_task,
-                                                   get_adaptive_cp_mask_list_by_user,
-                                                   get_adaptive_cp_grid_mask_by_user,
-                                                   generate_adaptive_cp_mask_list_by_user,
-                                                   generate_adaptive_cp_grid_mask_by_user)
+#from mindspeed.core.context_parallel.utils import (set_scheduling_info,
+#                                                   set_remapped_seq_order,
+#                                                   adaptive_reschedule_task,
+#                                                   get_adaptive_cp_mask_list_by_user,
+#                                                   get_adaptive_cp_grid_mask_by_user,
+#                                                   generate_adaptive_cp_mask_list_by_user,
+#                                                   generate_adaptive_cp_grid_mask_by_user)
 from mindspeed.model.transformer import set_attention_mask, get_attention_mask
 
 ""","""         print(message, flush=True)
 
 
-def get_batch_on_this_cp_rank(batch):
-    \"\"\" Slice batch input along sequence dimension into multiple chunks,
-        which are parallelized across GPUs in a context parallel group.
-    \"\"\"
-
-    # With causal masking, each token only attends to its prior tokens. Simply split
-    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
-    # at the end of sequence have bigger workload than others. To address this issue,
-    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
-    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
-    # that we can get balanced workload among GPUs in a context parallel group.
-    from megatron.training import get_args
-
-    args = get_args()
-
-    if args.reset_attention_mask:
-        position_ids = batch['position_ids']
-        position_ids = position_ids.transpose(0, 1).contiguous()
-        set_position_ids(position_ids)    
- 
-    tp_y_cp_size = args.context_parallel_size * args.tp_y if args.tp_2d else args.context_parallel_size
-    if not tp_y_cp_size > 1:
-        return batch
-
-    cp_expanded_by_2d_tp = args.tp_y > 1
-    if args.context_parallel_algo == 'megatron_cp_algo':
-        if args.cp_attention_mask_type == 'general':
-            batch = _get_batch_on_this_cp_rank_in_megatron_cp_general(batch)
-        elif cp_expanded_by_2d_tp:
-            batch = _get_batch_on_this_tp_y_cp_rank_in_megatron_cp(batch)
-        else:
-            batch = _get_batch_on_this_cp_rank_in_megatron_cp(batch)
-    elif args.context_parallel_algo == 'ulysses_cp_algo':
-        batch = _get_batch_on_this_cp_rank_in_ulysses_cp(batch)
-    elif args.context_parallel_algo == 'hybrid_cp_algo':
-        if args.cp_attention_mask_type == 'general':
-            batch = _get_batch_on_this_cp_rank_in_hybrid_cp_general(batch)
-        else:
-            batch = _get_batch_on_this_cp_rank_in_hybrid_cp(batch)
-    elif args.context_parallel_algo == 'adaptive_cp_algo':
-        batch = _get_batch_on_this_cp_rank_in_adaptive_cp(batch)
-    elif args.context_parallel_algo == 'hybrid_adaptive_cp_algo':
-        batch = _get_batch_on_this_cp_rank_in_hybrid_adaptive_cp(batch)
-    return batch
-
-
-def _get_batch_on_this_cp_rank_in_megatron_cp(batch):
-    cp_rank = mpu.get_context_parallel_rank()
-    cp_size = mpu.get_context_parallel_world_size()
-    for key, val in batch.items():
-        if key == 'attention_mask':
-            continue
-        if val is not None:
-            seq_dim = 1 if key != 'attention_mask' else 2
-            val = val.view(
-                *val.shape[0:seq_dim],
-                2 * cp_size,
-                val.shape[seq_dim] // (2 * cp_size),
-                *val.shape[(seq_dim + 1):],
-            )
-            index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=val.device)
-            val = val.index_select(seq_dim, index)
-            val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
-            batch[key] = val
-
-    return batch
-
-
-def _get_batch_on_this_cp_rank_in_megatron_cp_general(batch):
-    cp_rank = mpu.get_context_parallel_rank()
-    cp_size = mpu.get_context_parallel_world_size()
-
-    attention_mask = get_attention_mask()
-    if attention_mask is not None:
-        if len(attention_mask.shape) != 2:
-            raise AssertionError("The fusion attention operator currently only support 2D attention mask.")
-        seq_dim = 0
-        mask_row = attention_mask.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
-        from megatron.training import get_args
-        if get_args().attention_mask_on_cpu:
-            mask_list = [m.contiguous().npu(non_blocking=True) for m in mask_row.chunk(cp_size, dim=1)]
-        else:
-            mask_list = [m.contiguous() for m in mask_row.chunk(cp_size, dim=1)]
-        batch['attention_mask'] = mask_list
-        set_attention_mask(mask_list)
-
-    for key, val in batch.items():
-        if key != 'attention_mask' and val is not None:
-            seq_dim = 1
-            val = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
-            batch[key] = val
-
-    return batch
-
-
-def _get_batch_on_this_cp_rank_in_ulysses_cp(batch):
-    cp_rank = mpu.get_context_parallel_rank()
-    cp_size = mpu.get_context_parallel_world_size()
-    for key, val in batch.items():
-        if key == 'attention_mask':
-            continue
-        if val is not None:
-            seq_dim = 1 if key != 'attention_mask' else 2
-            val = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
-            batch[key] = val
-
-    return batch
-
-
-def _get_batch_on_this_cp_rank_in_hybrid_cp(batch):
-    u_size = get_context_parallel_for_hybrid_ulysses_world_size()
-    r_size = get_context_parallel_for_hybrid_ring_world_size()
-
-    u_rank = get_context_parallel_for_hybrid_ulysses_rank()
-    r_rank = get_context_parallel_for_hybrid_ring_rank()
-
-    for key, val in batch.items():
-        if key == 'attention_mask':
-            continue
-        if val is not None:
-            seq_dim = 1 if key != 'attention_mask' else 2
-            val = val.view(
-                *val.shape[0:seq_dim],
-                2 * r_size,
-                val.shape[seq_dim] // (2 * r_size),
-                *val.shape[(seq_dim + 1):],
-            )
-            index = torch.tensor([r_rank, (2 * r_size - r_rank - 1)], device=val.device)
-            val = val.index_select(seq_dim, index)
-            val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
-            val = val.chunk(u_size, dim=seq_dim)[u_rank].contiguous()
-            batch[key] = val
-
-    return batch
-
-
-def _get_batch_on_this_cp_rank_in_hybrid_cp_general(batch):
-    u_size = get_context_parallel_for_hybrid_ulysses_world_size()
-    r_size = get_context_parallel_for_hybrid_ring_world_size()
-
-    u_rank = get_context_parallel_for_hybrid_ulysses_rank()
-    r_rank = get_context_parallel_for_hybrid_ring_rank()
-
-    attention_mask = get_attention_mask()
-    if attention_mask is not None:
-        if len(attention_mask.shape) != 2:
-            raise AssertionError("The fusion attention operator currently only support 2D attention mask.")
-        seq_dim = 0
-        mask_row = attention_mask.chunk(r_size, dim=seq_dim)[r_rank].contiguous()
-        from megatron.training import get_args
-        if get_args().attention_mask_on_cpu:
-            mask_list = [m.contiguous().npu(non_blocking=True) for m in mask_row.chunk(r_size, dim=1)]
-        else:
-            mask_list = [m.contiguous() for m in mask_row.chunk(r_size, dim=1)]
-        batch['attention_mask'] = mask_list
-        set_attention_mask(mask_list)
-
-    for key, val in batch.items():
-        if key != 'attention_mask' and val is not None:
-            seq_dim = 1
-            val = val.chunk(r_size, dim=seq_dim)[r_rank].contiguous()
-            val = val.chunk(u_size, dim=seq_dim)[u_rank].contiguous()
-            batch[key] = val
-
-    return batch
-
-
-def _get_batch_on_this_cp_rank_in_adaptive_cp(batch):
-    from megatron.training import get_args
-    args = get_args()
-    cp_rank = mpu.get_context_parallel_rank()
-    cp_size = mpu.get_context_parallel_world_size()
-
-    attention_mask = get_attention_mask()
-    if args.adaptive_cp_manually_set_mask_list:
-        if not args.adaptive_cp_only_reschedule:
-            raise AssertionError("No sequence remapping allowed if manually set mast list, enable "
-                                 "--adaptive-cp-only-reschedule")
-        remapped_seq_order = list(range(args.seq_length))
-        generate_adaptive_cp_grid_mask_by_user(cp_size)
-        grid_mask = get_adaptive_cp_grid_mask_by_user()
-        scheduling = adaptive_reschedule_task(grid_mask, cp_size)
-        generate_adaptive_cp_mask_list_by_user(remapped_seq_order, scheduling, cp_rank, cp_size)
-        mask_list = get_adaptive_cp_mask_list_by_user()
-    else:
-        if attention_mask is None:
-            raise AssertionError("Do not use adaptive cp with full mask")
-        if len(attention_mask.shape) != 2:
-            raise AssertionError("The fusion attention operator currently only support 2D attention mask.")
-        from mindspeed.core.context_parallel.utils import adaptive_cp_ops
-        remapped_seq_order, scheduling = adaptive_cp_ops.get_adaptive_cp_info(attention_mask, cp_size)
-        mask_list = adaptive_cp_ops.get_mask_list(attention_mask, scheduling, remapped_seq_order, cp_rank, cp_size)
-
-    batch['attention_mask'] = mask_list
-    set_attention_mask(mask_list)
-    set_scheduling_info(torch.distributed.get_rank(), scheduling)
-    set_remapped_seq_order(remapped_seq_order)
-
-    for key, val in batch.items():
-        if key != 'attention_mask' and val is not None:
-            seq_dim = 1
-            per = val.shape[seq_dim] // cp_size
-            index = torch.tensor(remapped_seq_order[cp_rank * per:(cp_rank + 1) * per], device=val.device,
-                                 dtype=torch.int)
-            val = val.index_select(seq_dim, index)
-            batch[key] = val
-    return batch
-
-
-def _get_batch_on_this_cp_rank_in_hybrid_adaptive_cp(batch):
-    from megatron.training import get_args
-    args = get_args()
-    ulys_size = get_context_parallel_for_hybrid_ulysses_world_size()
-    adap_size = get_context_parallel_for_hybrid_ring_world_size()
-    ulys_rank = get_context_parallel_for_hybrid_ulysses_rank()
-    adap_rank = get_context_parallel_for_hybrid_ring_rank()
-
-    attention_mask = get_attention_mask()
-    if args.adaptive_cp_manually_set_mask_list:
-        if not args.adaptive_cp_only_reschedule:
-            raise AssertionError("No sequence remapping allowed if manually set mast list, enable "
-                                 "--adaptive-cp-only-reschedule")
-        remapped_seq_order = list(range(args.seq_length))
-        generate_adaptive_cp_grid_mask_by_user(adap_size)
-        grid_mask = get_adaptive_cp_grid_mask_by_user()
-        scheduling = adaptive_reschedule_task(grid_mask, adap_size)
-        generate_adaptive_cp_mask_list_by_user(remapped_seq_order, scheduling, adap_rank, adap_size)
-        mask_list = get_adaptive_cp_mask_list_by_user()
-    else:
-        if attention_mask is None:
-            raise AssertionError("Do not use adaptive cp with full mask")
-        if len(attention_mask.shape) != 2:
-            raise AssertionError("The fusion attention operator currently only support 2D attention mask.")
-        from mindspeed.core.context_parallel.utils import adaptive_cp_ops
-        remapped_seq_order, scheduling = adaptive_cp_ops.get_adaptive_cp_info(attention_mask, adap_size)
-        mask_list = adaptive_cp_ops.get_mask_list(attention_mask, scheduling, remapped_seq_order, adap_rank, adap_size)
-
-    batch['attention_mask'] = mask_list
-    set_scheduling_info(torch.distributed.get_rank(), scheduling)
-    set_remapped_seq_order(remapped_seq_order)
-    set_attention_mask(mask_list)
-
-    for key, val in batch.items():
-        if key != 'attention_mask' and val is not None:
-            seq_dim = 1
-            per = val.shape[seq_dim] // adap_size // ulys_size
-            which_per = adap_rank * ulys_size + ulys_rank
-            index = torch.tensor(remapped_seq_order[which_per * per:(which_per + 1) * per], device=val.device)
-            val = val.index_select(seq_dim, index)
-            batch[key] = val
-    return batch
+#def get_batch_on_this_cp_rank(batch):
+#    \"\"\" Slice batch input along sequence dimension into multiple chunks,
+#        which are parallelized across GPUs in a context parallel group.
+#    \"\"\"
+#
+#    # With causal masking, each token only attends to its prior tokens. Simply split
+#    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
+#    # at the end of sequence have bigger workload than others. To address this issue,
+#    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
+#    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
+#    # that we can get balanced workload among GPUs in a context parallel group.
+#    from megatron.training import get_args
+#
+#    args = get_args()
+#
+#    if args.reset_attention_mask:
+#        position_ids = batch['position_ids']
+#        position_ids = position_ids.transpose(0, 1).contiguous()
+#        set_position_ids(position_ids)
+#
+#    tp_y_cp_size = args.context_parallel_size * args.tp_y if args.tp_2d else args.context_parallel_size
+#    if not tp_y_cp_size > 1:
+#        return batch
+#
+#    cp_expanded_by_2d_tp = args.tp_y > 1
+#    if args.context_parallel_algo == 'megatron_cp_algo':
+#        if args.cp_attention_mask_type == 'general':
+#            batch = _get_batch_on_this_cp_rank_in_megatron_cp_general(batch)
+#        elif cp_expanded_by_2d_tp:
+#            batch = _get_batch_on_this_tp_y_cp_rank_in_megatron_cp(batch)
+#        else:
+#            batch = _get_batch_on_this_cp_rank_in_megatron_cp(batch)
+#    elif args.context_parallel_algo == 'ulysses_cp_algo':
+#        batch = _get_batch_on_this_cp_rank_in_ulysses_cp(batch)
+#    elif args.context_parallel_algo == 'hybrid_cp_algo':
+#        if args.cp_attention_mask_type == 'general':
+#            batch = _get_batch_on_this_cp_rank_in_hybrid_cp_general(batch)
+#        else:
+#            batch = _get_batch_on_this_cp_rank_in_hybrid_cp(batch)
+#    elif args.context_parallel_algo == 'adaptive_cp_algo':
+#        batch = _get_batch_on_this_cp_rank_in_adaptive_cp(batch)
+#    elif args.context_parallel_algo == 'hybrid_adaptive_cp_algo':
+#        batch = _get_batch_on_this_cp_rank_in_hybrid_adaptive_cp(batch)
+#    return batch
+#
+#
+#def _get_batch_on_this_cp_rank_in_megatron_cp(batch):
+#    cp_rank = mpu.get_context_parallel_rank()
+#    cp_size = mpu.get_context_parallel_world_size()
+#    for key, val in batch.items():
+#        if key == 'attention_mask':
+#            continue
+#        if val is not None:
+#            seq_dim = 1 if key != 'attention_mask' else 2
+#            val = val.view(
+#                *val.shape[0:seq_dim],
+#                2 * cp_size,
+#                val.shape[seq_dim] // (2 * cp_size),
+#                *val.shape[(seq_dim + 1):],
+#            )
+#            index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=val.device)
+#            val = val.index_select(seq_dim, index)
+#            val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
+#            batch[key] = val
+#
+#    return batch
+#
+#
+#def _get_batch_on_this_cp_rank_in_megatron_cp_general(batch):
+#    cp_rank = mpu.get_context_parallel_rank()
+#    cp_size = mpu.get_context_parallel_world_size()
+#
+#    attention_mask = get_attention_mask()
+#    if attention_mask is not None:
+#        if len(attention_mask.shape) != 2:
+#            raise AssertionError("The fusion attention operator currently only support 2D attention mask.")
+#        seq_dim = 0
+#        mask_row = attention_mask.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
+#        from megatron.training import get_args
+#        if get_args().attention_mask_on_cpu:
+#            mask_list = [m.contiguous().npu(non_blocking=True) for m in mask_row.chunk(cp_size, dim=1)]
+#        else:
+#            mask_list = [m.contiguous() for m in mask_row.chunk(cp_size, dim=1)]
+#        batch['attention_mask'] = mask_list
+#        set_attention_mask(mask_list)
+#
+#    for key, val in batch.items():
+#        if key != 'attention_mask' and val is not None:
+#            seq_dim = 1
+#            val = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
+#            batch[key] = val
+#
+#    return batch
+#
+#
+#def _get_batch_on_this_cp_rank_in_ulysses_cp(batch):
+#    cp_rank = mpu.get_context_parallel_rank()
+#    cp_size = mpu.get_context_parallel_world_size()
+#    for key, val in batch.items():
+#        if key == 'attention_mask':
+#            continue
+#        if val is not None:
+#            seq_dim = 1 if key != 'attention_mask' else 2
+#            val = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
+#            batch[key] = val
+#
+#    return batch
+#
+#
+#def _get_batch_on_this_cp_rank_in_hybrid_cp(batch):
+#    u_size = get_context_parallel_for_hybrid_ulysses_world_size()
+#    r_size = get_context_parallel_for_hybrid_ring_world_size()
+#
+#    u_rank = get_context_parallel_for_hybrid_ulysses_rank()
+#    r_rank = get_context_parallel_for_hybrid_ring_rank()
+#
+#    for key, val in batch.items():
+#        if key == 'attention_mask':
+#            continue
+#        if val is not None:
+#            seq_dim = 1 if key != 'attention_mask' else 2
+#            val = val.view(
+#                *val.shape[0:seq_dim],
+#                2 * r_size,
+#                val.shape[seq_dim] // (2 * r_size),
+#                *val.shape[(seq_dim + 1):],
+#            )
+#            index = torch.tensor([r_rank, (2 * r_size - r_rank - 1)], device=val.device)
+#            val = val.index_select(seq_dim, index)
+#            val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
+#            val = val.chunk(u_size, dim=seq_dim)[u_rank].contiguous()
+#            batch[key] = val
+#
+#    return batch
+#
+#
+#def _get_batch_on_this_cp_rank_in_hybrid_cp_general(batch):
+#    u_size = get_context_parallel_for_hybrid_ulysses_world_size()
+#    r_size = get_context_parallel_for_hybrid_ring_world_size()
+#
+#    u_rank = get_context_parallel_for_hybrid_ulysses_rank()
+#    r_rank = get_context_parallel_for_hybrid_ring_rank()
+#
+#    attention_mask = get_attention_mask()
+#    if attention_mask is not None:
+#        if len(attention_mask.shape) != 2:
+#            raise AssertionError("The fusion attention operator currently only support 2D attention mask.")
+#        seq_dim = 0
+#        mask_row = attention_mask.chunk(r_size, dim=seq_dim)[r_rank].contiguous()
+#        from megatron.training import get_args
+#        if get_args().attention_mask_on_cpu:
+#            mask_list = [m.contiguous().npu(non_blocking=True) for m in mask_row.chunk(r_size, dim=1)]
+#        else:
+#            mask_list = [m.contiguous() for m in mask_row.chunk(r_size, dim=1)]
+#        batch['attention_mask'] = mask_list
+#        set_attention_mask(mask_list)
+#
+#    for key, val in batch.items():
+#        if key != 'attention_mask' and val is not None:
+#            seq_dim = 1
+#            val = val.chunk(r_size, dim=seq_dim)[r_rank].contiguous()
+#            val = val.chunk(u_size, dim=seq_dim)[u_rank].contiguous()
+#            batch[key] = val
+#
+#    return batch
+#
+#
+#def _get_batch_on_this_cp_rank_in_adaptive_cp(batch):
+#    from megatron.training import get_args
+#    args = get_args()
+#    cp_rank = mpu.get_context_parallel_rank()
+#    cp_size = mpu.get_context_parallel_world_size()
+#
+#    attention_mask = get_attention_mask()
+#    if args.adaptive_cp_manually_set_mask_list:
+#        if not args.adaptive_cp_only_reschedule:
+#            raise AssertionError("No sequence remapping allowed if manually set mast list, enable "
+#                                 "--adaptive-cp-only-reschedule")
+#        remapped_seq_order = list(range(args.seq_length))
+#        generate_adaptive_cp_grid_mask_by_user(cp_size)
+#        grid_mask = get_adaptive_cp_grid_mask_by_user()
+#        scheduling = adaptive_reschedule_task(grid_mask, cp_size)
+#        generate_adaptive_cp_mask_list_by_user(remapped_seq_order, scheduling, cp_rank, cp_size)
+#        mask_list = get_adaptive_cp_mask_list_by_user()
+#    else:
+#        if attention_mask is None:
+#            raise AssertionError("Do not use adaptive cp with full mask")
+#        if len(attention_mask.shape) != 2:
+#            raise AssertionError("The fusion attention operator currently only support 2D attention mask.")
+#        from mindspeed.core.context_parallel.utils import adaptive_cp_ops
+#        remapped_seq_order, scheduling = adaptive_cp_ops.get_adaptive_cp_info(attention_mask, cp_size)
+#        mask_list = adaptive_cp_ops.get_mask_list(attention_mask, scheduling, remapped_seq_order, cp_rank, cp_size)
+#
+#    batch['attention_mask'] = mask_list
+#    set_attention_mask(mask_list)
+#    set_scheduling_info(torch.distributed.get_rank(), scheduling)
+#    set_remapped_seq_order(remapped_seq_order)
+#
+#    for key, val in batch.items():
+#        if key != 'attention_mask' and val is not None:
+#            seq_dim = 1
+#            per = val.shape[seq_dim] // cp_size
+#            index = torch.tensor(remapped_seq_order[cp_rank * per:(cp_rank + 1) * per], device=val.device,
+#                                 dtype=torch.int)
+#            val = val.index_select(seq_dim, index)
+#            batch[key] = val
+#    return batch
+#
+#
+#def _get_batch_on_this_cp_rank_in_hybrid_adaptive_cp(batch):
+#    from megatron.training import get_args
+#    args = get_args()
+#    ulys_size = get_context_parallel_for_hybrid_ulysses_world_size()
+#    adap_size = get_context_parallel_for_hybrid_ring_world_size()
+#    ulys_rank = get_context_parallel_for_hybrid_ulysses_rank()
+#    adap_rank = get_context_parallel_for_hybrid_ring_rank()
+#
+#    attention_mask = get_attention_mask()
+#    if args.adaptive_cp_manually_set_mask_list:
+#        if not args.adaptive_cp_only_reschedule:
+#            raise AssertionError("No sequence remapping allowed if manually set mast list, enable "
+#                                 "--adaptive-cp-only-reschedule")
+#        remapped_seq_order = list(range(args.seq_length))
+#        generate_adaptive_cp_grid_mask_by_user(adap_size)
+#        grid_mask = get_adaptive_cp_grid_mask_by_user()
+#        scheduling = adaptive_reschedule_task(grid_mask, adap_size)
+#        generate_adaptive_cp_mask_list_by_user(remapped_seq_order, scheduling, adap_rank, adap_size)
+#        mask_list = get_adaptive_cp_mask_list_by_user()
+#    else:
+#        if attention_mask is None:
+#            raise AssertionError("Do not use adaptive cp with full mask")
+#        if len(attention_mask.shape) != 2:
+#            raise AssertionError("The fusion attention operator currently only support 2D attention mask.")
+#        from mindspeed.core.context_parallel.utils import adaptive_cp_ops
+#        remapped_seq_order, scheduling = adaptive_cp_ops.get_adaptive_cp_info(attention_mask, adap_size)
+#        mask_list = adaptive_cp_ops.get_mask_list(attention_mask, scheduling, remapped_seq_order, adap_rank, adap_size)
+#
+#    batch['attention_mask'] = mask_list
+#    set_scheduling_info(torch.distributed.get_rank(), scheduling)
+#    set_remapped_seq_order(remapped_seq_order)
+#    set_attention_mask(mask_list)
+#
+#    for key, val in batch.items():
+#        if key != 'attention_mask' and val is not None:
+#            seq_dim = 1
+#            per = val.shape[seq_dim] // adap_size // ulys_size
+#            which_per = adap_rank * ulys_size + ulys_rank
+#            index = torch.tensor(remapped_seq_order[which_per * per:(which_per + 1) * per], device=val.device)
+#            val = val.index_select(seq_dim, index)
+#            batch[key] = val
+#    return batch
 
 
 def _broadcast(item):""","""             for send in send_part:
                 send.untyped_storage().resize_(0)
 
-        recv_tensor[start_index:end_index] = recv_part_cpu
\\ No newline at end of file
+        recv_tensor[start_index:end_index] = recv_part_cpu
"""
],

}


LINE_RULES_peft = {
"src/peft/__init__.py": ["""     LoraModel,
     LoHaConfig,
     LoHaModel,
-    LoKrConfig,
-    LoKrModel,
     IA3Config,
     IA3Model,
     AdaLoraConfig,
"""
],
"src/peft/mapping.py": ["""     IA3Model,
     LoHaConfig,
     LoHaModel,
-    LoKrConfig,
-    LoKrModel,
     LoraConfig,
     LoraModel,
     MultitaskPromptTuningConfig,""","""     "P_TUNING": PromptEncoderConfig,
     "LORA": LoraConfig,
     "LOHA": LoHaConfig,
-    "LOKR": LoKrConfig,
     "ADALORA": AdaLoraConfig,
     "IA3": IA3Config,
     "MULTITASK_PROMPT_TUNING": MultitaskPromptTuningConfig,""",""" PEFT_TYPE_TO_TUNER_MAPPING = {
     "LORA": LoraModel,
     "LOHA": LoHaModel,
-    "LOKR": LoKrModel,
     "ADALORA": AdaLoraModel,
     "IA3": IA3Model,
     "OFT": OFTModel,
"""
],
"src/peft/mixed_model.py": [""" from typing import Any, Optional, Union
 
 import torch
-from accelerate.hooks import remove_hook_from_submodules
+#from accelerate.hooks import remove_hook_from_submodules
 from torch import nn
 from transformers.utils import PushToHubMixin
 ""","""     AdaLoraModel,
     IA3Model,
     LoHaModel,
-    LoKrModel,
     LoraModel,
     MixedModel,
     OFTModel,""",""" PEFT_TYPE_TO_MODEL_MAPPING = {
     PeftType.LORA: LoraModel,
     PeftType.LOHA: LoHaModel,
-    PeftType.LOKR: LoKrModel,
     PeftType.ADALORA: AdaLoraModel,
     PeftType.IA3: IA3Model,
     PeftType.OFT: OFTModel,
"""
],
"src/peft/peft_model.py": [""" import packaging.version
 import torch
 import transformers
-from accelerate import dispatch_model, infer_auto_device_map
-from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
-from accelerate.utils import get_balanced_memory
+#from accelerate import dispatch_model, infer_auto_device_map
+#from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
+#from accelerate.utils import get_balanced_memory
 from huggingface_hub import ModelCard, ModelCardData, hf_hub_download
 from safetensors.torch import save_file as safe_save_file
 from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss""","""     AdaptionPromptModel,
     IA3Model,
     LoHaModel,
-    LoKrModel,
     LoraModel,
     MultitaskPromptEmbedding,
     OFTModel,""",""" PEFT_TYPE_TO_MODEL_MAPPING = {
     PeftType.LORA: LoraModel,
     PeftType.LOHA: LoHaModel,
-    PeftType.LOKR: LoKrModel,
     PeftType.PROMPT_TUNING: PromptEmbedding,
     PeftType.P_TUNING: PromptEncoder,
     PeftType.PREFIX_TUNING: PrefixEncoder,
"""
],
"src/peft/tuners/__init__.py": [""" from .adaption_prompt import AdaptionPromptConfig, AdaptionPromptModel
 from .lora import LoraConfig, LoraModel, LoftQConfig
 from .loha import LoHaConfig, LoHaModel
-from .lokr import LoKrConfig, LoKrModel
+#from .lokr import LoKrConfig, LoKrModel
 from .ia3 import IA3Config, IA3Model
 from .adalora import AdaLoraConfig, AdaLoraModel
 from .p_tuning import PromptEncoder, PromptEncoderConfig, PromptEncoderReparameterizationType
"""
],
"src/peft/tuners/mixed/model.py": [""" from torch import nn
 from tqdm import tqdm
 
-from peft.tuners import adalora, loha, lokr, lora, oft
+from peft.tuners import lora
+#from peft.tuners import adalora, loha, lokr, lora, oft
 from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
 from peft.utils import (
     TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,""",""" 
 # Collection of constants used for all tuners
 COMPATIBLE_TUNER_TYPES = (PeftType.LORA, PeftType.LOHA, PeftType.LOKR, PeftType.ADALORA, PeftType.OFT)
-PREFIXES = [lora.LoraModel.prefix, lokr.LoKrModel.prefix, loha.LoHaModel.prefix, oft.OFTModel.prefix]
-Configs = Union[lora.LoraConfig, loha.LoHaConfig, lokr.LoKrConfig, adalora.AdaLoraConfig, oft.OFTConfig]
-Layers = (lora.layer.LoraLayer, loha.layer.LoHaLayer, lokr.layer.LoKrLayer, adalora.layer.AdaLoraLayer, oft.OFTLayer)
-
+#PREFIXES = [lora.LoraModel.prefix, lokr.LoKrModel.prefix, loha.LoHaModel.prefix, oft.OFTModel.prefix]
+#Configs = Union[lora.LoraConfig, loha.LoHaConfig, lokr.LoKrConfig, adalora.AdaLoraConfig, oft.OFTConfig]
+#Layers = (lora.layer.LoraLayer, loha.layer.LoHaLayer, lokr.layer.LoKrLayer, adalora.layer.AdaLoraLayer, oft.OFTLayer)
+PREFIXES = [lora.LoraModel.prefix]
+Configs = Union[lora.LoraConfig]
+Layers = (lora.layer.LoraLayer)
 
 class MixedModel(BaseTuner):
     \"\"\"""","""         *args: Any,
         **kwargs: Any,
     ) -> None:
+        
+        if isinstance(config, lora.LoraConfig):
+            lora.LoraModel._create_and_replace(self, config, *args, **kwargs)
+        else:
+            raise ValueError(f"Unsupported config type {type(config)}, should be one of {COMPATIBLE_TUNER_TYPES}.")
+        ''' 
         if isinstance(config, adalora.AdaLoraConfig):
             adalora.AdaLoraModel._create_and_replace(self, config, *args, **kwargs)
         elif isinstance(config, lora.LoraConfig):""","""             lokr.LoKrModel._create_and_replace(self, config, *args, **kwargs)
         elif isinstance(config, oft.OFTConfig):
             oft.OFTModel._create_and_replace(self, config, *args, **kwargs)
+        
         else:
             raise ValueError(f"Unsupported config type {type(config)}, should be one of {COMPATIBLE_TUNER_TYPES}.")
+        '''
 
     def _replace_module(self, parent, child_name, new_module, child) -> None:
         setattr(parent, child_name, new_module)""","""         loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)
         if loaded_in_8bit or loaded_in_4bit:
             raise ValueError(f"8bit and 4bit quantization not supported for {config.peft_type.value} (yet).")
-
+        
+        if isinstance(config, lora.LoraConfig):
+            new_module = lora.LoraModel._create_new_module(config, adapter_name, target, **kwargs)
+        else:
+            raise ValueError(f"Unknown config type {type(config)}, should be one of {COMPATIBLE_TUNER_TYPES}.")
+        '''
         if isinstance(config, adalora.AdaLoraConfig):
             new_module = adalora.AdaLoraModel._create_new_module(config, adapter_name, target, **kwargs)
         elif isinstance(config, lora.LoraConfig):""","""             new_module = lokr.LoKrModel._create_new_module(config, adapter_name, target, **kwargs)
         elif isinstance(config, oft.OFTConfig):
             new_module = oft.OFTModel._create_new_module(config, adapter_name, target, **kwargs)
+        
         else:
             raise ValueError(f"Unknown config type {type(config)}, should be one of {COMPATIBLE_TUNER_TYPES}.")
+        '''
         return new_module
 
     def _set_adapter_layers(self, enabled=True):
"""
],
"src/peft/utils/other.py": [""" import warnings
 from typing import Optional, Tuple
 
-import accelerate
+#import accelerate
 import torch
-from accelerate.hooks import add_hook_to_module, remove_hook_from_module
-from accelerate.utils import is_npu_available, is_xpu_available
+#from accelerate.hooks import add_hook_to_module, remove_hook_from_module
+#from accelerate.utils import is_npu_available, is_xpu_available
 from safetensors.torch import storage_ptr, storage_size
 
 from ..import_utils import is_auto_gptq_available, is_torch_tpu_available
"""
],

}


LINE_RULES = {
"MindSpeed-LLM": LINE_RULES_MindSpeed_LLM,
"MindSpeed-MM": LINE_RULES_MindSpeed_MM,
"transformers": LINE_RULES_transformers,
"megatron": LINE_RULES_megatron,
"mindspeed_llm": LINE_RULES_acclerate_mindspeed_llm,
"mindspeed_mm": LINE_RULES_acclerate_mindspeed_mm,
"peft": LINE_RULES_peft,
}



