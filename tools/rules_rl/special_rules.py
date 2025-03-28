

einops_ms_backend = """
class MsadaptorBackend(AbstractBackend):
    framework_name = "msadaptor"

    def __init__(self):
        import msadaptor

        self.msadaptor = msadaptor
        # importing would register operations in torch._dynamo for torch.compile
        # from . import _torch_specific  # noqa

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.msadaptor.Tensor)

    def from_numpy(self, x):
        variable = self.msadaptor.from_numpy(x)
        if self.is_float_type(variable):
            # attach grad only to floating types
            variable.requires_grad = True
        return variable

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def arange(self, start, stop):
        return self.msadaptor.arange(start, stop, dtype=self.msadaptor.int64)

    def reduce(self, x, operation, reduced_axes):
        if operation == "min":
            return x.amin(dim=reduced_axes)
        elif operation == "max":
            return x.amax(dim=reduced_axes)
        elif operation == "sum":
            return x.sum(dim=reduced_axes)
        elif operation == "mean":
            return x.mean(dim=reduced_axes)
        elif operation in ("any", "all", "prod"):
            # msadaptor supports reducing only one operation at a time
            for i in list(sorted(reduced_axes))[::-1]:
                x = getattr(x, operation)(dim=i)
            return x
        else:
            raise NotImplementedError("Unknown reduction ", operation)

    def transpose(self, x, axes):
        return x.permute(axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.msadaptor.stack(tensors)

    def add_axes(self, x, n_axes, pos2len):
        repeats = [-1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = self.add_axis(x, axis_position)
            repeats[axis_position] = axis_length
        return x.expand(repeats)

    def tile(self, x, repeats):
        return x.repeat(repeats)

    def concat(self, tensors, axis: int):
        return self.msadaptor.cat(tensors, dim=axis)

    def add_axis(self, x, new_position):
        return self.msadaptor.unsqueeze(x, new_position)

    def is_float_type(self, x):
        return x.dtype in [self.msadaptor.float16, self.msadaptor.float32, self.msadaptor.float64, self.msadaptor.bfloat16]

    def einsum(self, pattern, *x):
        return self.msadaptor.einsum(pattern, *x)
"""

SPECIAL_RULES = {
"megatron":{
    "core/distributed/distributed_data_parallel.py": 
        [[r"param_tmp = param\.expand_as\(param\)", ""],
         [r"grad\_acc.*\n .*grad_acc\.register_hook.*\n.*self\.grad_accs.*\n", "param.register_hook(self._make_param_hook(param, self.param_to_buffer))\n"],
         [r"param\.main_grad\.add_\(param\.grad\.data\)", "param.main_grad.add_(*unused)"],
         ],
    "core/distributed/param_and_grad_buffer.py":
        [[r"self\.grad_data \*= self\.gradient\_scaling\_factor", "self.grad_data.copy_(self.grad_data * self.gradient_scaling_factor)"],],
    "core/models/common/embeddings/rotary_pos_embedding.py": 
        [[r"__all__ = \['RotaryEmbedding', 'apply_rotary_pos_emb']", "__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']\n\n_ROTATION_MATRIX = None\ndef get_rotation_matrix(x):\n    global _ROTATION_MATRIX\n    if _ROTATION_MATRIX is None:\n        import numpy as np\n        dim = x.shape[-1]\n        index1 = np.ones(dim)\n        index1[::2] = 0\n        index2 = np.zeros(dim)\n        index2[::2] = -1\n        rotation_matrix = np.eye(dim, k=1) * index1 + np.eye(dim, k=-1) * index2\n        _ROTATION_MATRIX = (\n            torch.from_numpy(rotation_matrix[None, None, :, :]).to(x.dtype).to(x.device)\n        )\n    return _ROTATION_MATRIX"],
         [r"x1 = x.*\n.*x2 = x.*\n.*x\_new = torch\.stack.*\n.*return.*", "return torch.matmul(x, get_rotation_matrix(x))"]],
    "core/optimizer/__init__.py":
        [[r"from torch\.optim import AdamW.*", ""],
         [r"decoupled\_min\_lr=config\.decoupled\_min\_lr,\n.*\)", "decoupled_min_lr=config.decoupled_min_lr,\n    )\n\n    # Fake params to construct optmizer\n    if len(param_groups) == 0:\n        fake_params = torch.zeros([1,], dtype=torch.float, requires_grad=True)\n        fake_params.fake = True\n        fake_params.grad = fake_params.clone()\n        fake_params.main_grad = fake_params.clone()\n        param_groups.append({'params': fake_params, 'wd_mult': 0.0, 'lr_mult': 0.0, 'is_decoupled_lr': False})"]],
    "core/optimizer/distrib_optimizer.py":
        [[r"assert param\.requires\_grad", "# assert param.requires_grad"]],
    "core/optimizer/optimizer.py":
        [[r"assert self\.optimizer\, \'no optimizer is provided\.\'", "assert self.optimizer, 'no optimizer is provided.'\n        self.empty_optmizer = False\n        if getattr(self.optimizer.param_groups[0]['params'][0], 'fake', False):\n            self.empty_optmizer = True"],],
    "core/pipeline_parallel/schedules.py": 
        [[r"from torch\.autograd\.variable import Variable",                                                 "from mindspore.ops import composite as C\nfrom mindspore.common.api import _pynative_executor"],
         [r"set\_input\_tensor\(input_tensor\)",                                                             "set_input_tensor(input_tensor)\n\n    if not parallel_state.is_pipeline_first_stage() and input_tensor is not None:\n        input_tensor[0].retain_grad()\n\n    # run forward\n    num_tokens = torch.tensor(0, dtype=torch.int)\n    if input_tensor[0] is None:\n        input_tensor[0] = num_tokens"],
         [r"context\_manager = contextlib\.nullcontext\(\)",                                                 "context_manager = contextlib.nullcontext()\n    _pynative_executor.set_grad_flag(True)\n    _pynative_executor.new_graph(forward_step_func, input_tensor[0])"],
         [r"for x in input_tensor.*\n.*if x is not None.*\n.*x\.retain_grad.*",                              ""],
         [r"if output_tensor_grad\[0\].*\n.*\n.*\n.*\n.*\n.*\n.*torch\.autograd\.backward.*",                "if output_tensor_grad[0] is None and config.grad_scale_func is not None:\n        output_tensor_grad[0] = config.grad_scale_func(torch.ones_like(output_tensor[0]))\n    if output_tensor_grad[0] is None:\n        output_tensor_grad[0] = torch.ones_like(output_tensor[0])\n\n    # set input tensor for backpropagation\n    if not parallel_state.is_pipeline_first_stage():\n        model.module.set_input_tensor(input_tensor[0])\n\n    # run backward\n    grad_ = C.GradOperation(True, True, True)\n    weights = model.trainable_params()\n    _pynative_executor.check_run(grad_, config.forward_step_func, weights, None, input_tensor[0])\n    _pynative_executor.grad(config.forward_step_func, grad_, weights, None, input_tensor[0], output_tensor_grad[0])"],
         [r"input\_tensor\_grad\.append\(x\.grad\)",                                                         "input_tensor_grad.append(x.grad)\n\n    if not parallel_state.is_pipeline_first_stage():\n        model.module.set_input_tensor(None)"],
         [r"config = get\_model\_config\(model\)",                                                           "config = get_model_config(model)\n    config.forward_step_func = forward_step_func"],
         [r"input\_tensor, output\_tensor\_grad = None\, None",                                              "input_tensor, output_tensor_grad = [None], [None]"],
         [r"backward_step\(\n?.*input_tensor, output_tensor, output_tensor_grad, model_type, config\n?.*\)", "backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config, model)"]
         ],
    "core/tensor_parallel/cross_entropy.py":
        [(r"masked_target\[target_mask\] = 0", "masked_target *= (1-target_mask)"),
         (r"predicted_logits\[target_mask\] = 0\.0", "predicted_logits *= (1-target_mask)"),
        ],
    "core/tensor_parallel/mappings.py":
        [("import torch", "import torch\nimport mindspore"),
         (r"output = input\.new_empty\(\n?.*\n?.*\n.*device=torch\.cuda\.current_device\(\).*\n?.*\)", "output = input.new_empty(\n                size=[int(sum(output_split_sizes))] + list(input.size()[1:]),\n                dtype=input.dtype,\n            )"),
         (r"torch\.distributed\.all_to_all_single\(\n?.*\n?.*\n?.*\n?.*\n.*group=group,\n.*\)", "mindspore.mint.distributed.all_to_all_single(\n            output,\n            input,\n            output_split_sizes=output_split_sizes.tolist(),\n            input_split_sizes=input_split_sizes.tolist(),\n            group=group._name,)")
        ],
    "core/tensor_parallel/random.py":
        [(r"if hasattr\(\n?.*\n?.*\n?.*\n?.*\n?.*\n?.*\n?.*\n?.*\n?.*\n?.*\n?.*\n?.*\n?.*\n?.*\n?.*\n?.*\n?.*\n?.*\n?.*\n?.*\n?.*\n?.*\n?.*\n?.*_lazy_call\(cb\)", "torch.cuda.set_rng_state(new_state)"),
        ],
    "core/transformer/custom_layers/transformer_engine.py":
        [(r"_te_version = packaging\.version\.Version\(version\(\"transformer-engine\"\)\)", "_te_version = packaging.version.parse(te.__version__)"),
        #  (r"te\.pytorch\.", "te.")
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
    "core/context_parallel/ring_context_parallel.py":
        [(r"import torch_npu", "import torch_npu\nimport mindspore"),
         (r"= cp_para", "= self.cp_para"),
         (r"AttentionWithCp\(torch\.autograd\.Function.*\n.*", "AttentionWithCp(mindspore.nn.Cell):\n\n    def __init__(self, cp_para):\n        super().__init__()\n        self.block_size = None\n        self.batch_size = None\n        self.cp_para = cp_para"),
         (r"ctx\.", "self."),
         (r"@staticmethod", ""),
         (r"def forward.*cp_para", "def construct(self, q, k, v, n"),
         (r"self\.save_for_backward.*", "self.k = k\n        self.v = v\n        self.attn_mask = attn_mask\n        self.softmax_max = softmax_max\n        self.softmax_sum = softmax_sum"),
         (r"def backward\(ctx.*\n.*", "def bprop(self, q, k, v, n, softmax_scale, attn_mask, dropout_p, actual_seq_qlen, actual_seq_kvlen, attn_out, dout):\n        k = self.k\n        v = self.v\n        cp_para = self.cp_para\n        softmax_max = self.softmax_max\n        softmax_sum = self.softmax_sum\n        attn_mask = self.attn_mask"),
         (r"AttentionWithCp\.block_size", "attn_with_cp = AttentionWithCp(cp_para)\n    attn_with_cp.block_size"),
         (r"AttentionWithCp\.batch_size =", "attn_with_cp.batch_size ="),
         (r"AttentionWithCp.*\n.*cp_para", "attn_with_cp(\n        q, k, v, n")
        ],
    "model/transformer.py":
        [(r"dtype=bool", "dtype=torch.bool")
        ],
    "op_builder/algorithm_builder.py":
        [(r"from mindspeed\.op_builder\.builder.*", "from mindspeed.op_builder.builder import MindSpeedOpBuilder\nfrom mindspore.utils import _reuse_data_ptr"),
         (r"OP_NAME = \"algorithm\"", 'OP_NAME = \"algorithm\"\n    reuse_data_ptr = _reuse_data_ptr'),
         (r"return \[\'ops.*", "raise NotImplementedError\n    def load(self, verbose=True):\n        return self")
        ],
    "core/transformer/moe/token_dispatcher.py":
        [(r"\.to\(\n?torch\.device\(\"cpu\"\)\)\n?", ""),
         (r"\.to\(\n?.*torch\.device\(\"cpu\"\),.*\n?.*\)", ""),
         (r"torch\.cuda\.current_stream\(", "mindspore.runtime.current_stream("),
         (r"torch\.cuda\.Stream\(", "mindspore.runtime.Stream("),
         (r"torch\.cuda\.stream\(", "mindspore.runtime.StreamCtx("),
         ],
    # "optimizer/distrib_optimizer.py":
    #     [(r"param_data\.device\)", "param_data.device).stub_sync()"),]
},

"mindspeed-llm":{
    
},

"mindspeed_mm":{
    
},

"transformers":{
    "pytorch_utils.py":
        [(r"from safetensors\.torch import storage_ptr\, storage_size", "")]
},

"einops":{
    "_backends.py":
        [("", einops_ms_backend)]
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
