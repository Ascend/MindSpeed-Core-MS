
import torch
import torch.nn.functional as F
import megatron
from megatron import get_args
from megatron.core import tensor_parallel
from megatron.model.utils import openai_gelu, erf_gelu
from megatron.model.transformer import ParallelMLP
import torch_npu


def parallel_mlp_init(self, config, is_expert=False):
    super(ParallelMLP, self).__init__()
    args = get_args()

    self.add_bias = config.add_bias_linear

    ffn_hidden_size = config.ffn_hidden_size
    if config.gated_linear_unit:
        ffn_hidden_size *= 2

    # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
    self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
        config.hidden_size,
        ffn_hidden_size,
        config=config,
        init_method=config.init_method,
        bias=self.add_bias,
        gather_output=False,
        skip_bias_add=True,
        is_expert=is_expert,
    )

    self.bias_gelu_fusion = False
    self.activation_func = None
    self.swiglu = args.swiglu

    if args.openai_gelu:
        self.activation_func = openai_gelu
    elif args.onnx_safe:
        self.activation_func = erf_gelu
    elif args.swiglu:
        def swiglu(x):
            if args.use_fused_swiglu:
                return torch_npu.npu_swiglu(x, dim=-1)
            else:
                x = torch.chunk(x, 2, dim=-1)
                return F.silu(x[0]) * x[1]
        self.activation_func = swiglu
    elif args.squared_relu:
        def squared_relu(x):
            return torch.pow(F.relu(x), 2)
        self.activation_func = squared_relu
    else:
        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu

    # Project back to h.
    self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
        config.ffn_hidden_size,
        config.hidden_size,
        config=config,
        init_method=config.output_layer_init_method,
        bias=self.add_bias,
        skip_bias_add=True,
        input_is_parallel=True,
        is_expert=is_expert,
    )