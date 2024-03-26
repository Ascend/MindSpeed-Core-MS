import math
import torch
import torch.nn.functional as F
import megatron
from megatron import get_args
from megatron.core import tensor_parallel
from megatron.model.utils import openai_gelu, erf_gelu
from megatron.model.transformer import ParallelMLP
from functools import wraps
import torch_npu

try:
    from einops import rearrange
except ImportError:
    rearrange = None


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


def flash_self_attention_forward(self, q, k, v):
    """Implements the multihead softmax attention.
    Arguments
    ---------
        q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
    """
    args = get_args()

    batch_size, seq_length, head_num, head_dim = q.shape[0], q.shape[1], q.shape[2], q.shape[3]

    if not hasattr(self, 'attention_mask'):
        self.attention_mask = (torch.tril(torch.ones([seq_length, seq_length]), diagonal=-(args.pre_tockens + 1)) \
                + torch.triu(torch.ones([seq_length, seq_length]), diagonal=args.next_tockens + 1)).bool().npu()
    if args.shape_order == 'BSH':
        q, k, v = [rearrange(x, 'b s h d -> b s (h d)') for x in [q, k, v]]
    elif args.shape_order == 'SBH':
        q, k, v = [rearrange(x, 'b s h d -> s b (h d)') for x in [q, k, v]]
    elif args.shape_order != 'BSND':
        raise ValueError('Invalid shape-order: {}, shape-order must be SBH or BSH or BSND'.format(args.shape_order))

    try:
        scale = 1.0 / math.sqrt(head_dim) if self.softmax_scale is None else self.softmax_scale
    except Exception as e:
        raise ValueError('Invalid head_dim: {}'.format(head_dim)) from e

    output = torch_npu.npu_fusion_attention( \
        q, k, v, head_num, args.shape_order, \
        pse=None, \
        padding_mask=None, \
        atten_mask=self.attention_mask, \
        scale=scale, \
        pre_tockens=args.pre_tockens, \
        next_tockens=args.next_tockens, \
        keep_prob=1 - self.dropout_p, \
        inner_precise=0
    )[0]

    if args.shape_order == 'BSH':
        output = rearrange(output, 'b s (h d) -> b s h d', h=head_num)
    elif args.shape_order == 'SBH':
        output = rearrange(output, 's b (h d) -> b s h d', h=head_num)
    elif args.shape_order != 'BSND':
        raise ValueError('Invalid shape-order: {}, shape-order must be SBH or BSH or BSND'.format(args.shape_order))

    return output
    
    
def ParallelAttention_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        config = args[0]
        query_projection_size = config.kv_channels * config.num_attention_heads
        _args = get_args()
        if _args.group_query_attention:
            kv_projection_size = _args.kv_channels * _args.num_query_groups
        else:
            kv_projection_size = _args.kv_channels * _args.num_attention_heads
        # qkv bias
        bias = _args.add_qkv_bias or _args.add_bias_linear
        self.query_key_value = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            query_projection_size + 2 * kv_projection_size,
            config=config,
            init_method=config.init_method,
            bias=bias,
            gather_output=False)
        # dense bias
        bias = _args.add_dense_bias or _args.add_bias_linear
        skip_bias_add = _args.skip_bias_add
        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            query_projection_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=skip_bias_add)
    return wrapper
