from functools import wraps

from megatron.core.transformer.custom_layers.transformer_engine import TENorm


def get_gpt_layer_local_spec_wrapper(fn):
    @wraps(fn)
    def wrapper(num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False):
        res = fn(num_experts, moe_grouped_gemm, qk_layernorm)
        res.submodules.input_layernorm = TENorm
        if qk_layernorm:
            res.submodules.self_attention.submodules.q_layernorm = TENorm
            res.submodules.self_attention.submodules.k_layernorm = TENorm
        res.submodules.pre_mlp_layernorm = TENorm
        return res

    return wrapper
