from functools import wraps

from megatron.core.transformer.custom_layers.transformer_engine import TENorm


def get_gpt_layer_local_spec_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        res.submodules.input_layernorm = TENorm
        res.submodules.self_attention.submodules.q_layernorm = TENorm
        res.submodules.self_attention.submodules.k_layernorm = TENorm
        res.submodules.pre_mlp_layernorm = TENorm
        return res

    return wrapper
