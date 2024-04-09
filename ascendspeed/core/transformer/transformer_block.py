from functools import wraps
from megatron.legacy.model.utils import get_norm


class DummyTENorm:
    def __init__(self, *args, **kwargs):
        pass


def final_layernorm_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        fn(*args, **kwargs)
        if hasattr(args[0], 'final_layernorm'):
            args[0].final_layernorm = get_norm(args[0].config)
    return wrapper