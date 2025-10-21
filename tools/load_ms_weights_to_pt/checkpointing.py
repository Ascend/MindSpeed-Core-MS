from functools import wraps
 
def load_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            res = fn(*args, **kwargs)
        except:
            from tools.load_ms_weights_to_pt.serialization import load_ms_weights
            res = load_ms_weights(*args, **kwargs)
        return res
    return wrapper