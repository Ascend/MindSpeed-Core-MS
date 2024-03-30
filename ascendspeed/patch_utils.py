import importlib
import sys
import types


def get_func_name(func):
    if isinstance(func, str):
        return func
    return '.'.join((func.__module__, func.__qualname__))


def dummy_function_wrapper(func_name):
    def dummy_function(*args, **kwargs):
        raise RuntimeError('function {} no exist'.format(func_name))

    return dummy_function


class Patch:
    def __init__(self, orig_func_name, new_func):
        self.orig_module_name, self.orig_func_name = orig_func_name.rsplit('.', 1)
        self.orig_module = None
        self.orig_func = None

        self.patch_func = None
        self.wrappers = []
        if new_func is None:
            new_func = dummy_function_wrapper(orig_func_name)
        self.set_patch_func(new_func)

    @property
    def orig_func_id(self):
        return id(self.orig_func)

    @property
    def patch_func_id(self):
        return id(self.patch_func)

    def set_patch_func(self, new_func, force_patch=False):
        if new_func.__name__.endswith(('wrapper', 'decorator')):
            self.wrappers.append(new_func)
        else:
            if self.patch_func and not force_patch:
                raise RuntimeError('the patch of {} exist !'.format(self.orig_func_name))
            self.patch_func = new_func

    def apply_patch(self):
        self.orig_module, self.orig_func = Patch.parse_path(self.orig_module_name, self.orig_func_name)
        if self.patch_func is None:
            self.patch_func = self.orig_func

        for wrapper in self.wrappers:
            self.patch_func = wrapper(self.patch_func)

        setattr(self.orig_module, self.orig_func_name, self.patch_func)
        for key, value in sys.modules.items():
            if hasattr(value, self.orig_func_name) and id(getattr(value, self.orig_func_name)) == self.orig_func_id:
                setattr(value, self.orig_func_name, self.patch_func)

    @staticmethod
    def parse_path(module_path, function_name):
        modules = module_path.split('.')
        for i in range(1, len(modules) + 1):
            parent = '.'.join(modules[:i - 1])
            path = '.'.join(modules[:i])
            try:
                importlib.import_module(path)
            except ModuleNotFoundError:
                if not parent or not hasattr(importlib.import_module(parent), modules[i - 1]):
                    sys.modules[path] = types.ModuleType(path)
                    sys.modules[path].__file__ = 'ascendspeed.dummy_module.py'
                    if parent:
                        setattr(importlib.import_module(parent), modules[i - 1], sys.modules[path])
                else:
                    module = getattr(importlib.import_module(parent), modules[i - 1])
                    if hasattr(module, function_name):
                        return module, getattr(module, function_name)
                    else:
                        raise RuntimeError('no support this type patch!')

        if not hasattr(sys.modules[module_path], function_name):
            setattr(sys.modules[module_path], function_name, None)
        return sys.modules[module_path], getattr(sys.modules[module_path], function_name)


class AscendSpeedPatchesManager:
    patches_info = {}

    @staticmethod
    def register_patch(orig_func_name, new_func=None, force_patch=False):
        if orig_func_name not in AscendSpeedPatchesManager.patches_info:
            AscendSpeedPatchesManager.patches_info[orig_func_name] = Patch(orig_func_name, new_func)
        else:
            AscendSpeedPatchesManager.patches_info.get(orig_func_name).set_patch_func(new_func, force_patch)

    @staticmethod
    def apply_patches():
        for patch in AscendSpeedPatchesManager.patches_info.values():
            patch.apply_patch()
