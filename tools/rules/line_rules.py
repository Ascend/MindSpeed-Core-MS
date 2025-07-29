LINE_RULES = {
    "megatron": {
        "core/utils.py": ['''def is_te_min_version(version, check_equality=True):
     """Check if minimum version of `transformer-engine` is installed."""
-    if check_equality:
-        return get_te_version() >= PkgVersion(version)
-    return get_te_version() > PkgVersion(version)
+    return False''']
    }
}

SPECIAL_RULES = {}

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