
SHELL_RULES = [
    ["torchrun", "msrun"],
    ["--nproc_per_node", "--local_worker_num"],
    ["--nnodes $NNODES", "--worker_num $WORLD_SIZE"]
]
