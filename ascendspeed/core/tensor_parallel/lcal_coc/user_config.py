import torch
import torch_npu

from .min_comm_cfg import min_comm_config, ModuleType
from .cc_parallel_linears_all_reduce import CCColumnAllReduceFunction, CCRowAllReduceFunction
from .cc_parallel_linears_sequence_parallel import CCColumnSeqParallelFunction, CCRowSeqParallelFunction
from .rewrite_parallel_linears_all_reduce import RewriteColumnAllReduceFunction, RewriteRowAllReduceFunction
from .rewrite_parallel_linears_sequence_parallel import RewriteColumnSeqParallelFunction, RewriteRowSeqParallelFunction
from .cc_sequence_parallel_fused_coc import FusedCCColumnSeqParallelFunction, FusedCCRowSeqParallelFunction


cc_cfgs = {
    'recompute_all_gather': True,
    'matmul_soc_friendly': True,
    'print_tensor_value_open': False,
    'customized_cc': {},
    'enable_cc_in_column_backward': False,
    'k_min': 1024,
    'k_max': 4096,
}


def check_config_valid():
    if min_comm_config.sequence_parallel_enabled:
        if min_comm_config.module_type not in [ModuleType.ORIGINAL_SEQ_PARALLEL,
                                               ModuleType.REWRITE_SEQ_PARALLEL,
                                               ModuleType.CC_FOR_SEQ_PARALLEL]:
            raise ValueError("In CoC, the config of sequence parallel is not valid")
    else:
        if min_comm_config.module_type not in [ModuleType.ORIGINAL_ALL_REDUCE,
                                               ModuleType.REWRITE_ALL_REDUCE,
                                               ModuleType.CC_FOR_ALL_REDUCE]:
            raise ValueError("In CoC, the config of sequence parallel is not valid")


def get_value_from_cfg(attr_name):
    if attr_name not in cc_cfgs.keys():
        raise RuntimeError("Lack attr_name: ", attr_name)
    return cc_cfgs[attr_name]


def initialize_cc_from_cfg(cfg):
    from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
    from megatron.core.parallel_state import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_world_size,
        get_tensor_model_parallel_rank
    )
    from megatron.core.tensor_parallel.mappings import (
        _reduce,
        _reduce_scatter_along_first_dim,
        _gather_along_first_dim
    )
    min_comm_config.register_tp_get_functions(get_tensor_model_parallel_group,
                                              get_tensor_model_parallel_world_size,
                                              get_tensor_model_parallel_rank)
    min_comm_config.register_class(ColumnParallelLinear,
                                   RowParallelLinear)
    min_comm_config.register_mappings(_reduce,
                                      _reduce_scatter_along_first_dim,
                                      _gather_along_first_dim)
    min_comm_config.register_sequence_parallel_switch(cfg.sequence_parallel)

    min_comm_config.register_customized_cc(get_value_from_cfg('customized_cc'))
    min_comm_config.register_matmul_soc_friendly_setting(get_value_from_cfg('matmul_soc_friendly'),
                                                         int(get_value_from_cfg('k_min')),
                                                         int(get_value_from_cfg('k_max')))
    min_comm_config.register_all_gather_recomputation_switch(get_value_from_cfg('recompute_all_gather'))
    min_comm_config.register_print_tensor_value_switch(get_value_from_cfg('print_tensor_value_open'))
    min_comm_config.register_column_backward_cc_switch(get_value_from_cfg('enable_cc_in_column_backward'))
    min_comm_config.register_check_fcn(check_config_valid)
    min_comm_config.acquire_module_type(cfg.tensor_model_parallel_size)

    map_type2autograd_class = {
        ModuleType.REWRITE_SEQ_PARALLEL: [RewriteColumnSeqParallelFunction,
                                          RewriteRowSeqParallelFunction],
        ModuleType.REWRITE_ALL_REDUCE: [RewriteColumnAllReduceFunction,
                                        RewriteRowAllReduceFunction],
        ModuleType.CC_FOR_SEQ_PARALLEL: [CCColumnSeqParallelFunction,
                                         CCRowSeqParallelFunction],
        ModuleType.CC_FOR_ALL_REDUCE: [CCColumnAllReduceFunction,
                                       CCRowAllReduceFunction]
    }

    curr_rank = torch.npu.current_device()
    if min_comm_config.cc_fused_kernel:
        if torch.npu.current_device() == 0:
            print("COC REPLACE WITH CC FUSED KERNEL SCRIPT!")
        min_comm_config.replace_forward_functions_by_autograd_class(FusedCCColumnSeqParallelFunction,
                                                                    FusedCCRowSeqParallelFunction)
    elif "ORIGINAL" not in min_comm_config.module_type.name:
        if torch.npu.current_device() == 0:
            print("COC REPLACE WITH NORMAL CC SCRIPT!")
        parallel_linear_autograd_class = map_type2autograd_class.get(min_comm_config.module_type)
        if parallel_linear_autograd_class is None:
            raise RuntimeError("Module type is not matched.")
        min_comm_config.replace_forward_functions_by_autograd_class(parallel_linear_autograd_class[0],
                                                                    parallel_linear_autograd_class[1])
    else:
        if torch.npu.current_device() == 0:
            print("COC REPLACE NONE!")
