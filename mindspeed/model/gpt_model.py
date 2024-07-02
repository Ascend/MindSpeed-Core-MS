from functools import wraps
from megatron.training import get_args
from megatron.core import parallel_state
from mindspeed.core.tensor_parallel.checkpoint_manager import get_pipeline_checkpoint_manager


def post_language_model_processing_wrapper(post_language_model_processing):
    @wraps(post_language_model_processing)
    def wrapper(lm_output, labels, logit_weights, parallel_output, fp16_lm_cross_entropy):
        args = get_args()
        if args.recompute_in_advance and lm_output.requires_grad:
            pipeline_checkpoint_manager = get_pipeline_checkpoint_manager(
                num_of_chunks=parallel_state.get_virtual_pipeline_model_parallel_world_size())

            def recompute(grad):
                pipeline_checkpoint_manager.recompute_next(parallel_state.get_virtual_pipeline_model_parallel_rank())

            lm_output.register_hook(recompute)
        return post_language_model_processing(lm_output, labels, logit_weights, parallel_output, fp16_lm_cross_entropy)

    return wrapper
