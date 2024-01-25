import os
from functools import wraps

import torch
import torch.nn.functional as F
import megatron
from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region


def VocabParallelEmbeddingForward(self, input_):

    if torch.any(
        (input_ < 0) | (input_ >= self.num_embeddings)
    ):
        raise ValueError("An input token is out of bounds of the embedding table")
    if self.tensor_model_parallel_size > 1:
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | \
                     (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input *= ~input_mask
    else:
        masked_input = input_
        # Get the embeddings.
    output_parallel = self.weight[masked_input]

    # Mask the output embedding.
    if self.tensor_model_parallel_size > 1:
        output_parallel *= ~input_mask[..., None]
    # Reduce across all the model parallel GPUs.
    output = reduce_from_tensor_model_parallel_region(output_parallel)
    return output


def embedding_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        res = fn(self, *args, **kwargs)
        if int(os.getenv('NPU_DETECT', '0')):
            from torch_npu.hook_module.hook_module import HOOKModule

            def embedding_hook(grad):
                HOOKModule.embedding_list.append(torch.norm(grad))

            self.weight.register_hook(embedding_hook)
        return res

    return wrapper


def linear_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        res = fn(self, *args, **kwargs)
        if int(os.getenv('NPU_DETECT', '0')):
            from torch_npu.hook_module.hook_module import HOOKModule

            def linear_hook(grad):
                HOOKModule.linear_list.append(torch.norm(grad))

            self.weight.register_hook(linear_hook)
        return res

    return wrapper