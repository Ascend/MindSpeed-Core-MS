# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Dataset."""

import torch
import mindspore.dataset as ds
from mindspeed_ms.core.parallel_state import get_data_parallel_world_size

def check_mismatch(tokens, labels):
    if tokens is None or labels is None:
        return
    mid_idx = tokens.size(1) // 2  # 找到中间的索引
    for i in range(tokens.size(0)):
        if tokens[i, mid_idx+1] != labels[i, mid_idx] and labels[i, mid_idx] != -100:
            print(f"Mismatch found for sample {i} after broadcast:")
            print(f"tokens: {tokens[i]}")
            print(f"labels: {labels[i]}")

def get_batch(batch, args, eod):
    """Generate a batch."""
    has_negative = (batch['tokens'] < 0).any()
    # add support for reset_attention_mask, reset_position_ids, eod_mask_loss for pack data
    attention_mask, loss_mask, position_ids = get_ltor_reset_masks_and_position_ids(
        batch['tokens'].unsqueeze(0),
        batch['loss_mask'],
        eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        has_negative,
        batch['labels'])
    batch['attention_mask'] = attention_mask
    batch['loss_mask'] = loss_mask
    batch['position_ids'] = position_ids
    #set_attention_mask(batch['attention_mask'])
    if has_negative:
        batch['tokens'] = torch.abs(batch['tokens'])
        batch['labels'] = torch.abs(batch['labels'])
    #check_mismatch(batch['tokens'], batch['labels'])
    tokens = batch['tokens'].cpu().numpy()
    labels = batch['labels'].cpu().numpy()
    attention_mask = batch['attention_mask'].cpu().numpy().reshape((1, 4096, 4096))
    loss_mask = batch['loss_mask'].cpu().numpy()
    position_ids = batch['position_ids'].cpu().numpy().reshape((4096))

    return tokens, labels, attention_mask, loss_mask, position_ids

def get_ltor_reset_masks_and_position_ids(data,
                                          loss_mask,
                                          eod_token,
                                          reset_position_ids,
                                          reset_attention_mask,
                                          eod_mask_loss,
                                          has_negative,
                                          labels):
    """Build masks and position id for left to right model."""
    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    if has_negative:
        loss_mask = torch.where(labels < 0, 0, 1) #這裏應該根據label進行設定
    #loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modified based on batch index.
    if reset_position_ids:
        position_ids = position_ids.copy()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indices where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indices from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.copy()

            # Loop through EOD indices:
            prev_index = 0
            pre_eod_idx = -1
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                if i == pre_eod_idx:
                    break
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                    pre_eod_idx = i + 1

                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1
    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids

# Todo (xiaohui): configurarize this api...
def get_dataset(config, generator_lst, tokenizer, args):
    """
    Create dataset

    Inputs:
        batch_size: batch size
        data_path: path of your MindRecord files
        device_num: total device number
        rank: current rank id
        drop: whether drop remainder
        column_name: the column name of the mindrecord file. Default is input_ids
        epoch: The repeat times of the dataset
    Returns:
        dataset_restore: the dataset for training or evaluating
    """

    # Control the size of data queue in the consideration of the memory
    ds.config.set_prefetch_size(1)
    ds.config.set_num_parallel_workers(1)

    # data_parallel = get_dp_world_size() if is_initialized() else 1
    data_parallel = get_data_parallel_world_size()
    # mp_size = get_tensor_model_parallel_world_size()
    micro_batch_num = config.micro_batch_num if config.micro_batch_num else 1
    print("batch_size is {} micro_batch_num is {} data_parallel is {}" \
          .format(config.batch_size, micro_batch_num, data_parallel))
    global_batch_size = config.batch_size * micro_batch_num * data_parallel

    # local_device_num = LOCAL_DEVICE_NUM if LOCAL_DEVICE_NUM < data_parallel else data_parallel
    # rank = get_data_parallel_rank() if data_parallel > 1 else 0

    train_dataset = ds.GeneratorDataset(generator_lst[0], column_names=["dataset"])
    dataset_num = len(train_dataset)
    print(f"Total dataset num: {dataset_num}")

    batch_size = global_batch_size // (data_parallel)

    # if if_gather:
    #     batch_size = global_batch_size // (data_parallel) #Todo:there can be a chance that the it can not be divided.
    # else:
    #     batch_size = global_batch_size // (data_parallel // LOCAL_DEVICE_NUM)

    map_func = (
        lambda batch: get_batch(batch, args=args, eod=tokenizer.eod))

    train_dataset = train_dataset.map(operations=map_func,
                                      input_columns=["dataset"],
                                      output_columns=['tokens',
                                                      'labels',
                                                      'attention_mask',
                                                      'loss_mask',
                                                      'position_ids'],
                                      )
    train_dataset = train_dataset.batch(batch_size, True)
    return train_dataset
