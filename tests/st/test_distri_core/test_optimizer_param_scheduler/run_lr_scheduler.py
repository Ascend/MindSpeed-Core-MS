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
"""Pipeline parallel test"""

import argparse
import mindspore as ms
import mindspore.dataset as ds
from mindspore.mint.optim import AdamW
from mindspore.communication import init

from mindspeed_ms.core.dist_checkpointing import load_checkpoint
from mindspeed_ms.core.optimizer import get_optimizer_param_scheduler
from mindspeed_ms.core.parallel_state import initialize_model_parallel
from mindspeed_ms.training import TrainOneStepCell, train
from mindspeed_ms.training import parse_args, core_transformer_config_from_yaml
from tests.st.test_distri_core.test_pipeline_parallel.test_pipeline_net import PipelineTestNet, FakeData

ms.set_seed(2024)


def run_lr_scheduler():
    """main function."""
    args = parse_args()
    config = core_transformer_config_from_yaml(args)
    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
    init()

    initialize_model_parallel()

    # generate dataset
    dataset = FakeData(data_num=32, seq_length=args.seq_length)
    dataset_parallel = ds.GeneratorDataset(dataset, column_names=['input_ids', 'labels'], shuffle=False)
    # calculate global batch size
    dataset_parallel = dataset_parallel.batch(args.global_batch_size)
    print("global batch size: ", args.global_batch_size, flush=True)

    # init model
    network = PipelineTestNet(config)

    optimizer = AdamW(params=network.trainable_params(), lr=0.001)

    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    # init train one step cell
    train_one_step_cell = TrainOneStepCell(network, optimizer, opt_param_scheduler, config)

    train(train_one_step_cell, dataset_parallel, config)

    if args.yaml_cfg == 'test_iteration_tarining.yaml':
        assert optimizer.param_groups[0]['lr'] == 2.9999999999999997e-06
        load_checkpoint(config, network, optimizer, opt_param_scheduler, f"./output")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml-cfg', type=str, default='test_iteration_tarining.yaml',
                        help="test_iteration_tarining.yaml: test_lr_base_iteration_tarining")
    run_lr_scheduler()
