# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

from functools import partial

from mmengine.config import Config
import torch

import mindspeed.megatron_adaptor
from mindspeed.utils import get_batch_on_this_cp_rank

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, get_timers, print_rank_0, pretrain
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.training.arguments import core_transformer_config_from_args

from opensora.utils.train_utils import update_ema
from opensora.utils.misc import requires_grad
from opensora.datasets import DatasetFromCSV, get_transforms_video, get_transforms_image, prepare_dataloader
from opensora.registry import MODELS, SCHEDULERS, build_module

scheduler = None
vae = None
text_encoder = None
cfg = None


def initialize():
    args = get_args()
    dtype = args.params_dtype

    def initialize_models():
        global cfg
        global vae
        global text_encoder

        vae_cfg = cfg.vae
        text_encoder_cfg = cfg.text_encoder
        vae = build_module(vae_cfg, MODELS)
        text_encoder = build_module(text_encoder_cfg, MODELS, device=torch.cuda.current_device())

        vae = vae.to(torch.cuda.current_device(), dtype)

    def initialize_scheduler():
        global scheduler
        scheduler = build_module(cfg.scheduler, SCHEDULERS)

    def initialize_config():
        args = get_args()
        global cfg
        cfg = Config.fromfile(args.additional_config)

    initialize_config()
    initialize_scheduler()
    initialize_models()


def initialize_pipeline_tensor_shapes(hidden_size):
    args = get_args()
    micro_batch_size = args.micro_batch_size
    dtype = args.params_dtype
    latent_size = vae.get_latent_size((cfg.num_frames, *cfg.image_size))
    text_encoder_maxlen = text_encoder.model_max_length
    args.pipeline_tensor_shapes = [
                {'shape': (micro_batch_size, text_encoder.output_dim, hidden_size), 'dtype': dtype},
                {'shape': (micro_batch_size, vae.out_channels, *latent_size), 'dtype': dtype},
                {'shape': (micro_batch_size, 1, text_encoder_maxlen, hidden_size), 'dtype': dtype},
                {'shape': (micro_batch_size,), 'dtype': torch.float32},
                {'shape': (micro_batch_size, hidden_size * 6), 'dtype': dtype},
                {'shape': (micro_batch_size, text_encoder_maxlen), 'dtype': torch.float32},
                {'shape': (micro_batch_size, vae.out_channels, *latent_size), 'dtype': dtype},
                {'shape': (micro_batch_size, vae.out_channels, *latent_size), 'dtype': dtype},
                {'shape': (micro_batch_size, hidden_size), 'dtype': dtype}
            ]
    setattr(forward_step, 'pipeline_tensor_shapes', args.pipeline_tensor_shapes)


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()
    dtype = args.params_dtype
    latent_size = vae.get_latent_size((cfg.num_frames, *cfg.image_size))
    stdit = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        dtype=dtype,
    )

    ema = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        dtype=dtype,
    )

    requires_grad(ema, False)
    stdit.ema = ema
    update_ema(ema, stdit, decay=0, sharded=False)
    ema.eval()

    initialize_pipeline_tensor_shapes(stdit.hidden_size)
    stdit.config = core_transformer_config_from_args(get_args())
    return stdit


def get_batch_on_this_tp_rank(data_iterator):
    global vae
    global text_encoder
    args = get_args()
    dtype = args.params_dtype

    if data_iterator is not None:
        batch = next(data_iterator)
    else:
        batch = None
    # x.shape: [B, C, T, H/P, W/P]
    x = batch['video'].to(torch.cuda.current_device(), dtype)
    y = batch['text']

    with torch.no_grad():
        # Prepare visual inputs
        # x.shape: [B, C, T, H/P, W/P]
        x = vae.encode(x).contiguous()
        # Prepare text inputs
        encoded_text = text_encoder.encode(y)
    y = encoded_text['y'].contiguous()
    mask = encoded_text['mask'].contiguous()

    batch = {
        'x': x,
        'y': y,
        'mask': mask
    }
    return batch


def get_batch(data_iterator):
    """Build the batch."""

    if mpu.is_pipeline_first_stage():
        batch = get_batch_on_this_tp_rank(data_iterator)
        return batch['x'], batch['y'], batch['mask']
    else:
        return None, None, None


def loss_func(x_t, x_0, t, noise, output_tensor):
    loss_dict = scheduler.training_losses(output_tensor[0], x_t, x_0, t, noise=noise)
    loss = loss_dict["loss"].mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)
    return loss, {"loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    dtype = args.params_dtype

    # Get the batch.
    x, y, mask = get_batch(data_iterator)

    num_timesteps = 1000
    micro_bs = args.micro_batch_size
    timestep = None
    x_0 = None
    noise = None

    if mpu.is_pipeline_first_stage():
        x_0 = x.clone()
        timestep = torch.randint(0, num_timesteps, (micro_bs,), device=torch.cuda.current_device(), dtype=torch.int64)
        noise = torch.randn_like(x)
        noise = noise.to(device=torch.cuda.current_device(), dtype=dtype)
        x = scheduler.q_sample(x, timestep, noise=noise)
        x_t = x.clone()

    if mpu.get_pipeline_model_parallel_world_size() > 1:
        x, x_t, y, timestep, t0, mask, x_0, noise, t = model(x, timestep, y, x_0, noise, mask)
        output_tensor_wrap = [x, x_t, y, timestep, t0, mask, x_0, noise, t]
    else:
        x = model(x, timestep, y, x_0, noise, mask)
        output_tensor_wrap = [x]

    return output_tensor_wrap, partial(loss_func, x_t, x_0, timestep, noise)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    dataset = DatasetFromCSV(
        args.data_path[0],
        transform=(
            get_transforms_video(cfg.image_size[0])
            if not cfg.use_image_transform
            else get_transforms_image(cfg.image_size[0])
        ),
        num_frames=cfg.num_frames,
        frame_interval=cfg.frame_interval,
        root=cfg.root,
    )

    dataloader = prepare_dataloader(
        dataset,
        batch_size=args.micro_batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=mpu.get_data_parallel_group(),
    )

    return iter(dataloader), None, None


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'dataloader_type': 'external',
                       'init_func': initialize}
    )
