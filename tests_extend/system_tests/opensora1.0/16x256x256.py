num_frames = 16
frame_interval = 3
image_size = (256, 256)

# Define dataset
root = None
use_image_transform = False
num_workers = 1

# Define model
model = dict(
    type="STDiT-XL/2/mindspeed",
    space_scale=0.5,
    time_scale=1.0,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL_mindspeed",
    from_pretrained="sd-vae-ft-ema",
    enable_sequence_parallelism=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=120,
    shardformer=False,
)
scheduler = dict(
    type="iddpm_mindspeed",
    timestep_respacing="",
)