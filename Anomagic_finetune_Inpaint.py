import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch.nn as nn
import math
import random
import argparse
from pathlib import Path
import json
import itertools
import time
from peft import LoraConfig, LoraModel
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers import StableDiffusionInpaintPipelineLegacy
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
import numpy as np
from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available
from torch.utils.data.distributed import DistributedSampler

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, datasets_root, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05,
                 ti_drop_rate=0.05,
                 use_analysis_text=True, use_short_text=True):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.use_analysis_text = use_analysis_text
        self.use_short_text = use_short_text
        self.datasets_root = datasets_root

        # Load and flatten the dataset structure
        with open(json_file) as f:
            data = json.load(f)

        # Extract all images from all datasets
        self.data = []
        for dataset in data["datasets"]:
            for image_info in dataset["images"]:
                self.data.append(image_info)

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()

    def __getitem__(self, idx):
        max_retries = len(self.data)
        retries = 0
        while retries < max_retries:
            try:
                item = self.data[idx]

                text = None
                if self.use_analysis_text and item.get("analysis_files"):
                    analysis_path = os.path.join(self.datasets_root, item["analysis_files"])
                    if not os.path.exists(analysis_path):
                        raise FileNotFoundError(f"Analysis text file does not exist: {analysis_path}")
                    if not os.access(analysis_path, os.R_OK):
                        raise PermissionError(f"No permission to read analysis text file: {analysis_path}")
                    with open(analysis_path, "r") as f:
                        text = f.read().strip()

                if text is None and self.use_short_text and item.get("analysis_files_short"):
                    short_path = os.path.join(self.datasets_root, item["analysis_files_short"])
                    if not os.path.exists(short_path):
                        raise FileNotFoundError(f"Short text file does not exist: {short_path}")
                    if not os.access(short_path, os.R_OK):
                        raise PermissionError(f"No permission to read short text file: {short_path}")
                    with open(short_path, "r") as f:
                        text = f.read().strip()

                if text is None:
                    text = f"A high-quality image showing {item.get('defect_name', 'unknown')} defect on {item.get('category', 'unknown')}"

                image_path = os.path.join(self.datasets_root, item["image_path"])
                mask_path = os.path.join(self.datasets_root, item["mask_path"])

                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file does not exist: {image_path}")
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Mask file does not exist: {mask_path}")

                with Image.open(image_path) as img:
                    img.verify()
                    raw_image = Image.open(image_path)

                with Image.open(mask_path) as msk:
                    msk.verify()
                    mask = Image.open(mask_path)

                mask = mask.resize((64, 64))
                mask = mask.convert('L')
                mask = torch.tensor(np.array(mask), dtype=torch.float32)
                mask = (mask > 0.5).float()

                image = self.transform(raw_image.convert("RGB"))
                clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values

                drop_image_embed = 0
                rand_num = random.random()
                if rand_num < self.i_drop_rate:
                    drop_image_embed = 1
                elif rand_num < (self.i_drop_rate + self.t_drop_rate):
                    text = ""
                elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
                    text = ""
                    drop_image_embed = 1

                text_input_ids = self.tokenizer(
                    text,
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids

                return {
                    "image": image,
                    "mask": mask,
                    "text_input_ids": text_input_ids,
                    "clip_image": clip_image,
                    "drop_image_embed": drop_image_embed,
                }

            except (Image.UnidentifiedImageError, FileNotFoundError, OSError, PermissionError) as e:
                print(f"Sample {idx} loading failed: {str(e)}, skipping this sample")
                retries += 1
                idx = (idx + 1) % len(self.data)
            except Exception as e:
                print(f"Sample {idx} encountered unknown error: {str(e)}, skipping this sample")
                retries += 1
                idx = (idx + 1) % len(self.data)

        raise RuntimeError(f"Failed to load {max_retries} samples after attempts, please check dataset integrity")

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    masks = torch.stack([example["mask"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]

    return {
        "images": images,
        "masks": masks,
        "text_input_ids": text_input_ids,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds
    }


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.proj_out = nn.Linear(1280, 1024)

    def forward(self, x, mask=None):
        x = x.permute(0, 2, 1)
        batch_size, channels, h = x.size()
        height = int(math.sqrt(h))
        width = height
        x = x.view(batch_size, channels, width, height)
        batch_size, channels, height, width = x.size()

        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)

        attention_scores = torch.bmm(q, k)

        if mask is not None:
            mask = nn.functional.interpolate(mask, size=(height, width), mode='nearest')
            mask = mask.view(batch_size, 1, height * width)
            large_constant = 1e6
            attention_scores = attention_scores - (1 - mask) * large_constant

        attention_weights = self.softmax(attention_scores)
        out = torch.bmm(v, attention_weights.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        out = out.view(batch_size, channels, height * width)
        out = out.permute(0, 2, 1)
        out = self.proj_out(out)

        return out


class Anomagic(torch.nn.Module):
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        anomagic_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, anomagic_tokens], dim=1)
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        orig_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        new_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        assert orig_proj_sum != new_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

    def save_checkpoint(self, save_path):
        state_dict = {
            "image_proj": self.image_proj_model.state_dict(),
            "ip_adapter": {k: v.clone() for k, v in self.adapter_modules.state_dict().items()}
        }
        torch.save(state_dict, save_path)
        print(f"Saved checkpoint to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default='/home/jiangyuxin/CODE/My_paper/Anomagic/models/ip-adapter_sd15.bin',
        help="Path to pretrained ip adapter model.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default="",
        help="Training data JSON file path",
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        default="",
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="",
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=10000)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save a checkpoint of the training state every X updates",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='The integration to report the results and logs to.',
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--finetune_mode",
        type=str,
        default="full",
        choices=["lora_only", "feature_extraction", "full"],
        help="Finetune mode: lora_only (only LoRA), feature_extraction (everything except LoRA), full (all parameters)"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def load_lora_model(unet, device, diffusion_model_learning_rate):
    for param in unet.parameters():
        param.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.add_adapter(unet_lora_config)
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    return unet, lora_layers


def encode_long_text(input_ids, tokenizer, text_encoder, max_length=77, device="cuda"):
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    batch_size = input_ids.size(0)
    hidden_dim = text_encoder.config.hidden_size
    combined_embeddings = torch.zeros(batch_size, hidden_dim, device=device)

    for batch_idx in range(batch_size):
        current_input_ids = input_ids[batch_idx]
        chunks = [
            current_input_ids[i:i + max_length]
            for i in range(0, len(current_input_ids), max_length)
        ]

        embeddings = []
        for chunk in chunks:
            chunk_len = len(chunk)
            padding_len = max_length - chunk_len

            chunk_input = {
                "input_ids": torch.cat([
                    chunk.unsqueeze(0).to(device),
                    torch.zeros(1, padding_len, dtype=torch.long, device=device)
                ], dim=1),
                "attention_mask": torch.cat([
                    torch.ones(1, chunk_len, dtype=torch.long, device=device),
                    torch.zeros(1, padding_len, dtype=torch.long, device=device)
                ], dim=1)
            }

            with torch.no_grad():
                chunk_emb = text_encoder(**chunk_input).last_hidden_state
                embeddings.append(chunk_emb[:, :chunk_len, :].mean(dim=1))

        if embeddings:
            combined_embeddings[batch_idx] = torch.mean(torch.cat(embeddings, dim=0), dim=0)
        else:
            combined_embeddings[batch_idx] = torch.zeros(hidden_dim, device=device)

    return combined_embeddings.unsqueeze(1)


def setup_finetune_mode(args, anomagic_model, attention_module, lora_layers):
    """根据选择的微调模式设置参数更新策略"""

    # 先把所有生成器转为列表
    image_proj_params = list(anomagic_model.image_proj_model.parameters())
    adapter_params = list(anomagic_model.adapter_modules.parameters())
    attention_params = list(attention_module.parameters())

    # lora_layers 可能已经是列表，如果不是就转换
    if not isinstance(lora_layers, list):
        lora_layers = list(lora_layers)

    if args.finetune_mode == "lora_only":
        print("Finetune mode: LoRA only")

        # 冻结其他参数
        for param in image_proj_params + adapter_params + attention_params:
            param.requires_grad = False

        # 只返回LoRA参数
        params_to_opt = lora_layers
        trainable_params = sum(p.numel() for p in lora_layers)

    elif args.finetune_mode == "feature_extraction":
        print("Finetune mode: Feature Extraction")

        # 冻结LoRA参数
        for param in lora_layers:
            param.requires_grad = False

        # 返回其他参数
        params_to_opt = image_proj_params + adapter_params + attention_params
        trainable_params = sum(p.numel() for p in params_to_opt)

    else:  # "full" 模式
        print("Finetune mode: Full")

        # 返回所有参数
        params_to_opt = image_proj_params + adapter_params + lora_layers + attention_params
        trainable_params = sum(p.numel() for p in params_to_opt)

    print(f"Number of trainable parameters: {trainable_params:,}")

    return params_to_opt  # 返回的是列表，不是生成器

def main():
    args = parse_args()

    # Initialize distributed training
    accelerator = Accelerator(
        gradient_accumulation_steps=1,  # Explicitly disable gradient accumulation
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.output_dir,
    )

    # Print gradient accumulation steps
    if accelerator.is_main_process:
        print(f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")
        print(f"Number of processes (GPUs): {accelerator.num_processes}")
        print(f"Selected finetune mode: {args.finetune_mode}")

    # Set random seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load model components
    # 设置数据类型
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # 加载VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=dtype,
    )

    # 创建噪声调度器
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    # 创建Inpainting管道
    print(f"Loading base model from: {args.pretrained_model_name_or_path}")
    pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
    )

    # 更换调度器以获得更好的采样效果
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # 提取各个组件
    unet = pipe.unet
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # 加载图像编码器
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

    # Freeze unnecessary parameters (这些在所有模式下都冻结)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # Initialize Anomagic projection model
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )

    # Initialize attention processors
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)

    # Add LoRA adapter
    unet, lora_layers = load_lora_model(unet, accelerator.device, 4e-4)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    # Initialize Anomagic model
    anomagic_model = Anomagic(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path)
    attention_module = SelfAttention(1280)

    # Set mixed precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move models to device
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    attention_module.to(accelerator.device, dtype=weight_dtype)

    # 根据微调模式设置优化参数
    params_to_opt = setup_finetune_mode(args, anomagic_model, attention_module, lora_layers)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Prepare dataset
    train_dataset = MyDataset(
        args.data_json_file,
        args.datasets_root,
        tokenizer=tokenizer,
        size=args.resolution,
        use_analysis_text=False,
        use_short_text=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True if args.dataloader_num_workers > 0 else False
    )

    # Prepare all components for distributed training
    anomagic_model, attention_module, optimizer, train_dataloader = accelerator.prepare(
        anomagic_model, attention_module, optimizer, train_dataloader
    )

    # 获取实际的可训练参数列表
    if args.finetune_mode == "lora_only":
        # 在accelerator.prepare之后，我们需要重新获取lora_layers
        lora_layers = filter(lambda p: p.requires_grad, anomagic_model.unet.parameters())
        params_to_opt = list(lora_layers)
    elif args.finetune_mode == "feature_extraction":
        # 重新设置参数列表，排除LoRA参数
        params_to_opt = list(itertools.chain(
            anomagic_model.image_proj_model.parameters(),
            anomagic_model.adapter_modules.parameters(),
            attention_module.parameters()
        ))
    # full模式保持原有参数列表

    # 为完整模式重新创建优化器（如果在setup之后参数有变化）
    if args.finetune_mode in ["lora_only", "feature_extraction"]:
        # 需要重新创建优化器，因为参数列表已改变
        optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer = accelerator.prepare(optimizer)

    # Training loop
    global_step = 0
    for epoch in range(args.num_train_epochs):
        begin = time.perf_counter()
        anomagic_model.train()
        attention_module.train()

        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin

            with accelerator.accumulate(anomagic_model):
                # Encode images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample random timesteps
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    # Get image embeddings
                    outputs = image_encoder(batch["clip_images"].to(weight_dtype))
                    image_embeds = outputs.image_embeds
                    last_feature_layer_output = outputs.last_hidden_state

                    # Process dropout
                    image_embeds_ = []
                    for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                        if drop_image_embed == 1:
                            image_embeds_.append(torch.zeros_like(image_embed))
                        else:
                            image_embeds_.append(image_embed)

                    # Encode text
                    encoder_hidden_states = encode_long_text(
                        batch["text_input_ids"],
                        tokenizer,
                        text_encoder,
                        device=accelerator.device
                    )

                # Process image features through attention module
                image_embeds = attention_module(last_feature_layer_output[:, :256, :], batch["masks"].unsqueeze(1))

                # Forward pass
                noise_pred = anomagic_model(noisy_latents, timesteps, encoder_hidden_states, image_embeds)

                # Compute loss
                loss = (F.mse_loss(noise_pred.float(), noise.float(), reduction="none") * batch["masks"].unsqueeze(
                    1)).mean([1, 2, 3]).mean()

                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {:.3f}s, time: {:.3f}s, step_loss: {:.6f}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, loss.item()))

            global_step += 1

            # Log records
            if accelerator.sync_gradients:
                if accelerator.is_main_process:
                    if global_step % 100 == 0:
                        print(
                            f"Epoch {epoch}, Step {global_step}, Loss: {loss.detach().item():.6f}, Mode: {args.finetune_mode}"
                        )

            # Save checkpoint
            if accelerator.is_main_process and global_step % args.save_steps == 0:
                print(f"Saving model at global_step {global_step}")
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                print(f"Checkpoint dir: {checkpoint_dir}")

                # 保存Anomagic模型
                anomagic_save_path = os.path.join(checkpoint_dir, f"anomagic-{global_step}.bin")
                anomagic_model.save_checkpoint(anomagic_save_path)

                # 保存attention module
                attention_module_save_path = os.path.join(checkpoint_dir, f"attention_module-{global_step}.bin")
                torch.save(attention_module.state_dict(), attention_module_save_path)

                # 如果是LoRA only模式，额外保存LoRA权重
                if args.finetune_mode == "lora_only":
                    lora_save_path = os.path.join(checkpoint_dir, f"lora-{global_step}.bin")
                    lora_state_dict = {k: v.clone() for k, v in anomagic_model.unet.state_dict().items() if 'lora' in k}
                    torch.save(lora_state_dict, lora_save_path)
                    print(f"Saved LoRA weights to {lora_save_path}")

            begin = time.perf_counter()

    # Final save
    if accelerator.is_main_process:
        checkpoint_dir = os.path.join(args.output_dir, "final")
        os.makedirs(checkpoint_dir, exist_ok=True)

        anomagic_save_path = os.path.join(checkpoint_dir, "anomagic-final.bin")
        attention_module_save_path = os.path.join(checkpoint_dir, "attention_module-final.bin")

        anomagic_model.save_checkpoint(anomagic_save_path)
        torch.save(attention_module.state_dict(), attention_module_save_path)

        # 如果是LoRA only模式，额外保存LoRA权重
        if args.finetune_mode == "lora_only":
            lora_save_path = os.path.join(checkpoint_dir, "lora-final.bin")
            lora_state_dict = {k: v.clone() for k, v in anomagic_model.unet.state_dict().items() if 'lora' in k}
            torch.save(lora_state_dict, lora_save_path)
            print(f"Saved final LoRA weights to {lora_save_path}")

        print(f"Training completed! Final models saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()