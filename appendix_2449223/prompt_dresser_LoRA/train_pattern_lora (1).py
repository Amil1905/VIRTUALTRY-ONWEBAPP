"""
PATTERN-ONLY LORA TRAINING FOR PROMPTDRESSER (NO STYLE FIELDS)
- Style/season (summer/winter vs.) tamamen kaldırıldı.
- Sadece (person, cloth, target, mask, pose) eşleşmeleri ile çalışır.
- Dataset'in döndürdüğü `prompt` alanı aynen kullanılır.
- VRAM dostu optimizasyonlar korunmuştur (fp16/bf16, grad checkpointing, xFormers/slicing, VAE tiling, LoRA regex).
"""

import os
import re
import sys
import yaml
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

# PromptDresser repo yolları
sys.path.append('/content/drive/MyDrive/PromptDresser')

# Dataset (prompts dahil, ama style yok)
from promptdresser_style_dataset import PromptDresserStyleDataset
from promptdresser.models.unet import UNet2DConditionModel
from promptdresser.models.cloth_encoder import ClothEncoder
from promptdresser.models.mutual_self_attention import ReferenceAttentionControl
from promptdresser.utils import load_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pattern_lora")

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank=8, alpha=16):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(1, rank)
        device = original_linear.weight.device
        dtype = original_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(original_linear.in_features, rank, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(rank, original_linear.out_features, device=device, dtype=dtype))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_B)

    def forward(self, x, scale: float = 1.0, **kwargs):
        out = self.original_linear(x)
        if self.training or scale != 0:
            x_c = torch.clamp(x, -10, 10)
            out = out + (x_c @ self.lora_A @ self.lora_B) * self.scaling * scale
        return out

def replace_linear_with_lora(model: nn.Module, target_patterns, rank=8, alpha=16):
    compiled = [re.compile(p) for p in target_patterns]
    lora_layers = {}
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(p.search(name) for p in compiled):
            *parents, attr = name.split(".")
            parent = model
            for p in parents:
                parent = getattr(parent, p)
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(parent, attr, lora_layer)
            lora_layers[name] = lora_layer
    return lora_layers

def create_time_ids(batch_size, device, dtype):
    original_size = (1024, 768)
    crops_coords_top_left = (0, 0)
    target_size = (1024, 768)
    time_ids = torch.tensor([
        original_size[0], original_size[1],
        crops_coords_top_left[0], crops_coords_top_left[1],
        target_size[0], target_size[1]
    ], dtype=torch.float32).unsqueeze(0)
    return time_ids.repeat(batch_size, 1).to(device=device, dtype=dtype)

def parse_args():
    ap = argparse.ArgumentParser(description="Pattern-Only LoRA Training for PromptDresser (VRAM-optimized)")
    ap.add_argument("--config", type=str, required=True, help="./configs/VITONHD_lora.yaml")
    ap.add_argument("--data_dir", type=str, default="./DATA/zalando-hd-resized", help="Dataset klasörü")
    ap.add_argument("--output_dir", type=str, default="./checkpoints/pattern_lora_training", help="Checkpoint klasörü")
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    ap.add_argument("--lora_scale", type=float, default=1.0)
    return ap.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading base models…")
    init_model_path = "./pretrained_models/stable-diffusion-xl-1.0-inpainting-0.1"
    init_vae_path   = "./pretrained_models/sdxl-vae-fp16-fix"
    init_cloth_encoder_path = "./pretrained_models/stable-diffusion-xl-base-1.0"

    noise_scheduler = DDPMScheduler.from_pretrained(init_model_path, subfolder="scheduler")
    tokenizer   = CLIPTokenizer.from_pretrained(init_model_path, subfolder="tokenizer")
    text_enc    = CLIPTextModel.from_pretrained(init_model_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(init_model_path, subfolder="tokenizer_2")
    text_enc_2  = CLIPTextModelWithProjection.from_pretrained(init_model_path, subfolder="text_encoder_2")

    vae  = AutoencoderKL.from_pretrained(init_vae_path)
    unet = UNet2DConditionModel.from_pretrained(init_model_path, subfolder="unet")

    detach_cloth = bool(config.get("detach_cloth_encoder", True))
    cloth_encoder = None
    if not detach_cloth:
        cloth_encoder = ClothEncoder.from_pretrained(init_cloth_encoder_path, subfolder="unet")

    if "model" in config and "params" in config["model"]:
        base_unet_path = config["model"]["params"].get("base_model_path", None)
        if base_unet_path and os.path.exists(base_unet_path):
            unet.load_state_dict(load_file(base_unet_path))
            logger.info("Loaded PromptDresser UNet weights.")
        cloth_path = config["model"]["params"].get("cloth_encoder_path", None)
        if cloth_path and (cloth_encoder is not None):
            cloth_encoder.load_state_dict(load_file(cloth_path), strict=False)
            logger.info("Loaded cloth encoder weights.")

    for m in filter(None, [unet, vae, text_enc, text_enc_2, cloth_encoder]):
        m.to(device, dtype=weight_dtype)

    unet.to(memory_format=torch.channels_last)
    vae.to(memory_format=torch.channels_last)

    try:
        unet.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled.")
    except Exception:
        logger.info("Gradient checkpointing not available; continuing.")

    try:
        unet.set_use_memory_efficient_attention_xformers(True)
        logger.info("xFormers attention enabled.")
    except Exception:
        try:
            unet.enable_xformers_memory_efficient_attention()
            logger.info("xFormers attention enabled (alt method).")
        except Exception:
            logger.info("xFormers not found; using attention slicing.")
            try:
                unet.set_attention_slice("max")
            except Exception:
                pass

    try:
        vae.enable_slicing(); vae.enable_tiling()
        logger.info("VAE slicing/tiling enabled.")
    except Exception:
        logger.info("VAE slicing/tiling not available; continuing.")

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_enc.requires_grad_(False)
    text_enc_2.requires_grad_(False)
    if cloth_encoder is not None:
        cloth_encoder.requires_grad_(False)

    reference_control_writer = None
    reference_control_reader = None
    if not detach_cloth and cloth_encoder is not None:
        reference_control_writer = ReferenceAttentionControl(
            cloth_encoder, do_classifier_free_guidance=False, mode="write", fusion_blocks="midup",
            batch_size=1, is_train=True,
        )
        reference_control_reader = ReferenceAttentionControl(
            unet, do_classifier_free_guidance=False, mode="read", fusion_blocks="midup",
            batch_size=1, is_train=True,
        )

    lora_rank  = int(args.lora_rank)
    lora_alpha = int(args.lora_alpha)

    target_patterns = (
        config.get("lora", {}).get("target_modules") or [
            r"mid_block\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d\.to_(q|k|v|out\.0)$",
            r"up_blocks\.(2|3)\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d\.to_(q|k|v|out\.0)$",
        ]
    )

    lora_layers = replace_linear_with_lora(unet, target_patterns, rank=lora_rank, alpha=lora_alpha)
    lora_params = [p for layer in lora_layers.values() for p in (layer.lora_A, layer.lora_B)]
    if len(lora_params) == 0:
        examples = [n for n, m in unet.named_modules() if isinstance(m, nn.Linear)]
        logger.error(f"No LoRA layers injected. Example Linear names: {examples[:15]}")
        raise RuntimeError("LoRA injection failed: no target modules matched your patterns.")

    train_dataset = PromptDresserStyleDataset(
        data_dir=args.data_dir,
        mode="fine",
        image_size=args.image_size
    )

    if args.max_train_samples and args.max_train_samples > 0:
        if hasattr(train_dataset, "pairs"):
            train_dataset.pairs = train_dataset.pairs[:args.max_train_samples]
            logger.info(f"Limited dataset to {len(train_dataset.pairs)} samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            lora_params, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01, eps=1e-6
        )
        logger.info("Using AdamW 8-bit (bitsandbytes).")
    except Exception:
        optimizer = torch.optim.AdamW(
            lora_params, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01, eps=1e-6
        )
        logger.info("bitsandbytes not found; using torch AdamW.")

    scaler = torch.cuda.amp.GradScaler(enabled=(weight_dtype == torch.float16))

    global_step = 0
    logger.info("***** Start Pattern-Only LoRA Training *****")
    logger.info(f"Examples={len(train_dataset)} | Epochs={args.num_train_epochs} | BS={args.batch_size} | Accum={args.gradient_accumulation_steps}")

    total_steps = args.num_train_epochs * math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps))
    pbar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True)

    unet.train()

    for epoch in range(args.num_train_epochs):
        epoch_loss = 0.0
        valid_steps = 0
        for step, batch in enumerate(train_loader):
            person_img = batch["person"].to(device, dtype=weight_dtype, non_blocking=True)
            mask       = batch["mask"].to(device, dtype=weight_dtype, non_blocking=True)
            pose       = batch.get("pose", None)
            if torch.is_tensor(pose):
                pose = pose.to(device, dtype=weight_dtype, non_blocking=True)
            cloth      = batch["cloth"].to(device, dtype=weight_dtype, non_blocking=True)
            target_img = batch["target"].to(device, dtype=weight_dtype, non_blocking=True)

            bsz = person_img.shape[0]
            prompts = batch.get("prompt", [""] * bsz)

            with torch.no_grad(), torch.autocast("cuda", dtype=weight_dtype, enabled=(weight_dtype != torch.float32)):
                person_latents = vae.encode(person_img).latent_dist.sample() * vae.config.scaling_factor
                target_latents = vae.encode(target_img).latent_dist.sample() * vae.config.scaling_factor
                cloth_latents  = vae.encode(cloth).latent_dist.sample() * vae.config.scaling_factor
                mask_latents = torch.nn.functional.interpolate(mask, size=person_latents.shape[-2:], mode="nearest")

                text_in  = tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
                text_in2 = tokenizer_2(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
                text_emb     = text_enc(text_in.input_ids)[0]
                te2_out      = text_enc_2(text_in2.input_ids)
                text_emb2    = te2_out.last_hidden_state
                pooled_text  = te2_out.text_embeds
                encoder_hidden_states = torch.cat([text_emb, text_emb2], dim=-1)

                if not detach_cloth and cloth_encoder is not None:
                    time_ids_cloth = create_time_ids(bsz, device, weight_dtype)
                    added_cond_kwargs_cloth = {'text_embeds': pooled_text, 'time_ids': time_ids_cloth}
                    _ = cloth_encoder(
                        cloth_latents,
                        timestep=torch.zeros(bsz, device=device, dtype=torch.long),
                        encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs_cloth,
                        return_dict=False
                    )[0]

            with torch.autocast("cuda", dtype=weight_dtype, enabled=(weight_dtype != torch.float32)):
                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

                unet_input = torch.cat([noisy_latents, mask_latents, person_latents], dim=1)
                time_ids = create_time_ids(bsz, device, weight_dtype)
                added_cond_kwargs = {'text_embeds': pooled_text, 'time_ids': time_ids}
                model_pred = unet(
                    sample=unet_input,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=True
                ).sample

                loss = nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.detach().float().item()
            valid_steps += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                for layer in lora_layers.values():
                    if layer.lora_A.grad is not None and torch.isnan(layer.lora_A.grad).any():
                        layer.lora_A.grad.zero_()
                    if layer.lora_B.grad is not None and torch.isnan(layer.lora_B.grad).any():
                        layer.lora_B.grad.zero_()

                torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                pbar.update(1)

                if global_step % args.logging_steps == 0 and valid_steps > 0:
                    avg_loss = epoch_loss / valid_steps * args.gradient_accumulation_steps
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{args.learning_rate:.2e}'})
                    logger.info(f"Step {global_step}: loss={avg_loss:.4f}")

                if global_step % args.save_steps == 0:
                    ckpt_dir = Path(args.output_dir) / f"pattern-checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    lora_state = {}
                    for name, layer in lora_layers.items():
                        lora_state[f"{name}.lora_A"] = layer.lora_A.detach().cpu()
                        lora_state[f"{name}.lora_B"] = layer.lora_B.detach().cpu()
                    torch.save({
                        "lora_weights": lora_state,
                        "optimizer": optimizer.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "config": config,
                        "lora_rank": lora_rank,
                        "lora_alpha": lora_alpha
                    }, ckpt_dir / "pattern_checkpoint.pt")
                    logger.info(f"Saved checkpoint: {ckpt_dir}")

                # Temizlik & VRAM guard
                del person_latents, target_latents, cloth_latents, mask_latents, unet_input, model_pred, noise
                torch.cuda.empty_cache()

            # batch tensörlerini bırak
            del person_img, mask, cloth, target_img

        if valid_steps > 0:
            avg_epoch = epoch_loss / valid_steps * args.gradient_accumulation_steps
            logger.info(f"Epoch {epoch+1}/{args.num_train_epochs} done | avg_loss={avg_epoch:.4f}")

    pbar.close()

    # Final LoRA ağırlıkları
    final_dir = Path(args.output_dir) / "final_pattern_lora"
    final_dir.mkdir(parents=True, exist_ok=True)

    lora_state = {}
    for name, layer in lora_layers.items():
        lora_state[f"{name}.lora_A"] = layer.lora_A.detach().cpu()
        lora_state[f"{name}.lora_B"] = layer.lora_B.detach().cpu()

    final_ckpt = {
        "lora_weights": lora_state,
        "config": config,
        "training_samples": len(train_dataset),
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha
    }
    torch.save(final_ckpt, final_dir / "pattern_lora_weights.pt")

    with open(final_dir / "pattern_info.txt", "w") as f:
        f.write("PATTERN-ONLY PROMPTDRESSER LORA (VRAM-OPTIMIZED)\n")
        f.write("===============================================\n\n")
        f.write(f"Samples: {len(train_dataset)}\n")
        f.write(f"LoRA rank: {lora_rank} | alpha: {lora_alpha}\n")
        f.write(f"Detach cloth encoder: {detach_cloth}\n")

    logger.info(f"Training complete. Final LoRA saved to: {final_dir}")

if __name__ == "__main__":
    main()
