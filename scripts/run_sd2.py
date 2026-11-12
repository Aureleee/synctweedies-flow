"""SD 2 base + SyncTweedies, without the ComfyUI graph: every step of the diffusion is written out here.

Uses ComfyUI only to load the weights (UNet, OpenCLIP, VAE). Schedule, CFG, Tweedie, DDIM step and
synchronisation are all explicit.

    python scripts/run_sd2.py --comfyui /path/to/ComfyUI --mode synctweedies
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from sync_core import Views, sync_sample  # noqa: E402


def load(ckpt_name):
    import comfy.sd
    import comfy.model_management as mm
    import folder_paths
    out = comfy.sd.load_checkpoint_guess_config(folder_paths.get_full_path("checkpoints", ckpt_name),
                                                output_vae=True, output_clip=True)
    mm.load_models_gpu([out[0]])
    return out[0], out[1], out[2]


def encode(clip, text):
    return clip.encode_from_tokens(clip.tokenize(text), return_pooled=True)[0]     # (1, 77, 1024)


def ddim_sigmas(model_sampling, steps):
    # ComfyUI convention: x = z_t / sqrt(a_bar) = x0 + sigma eps, sigma = sqrt((1 - a_bar) / a_bar)
    s = model_sampling.sigmas
    ss = max(len(s) // steps, 1)
    return torch.tensor([float(s[x]) for x in range(1, len(s), ss)][::-1] + [0.0])


class Denoiser:
    """x -> eps_hat (two UNet passes, CFG) -> x0_hat (Tweedie)."""

    def __init__(self, model, cond, uncond, cfg):
        import comfy.model_management as mm
        self.unet = model.model.diffusion_model
        self.ms = model.get_model_object("model_sampling")
        self.dtype = model.model.get_dtype_inference()
        self.device = mm.get_torch_device()
        self.cond, self.uncond, self.cfg = cond, uncond, cfg

    @torch.no_grad()
    def __call__(self, x, sigma):
        b = x.shape[0]
        z_t = x / (sigma ** 2 + 1) ** 0.5                                   # network input: z_t = sqrt(a_bar) x
        t = self.ms.timestep(torch.full((b,), sigma)).to(self.device).float()
        zin = torch.cat([z_t, z_t]).to(self.device, self.dtype)
        ctx = torch.cat([self.cond.repeat(b, 1, 1), self.uncond.repeat(b, 1, 1)]).to(self.device, self.dtype)
        eps = self.unet(zin, torch.cat([t, t]), context=ctx).float()
        eps_c, eps_u = eps[:b], eps[b:]
        eps_hat = eps_u + self.cfg * (eps_c - eps_u)                        # classifier-free guidance
        return x - sigma * eps_hat.to(x.device)                             # Tweedie


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comfyui", required=True, help="path to the ComfyUI folder")
    p.add_argument("--ckpt", default="512-base-ema.safetensors")
    p.add_argument("--prompt", default="A photo of a mountain range at twilight")
    p.add_argument("--width", type=int, default=3072)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--view", type=int, default=512)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", default="synctweedies", choices=["synctweedies", "multidiffusion", "independent"])
    p.add_argument("--out", default="sd2_panorama.png")
    a = p.parse_args()

    sys.path.insert(0, a.comfyui)
    os.chdir(a.comfyui)
    import comfy.model_management as mm

    model, clip, vae = load(a.ckpt)
    cond, uncond = encode(clip, a.prompt), encode(clip, "")
    sigmas = ddim_sigmas(model.get_model_object("model_sampling"), a.steps)

    g = torch.Generator().manual_seed(a.seed)
    x = (torch.randn(1, 4, a.height // 8, a.width // 8, generator=g) * sigmas[0]).to(mm.get_torch_device())
    views = Views(x, a.view // 8, a.view // 8, a.stride // 8)
    print(f"{a.mode} | {len(views.pos)} views | {a.steps} steps | cfg {a.cfg}")

    z0 = sync_sample(Denoiser(model, cond, uncond, a.cfg), x, sigmas, views, a.mode,
                     lambda s: 1.0, lambda L: torch.cat(L, 0), batch=4)

    img = vae.decode_tiled(z0.cpu(), tile_x=64, tile_y=64, overlap=8)[0]
    Image.fromarray((255 * img.clamp(0, 1).numpy()).astype(np.uint8)).save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
