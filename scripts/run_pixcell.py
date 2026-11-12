"""PixCell + SyncTweedies on a 2D grid, without ComfyUI (torch + diffusers only).

    python scripts/run_pixcell.py --models /path/to/ComfyUI/models/diffusers --mode synctweedies
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from sync_core import Views, sync_sample  # noqa: E402

VAE_SCALE, VAE_SHIFT = 1.5305, 0.0609


def all_sigmas(T=1000):
    b = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)       # PixCell: linear DDPM betas
    a_bar = torch.cumprod(1 - b, 0)
    return ((1 - a_bar) / a_bar).sqrt().float()                  # sigma_t = sqrt((1 - a_bar) / a_bar)


def load(models, name, device, dtype):
    from diffusers import AutoencoderKL
    path = os.path.join(models, name, "transformer", "pixcell_transformer_2d.py")
    spec = importlib.util.spec_from_file_location("pixcell_transformer_2d", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tr = mod.PixCellTransformer2DModel.from_pretrained(os.path.join(models, name, "transformer"),
                                                       torch_dtype=dtype).to(device).eval()
    vae = AutoencoderKL.from_pretrained(os.path.join(models, "PixCell-SD3-VAE"),
                                        torch_dtype=torch.float32).to(device).eval()
    return tr, vae, tr.caption_projection.uncond_embedding.detach().clone()


def make_denoiser(tr, uncond, device, dtype):
    sig = all_sigmas()

    @torch.no_grad()
    def denoise(x, sigma):
        b = x.shape[0]
        z_t = (x / (sigma ** 2 + 1) ** 0.5).to(device, dtype)
        t = torch.full((b,), int((sig - sigma).abs().argmin()), dtype=torch.long, device=device)
        enc = uncond.to(device, dtype).unsqueeze(0).expand(b, -1, -1)
        out = tr(z_t, encoder_hidden_states=enc, timestep=t, added_cond_kwargs={}, return_dict=False)[0]
        return x - sigma * out[:, :16].float().to(x.device)          # Tweedie, eps = first 16 channels
    return denoise


@torch.no_grad()
def decode(vae, z, tile=64, overlap=16):
    _, _, H, W = z.shape
    out, wsum = torch.zeros(1, 3, 8 * H, 8 * W), torch.zeros(1, 1, 8 * H, 8 * W)
    step = tile - overlap
    ys = sorted(set(list(range(0, max(H - tile, 0) + 1, step)) + [max(H - tile, 0)]))
    xs = sorted(set(list(range(0, max(W - tile, 0) + 1, step)) + [max(W - tile, 0)]))
    th, tw = min(tile, H), min(tile, W)
    ramp = lambda n: torch.clamp(torch.minimum(torch.arange(n) + 1, n - torch.arange(n)).float() / (4 * overlap), 0.05, 1)
    w = (ramp(8 * th)[:, None] * ramp(8 * tw)[None, :])[None, None]
    for y in ys:
        for x in xs:
            img = vae.decode(z[..., y:y + th, x:x + tw].to(vae.device) / VAE_SCALE + VAE_SHIFT).sample.float().cpu()
            out[..., 8 * y:8 * (y + th), 8 * x:8 * (x + tw)] += img * w
            wsum[..., 8 * y:8 * (y + th), 8 * x:8 * (x + tw)] += w
    img = ((out / wsum).clamp(-1, 1) * 0.5 + 0.5)[0].permute(1, 2, 0).numpy()
    return Image.fromarray((255 * img).astype(np.uint8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True, help="folder containing PixCell-256/ and PixCell-SD3-VAE/")
    p.add_argument("--model", default="PixCell-256", choices=["PixCell-256", "PixCell-1024"])
    p.add_argument("--size", type=int, default=2048)
    p.add_argument("--stride", type=int, default=192)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--mode", default="synctweedies", choices=["synctweedies", "multidiffusion", "independent"])
    p.add_argument("--out", default="pixcell_grid.png")
    a = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16
    tr, vae, uncond = load(a.models, a.model, device, dtype)

    ts = torch.linspace(999, 0, a.steps).round().long()
    sigmas = torch.cat([all_sigmas()[ts], torch.zeros(1)])
    g = torch.Generator().manual_seed(a.seed)
    x = (torch.randn(1, 16, a.size // 8, a.size // 8, generator=g) * sigmas[0]).to(device)

    v = tr.config.sample_size                                    # view size in latent (32 or 128)
    views = Views(x, v, v, a.stride // 8)
    print(f"{a.mode} | {len(views.pos)} views of {8 * v}px | {a.steps} steps")
    z0 = sync_sample(make_denoiser(tr, uncond, device, dtype), x, [float(s) for s in sigmas], views, a.mode,
                     lambda s: 1.0, lambda L: torch.cat(L, 0), batch=a.batch)
    decode(vae, z0.cpu()).save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
