"""PixCell + SyncTweedies: large histology images on a 2D grid of views.

PixCell (Stony Brook CVLab, 2025): PixArt-like DiT (~600M), trained on TCGA histology tiles,
SD3 VAE latent (16 channels), eps prediction with a linear DDPM schedule (betas 1e-4 -> 0.02),
conditioned on UNI-2h embeddings. Here the generation is unconditional (learned empty embedding).

PixCell is a diffusers model, not a ComfyUI MODEL, so it has its own loader / denoiser / sampling / decode
nodes. The views and the synchronisation are the same nodes as for SD 2 and FLUX.

Expected weights in ComfyUI/models/diffusers/:  PixCell-256/  PixCell-1024/  PixCell-SD3-VAE/
"""
import importlib.util
import os

import torch
from tqdm.auto import trange

import comfy.latent_formats
import comfy.model_management as mm
import comfy.utils
import folder_paths
import latent_preview

from .sync_core import Views, sync_sample

DIFF_DIR = os.path.join(folder_paths.models_dir, "diffusers")
VAE_SCALE, VAE_SHIFT = 1.5305, 0.0609          # SD3 VAE: z_dit = (z_vae - shift) * scale


def all_sigmas(T=1000):
    """The 1000 sigmas of PixCell's linear DDPM schedule, sigma_t = sqrt((1 - a_bar_t) / a_bar_t)."""
    b = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
    a_bar = torch.cumprod(1 - b, 0)
    return ((1 - a_bar) / a_bar).sqrt().float()


def schedule(steps):
    """`steps` evenly spaced timesteps of the 1000, then 0."""
    ts = torch.linspace(999, 0, steps).round().long()
    return torch.cat([all_sigmas()[ts], torch.zeros(1)])


def make_denoiser(pixcell):
    """x (sigma convention) -> x0_hat: one DiT pass, then Tweedie."""
    tr, dev, dt = pixcell["transformer"], pixcell["device"], pixcell["dtype"]
    sig = all_sigmas()

    def denoise(xb, s):
        b = xb.shape[0]
        z_t = (xb / (s * s + 1) ** 0.5).to(dev, dt)                          # back to the DDPM scale
        t = torch.full((b,), int((sig - s).abs().argmin()), dtype=torch.long, device=dev)
        enc = pixcell["uncond"].to(dev, dt).unsqueeze(0).expand(b, -1, -1)   # unconditional
        out = tr(z_t, encoder_hidden_states=enc, timestep=t, added_cond_kwargs={}, return_dict=False)[0]
        eps = out[:, :pixcell["latent_ch"]].float()                         # 32 channels = eps + learned variance
        return xb - s * eps.to(xb.device)                                    # Tweedie
    return denoise


@torch.no_grad()
def decode_tiled(vae, z, tile=64, overlap=16):
    """VAE decoding by latent tiles (64 -> 512 px), blended on the overlaps."""
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
            zt = z[..., y:y + th, x:x + tw].to(vae.device, torch.float32) / VAE_SCALE + VAE_SHIFT
            img = vae.decode(zt).sample.float().cpu()
            out[..., 8 * y:8 * (y + th), 8 * x:8 * (x + tw)] += img * w
            wsum[..., 8 * y:8 * (y + th), 8 * x:8 * (x + tw)] += w
    return ((out / wsum).clamp(-1, 1) * 0.5 + 0.5).permute(0, 2, 3, 1)      # (1, H, W, 3) ComfyUI IMAGE


class PixCellLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": (["PixCell-256", "PixCell-1024"],), "dtype": (["float16", "float32"],)}}

    RETURN_TYPES = ("PIXCELL",)
    FUNCTION = "load"
    CATEGORY = "sampling/synctweedies/pixcell"

    def load(self, model, dtype):
        from diffusers import AutoencoderKL
        device = mm.get_torch_device()
        dt = torch.float16 if dtype == "float16" else torch.float32
        mdir = os.path.join(DIFF_DIR, model)
        spec = importlib.util.spec_from_file_location(
            "pixcell_transformer_2d", os.path.join(mdir, "transformer", "pixcell_transformer_2d.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tr = mod.PixCellTransformer2DModel.from_pretrained(os.path.join(mdir, "transformer"), torch_dtype=dt).to(device).eval()
        vae = AutoencoderKL.from_pretrained(os.path.join(DIFF_DIR, "PixCell-SD3-VAE"), torch_dtype=torch.float32).to(device).eval()
        return ({"transformer": tr, "vae": vae, "uncond": tr.caption_projection.uncond_embedding.detach().clone(),
                 "dtype": dt, "device": device, "view_px": 8 * tr.config.sample_size,
                 "latent_ch": tr.config.in_channels},)


class PixCellDenoiser:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pixcell": ("PIXCELL",)}}

    RETURN_TYPES = ("PIXCELL_DENOISER",)
    RETURN_NAMES = ("denoiser",)
    FUNCTION = "make"
    CATEGORY = "sampling/synctweedies/pixcell"

    def make(self, pixcell):
        return ({"pixcell": pixcell, "fn": make_denoiser(pixcell)},)


class PixCellSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"steps": ("INT", {"default": 50, "min": 1, "max": 1000})}}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get"
    CATEGORY = "sampling/synctweedies/pixcell"

    def get(self, steps):
        return (schedule(steps),)


class PixCellEmptyLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"width": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 64}),
                             "height": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 64})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "make"
    CATEGORY = "sampling/synctweedies/pixcell"

    def make(self, width, height):
        return ({"samples": torch.zeros(1, 16, height // 8, width // 8)},)


class PixCellSampling:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "noise": ("NOISE",),
            "denoiser": ("PIXCELL_DENOISER",),
            "sampler": ("SAMPLER", {"tooltip": "Output of the SyncTweedies Sampler node (views + mode)."}),
            "sigmas": ("SIGMAS",),
            "latent_image": ("LATENT",),
            "trajectory_every": ("INT", {"default": 10, "min": 0, "max": 1000,
                                         "tooltip": "Keep x0_hat every N steps in the 'trajectory' output (0 = off)."}),
        }}

    RETURN_TYPES = ("PIXCELL_LATENT", "PIXCELL_LATENT")
    RETURN_NAMES = ("latent", "trajectory")
    FUNCTION = "sample"
    CATEGORY = "sampling/synctweedies/pixcell"

    @torch.no_grad()
    def sample(self, noise, denoiser, sampler, sigmas, latent_image, trajectory_every):
        if not hasattr(sampler, "sync_views"):
            raise ValueError("Connect the output of the SyncTweedies Sampler node.")
        pixcell, cfg, mode = denoiser["pixcell"], sampler.sync_views, sampler.sync_mode
        dev, steps = pixcell["device"], len(sigmas) - 1
        x = (noise.generate_noise(latent_image) * float(sigmas[0])).to(dev)   # x_T = sigma_max * eps
        v = pixcell["view_px"] // 8                                            # view size is fixed by the model
        views = Views(x, v, v, max(1, cfg["stride"] // 8), cfg["weighting"])
        print(f"[PixCell] {mode} | {len(views.pos)} views of {pixcell['view_px']}px | {steps} steps")

        pbar = comfy.utils.ProgressBar(steps)
        previewer = latent_preview.get_previewer(dev, comfy.latent_formats.SD3())  # PixCell latent = SD3 latent
        traj = []

        def cb(i, x0_canvas, x_canvas, s):
            preview = previewer.decode_latent_to_preview_image("JPEG", x0_canvas) if previewer else None
            pbar.update_absolute(i + 1, steps, preview)
            if trajectory_every > 0 and ((i + 1) % trajectory_every == 0 or i == steps - 1):
                traj.append(x0_canvas.cpu())

        z0 = sync_sample(denoiser["fn"], x, [float(s) for s in sigmas], views, mode, lambda s: 1.0,
                         lambda L: torch.cat(L, 0), batch=cfg["views_per_batch"], callback=cb,
                         iterator=lambda k: trange(k, disable=not comfy.utils.PROGRESS_BAR_ENABLED)).cpu()
        return ({"samples": z0}, {"samples": torch.cat(traj, 0) if traj else z0})


class PixCellDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pixcell": ("PIXCELL",), "latent": ("PIXCELL_LATENT",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "sampling/synctweedies/pixcell"

    def decode(self, pixcell, latent):
        z = latent["samples"]
        return (torch.cat([decode_tiled(pixcell["vae"], z[i:i + 1]) for i in range(z.shape[0])]),)


NODE_CLASS_MAPPINGS = {"PixCellLoader": PixCellLoader, "PixCellDenoiser": PixCellDenoiser,
                       "PixCellSchedule": PixCellSchedule, "PixCellEmptyLatent": PixCellEmptyLatent,
                       "PixCellSampling": PixCellSampling, "PixCellDecode": PixCellDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"PixCellLoader": "PixCell Loader", "PixCellDenoiser": "PixCell Denoiser (DiT + Tweedie)",
                              "PixCellSchedule": "PixCell Schedule", "PixCellEmptyLatent": "PixCell Empty Latent",
                              "PixCellSampling": "PixCell Sampling (synchronised)", "PixCellDecode": "PixCell VAE Decode"}
