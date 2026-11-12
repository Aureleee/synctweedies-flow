"""ComfyUI nodes: SyncTweedies as a regular SAMPLER.

    CheckpointLoader -> CLIPTextEncode +/- -> CFGGuider ---------------------\
    RandomNoise --------------------------------------------------------------- SamplerCustomAdvanced -> VAEDecodeTiled
    BasicScheduler -> SIGMAS -------------------------------------------------/
    EmptyLatentImage (e.g. 3072 x 512) ---------------------------------------/
    SyncTweedies Views -> SyncTweedies Sampler (mode) -> SAMPLER ------------/

Works with eps models (SD 2, SDXL) and flow models (FLUX): the flow case is detected from model_sampling.
Progress bar and live preview are the standard ComfyUI ones.
"""
import torch
from tqdm.auto import trange

import comfy.model_management
import comfy.model_sampling
import comfy.samplers

from .sync_core import Views, sync_sample

MODES = ["synctweedies", "multidiffusion", "independent"]


def make_sync_sampler(views_cfg, mode):
    """Returns a k-diffusion style sampler function: (model, x, sigmas, extra_args, callback, disable) -> x0."""

    @torch.no_grad()
    def sampler_fn(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        base = model.inner_model.inner_model                      # BaseModel: network + model_sampling + latent_format
        flow = isinstance(base.model_sampling, comfy.model_sampling.CONST)
        a = (lambda s: 1.0 - s) if flow else (lambda s: 1.0)      # x = a(sigma) x0 + sigma eps
        r = getattr(base.latent_format, "spacial_downscale_ratio", 8)

        def denoise(xb, s):                                       # CFG + Tweedie are done by the guider
            return model(xb, xb.new_ones([xb.shape[0]]) * s, **extra_args)

        v = max(1, views_cfg["view_size"] // r)
        views = Views(x, v, v, max(1, views_cfg["stride"] // r), views_cfg["weighting"])
        print(f"[SyncTweedies] {mode} | {len(views.pos)} views of {views.h * r}x{views.w * r}px | "
              f"canvas {x.shape[-2] * r}x{x.shape[-1] * r}px | {len(sigmas) - 1} steps | flow={flow}")

        def cb(i, x0_canvas, x_canvas, s):
            if callback is not None:
                callback({"x": x_canvas, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": x0_canvas})

        return sync_sample(denoise, x, sigmas, views, mode, a, lambda L: torch.cat(L, 0),
                           batch=views_cfg["views_per_batch"], callback=cb,
                           iterator=lambda n: trange(n, disable=disable))
    return sampler_fn


class SyncTweediesViews:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "view_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8,
                                  "tooltip": "View size in pixels (native resolution of the model)."}),
            "stride": ("INT", {"default": 128, "min": 8, "max": 2048, "step": 8,
                               "tooltip": "Offset between two views in pixels. 128 = paper setup (21 views for 512x3072)."}),
            "weighting": (["uniform", "gaussian"],),
            "views_per_batch": ("INT", {"default": 4, "min": 1, "max": 256,
                                        "tooltip": "Views sent to the model together. Lower it if memory is tight."}),
        }}

    RETURN_TYPES = ("SYNC_VIEWS",)
    RETURN_NAMES = ("views",)
    FUNCTION = "make"
    CATEGORY = "sampling/synctweedies"

    def make(self, view_size, stride, weighting, views_per_batch):
        return ({"view_size": view_size, "stride": stride, "weighting": weighting, "views_per_batch": views_per_batch},)


class SyncTweediesSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"views": ("SYNC_VIEWS",), "mode": (MODES,)}}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/synctweedies"

    def get_sampler(self, views, mode):
        sampler = comfy.samplers.KSAMPLER(make_sync_sampler(views, mode))
        sampler.sync_views, sampler.sync_mode = views, mode       # read by PixCell Sampling (model outside ComfyUI)
        return (sampler,)


NODE_CLASS_MAPPINGS = {"SyncTweediesViews": SyncTweediesViews, "SyncTweediesSampler": SyncTweediesSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"SyncTweediesViews": "SyncTweedies Views",
                              "SyncTweediesSampler": "SyncTweedies Sampler"}
