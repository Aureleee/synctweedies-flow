"""SyncTweedies / MultiDiffusion core. Pure torch, no ComfyUI dependency.

Noise parametrisation (k-diffusion / ComfyUI convention):
    eps models  (SD 1.5, SD 2, SDXL, PixCell) : x = x0 + sigma * eps             -> a(sigma) = 1
    flow models (FLUX, SD3, Z-Image)          : x = (1 - sigma) x0 + sigma * eps -> a(sigma) = 1 - sigma

`denoise(x, sigma)` returns x0_hat (Tweedie). One deterministic step from sigma to sigma_next is
    eps_hat = (x - a(sigma) x0_hat) / sigma
    x_next  = a(sigma_next) x0_hat + sigma_next * eps_hat
which is DDIM (eta = 0) for eps models and an Euler step for flow models.

Modes:
    synctweedies   : average x0_hat over the views, recompute each view's noise from the synced x0_hat, then step
    multidiffusion : each view does its own step, then x_next is averaged over the views
    independent    : no synchronisation (baseline, visible seams)
"""
import math


def positions(length, window, stride):
    if length <= window:
        return [0]
    p = list(range(0, length - window + 1, stride))
    if p[-1] != length - window:
        p.append(length - window)
    return p


def weight_window(h, w, kind, like):
    """(1, 1, h, w) weight map: uniform, or gaussian (the centre of a view counts more than its border)."""
    win = like[..., :h, :w][:1, :1] * 0 + 1
    if kind == "gaussian":
        for i in range(h):
            for j in range(w):
                dy = (i - (h - 1) / 2) / (h / 2)
                dx = (j - (w - 1) / 2) / (w / 2)
                win[..., i, j] = math.exp(-0.5 * (dy * dy + dx * dx) / 0.25)
    return win


class Views:
    """Overlapping crops of a canvas (canonical space) and their weighted average back onto it."""

    def __init__(self, canvas_like, h, w, stride, weighting="uniform"):
        _, _, self.H, self.W = canvas_like.shape
        self.h, self.w = min(h, self.H), min(w, self.W)
        self.pos = [(y, x) for y in positions(self.H, self.h, stride) for x in positions(self.W, self.w, stride)]
        self.win = weight_window(self.h, self.w, weighting, canvas_like)
        self.count = canvas_like[:1, :1] * 0
        for y, x in self.pos:
            self.count[..., y:y + self.h, x:x + self.w] += self.win

    def crop(self, canvas):
        return [canvas[..., y:y + self.h, x:x + self.w] for y, x in self.pos]

    def aggregate(self, tiles, canvas_like):
        acc = canvas_like * 0
        for (y, x), t in zip(self.pos, tiles):
            acc[..., y:y + self.h, x:x + self.w] += t * self.win
        return acc / self.count


def sync_sample(denoise, x, sigmas, views, mode, a, cat, batch=4, callback=None, iterator=range):
    """x: noised canvas (1, C, H, W). sigmas: decreasing, last = 0.
    cat: concatenates a list of tiles into a batch. iterator: range, or tqdm.trange for a progress bar."""
    xs = [t.clone() for t in views.crop(x)]
    n = len(xs)
    for i in iterator(len(sigmas) - 1):
        s, sn = float(sigmas[i]), float(sigmas[i + 1])

        den = []
        for k in range(0, n, batch):
            chunk = xs[k:k + batch]
            out = denoise(cat(chunk), s)
            den += [out[j:j + 1] for j in range(len(chunk))]
        eps = [(xi - a(s) * di) / s for xi, di in zip(xs, den)]
        x0_canvas = views.aggregate(den, x)

        if mode == "synctweedies":
            # recompute each view's noise from the synced x0_hat.
            # keeping the view's own eps_hat makes the views drift apart on the overlaps (patchwork).
            x0_sync = views.crop(x0_canvas)
            xs = [a(sn) * x0 + sn * (xi - a(s) * x0) / s for x0, xi in zip(x0_sync, xs)]
        elif mode == "multidiffusion":
            x_next = views.aggregate([a(sn) * d + sn * e for d, e in zip(den, eps)], x)
            xs = views.crop(x_next)
        else:
            xs = [a(sn) * d + sn * e for d, e in zip(den, eps)]

        if callback is not None:
            callback(i, x0_canvas, views.aggregate(xs, x), s)
    return views.aggregate(xs, x)
