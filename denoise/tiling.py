"""Overlapping tile processing with feathered blending (for large images)."""
import numpy as np


def _ramp(n: int, overlap: int) -> np.ndarray:
    w = np.ones(n, dtype=np.float32)
    if overlap > 0:
        r = np.linspace(0, 1, overlap + 2, dtype=np.float32)[1:-1]
        w[:overlap] = r
        w[-overlap:] = r[::-1]
    return w


def process_tiled(img: np.ndarray, fn, tile: int = 256, overlap: int = 32, out_channels: int = None) -> np.ndarray:
    """Apply fn to (H, W, C_in) tiles and blend into (H, W, C_out)."""
    h, w, c_in = img.shape
    out = None
    acc = np.zeros((h, w, 1), np.float32)
    step = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, step))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if ys[-1] + tile < h: ys.append(h - tile)
    if xs[-1] + tile < w: xs.append(w - tile)

    for y in ys:
        for x in xs:
            th, tw = min(tile, h), min(tile, w)
            t = fn(img[y:y + th, x:x + tw])

            if out is None:
                c_out = out_channels if out_channels is not None else t.shape[-1]
                out = np.zeros((h, w, c_out), np.float32)

            wt = np.outer(_ramp(th, overlap), _ramp(tw, overlap))[..., None]
            out[y:y + th, x:x + tw] += t * wt
            acc[y:y + th, x:x + tw] += wt

    return out / np.maximum(acc, 1e-8)
