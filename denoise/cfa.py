"""CFA packing helpers: Bayer mosaic <-> 4-channel half-resolution tensor."""
import numpy as np


def pack_bayer(mosaic: np.ndarray) -> np.ndarray:
    """(H, W) RGGB mosaic -> (4, H/2, W/2) channels [R, G1, G2, B]."""
    h, w = mosaic.shape
    if h % 2 or w % 2:
        raise ValueError("mosaic dimensions must be even")
    return np.stack([mosaic[0::2, 0::2], mosaic[0::2, 1::2],
                     mosaic[1::2, 0::2], mosaic[1::2, 1::2]])


def unpack_bayer(packed: np.ndarray) -> np.ndarray:
    """Inverse of pack_bayer."""
    _, h, w = packed.shape
    out = np.empty((h * 2, w * 2), dtype=packed.dtype)
    out[0::2, 0::2], out[0::2, 1::2] = packed[0], packed[1]
    out[1::2, 0::2], out[1::2, 1::2] = packed[2], packed[3]
    return out
