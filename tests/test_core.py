import numpy as np
from denoise.cfa import pack_bayer, unpack_bayer
from denoise.tiling import process_tiled
from denoise.metrics import psnr


def test_cfa_roundtrip():
    m = np.random.rand(64, 48).astype(np.float32)
    assert np.array_equal(unpack_bayer(pack_bayer(m)), m)


def test_tiling_identity_no_seams():
    img = np.random.rand(300, 410, 3).astype(np.float32)
    out = process_tiled(img, lambda t: t, tile=128, overlap=16)
    assert np.allclose(out, img, atol=1e-5)


def test_psnr_identical_is_high():
    a = np.random.rand(32, 32).astype(np.float32)
    assert psnr(a, a + 1e-4) > 60
