import os
import numpy as np
import torch
from models.nafnet import NAFNet
from denoise.tiling import process_tiled
from denoise.dng_writer import write_linear_dng


def test_fast_inference_tile_and_dng(tmp_path):
    out_dng = str(tmp_path / "test_mini.dng")
    device = torch.device("cpu")
    model = NAFNet(img_channel=5, out_channel=4, width=16)
    model.eval()

    # Synthetic 256x256 half-res packed raw patch (equivalent to 512x512 full mosaic)
    inp_hwc = np.random.uniform(0.0, 1.0, (256, 256, 5)).astype(np.float32)

    def process_tile_fn(tile_hwc: np.ndarray) -> np.ndarray:
        t_in = torch.from_numpy(np.transpose(tile_hwc, (2, 0, 1))).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(t_in)
        pred_np = pred.squeeze(0).cpu().float().numpy()
        return np.transpose(pred_np, (1, 2, 0))

    denoised_packed_hwc = process_tiled(inp_hwc, process_tile_fn, tile=128, overlap=16)
    assert denoised_packed_hwc.shape == (256, 256, 4)

    # 3-channel linear RGB
    rgb = np.stack([denoised_packed_hwc[..., 0],
                    (denoised_packed_hwc[..., 1] + denoised_packed_hwc[..., 2]) * 0.5,
                    denoised_packed_hwc[..., 3]], axis=-1)

    write_linear_dng(out_dng, rgb, camera_model="Canon EOS 6D")
    assert os.path.exists(out_dng)
    assert os.path.getsize(out_dng) > 10000
