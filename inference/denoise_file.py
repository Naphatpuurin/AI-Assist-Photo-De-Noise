"""Inference pipeline: Denoise Canon 6D RAW to Linear DNG using trained NAFNet."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import rawpy
import torch
from PIL import Image
from denoise.cfa import pack_bayer, unpack_bayer
from denoise.dng_writer import write_linear_dng
from denoise.tiling import process_tiled
from models.nafnet import NAFNet


def extract_iso(path: str) -> int:
    try:
        ex = Image.open(path).getexif()
        return int(ex.get_ifd(0x8769).get(0x8827, 800))
    except Exception:
        return 800


def denoise_cr2_to_dng(
    cr2_path: str,
    out_dng_path: str,
    model: torch.nn.Module,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 64,
):
    start_total = time.time()
    iso = extract_iso(cr2_path)
    print(f"Processing: {cr2_path} (Detected ISO: {iso})")

    # Read RAW with rawpy
    t0 = time.time()
    with rawpy.imread(cr2_path) as r:
        raw = r.raw_image
        top = r.sizes.top_margin
        left = r.sizes.left_margin
        h = (r.sizes.height // 2) * 2
        w = (r.sizes.width // 2) * 2
        active = raw[top:top + h, left:left + w].astype(np.float32)
        black = float(r.black_level_per_channel[0])
        white = float(r.white_level)
        # Extract camera color matrix and white balance
        color_matrix = r.color_matrix[:3, :3].flatten()
        daylight_wb = r.daylight_whitebalance
        wb_neutral = (1.0 / daylight_wb[0], 1.0 / daylight_wb[1], 1.0 / daylight_wb[2])

    read_time = time.time() - t0
    active_range = white - black

    # Normalize to [0, 1]
    norm = np.clip((active - black) / active_range, 0.0, 1.0)
    # Pack Bayer RGGB to (4, H/2, W/2)
    packed = pack_bayer(norm)
    c, ph, pw = packed.shape

    # Prepare input channels: 4 CFA + 1 ISO condition
    iso_cond = np.full((1, ph, pw), np.log2(max(iso, 100) / 100.0) / 10.0, dtype=np.float32)
    inp_np = np.concatenate([packed, iso_cond], axis=0)  # (5, ph, pw)
    inp_hwc = np.transpose(inp_np, (1, 2, 0))  # (ph, pw, 5)

    print(f"  RAW decode & packing: {read_time:.2f}s (Mosaic: {ph * 2}x{pw * 2}, Half-res: {ph}x{pw})")

    # Define tile processing function
    model.eval()

    def process_tile_fn(tile_hwc: np.ndarray) -> np.ndarray:
        # tile_hwc: (th, tw, 5) -> torch (1, 5, th, tw)
        t_in = torch.from_numpy(np.transpose(tile_hwc, (2, 0, 1))).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                pred = model(t_in)
        pred_np = pred.squeeze(0).cpu().float().numpy()  # (4, th, tw)
        return np.transpose(pred_np, (1, 2, 0))  # (th, tw, 4)

    # Run tiled inference with feathered blending
    t_infer = time.time()
    denoised_packed_hwc = process_tiled(inp_hwc, process_tile_fn, tile=tile_size, overlap=overlap)
    infer_time = time.time() - t_infer
    print(f"  NAFNet Tiled Denoise: {infer_time:.2f}s")

    # Transpose back to (4, ph, pw) and clip
    denoised_packed = np.clip(np.transpose(denoised_packed_hwc, (2, 0, 1)), 0.0, 1.0)

    # Simple fast linear demosaic (or unpack to Bayer)
    # For Linear DNG: R = ch0, G = (ch1 + ch2)/2, B = ch3 upsampled to full resolution
    # or bilinear interpolate channels
    import scipy.ndimage
    r_ch = scipy.ndimage.zoom(denoised_packed[0], 2.0, order=1)
    g_ch = scipy.ndimage.zoom((denoised_packed[1] + denoised_packed[2]) * 0.5, 2.0, order=1)
    b_ch = scipy.ndimage.zoom(denoised_packed[3], 2.0, order=1)
    rgb_linear = np.stack([r_ch, g_ch, b_ch], axis=-1)

    # Write Linear DNG
    t_dng = time.time()
    write_linear_dng(
        out_dng_path,
        rgb_linear,
        camera_model="Canon EOS 6D",
        as_shot_neutral=wb_neutral,
    )
    dng_time = time.time() - t_dng
    total_time = time.time() - start_total
    print(f"  Linear DNG write:     {dng_time:.2f}s")
    print(f"==> Total Time: {total_time:.2f}s (Inference alone: {infer_time:.2f}s)")
    print(f"Saved: {out_dng_path}")


def main():
    parser = argparse.ArgumentParser(description="Denoise CR2 RAW file to Linear DNG")
    parser.add_argument("input", type=str, help="Input .CR2 file")
    parser.add_argument("output", type=str, help="Output .DNG file")
    parser.add_argument("--weights", type=str, default=None, help="Trained weights path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tile", type=int, default=512, help="Tile size")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = NAFNet(img_channel=5, out_channel=4, width=32).to(device)

    if args.weights and os.path.exists(args.weights):
        state = torch.load(args.weights, map_location=device)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        print(f"Loaded weights from {args.weights}")
    else:
        print("Running with initialized model (benchmark/dry-run mode)")

    denoise_cr2_to_dng(args.input, args.output, model, device, tile_size=args.tile)


if __name__ == "__main__":
    main()
