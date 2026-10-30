"""Pre-extract diverse RAW patches into a compact binary cache for ultra-fast GPU training."""
import glob
import os
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import rawpy
from denoise.cfa import pack_bayer


def prepare_cache(
    raw_dir: str = "data/RAW",
    out_cache: str = "data/train_crops.npy",
    num_images: int = 120,
    crops_per_image: int = 10,
    crop_size: int = 128,  # 128 half-res = 256 full mosaic
):
    files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.CR2"), recursive=True))
    if not files:
        print(f"No .CR2 files found in {raw_dir}")
        return

    random.seed(42)
    selected = random.sample(files, min(num_images, len(files)))
    total_crops = len(selected) * crops_per_image

    print(f"Extracting {total_crops} patches ({crop_size}x{crop_size}) from {len(selected)} RAW files...")
    t0 = time.time()

    crops = np.zeros((total_crops, 4, crop_size, crop_size), dtype=np.float32)
    idx = 0

    for i, f in enumerate(selected):
        try:
            with rawpy.imread(f) as r:
                raw = r.raw_image
                top = r.sizes.top_margin
                left = r.sizes.left_margin
                h = (r.sizes.height // 2) * 2
                w = (r.sizes.width // 2) * 2
                active = raw[top:top + h, left:left + w].astype(np.float32)
                black = float(r.black_level_per_channel[0])
                active_range = float(r.white_level) - black

            norm = np.clip((active - black) / active_range, 0.0, 1.0)
            packed = pack_bayer(norm)
            _, ph, pw = packed.shape

            for _ in range(crops_per_image):
                if ph > crop_size and pw > crop_size:
                    y = random.randint(0, ph - crop_size)
                    x = random.randint(0, pw - crop_size)
                    crops[idx] = packed[:, y:y + crop_size, x:x + crop_size]
                else:
                    crops[idx] = packed[:, :crop_size, :crop_size]
                idx += 1

            if (i + 1) % 20 == 0 or (i + 1) == len(selected):
                print(f"  Processed [{i + 1}/{len(selected)}] files ({idx}/{total_crops} patches) - {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"Error on {f}: {e}")

    crops = crops[:idx]
    os.makedirs(os.path.dirname(out_cache), exist_ok=True)
    np.save(out_cache, crops)

    mb = os.path.getsize(out_cache) / (1024 * 1024)
    print(f"\nCache successfully saved to {out_cache} ({crops.shape[0]} patches, {mb:.1f} MB) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    prepare_cache()
