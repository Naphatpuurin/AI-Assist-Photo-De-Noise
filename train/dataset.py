"""PyTorch Dataset for RAW denoising with optional memory cache."""
import glob
import os
import random
import numpy as np
import rawpy
import torch
from torch.utils.data import Dataset
from denoise.cfa import pack_bayer
from denoise.noise_model import SensorNoiseModel


class RawDenoiseDataset(Dataset):
    def __init__(
        self,
        raw_dir: str = "data/RAW",
        cache_file: str = "data/train_crops.npy",
        crop_size: int = 128,
        target_isos: list = None,
        profile_path: str = "calibration/canon_6d_profile.json",
        max_samples: int = None,
    ):
        self.crop_size = crop_size
        self.noise_model = SensorNoiseModel(profile_path)
        self.target_isos = target_isos or [1600, 3200, 6400, 12800, 25600]
        self.cached_crops = None

        if cache_file and os.path.exists(cache_file):
            print(f"Loading memory cache from {cache_file}...")
            self.cached_crops = np.load(cache_file)
            print(f"  Loaded {len(self.cached_crops)} patches into RAM ({self.cached_crops.nbytes / 1e6:.1f} MB)")
        else:
            self.files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.CR2"), recursive=True))
            if max_samples:
                self.files = self.files[:max_samples]
            self.black = self.noise_model.black_level
            self.range = self.noise_model.active_range

    def __len__(self):
        if self.cached_crops is not None:
            return len(self.cached_crops)
        return len(self.files)

    def __getitem__(self, idx):
        if self.cached_crops is not None:
            clean_patch = self.cached_crops[idx].copy()
        else:
            file_path = self.files[idx]
            with rawpy.imread(file_path) as r:
                raw = r.raw_image
                top = r.sizes.top_margin
                left = r.sizes.left_margin
                h = (r.sizes.height // 2) * 2
                w = (r.sizes.width // 2) * 2
                active = raw[top:top + h, left:left + w].astype(np.float32)

            norm = np.clip((active - self.black) / self.range, 0.0, 1.0)
            packed = pack_bayer(norm)
            _, ph, pw = packed.shape

            cs = self.crop_size
            if ph > cs and pw > cs:
                y = random.randint(0, ph - cs)
                x = random.randint(0, pw - cs)
                clean_patch = packed[:, y:y + cs, x:x + cs]
            else:
                clean_patch = packed

        # Dynamic data augmentation: random flips and 90-deg rotations
        if random.random() < 0.5:
            clean_patch = np.flip(clean_patch, axis=1)
        if random.random() < 0.5:
            clean_patch = np.flip(clean_patch, axis=2)
        rot = random.choice([0, 1, 2, 3])
        if rot > 0:
            clean_patch = np.rot90(clean_patch, rot, axes=(1, 2))

        clean_patch = np.ascontiguousarray(clean_patch)

        # Dynamic noise synthesis
        target_iso = random.choice(self.target_isos)
        noisy_patch = self.noise_model.add_noise(clean_patch, target_iso=target_iso)

        # Condition channel (normalized log ISO)
        iso_cond = np.full(
            (1, clean_patch.shape[1], clean_patch.shape[2]),
            np.log2(target_iso / 100.0) / 10.0,
            dtype=np.float32,
        )

        noisy_with_cond = np.concatenate([noisy_patch, iso_cond], axis=0)

        return (
            torch.from_numpy(noisy_with_cond),
            torch.from_numpy(clean_patch),
            torch.tensor(target_iso, dtype=torch.int32),
        )
