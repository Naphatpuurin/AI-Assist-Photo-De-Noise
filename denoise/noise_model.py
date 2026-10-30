"""Physics-based noise synthesizer calibrated for Canon EOS 6D sensor."""
import json
import os
import numpy as np


class SensorNoiseModel:
    def __init__(self, profile_path: str = "calibration/canon_6d_profile.json"):
        with open(profile_path, "r") as f:
            self.profile = json.load(f)
        self.black_level = float(self.profile["black_level_nominal"])
        self.white_level = float(self.profile["white_level_nominal"])
        self.active_range = self.white_level - self.black_level
        self.iso_profiles = self.profile["iso_profiles"]

    def get_iso_params(self, target_iso: int):
        """Lookup or interpolate noise parameters for a target ISO."""
        isos = sorted([int(k) for k in self.iso_profiles.keys()])
        if str(target_iso) in self.iso_profiles:
            p = self.iso_profiles[str(target_iso)]
            return {
                "read_std": p["read_noise_std_dn"] / self.active_range,
                "row_std": p["row_noise_std_dn"] / self.active_range,
                "col_std": p["col_noise_std_dn"] / self.active_range,
                "random_std": p["random_noise_std_dn"] / self.active_range,
            }

        # Interpolate between nearest ISOs
        if target_iso <= isos[0]:
            p = self.iso_profiles[str(isos[0])]
        elif target_iso >= isos[-1]:
            p = self.iso_profiles[str(isos[-1])]
        else:
            # Linear in log2(ISO) space
            log_target = np.log2(target_iso)
            for i in range(len(isos) - 1):
                if isos[i] <= target_iso <= isos[i + 1]:
                    w = (log_target - np.log2(isos[i])) / (np.log2(isos[i + 1]) - np.log2(isos[i]))
                    p1 = self.iso_profiles[str(isos[i])]
                    p2 = self.iso_profiles[str(isos[i + 1])]
                    std_total = (1 - w) * p1["read_noise_std_dn"] + w * p2["read_noise_std_dn"]
                    std_row = (1 - w) * p1["row_noise_std_dn"] + w * p2["row_noise_std_dn"]
                    std_col = (1 - w) * p1["col_noise_std_dn"] + w * p2["col_noise_std_dn"]
                    std_rand = (1 - w) * p1["random_noise_std_dn"] + w * p2["random_noise_std_dn"]
                    return {
                        "read_std": std_total / self.active_range,
                        "row_std": std_row / self.active_range,
                        "col_std": std_col / self.active_range,
                        "random_std": std_rand / self.active_range,
                    }

        return {
            "read_std": p["read_noise_std_dn"] / self.active_range,
            "row_std": p["row_noise_std_dn"] / self.active_range,
            "col_std": p["col_noise_std_dn"] / self.active_range,
            "random_std": p["random_noise_std_dn"] / self.active_range,
        }

    def add_noise(self, clean_packed: np.ndarray, target_iso: int, source_iso: int = 800) -> np.ndarray:
        """Add calibrated shot + read + banding noise to a 4-channel packed RAW tensor in [0.0, 1.0].

        clean_packed: (4, H, W) float32 in [0.0, 1.0] (black-subtracted and normalized)
        target_iso: e.g. 6400, 12800, 25600
        source_iso: ISO of the source image (e.g. 800)
        """
        c, h, w = clean_packed.shape
        params = self.get_iso_params(target_iso)

        # Photon shot noise: Poisson process
        # System gain K (electrons per DN) for Canon 6D:
        # At ISO 100, gain ~ 4.2 e-/DN. Gain scales inversely with ISO.
        gain_target = 4.2 * (100.0 / target_iso)
        # Shot noise variance in normalized [0, 1] units:
        # Var(S) = S / (gain * active_range)
        shot_var = np.maximum(clean_packed, 1e-6) / (gain_target * self.active_range)
        shot_noise = np.random.normal(0.0, np.sqrt(shot_var)).astype(np.float32)

        # Random Gaussian read noise
        rand_noise = np.random.normal(0.0, params["random_std"], (c, h, w)).astype(np.float32)

        # Row banding noise (correlated across horizontal rows within each CFA channel)
        row_noise = np.random.normal(0.0, params["row_std"], (c, h, 1)).astype(np.float32)

        # Col banding noise
        col_noise = np.random.normal(0.0, params["col_std"], (c, 1, w)).astype(np.float32)

        noisy = clean_packed + shot_noise + rand_noise + row_noise + col_noise
        return np.clip(noisy, 0.0, 1.0).astype(np.float32)
