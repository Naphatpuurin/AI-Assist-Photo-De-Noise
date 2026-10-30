"""Calibrate camera sensor noise parameters from dark frames (Canon EOS 6D)."""
import glob
import json
import os
from collections import defaultdict
import numpy as np
from PIL import Image
import rawpy


def get_exif_iso_exp(path: str):
    try:
        ex = Image.open(path).getexif()
        ifd = ex.get_ifd(0x8769)
        iso = ifd.get(0x8827)
        exp = float(ifd.get(0x829a, 0.0))
        return iso, exp
    except Exception:
        return None, None


def analyze_dark_frame(path: str):
    """Analyze a single dark frame to extract noise components."""
    with rawpy.imread(path) as r:
        raw = r.raw_image.astype(np.float64)
        top = r.sizes.top_margin
        left = r.sizes.left_margin
        h = r.sizes.height
        w = r.sizes.width
        # Ensure dimensions are even
        h = (h // 2) * 2
        w = (w // 2) * 2
        active = raw[top:top + h, left:left + w]

        # Per-channel analysis (RGGB)
        # channel 0: (0,0) R, channel 1: (0,1) G1, channel 2: (1,0) G2, channel 3: (1,1) B
        channels = [
            active[0::2, 0::2],
            active[0::2, 1::2],
            active[1::2, 0::2],
            active[1::2, 1::2],
        ]

        ch_stats = []
        for ch in channels:
            mean = float(np.mean(ch))
            var_total = float(np.var(ch))

            # Row noise: variance of row means
            row_means = np.mean(ch, axis=1)
            var_row = float(np.var(row_means))

            # Col noise: variance of col means
            col_means = np.mean(ch, axis=0)
            var_col = float(np.var(col_means))

            # Random pixel noise = total - row - col (approx)
            var_random = max(0.0, var_total - var_row - var_col)

            ch_stats.append({
                "mean": mean,
                "std_total": float(np.sqrt(var_total)),
                "std_row": float(np.sqrt(var_row)),
                "std_col": float(np.sqrt(var_col)),
                "std_random": float(np.sqrt(var_random)),
            })

        return {
            "overall_mean": float(np.mean(active)),
            "overall_std": float(np.std(active)),
            "channels": ch_stats,
        }


def run_calibration(input_dir: str, output_json: str):
    files = glob.glob(os.path.join(input_dir, "**", "*.CR2"), recursive=True)
    if not files:
        print(f"No .CR2 files found in {input_dir}")
        return

    print(f"Found {len(files)} files in {input_dir}. Grouping by ISO and exposure...")
    by_iso = defaultdict(list)
    for f in files:
        iso, exp = get_exif_iso_exp(f)
        if iso is not None:
            by_iso[iso].append((exp, f))

    iso_results = {}
    for iso, file_list in sorted(by_iso.items()):
        print(f"\nProcessing ISO {iso} ({len(file_list)} frames)...")
        # Sort by exposure time ascending (fastest shutter speed first)
        file_list.sort(key=lambda x: x[0])
        fastest_exp, fastest_file = file_list[0]

        stats_list = []
        for exp, f in file_list:
            stats = analyze_dark_frame(f)
            stats["exp"] = exp
            stats["filename"] = os.path.basename(f)
            stats_list.append(stats)

        # Baseline read noise from the fastest shutter frame (minimal thermal noise)
        base_stats = stats_list[0]

        iso_results[str(iso)] = {
            "fastest_exp_sec": fastest_exp,
            "read_noise_std_dn": base_stats["overall_std"],
            "read_noise_var_dn": base_stats["overall_std"] ** 2,
            "row_noise_std_dn": float(np.mean([ch["std_row"] for ch in base_stats["channels"]])),
            "col_noise_std_dn": float(np.mean([ch["std_col"] for ch in base_stats["channels"]])),
            "random_noise_std_dn": float(np.mean([ch["std_random"] for ch in base_stats["channels"]])),
            "per_channel_std": [ch["std_total"] for ch in base_stats["channels"]],
            "black_level_measured": base_stats["overall_mean"],
            "frames_analyzed": len(file_list),
        }
        print(f"  ISO {iso:5d} -> Read Noise std: {base_stats['overall_std']:7.2f} DN | Row: {iso_results[str(iso)]['row_noise_std_dn']:5.2f} | Random: {iso_results[str(iso)]['random_noise_std_dn']:5.2f}")

    profile = {
        "camera": "Canon EOS 6D",
        "cfa_pattern": "RGGB",
        "black_level_nominal": 2048,
        "white_level_nominal": 15490,
        "active_range": 15490 - 2048,
        "iso_profiles": iso_results,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\nCalibration complete! Saved profile to: {output_json}")


if __name__ == "__main__":
    import sys
    in_dir = sys.argv[1] if len(sys.argv) > 1 else "data/101CANON"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "calibration/canon_6d_profile.json"
    run_calibration(in_dir, out_file)
