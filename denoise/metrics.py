"""Image quality metrics for evaluation."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(ref: np.ndarray, test: np.ndarray, data_range: float = 1.0) -> float:
    return float(peak_signal_noise_ratio(ref, test, data_range=data_range))


def ssim(ref: np.ndarray, test: np.ndarray, data_range: float = 1.0) -> float:
    return float(structural_similarity(ref, test, data_range=data_range,
                                       channel_axis=-1 if ref.ndim == 3 else None))
