import numpy as np
from denoise.noise_model import SensorNoiseModel


def test_noise_model_generation():
    model = SensorNoiseModel("calibration/canon_6d_profile.json")
    clean = np.full((4, 64, 64), 0.2, dtype=np.float32)

    # Test noise addition at ISO 6400
    noisy_6400 = model.add_noise(clean, target_iso=6400)
    assert noisy_6400.shape == clean.shape
    assert not np.array_equal(clean, noisy_6400)
    std_6400 = np.std(noisy_6400 - clean)

    # Test noise addition at ISO 25600 (should have higher std)
    noisy_25600 = model.add_noise(clean, target_iso=25600)
    std_25600 = np.std(noisy_25600 - clean)

    assert std_25600 > std_6400, f"Expected higher noise at ISO 25600: {std_25600} vs {std_6400}"
