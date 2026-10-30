import os
import numpy as np
import tifffile
from denoise.dng_writer import write_linear_dng


def test_linear_dng_write_and_read(tmp_path):
    out_file = str(tmp_path / "test_out.dng")
    # Synthetic 128x128 linear RGB patch
    patch = np.random.uniform(0.0, 1.0, (128, 128, 3)).astype(np.float32)
    write_linear_dng(out_file, patch)

    assert os.path.exists(out_file)
    with tifffile.TiffFile(out_file) as tif:
        page = tif.pages[0]
        assert page.shape == (128, 128, 3)
        assert page.dtype == np.uint16
        # Check DNG tag
        assert 50706 in page.tags  # DNGVersion
        assert bytes(page.tags[50706].value) == b"\x01\x04\x00\x00"
        assert 50708 in page.tags  # UniqueCameraModel
        assert page.tags[50708].value == "Canon EOS 6D"
