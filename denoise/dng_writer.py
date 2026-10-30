"""DNG Writer: export denoised linear RGB or Bayer RAW to Adobe DNG format."""
import numpy as np
import tifffile


# Standard DNG Tag IDs
TAG_DNG_VERSION = 50706
TAG_DNG_BACKWARD_VERSION = 50707
TAG_UNIQUE_CAMERA_MODEL = 50708
TAG_BLACK_LEVEL = 50714
TAG_WHITE_LEVEL = 50717
TAG_COLOR_MATRIX1 = 50721
TAG_AS_SHOT_NEUTRAL = 50728
TAG_CALIBRATION_ILLUMINANT1 = 50778


def write_linear_dng(
    out_path: str,
    rgb_linear: np.ndarray,
    camera_model: str = "Canon EOS 6D",
    color_matrix1: tuple = None,
    as_shot_neutral: tuple = (0.47, 1.0, 0.65),
):
    """Write a 16-bit Linear DNG (demosaiced + denoised RGB).

    rgb_linear: (H, W, 3) float32 in [0.0, 1.0] linear sensor RGB.
    """
    h, w, c = rgb_linear.shape
    assert c == 3, "Linear DNG must have 3 RGB channels"

    # Scale to 16-bit unsigned integer [0, 65535]
    img_u16 = np.clip(rgb_linear * 65535.0, 0, 65535).astype(np.uint16)

    # Standard Canon 6D ColorMatrix1 (XYZ to Camera RGB D65 approximate)
    if color_matrix1 is None:
        cm = [
            (6722, 10000), (-635, 10000), (-963, 10000),
            (-4285, 10000), (12486, 10000), (2028, 10000),
            (-908, 10000), (2162, 10000), (5668, 10000),
        ]
    else:
        cm = [(int(x * 10000), 10000) for x in color_matrix1]

    # AsShotNeutral: R, G, B neutral reciprocal
    asn = [(int(x * 10000), 10000) for x in as_shot_neutral]

    extratags = [
        # DNGVersion: 1.4.0.0 (BYTE, count 4)
        (TAG_DNG_VERSION, "B", 4, (1, 4, 0, 0), False),
        # DNGBackwardVersion: 1.3.0.0 (BYTE, count 4)
        (TAG_DNG_BACKWARD_VERSION, "B", 4, (1, 3, 0, 0), False),
        # UniqueCameraModel (ASCII)
        (TAG_UNIQUE_CAMERA_MODEL, "s", len(camera_model) + 1, camera_model, False),
        # ColorMatrix1 (SRATIONAL, count 9)
        (TAG_COLOR_MATRIX1, "2i", 9, cm, False),
        # CalibrationIlluminant1 (SHORT, 21 = D65)
        (TAG_CALIBRATION_ILLUMINANT1, "H", 1, 21, False),
        # AsShotNeutral (RATIONAL, count 3)
        (TAG_AS_SHOT_NEUTRAL, "2I", 3, asn, False),
        # BlackLevel (RATIONAL: 0 for pre-normalized linear DNG)
        (TAG_BLACK_LEVEL, "2I", 1, (0, 1), False),
        # WhiteLevel (SHORT: 65535)
        (TAG_WHITE_LEVEL, "H", 1, 65535, False),
    ]

    tifffile.imwrite(
        out_path,
        img_u16,
        photometric="rgb",
        extratags=extratags,
        metadata={"axes": "YXS"},
    )
    return out_path
