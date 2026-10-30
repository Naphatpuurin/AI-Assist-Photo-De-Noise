"""Scan a RAW folder and summarise camera / ISO / black level."""
import sys, glob, os, collections
import rawpy
from PIL import Image


def iso_of(path):
    try:
        ex = Image.open(path).getexif()
        return ex.get_ifd(0x8769).get(0x8827)  # ExifIFD ISOSpeedRatings
    except Exception:
        return None


def main(root):
    files = glob.glob(os.path.join(root, "**", "*.CR2"), recursive=True)
    isos = collections.Counter()
    for f in files:
        isos[iso_of(f)] += 1
    print(len(files), "files")
    for k, v in sorted(isos.items(), key=lambda kv: (kv[0] is None, kv[0])):
        print("ISO", k, ":", v)
    with rawpy.imread(files[0]) as r:
        print("sample:", os.path.basename(files[0]), r.raw_image.shape,
              "pattern", r.raw_pattern.tolist(), "black", r.black_level_per_channel,
              "white", r.white_level)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
