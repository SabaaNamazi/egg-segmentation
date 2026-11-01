"""
Task 2: run the final classical pipeline on a folder of images.

    python task2_segment.py --images data --out outputs/task2_pred [--split split.csv --which eval]

Writes one binary PNG (255 = egg) per image, named like the input image.
"""
import argparse
import os
from pathlib import Path

import cv2
import pandas as pd

import egg_pipeline as ep

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder with the images (searched recursively)")
    ap.add_argument("--out", default="outputs/task2_pred")
    ap.add_argument("--split", default=None, help="split.csv (columns file, split); needs --which")
    ap.add_argument("--which", choices=["dev", "eval"], default=None)
    args = ap.parse_args()

    paths = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if args.split and args.which:
        keep = set(pd.read_csv(args.split).query("split == @args.which").file)
        paths = [p for p in paths if p.name in keep]
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(paths)} images")
    for p in paths:
        img = cv2.imread(str(p))
        mask = ep.segment(img, min_score=0.9, min_score_lo=0.85, ring=0.8, iters=10)
        cv2.imwrite(str(Path(args.out) / f"{p.stem}.png"), mask.astype("uint8") * 255)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
