"""
Task 2: reproduce the experiment log (trials T0 to T11) on the dev split.

    python task2_experiments.py --data data --masks outputs/task1/masks --split split.csv \
        --trials T0,T3,T6,T9d,T11 --out results/experiments_reproduced.csv

Every trial changes one thing compared with the best kept trial before it; see the report for
the reason of each change. Scores are the IoU between the predicted union mask and the Task 1
union mask, averaged over the 88 dev images (the eval images are never used here).
GrabCut trials use a fixed OpenCV seed, so small differences from the original log (about
0.005 mean IoU) are expected: the original trials were run before the seed was fixed.
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import egg_pipeline as ep


# --------------------------------------------------------------------------
# T0 - T2: single threshold
# --------------------------------------------------------------------------
def clean(m, min_area_frac=0.01):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    keep = np.zeros_like(m, bool)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area_frac * m.size:
            keep |= lab == i
    return ndi.binary_fill_holes(keep)


def seg_otsu(img, blur=5, min_area_frac=0.01):                       # T0
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (blur, blur), 0)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return clean(m > 0, min_area_frac)


def seg_bgdist(img, blur=5, border_frac=0.06, min_area_frac=0.01):  # T1
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (blur, blur), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    b = max(2, int(border_frac * min(h, w)))
    strip = np.zeros((h, w), bool)
    strip[:b, :] = strip[-b:, :] = strip[:, :b] = strip[:, -b:] = True
    bg = np.median(lab[strip], axis=0)
    d8 = cv2.normalize(np.linalg.norm(lab - bg, axis=2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return clean(m > 0, min_area_frac)


def seg_otsu_prior(img, blur=5, min_area=0.02, max_area=0.40, min_solidity=0.9, reject_border=True):  # T2
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (blur, blur), 0)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ndi.binary_fill_holes(m > 0).astype(np.uint8)
    H, W = m.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros((H, W), bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (min_area <= a / (H * W) <= max_area):
            continue
        if reject_border and (x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1):
            continue
        comp = (lab == i).astype(np.uint8)
        cnt, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.contourArea(cv2.convexHull(max(cnt, key=cv2.contourArea)))
        if hull > 0 and a / hull < min_solidity:
            continue
        out |= comp.astype(bool)
    return out


# --------------------------------------------------------------------------
# T3 - T11: candidate sweep, acceptance, refinement (see egg_pipeline.py)
# --------------------------------------------------------------------------
def seg_sweep(img):                       # T3: grey-level candidates, ellipse + edge score
    return ep.accept(ep.prepare(img, 0.02, grey_only=True), min_score=0.9, max_eggs=3)


def seg_sweep_mc(img):                    # T4: + Lab a, Lab b, saturation candidates
    return ep.accept(ep.prepare(img, 0.02), min_score=0.9, max_eggs=3)


def seg_rank(img):                        # T5: brightness in the ranking only
    return ep.accept(ep.prepare(img, 0.02), w_bright=0.5, min_score=0.9, max_eggs=3)


def seg_gate(img):                        # T6: brightness gate on acceptance
    return ep.segment_blobs(img, retry=False, fallback=False)


def seg_fb(img):                          # T7: + fallback when empty
    return ep.segment_blobs(img, retry=False, fallback=True)


def seg_ell(img):                         # T8: blobs replaced by their fitted ellipse
    return ep.fill_ellipses(seg_fb(img))


def seg_gc(img, ring=0.4, min_score=0.9):  # T9 - T10: GrabCut on the T7 blobs
    cv2.setRNGSeed(0)
    base = ep.segment_blobs(img, min_score=min_score, retry=False, fallback=True)
    return ep.refine_grabcut(img, base, ring, 10) if base.sum() else base


def seg_final(img):                       # T11 = frozen final pipeline
    return ep.segment(img, min_score=0.9, min_score_lo=0.85, ring=0.8, iters=10)


TRIALS = {
    "T0":  seg_otsu,
    "T1":  seg_bgdist,
    "T2":  seg_otsu_prior,
    "T3":  seg_sweep,
    "T4":  seg_sweep_mc,
    "T5":  seg_rank,
    "T6":  seg_gate,
    "T7":  seg_fb,
    "T8":  seg_ell,
    "T9":  lambda im: seg_gc(im, ring=0.4),
    "T9a": lambda im: seg_gc(im, ring=0.2),
    "T9c": lambda im: seg_gc(im, ring=0.6),
    "T9d": lambda im: seg_gc(im, ring=0.8),
    "T9e": lambda im: seg_gc(im, ring=1.0),
    "T10": lambda im: seg_gc(im, ring=0.8, min_score=0.85),
    "T11": seg_final,
}


def iou(p, g):
    u = np.logical_or(p, g).sum()
    return np.logical_and(p, g).sum() / u if u else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="folder with the dataset images")
    ap.add_argument("--masks", default="outputs/task1/masks", help="Task 1 instance masks")
    ap.add_argument("--split", default="split.csv")
    ap.add_argument("--trials", default="T0,T3,T4,T6,T7,T9d,T11", help="comma separated, or all")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N dev images (quick test)")
    ap.add_argument("--out", default="results/experiments_reproduced.csv")
    args = ap.parse_args()

    split = pd.read_csv(args.split)
    dev = split[split.split == "dev"].file.tolist()
    if args.limit:
        dev = dev[:args.limit]
    paths = {os.path.basename(p): p for p in glob.glob(os.path.join(args.data, "**", "*.jpg"), recursive=True)}
    imgs = {f: cv2.imread(paths[f]) for f in dev}
    gts = {f: cv2.imread(os.path.join(args.masks, os.path.splitext(f)[0] + ".png"), cv2.IMREAD_UNCHANGED) > 0
           for f in dev}
    names = list(TRIALS) if args.trials == "all" else args.trials.split(",")

    rows = []
    for name in names:
        t0 = time.time()
        ious = np.array([iou(TRIALS[name](imgs[f]), gts[f]) for f in dev])
        rows.append(dict(trial=name, n_images=len(dev), mean_iou=round(float(ious.mean()), 4),
                         median_iou=round(float(np.median(ious)), 4), fails_below_0_5=int((ious < 0.5).sum()),
                         seconds=round(time.time() - t0)))
        print(rows[-1], flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
