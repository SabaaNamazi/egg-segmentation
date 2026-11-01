"""
Classical egg segmentation pipeline.

This is the final Task 2 pipeline (trial T11 with a fixed GrabCut seed) and it is
also the detector used for the video in Task 4.

Steps
1. Candidates: threshold sweeps (bright and dark side) over the grey channel,
   Lab a, Lab b and HSV saturation; keep connected components of 2-40 % of the
   image that do not touch the border.
2. Score: IoU with the best-fit ellipse + 0.5 * mean edge strength on the contour.
3. Accept: score >= 0.9 and mean grey brightness inside >= 0.6, up to 3 eggs.
   If nothing is accepted: retry once at score 0.85 (one egg); if still nothing,
   retry with brightness >= 0.3 (one egg).
4. Refine each accepted blob with GrabCut.
5. Output the union of the refined masks.
"""
import hashlib

import cv2
import numpy as np
from scipy import ndimage as ndi


# --------------------------------------------------------------------------
# candidate generation
# --------------------------------------------------------------------------
def norm8(ch):
    lo, hi = np.percentile(ch, (1, 99))
    return np.clip((ch - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)


def threshold_candidates(g, thresholds, min_area, max_area):
    """Components of (g > t) and (g < t) that pass the size and border rules."""
    H, W = g.shape
    out = []
    k = np.ones((5, 5), np.uint8)
    for t in thresholds:
        for polarity in (0, 1):                      # bright side, dark side
            m = (g > t) if polarity == 0 else (g < t)
            m = cv2.morphologyEx(ndi.binary_fill_holes(m).astype(np.uint8), cv2.MORPH_OPEN, k)
            n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
            for i in range(1, n):
                x, y, w, h, a = st[i]
                if not (min_area <= a / (H * W) <= max_area):
                    continue
                if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
                    continue
                out.append(lab == i)
    return out


def make_candidates(img, blur=5, n_thr=24, n_thr_extra=12, min_area=0.02, max_area=0.40,
                    grey_only=False):
    b = cv2.GaussianBlur(img, (blur, blur), 0)
    gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    out = threshold_candidates(gray, np.linspace(20, 235, n_thr), min_area, max_area)
    if not grey_only:
        lab = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)
        sat = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[:, :, 1]
        for ch in (lab[:, :, 1], lab[:, :, 2], sat):
            out += threshold_candidates(norm8(ch), np.linspace(20, 235, n_thr_extra), min_area, max_area)
    return gray, out


# --------------------------------------------------------------------------
# candidate features
# --------------------------------------------------------------------------
def edge_magnitude(gray):
    mag = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    return np.clip(mag / (np.percentile(mag, 99) + 1e-6), 0, 1)


def candidate_features(c, mag, gray):
    """(ellipse-fit IoU, mean edge strength on the contour, mean brightness inside / 255)."""
    cnts, _ = cv2.findContours(c.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return 0.0, 0.0, 0.0
    em = np.zeros(c.shape, np.uint8)
    cv2.ellipse(em, cv2.fitEllipse(cnt), 1, -1)
    em = em.astype(bool)
    e_iou = np.logical_and(c, em).sum() / max(np.logical_or(c, em).sum(), 1)
    p = cnt[:, 0, :]
    return e_iou, float(mag[p[:, 1], p[:, 0]].mean()), float(gray[c].mean() / 255)


def _pack(c):
    return np.packbits(c)


def _unpack(p, shape):
    return np.unpackbits(p)[:shape[0] * shape[1]].reshape(shape).astype(bool)


_CACHE = {}


def prepare(img, cand_min_area=0.02, grey_only=False, use_cache=True):
    """Candidates and their features for one image: (shape, packed masks, features)."""
    key = (hashlib.md5(img.tobytes()).hexdigest(), cand_min_area, grey_only)
    if use_cache and key in _CACHE:
        return _CACHE[key]
    gray, cands = make_candidates(img, min_area=cand_min_area, grey_only=grey_only)
    mag = edge_magnitude(gray)
    F = np.array([candidate_features(c, mag, gray) for c in cands]) if cands else np.zeros((0, 3))
    res = (gray.shape, [_pack(c) for c in cands], F)
    if use_cache:
        if len(_CACHE) >= 16:
            _CACHE.clear()
        _CACHE[key] = res
    return res


# --------------------------------------------------------------------------
# acceptance
# --------------------------------------------------------------------------
def accept(prep, w_grad=0.5, w_bright=0.0, min_bright=0.0, min_score=0.9, max_eggs=3, overlap=0.3):
    """Union of the accepted candidates.

    score = ellipse IoU + w_grad * edge; a candidate is accepted if score >= min_score and its
    brightness >= min_bright. w_bright only changes the order (trial T5)."""
    shape, packed, F = prep
    out = np.zeros(shape, bool)
    if len(packed) == 0:
        return out
    base = F[:, 0] + w_grad * F[:, 1]
    rank = base + w_bright * F[:, 2]
    kept = []
    for i in np.argsort(-rank):
        if base[i] < min_score or F[i, 2] < min_bright:
            continue
        if len(kept) >= max_eggs:
            break
        c = _unpack(packed[i], shape)
        if any(np.logical_and(c, k).sum() / min(c.sum(), k.sum()) > overlap for k in kept):
            continue
        kept.append(c)
        out |= c
    return out


def segment_blobs(img, w_grad=0.5, min_bright=0.6, fb_bright=0.3, min_score=0.9, min_score_lo=0.85,
                  max_eggs=3, overlap=0.3, retry=True, fallback=True, cand_min_area=0.02,
                  use_cache=True):
    """Accepted blobs before GrabCut (with the retry and fallback stages)."""
    prep = prepare(img, cand_min_area, use_cache=use_cache)
    kw = dict(w_grad=w_grad, overlap=overlap)
    base = accept(prep, min_bright=min_bright, min_score=min_score, max_eggs=max_eggs, **kw)
    if base.sum() == 0 and retry:
        base = accept(prep, min_bright=min_bright, min_score=min_score_lo, max_eggs=1, **kw)
    if base.sum() == 0 and fallback:
        base = accept(prep, min_bright=fb_bright, min_score=min_score, max_eggs=1, **kw)
    return base


# --------------------------------------------------------------------------
# refinement
# --------------------------------------------------------------------------
def refine_grabcut(img, base, ring=0.8, iters=10):
    """Run GrabCut on every connected blob of `base`.

    sure foreground = blob eroded by ~a quarter of its radius, probable foreground = blob,
    probable background = a ring of ring*radius around the blob, sure background beyond it."""
    out = np.zeros(base.shape, bool)
    n, lab = cv2.connectedComponents(base.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        blob = (lab == i).astype(np.uint8)
        r = np.sqrt(blob.sum() / np.pi)
        k_in = max(3, int(0.25 * r))
        k_out = max(3, int(ring * r))
        fg = cv2.erode(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_in, k_in)))
        outer = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k_out + 1, 2 * k_out + 1)))
        mask = np.full(blob.shape, cv2.GC_BGD, np.uint8)
        mask[outer > 0] = cv2.GC_PR_BGD
        mask[blob > 0] = cv2.GC_PR_FGD
        mask[fg > 0] = cv2.GC_FGD
        if fg.sum() < 50:
            out |= blob.astype(bool)
            continue
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(img, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            out |= blob.astype(bool)
            continue
        out |= (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    return out


def fill_ellipses(base):
    """Replace each blob by its filled best-fit ellipse (trial T8)."""
    out = np.zeros(base.shape, np.uint8)
    n, lab = cv2.connectedComponents(base.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        if len(cnt) >= 5:
            cv2.ellipse(out, cv2.fitEllipse(cnt), 1, -1)
        else:
            out |= (lab == i).astype(np.uint8)
    return out.astype(bool)


# --------------------------------------------------------------------------
# final pipeline
# --------------------------------------------------------------------------
def segment(img, ring=0.8, iters=10, seed=0, refine=True, **kw):
    """Final pipeline. `img` is a BGR image; returns a boolean mask (union of all eggs).

    seed: GrabCut uses random initialisation; the OpenCV seed is fixed so that repeated runs
    give identical masks (None keeps OpenCV's state)."""
    if seed is not None:
        cv2.setRNGSeed(seed)
    base = segment_blobs(img, **kw)
    if not refine or base.sum() == 0:
        return base
    return refine_grabcut(img, base, ring, iters)
