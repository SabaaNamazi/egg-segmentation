"""
Task 3: evaluate the classical predictions against the Task 1 masks (ground truth).

    python task3_evaluate.py --images data --gt outputs/task1/masks --pred outputs/task2_pred \
        --split split.csv --which eval --dev-iou results/per_image_iou.csv --dev-column FINAL_dev

Writes eval_metrics.csv, task3_plots.png (needs --images) and task3_worst12.png (needs --images)
into --out, and prints the summary tables used in the report.

Metrics: IoU, Dice, precision, recall, boundary F1 (3 px tolerance), area ratio on the union
mask of each image; instance level TP/FP/FN with one-to-one matching at IoU >= 0.5, where
predicted components under 1 % of the image (4096 px for 640x640) are ignored.
"""
import argparse
import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

MIN_COMP_FRAC = 0.01


def miou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 1.0


def contour(m):
    m8 = m.astype(np.uint8)
    return (m8 - cv2.erode(m8, np.ones((3, 3), np.uint8))) > 0


def boundary_f1(p, g, tol=3):
    cp, cg = contour(p), contour(g)
    if cp.sum() == 0 or cg.sum() == 0:
        return 0.0
    dg = ndi.distance_transform_edt(~cg)
    dp = ndi.distance_transform_edt(~cp)
    pr, rc = (dg[cp] <= tol).mean(), (dp[cg] <= tol).mean()
    return 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)


def failure_type(iou_, prec, rec, empty):
    """Heuristic label used only for sorting the failures."""
    if iou_ >= 0.5:
        return "ok"
    if empty:
        return "empty"
    if rec < 0.2:
        return "wrong object"
    if prec >= 0.7 and rec < 0.6:
        return "partial egg"
    if rec >= 0.6 and prec < 0.6:
        return "leak / extra region"
    return "mixed"


def image_metrics(gt_inst, pred):
    g = gt_inst > 0
    inter = np.logical_and(pred, g).sum()
    prec, rec = inter / max(pred.sum(), 1), inter / max(g.sum(), 1)
    gl = [gt_inst == k for k in np.unique(gt_inst)[1:]]
    n, lab = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    pl_all = [lab == i for i in range(1, n)]
    pl = [c for c in pl_all if c.sum() >= MIN_COMP_FRAC * pred.size]
    pairs = sorted(((miou(a, b), i, j) for i, a in enumerate(gl) for j, b in enumerate(pl)), reverse=True)
    used_g, used_p, tp = set(), set(), 0
    for s, i, j in pairs:
        if s < 0.5:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        tp += 1
    i_ = miou(pred, g)
    return dict(iou=i_, dice=2 * inter / max(pred.sum() + g.sum(), 1), precision=prec, recall=rec,
                boundary_f1=boundary_f1(pred, g), area_ratio=pred.sum() / max(g.sum(), 1),
                n_gt=len(gl), n_pred_raw=len(pl_all), n_pred=len(pl), tp=tp, fp=len(pl) - tp,
                fn=len(gl) - tp, failure=failure_type(i_, prec, rec, pred.sum() == 0))


def overlay(img_path, gt_inst, pred):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    for mk, c in (((gt_inst > 0).astype(np.uint8), (0, 255, 0)), (pred.astype(np.uint8), (255, 0, 0))):
        cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, c, 3)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data")
    ap.add_argument("--gt", default="outputs/task1/masks")
    ap.add_argument("--pred", default="outputs/task2_pred")
    ap.add_argument("--split", default="split.csv")
    ap.add_argument("--which", default="eval", choices=["dev", "eval"])
    ap.add_argument("--dev-iou", default=None, help="per_image_iou.csv with the dev IoU of every trial")
    ap.add_argument("--dev-column", default="FINAL_dev")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    split = pd.read_csv(args.split)
    files = split[split.split == args.which].file.tolist()
    stem = lambda f: os.path.splitext(f)[0]
    rows = []
    for f in files:
        gt = cv2.imread(os.path.join(args.gt, stem(f) + ".png"), cv2.IMREAD_UNCHANGED)
        pr = cv2.imread(os.path.join(args.pred, stem(f) + ".png"), cv2.IMREAD_UNCHANGED) > 0
        rows.append(dict(img=f.split("_")[2], file=f, **image_metrics(gt, pr)))
    m = pd.DataFrame(rows).merge(split[["file", "group"]], on="file")
    m.to_csv(os.path.join(args.out, "eval_metrics.csv"), index=False)

    print(f"PIXEL-LEVEL ({args.which}, {len(m)} images)  mean / median")
    for c in ["iou", "dice", "precision", "recall", "boundary_f1"]:
        print(f"  {c:12s} {m[c].mean():.3f} / {m[c].median():.3f}")
    TP, FP, FN = m.tp.sum(), m.fp.sum(), m.fn.sum()
    P, R = TP / max(TP + FP, 1), TP / max(TP + FN, 1)
    print(f"\nINSTANCE-LEVEL (IoU>=0.5, components under {MIN_COMP_FRAC:.0%} of the image ignored)")
    print(f"  GT eggs {m.n_gt.sum()} | predicted components: {m.n_pred_raw.sum()} raw, {m.n_pred.sum()} after the size rule")
    print(f"  TP {TP}, FP {FP}, FN {FN} | precision {P:.3f} | recall {R:.3f} | F1 {2*P*R/max(P+R, 1e-9):.3f}")
    print("  images with the right egg count:", int((m.n_gt == m.n_pred).sum()), "/", len(m))
    print("\nFAILURE TYPES (heuristic):", m.failure.value_counts().to_dict())

    imgs = {os.path.basename(p): p for p in glob.glob(os.path.join(args.images, "**", "*.jpg"), recursive=True)}
    if not imgs:
        print("no images found under --images; skipping the plots")
        return

    # contrast between the egg and a ring around it, from the Task 1 mask
    feats = []
    for f in m.file:
        gray = cv2.cvtColor(cv2.imread(imgs[f]), cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.imread(os.path.join(args.gt, stem(f) + ".png"), cv2.IMREAD_UNCHANGED) > 0
        ring = (cv2.dilate(g.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0) & ~g
        feats.append((g.mean() * 100, float(gray[g].mean() - gray[ring].mean())))
    m["egg_area_pct"], m["contrast"] = zip(*feats)

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    bins = np.linspace(0, 1, 11)
    if args.dev_iou and os.path.exists(args.dev_iou):
        dev = pd.read_csv(args.dev_iou, index_col=0)[args.dev_column].values
        ax[0, 0].hist(dev, bins=bins, alpha=.55, density=True, label=f"dev (n={len(dev)})")
    ax[0, 0].hist(m.iou, bins=bins, alpha=.55, density=True, label=f"{args.which} (n={len(m)})")
    ax[0, 0].axvline(.5, color="k", ls=":")
    ax[0, 0].set(xlabel="IoU with Task 1 mask", ylabel="share of images", title="IoU distribution")
    ax[0, 0].legend()
    ax[0, 1].scatter(m.recall, m.precision, c=m.iou, cmap="viridis", s=60)
    for _, r in m.iterrows():
        ax[0, 1].annotate(str(r.img), (r.recall, r.precision), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[0, 1].set(xlabel="recall", ylabel="precision", xlim=(-.03, 1.05), ylim=(-.03, 1.05),
                 title=f"precision vs recall, {args.which} images")
    scene = np.where(m.group == m.group.value_counts().index[0], "largest scene group", "other scenes")
    scene_names = sorted(set(scene))
    for k, name in enumerate(scene_names):
        sub = m[scene == name]
        ax[1, 0].scatter(k + np.random.RandomState(0).uniform(-.1, .1, len(sub)), sub.iou, s=50)
        ax[1, 0].hlines(sub.iou.mean(), k - .25, k + .25, color="k")
        print(f"{name}: n={len(sub)}, mean IoU {sub.iou.mean():.3f}")
    ax[1, 0].set_xticks(range(len(scene_names)))
    ax[1, 0].set_xticklabels(scene_names)
    ax[1, 0].set(ylabel="IoU", title="IoU by scene (black bar = mean)")
    ax[1, 1].scatter(m.contrast, m.iou, s=60)
    for _, r in m.iterrows():
        ax[1, 1].annotate(str(r.img), (r.contrast, r.iou), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[1, 1].set(xlabel="egg brightness minus surrounding ring (grey levels)", ylabel="IoU",
                 title="IoU vs egg-background contrast")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "task3_plots.png"), dpi=140)
    plt.close()
    print(m[["iou", "egg_area_pct", "contrast"]].corr().round(2).to_string())

    worst = m.sort_values("iou").head(12)
    fig, ax = plt.subplots(3, 4, figsize=(16, 12))
    for a, (_, r) in zip(ax.ravel(), worst.iterrows()):
        gt = cv2.imread(os.path.join(args.gt, stem(r.file) + ".png"), cv2.IMREAD_UNCHANGED)
        pr = cv2.imread(os.path.join(args.pred, stem(r.file) + ".png"), cv2.IMREAD_UNCHANGED) > 0
        a.imshow(overlay(imgs[r.file], gt, pr))
        a.set_title(f"image {r.img}  IoU={r.iou:.2f}  P={r.precision:.2f} R={r.recall:.2f}", fontsize=9)
        a.axis("off")
    plt.suptitle("green = Task 1 mask (ground truth), red = classical pipeline")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "task3_worst12.png"), dpi=130)
    plt.close()
    print("saved metrics and plots to", args.out)


if __name__ == "__main__":
    main()
