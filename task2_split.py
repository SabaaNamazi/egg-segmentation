"""
Task 2: leakage-safe 4:1 dev/eval split by scene.

Many photos share the same scene (floor, tiles, steel pan, white bag, plate), so a random split
would put near-identical scenes on both sides. Instead the images are grouped by scene and whole
groups go to dev or eval.

    python task2_split.py --data data --masks outputs/task1/masks

1. Hide each egg (Task 1 mask filled with the mean background colour).
2. Embed the images with ResNet-18 (ImageNet weights) and cluster them (agglomerative,
   cosine distance, average linkage, threshold 0.25).
3. Eval = the group with the most multi-egg images among groups of at most 12 images, plus small
   groups (at most 4 images) until eval has 20 % of the images, using as many different groups as
   possible. Every other group is dev.

The exact split used in the report is shipped as results/split.csv; use it to reproduce the
numbers. Re-running this script may give a slightly different split if library versions differ.
"""
import argparse
import glob
import itertools
import os
import random

import cv2
import numpy as np
import pandas as pd


def hide_egg(path, mask_path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    inst = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    m = inst > 0
    if (~m).any():
        img[m] = img[~m].mean(axis=0).astype(np.uint8)
    return img, int(len(np.unique(inst)) - 1)


def embed(images):
    import torch
    from PIL import Image
    from torchvision import models, transforms

    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = torch.nn.Identity()
    net.eval()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.no_grad():
        emb = torch.cat([net(tf(Image.fromarray(im)).unsqueeze(0)) for im in images]).numpy()
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def choose_eval_groups(groups, eval_frac=0.2, forced_max_size=12, small_max=4, seed=0):
    """groups: DataFrame with columns group, n (images) and multi (images with more than one egg)."""
    total = int(groups.n.sum())
    target = int(round(eval_frac * total))
    pool = groups[groups.n <= forced_max_size].sort_values(["multi", "n"], ascending=False)
    forced = int(pool.index[0])
    need = target - int(groups.loc[forced, "n"])
    cands = [g for g in groups.index if g != forced and groups.loc[g, "n"] <= small_max]
    best = []
    for k in range(1, 9):
        for combo in itertools.combinations(cands, k):
            if groups.loc[list(combo), "n"].sum() == need:
                best.append(combo)
    if not best:
        raise SystemExit("no valid combination of small groups found; relax small_max or eval_frac")
    kmax = max(len(c) for c in best)
    best = [c for c in best if len(c) == kmax]
    random.Random(seed).shuffle(best)
    return {forced, *best[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--masks", default="outputs/task1/masks")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--out-groups", default="groups.csv")
    ap.add_argument("--out-split", default="split.csv")
    args = ap.parse_args()

    from sklearn.cluster import AgglomerativeClustering

    files = sorted(glob.glob(os.path.join(args.data, "**", "*.jpg"), recursive=True))
    ims, n_eggs = zip(*[hide_egg(f, os.path.join(args.masks, os.path.splitext(os.path.basename(f))[0] + ".png"))
                        for f in files])
    emb = embed(list(ims))
    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=args.threshold,
                                     metric="cosine", linkage="average").fit(emb).labels_
    df = pd.DataFrame({"file": [os.path.basename(f) for f in files], "group": labels, "n_eggs": n_eggs})
    groups = df.groupby("group").agg(n=("file", "size"), multi=("n_eggs", lambda x: int((np.array(x) > 1).sum())))
    print(f"{len(df)} images, {len(groups)} scene groups, sizes: {sorted(groups.n, reverse=True)}")

    eval_groups = choose_eval_groups(groups)
    df["split"] = df.group.map(lambda g: "eval" if g in eval_groups else "dev")
    assert not set(df[df.split == "dev"].group) & set(df[df.split == "eval"].group)
    print(df.split.value_counts().to_dict(), "| eval groups:", sorted(eval_groups))
    print("multi-egg images per split:", df[df.n_eggs > 1].split.value_counts().to_dict())
    df[["file", "group"]].to_csv(args.out_groups, index=False)
    df[["file", "group", "n_eggs", "split"]].to_csv(args.out_split, index=False)


if __name__ == "__main__":
    main()
