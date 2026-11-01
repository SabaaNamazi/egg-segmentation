"""
Task 4: track the individual eggs in a video.

    python task4_track.py --video video.mp4 --out results/tracked_v2.mp4 --start 240 --end 1000 --step 2

Detector: the final Task 2 pipeline (egg_pipeline.segment) on every processed frame, resized to
640x360, with the video-specific settings found in the experiments (candidate minimum area 1 %
of the frame, up to 6 accepted blobs, blobs above y = 150 (the shirt) and in the logo zone are
dropped, components under 800 px are dropped).
Tracker: each track keeps its last mask. Every (track, mask) pair costs
centroid distance - 300 * mask IoU; pairs with IoU < 0.2 and distance > 260 px are not allowed
unless the track is already missing; pairs are assigned greedily from the lowest cost; tracks
without a mask are marked missed and keep their last position. IDs are given from left to
right at the start frame (no new tracks are created later), so the video must start with two
eggs. The two eggs look the same, so identity comes from position and continuity only.

Outputs: the annotated video (Egg 1 red, Egg 2 green) and a CSV log (frame, id, x, y).
"""
import argparse
import os
import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd

import egg_pipeline as ep

W, H = 640, 360


def mask_iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def centroid(m):
    return np.argwhere(m).mean(0)[::-1]            # (x, y)


class Tracker:
    def __init__(self, init_masks, iou_thr=0.2, max_dist=260):
        order = sorted(range(len(init_masks)), key=lambda j: centroid(init_masks[j])[0])
        self.tracks = {k + 1: dict(mask=init_masks[j], missed=0) for k, j in enumerate(order)}
        self.iou_thr, self.max_dist = iou_thr, max_dist

    def update(self, masks):
        pairs = []
        for t, v in self.tracks.items():
            for b, m in enumerate(masks):
                iou_v = mask_iou(v["mask"], m)
                d = np.linalg.norm(centroid(v["mask"]) - centroid(m))
                if iou_v < self.iou_thr and d > self.max_dist and v["missed"] == 0:
                    continue
                pairs.append((d - 300 * iou_v, t, b))
        pairs.sort()
        assigned, used_t = {}, set()
        for _, t, b in pairs:
            if t in used_t or b in assigned:
                continue
            assigned[b] = t
            used_t.add(t)
        for t in self.tracks:
            if t not in used_t:
                self.tracks[t]["missed"] += 1
        for b, t in assigned.items():
            self.tracks[t] = dict(mask=masks[b], missed=0)
        return {t: masks[b] for b, t in assigned.items()}


def detect(frame, y_min=150, min_area=800, cand_min_area=0.01, max_eggs=6, logo=(520, 300)):
    """Egg masks of one 640x360 frame."""
    pred = ep.segment(frame, min_score=0.9, min_score_lo=0.85, ring=0.8, iters=10,
                      cand_min_area=cand_min_area, max_eggs=max_eggs, use_cache=False)
    n, lab, st, cen = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    keep = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area or cen[i][1] <= y_min:
            continue
        if logo and cen[i][0] >= logo[0] and cen[i][1] >= logo[1]:
            continue
        keep.append(lab == i)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="video.mp4")
    ap.add_argument("--out", default="results/tracked_v2.mp4")
    ap.add_argument("--log", default="results/track_log_v2.csv")
    ap.add_argument("--start", type=int, default=240, help="first frame (10 s: the eggs appear)")
    ap.add_argument("--end", type=int, default=1000, help="last frame (the text slides start after it)")
    ap.add_argument("--step", type=int, default=2, help="process every n-th frame")
    ap.add_argument("--y-min", type=int, default=150)
    ap.add_argument("--no-logo-zone", action="store_true")
    args = ap.parse_args()
    logo = None if args.no_logo_zone else (520, 300)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    ok, fr = cap.read()
    if not ok:
        raise SystemExit("cannot read the start frame")
    init = detect(cv2.resize(fr, (W, H)), y_min=args.y_min, logo=logo)
    if len(init) != 2:
        raise SystemExit(f"expected 2 eggs at frame {args.start}, found {len(init)}")
    tracker = Tracker(init)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps / args.step, (W, H))
    colours = {1: (0, 0, 255), 2: (0, 200, 0)}      # BGR: Egg 1 red, Egg 2 green
    log = []
    for fi in range(args.start, args.end, args.step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (W, H))
        res = tracker.update(detect(fr, y_min=args.y_min, logo=logo))
        for t, m in res.items():
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(fr, cs, -1, colours.get(t, (255, 255, 0)), 2)
            x, y = centroid(m)
            cv2.putText(fr, f"Egg {t}", (int(x) - 25, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        colours.get(t, (255, 255, 0)), 2)
            log.append((fi, t, int(x), int(y)))
        vw.write(fr)
    vw.release()
    if shutil.which("ffmpeg"):                       # H.264 so that the video plays in a browser
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p", args.out], check=True)
        os.remove(tmp)
    else:
        os.replace(tmp, args.out)
        print("ffmpeg not found: the video uses the mp4v codec and may not play in a browser")
    df = pd.DataFrame(log, columns=["frame", "id", "x", "y"])
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    df.to_csv(args.log, index=False)
    print("frames processed:", len(range(args.start, args.end, args.step)),
          "| IDs seen:", sorted(int(i) for i in df.id.unique()))
    print(df.groupby("id").agg(frames=("frame", "count"), x_min=("x", "min"), x_max=("x", "max")).to_string())


if __name__ == "__main__":
    main()
