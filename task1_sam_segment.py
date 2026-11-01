"""
Task 1 - Egg segmentation with pretrained models.

Pipeline: Grounding DINO (text prompt "egg") -> boxes -> SAM (box prompts) -> instance masks.
Why this and not SamAutomaticMaskGenerator: auto mode returns dozens of masks (fingers, palm,
background) and you'd need heuristics to pick the egg. Box-prompted SAM gives one clean mask per egg.

Usage:
    python task1_sam_segment.py --images data/images --out outputs/task1
Outputs (per image <name>):
    masks/<name>.png      uint16 instance map (0 = background, 1..N = egg id)
    overlays/<name>.jpg   red overlay like the example in the task PDF
    results.json          boxes + scores per image (for debugging / report)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                          SamModel, SamProcessor)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_models(gd_id, sam_id):
    gd_proc = AutoProcessor.from_pretrained(gd_id)
    gd = AutoModelForZeroShotObjectDetection.from_pretrained(gd_id).to(DEVICE).eval()
    sam_proc = SamProcessor.from_pretrained(sam_id)
    sam = SamModel.from_pretrained(sam_id).to(DEVICE).eval()
    return gd_proc, gd, sam_proc, sam


@torch.no_grad()
def detect_eggs(img, gd_proc, gd, prompt, box_thr, text_thr, max_area_frac):
    inputs = gd_proc(images=img, text=prompt, return_tensors="pt").to(DEVICE)
    out = gd(**inputs)
    res = gd_proc.post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=box_thr, text_threshold=text_thr,
        target_sizes=[img.size[::-1]],
    )[0]
    boxes, scores = res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()

    # drop boxes that cover almost the whole image (usually "hand holding egg" false hits)
    W, H = img.size
    keep = [i for i, b in enumerate(boxes)
            if (b[2] - b[0]) * (b[3] - b[1]) < max_area_frac * W * H]
    return boxes[keep], scores[keep]


@torch.no_grad()
def segment_boxes(img, boxes, sam_proc, sam):
    inputs = sam_proc(img, input_boxes=[boxes.tolist()], return_tensors="pt").to(DEVICE)
    out = sam(**inputs, multimask_output=False)
    masks = sam_proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu())[0]          # (n, 1, H, W) bool
    return masks[:, 0].numpy()


def make_overlay(img_np, instance_map, alpha=0.6):
    over = img_np.copy()
    red = np.array([255, 0, 0], dtype=np.float32)
    m = instance_map > 0
    over[m] = (over[m] * (1 - alpha) + red * alpha).astype(np.uint8)
    return over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="outputs/task1")
    ap.add_argument("--prompt", default="an egg.")      # must be lowercase + end with '.'
    ap.add_argument("--gd", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--sam", default="facebook/sam-vit-large")
    ap.add_argument("--box-thr", type=float, default=0.30)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--max-area-frac", type=float, default=0.85)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    gd_proc, gd, sam_proc, sam = load_models(args.gd, args.sam)
    paths = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMG_EXTS)
    print(f"{len(paths)} images, device={DEVICE}")

    log, no_det = {}, []
    for p in paths:
        img = Image.open(p).convert("RGB")
        boxes, scores = detect_eggs(img, gd_proc, gd, args.prompt,
                                    args.box_thr, args.text_thr, args.max_area_frac)
        if len(boxes) == 0:
            no_det.append(p.name)
            log[p.name] = {"boxes": [], "scores": []}
            continue

        masks = segment_boxes(img, boxes, sam_proc, sam)
        inst = np.zeros(masks.shape[1:], dtype=np.uint16)
        for i, m in enumerate(masks, start=1):
            inst[m & (inst == 0)] = i

        cv2.imwrite(str(out / "masks" / f"{p.stem}.png"), inst)
        ov = make_overlay(np.array(img), inst)
        cv2.imwrite(str(out / "overlays" / f"{p.stem}.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        log[p.name] = {"boxes": boxes.round(1).tolist(), "scores": scores.round(3).tolist()}

    (out / "results.json").write_text(json.dumps(log, indent=1))
    print(f"done. images with no detection: {len(no_det)}")
    for n in no_det:
        print("  -", n)


if __name__ == "__main__":
    main()
