# Egg segmentation and tracking (CV intern task)

Segmentation of eggs with a pretrained model (Task 1), with classical image processing (Task 2),
the evaluation of the classical method against the pretrained one (Task 3), and tracking of two
eggs in a video (Task 4).

- **Report (PDF):** [`report/report.pdf`](report/report.pdf)
- **Tracked video (public Google Drive link):** https://drive.google.com/file/d/1xBU2eQbkeQ98u3YnsAMKMuvQ8GlEF2BZ/view?usp=sharing

## Results in short

| Task | What | Result |
|---|---|---|
| 1 | Grounding DINO + SAM, text prompt `an egg.` | 110 images, 138 eggs, an egg found in every image |
| 2 | Classical pipeline, 16 scored trials (T0 to T11) | dev (88 images): mean IoU 0.714, median 0.807; eval (22 images): mean IoU 0.634, median 0.669 |
| 3 | Evaluation on the eval split | instance level: 21 TP, 7 FP, 6 FN (F1 0.76); the main failures are listed in the report |
| 4 | Video tracking | IDs 1 and 2 only, no frame with more than 2 masks, 380 frames processed |

Task 1 masks are model predictions, not human labels, and are used as ground truth in Tasks 2 and 3.

## Repository layout

| File | Task | Purpose |
|---|---|---|
| `task1_sam_segment.py` | 1 | Grounding DINO + SAM instance masks and red overlays |
| `egg_pipeline.py` | 2, 4 | the final classical pipeline (candidates, scoring, acceptance, GrabCut) |
| `task2_split.py` | 2 | scene groups and the leakage-safe 4:1 dev/eval split |
| `task2_segment.py` | 2 | run the final pipeline on a folder of images |
| `task2_experiments.py` | 2 | reproduce the dev scores of the trials T0 to T11 |
| `task3_evaluate.py` | 3 | metrics, plots and the overlays of the worst images |
| `task4_track.py` | 4 | detection + tracking in the video, annotated output and track log |
| `results/` | all | masks, split, experiment log and metrics produced during the work |
| `report/` | all | the PDF report |

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # Tasks 2, 3 and 4
pip install -r requirements-extra.txt  # Task 1 (torch, transformers), the split script, the video download
```

Task 1 needs a GPU for reasonable speed (it ran on a Colab T4). Tasks 2 to 4 run on a CPU.
OpenCV's GrabCut can differ slightly between versions; the numbers in the report were obtained
with OpenCV 5.0 on Colab.

## Data

The dataset (110 images, 640x640) is the zip linked in the task sheet (Roboflow export).

```bash
pip install gdown
gdown --fuzzy "https://drive.google.com/file/d/1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv/view" -O dataset.zip
unzip -q dataset.zip -d data          # data/train, data/valid, data/test
```

The scripts search `data/` recursively and ignore the Roboflow train/valid/test folders.

## Run

### Task 1: segmentation with a pretrained model

```bash
python task1_sam_segment.py --images data --out outputs/task1
```

Writes `outputs/task1/masks/*.png` (instance masks), `outputs/task1/overlays/*.jpg` and
`results.json`. The masks used in the report are in `results/task1_masks/`, so Tasks 2 and 3 can
be reproduced without a GPU.

### Task 2: classical segmentation

```bash
# the split used in the report is results/split.csv; this recomputes it (needs torch):
python task2_split.py --data data --masks results/task1_masks

# run the final pipeline on the eval images
python task2_segment.py --images data --out outputs/task2_pred --split results/split.csv --which eval

# reproduce the dev scores of the trials (some trials take a few minutes each on a CPU)
python task2_experiments.py --data data --masks results/task1_masks --split results/split.csv --trials all
```

`task2_experiments.py` prints the mean IoU, the median and the number of dev images below 0.5 for
every trial. The full log with the change, the reason, the parameters and the decision of every
trial is in `results/experiment_log.csv` and in the report. GrabCut uses random initialisation,
so the original trials differ by about 0.005 mean IoU from a rerun; the final pipeline fixes the
seed (`egg_pipeline.segment(..., seed=0)`).

### Task 3: evaluation

```bash
python task3_evaluate.py --images data --gt results/task1_masks --pred outputs/task2_pred \
    --split results/split.csv --which eval \
    --dev-iou results/per_image_iou.csv --dev-column FINAL_dev --out results/task3_reproduced
```

### Task 4: video tracking

Video: https://youtu.be/Y4JyV94y2_A (Mocomi, Raw and Boiled Egg Spinning). YouTube may block
downloads from cloud notebooks; on a laptop the following worked (it needs Deno for the JavaScript
challenge):

```bash
yt-dlp --remote-components ejs:github -f b -o video.mp4 "https://youtu.be/Y4JyV94y2_A"
python task4_track.py --video video.mp4 --out results/tracked_reproduced.mp4 \
    --log results/track_log_reproduced.csv --start 240 --end 1000 --step 2
```

Frames 240 to 1000 (10 s to 41.7 s) contain the eggs; every second frame is processed. The output
has Egg 1 in red and Egg 2 in green. The video settings (candidate minimum area 1 %, up to 6
blobs, the shirt and logo filters) were set for this video; see the report for the trials that led
to them.

## AI tools

I used Claude (Anthropic) as a coding and analysis assistant in all four tasks: to compare
approaches, to draft the code and the diagnostics, and to help write the report. I ran every
experiment myself, looked at the outputs, and made the keep or discard decisions.
