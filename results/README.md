# results/

Files produced while doing the tasks (copy them here from the Google Drive folder before committing).

| File / folder | Produced by | Content |
|---|---|---|
| `task1_masks/` | `task1_sam_segment.py` | Task 1 instance masks of the 110 images (uint16 PNG, 0 = background, 1..N = egg). They are the ground truth of Tasks 2 and 3 |
| `split.csv` | `task2_split.py` | dev/eval assignment, scene group and egg count of every image |
| `groups.csv` | `task2_split.py` | scene group of every image |
| `experiment_log.csv` | notebook | the experiment log T0 to T11 (change, why, parameters, dev results, decision, reason) |
| `per_image_iou.csv` | notebook | dev IoU of every image for every trial |
| `task2_eval_pred/` | `task2_segment.py` | predicted masks of the 22 eval images (255 = egg) |
| `eval_results.csv` | notebook | eval IoU per image |
| `eval_metrics.csv` | `task3_evaluate.py` | all Task 3 metrics per eval image |
| `task3_plots.png`, `task3_worst12.png` | `task3_evaluate.py` | figures of the Task 3 report |
| `track_log_v2.csv` | `task4_track.py` | frame, egg ID, x, y of the final tracking run |

The tracked video is too large for the repository; its public Google Drive link is in the report and in the main README.
