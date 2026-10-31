# Task 4: Real-Time Shape Detector & Frame-by-Frame Segmenter (OpenCV)

Detects circles, rectangles and triangles in video, draws boxes, and overlays **live count text** on every frame.

## Pipeline
Grayscale -> **Gaussian filter** -> **Adaptive threshold** -> morphology -> contours -> shape classification -> overlay counts -> per-frame metrics

## Structure
```
task4_vision/
├── make_video.py       # synthetic test video with exact ground truth
├── cv_pipeline.py      # the pipeline (works on video file or webcam)
├── experiments.py      # evaluation, noise x blur ablation, figures
├── build_report.py     # builds the whitepaper PDF
├── data/               # ground_truth.json (test_video.mp4 is created by make_video.py)
├── outputs/
│   ├── annotated_output.mp4   # result video with boxes + count overlay
│   ├── frame_metrics.csv      # per-frame latency, FPS, counts, TP/FP/FN
│   ├── results.json, fig_*.png, frames/
│   └── Task4_Whitepaper.pdf   # 6-page performance whitepaper
└── requirements.txt
```

## Run
```bash
pip install -r requirements.txt
python make_video.py
python cv_pipeline.py                  # on the test video
python cv_pipeline.py --source 0       # live webcam
python cv_pipeline.py --source my.mp4 --gt none   # your own video (no ground truth)
python experiments.py && python build_report.py
```

## Results (240-frame synthetic video)
| Metric | Value |
|---|---|
| Precision / Recall / F1 | 0.936 / 0.870 / 0.901 |
| Mean latency | 4.1 ms/frame (~245 FPS, CPU) |
| Exact-count accuracy | 45.4% (strict: all 3 counts right in a frame) |
| Per-class precision | 1.000 for all classes |

Main error source: touching or overlapping shapes merge into one blob. The Gaussian filter matters at high noise (F1 0.777 without vs 0.860 with, at sigma 25).

## Notes
- Evaluation uses a synthetic video, so real-world footage will give lower accuracy (see limitations in the whitepaper).
- YOLO was not used; the task allows OpenCV **or** a pre-trained model. YOLO is suggested as future work.
