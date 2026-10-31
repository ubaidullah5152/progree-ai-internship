"""Evaluation: main run, per-class metrics, noise x blur ablation, figures for the whitepaper."""
import json, csv, math
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_video import make_video
from cv_pipeline import run, summarize, process, match

CL = ["circle", "rectangle", "triangle"]
gt = json.load(open("data/ground_truth.json"))
rows = run("data/test_video.mp4", gt, "outputs/annotated_output.mp4")
with open("outputs/frame_metrics.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
main = summarize(rows)

# ---- per-class precision/recall (re-run detection to keep class labels) ----
cap = cv2.VideoCapture("data/test_video.mp4"); per = {c: [0, 0, 0] for c in CL}; f = 0
while True:
    ok, fr = cap.read()
    if not ok: break
    _, dets, _, _ = process(fr)
    for c in CL:
        tp, fp, fn = match([d for d in dets if d["cls"] == c], [g for g in gt[f] if g["cls"] == c])
        per[c][0] += tp; per[c][1] += fp; per[c][2] += fn
    f += 1
perclass = {c: {"precision": v[0] / max(v[0] + v[1], 1), "recall": v[0] / max(v[0] + v[2], 1),
                "f1": 2 * v[0] / max(2 * v[0] + v[1] + v[2], 1), "TP": v[0], "FP": v[1], "FN": v[2]}
            for c, v in per.items()}

# ---- ablation: noise level x Gaussian kernel ----
noises, kernels = [5, 12, 25], [1, 3, 5, 9]
abl = {}
for nz in noises:
    path = f"data/abl_noise{nz}.mp4"
    g2 = make_video(path, n_frames=100, noise=nz, seed=7)
    for k in kernels:
        s = summarize(run(path, g2, None, blur_k=k))
        abl[(nz, k)] = s
        print(f"noise={nz:2d} blur={k}: F1={s['f1']:.3f} P={s['precision']:.3f} R={s['recall']:.3f} ms={s['mean_ms']:.2f}")

json.dump({"main": main, "perclass": perclass,
           "ablation": {f"{n}_{k}": v for (n, k), v in abl.items()}}, open("outputs/results.json", "w"), indent=1)

# ---- figures ----
# 1 pipeline stages for one frame
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, (nm, t) in zip(axs, [("original", "Original frame"), ("mask", "Adaptive threshold mask"), ("annotated", "Detections + counts")]):
    im = cv2.imread(f"outputs/frames/frame120_{nm}.png")
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.set_title(t, fontsize=10); ax.axis("off")
plt.tight_layout(); plt.savefig("outputs/fig_stages.png", dpi=130); plt.close()

# 2 count timeline
fr = [r["frame"] for r in rows]
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
for ax, c, n in zip(axs, CL, [3, 3, 2]):
    ax.plot(fr, [r[f"det_{c}"] for r in rows], lw=1, label="detected"); ax.axhline(n, color="r", ls="--", label="ground truth")
    ax.set_title(f"{c.capitalize()} count per frame", fontsize=10); ax.set_xlabel("frame"); ax.set_ylim(0, 6); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("outputs/fig_counts.png", dpi=130); plt.close()

# 3 latency stages + per-frame latency
stages = ["gray", "blur", "thresh", "morph", "contours"]
means = [np.mean([r[f"ms_{s}"] for r in rows]) for s in stages]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.2))
axs[0].bar(stages, means, color="#6c8ebf"); axs[0].set_title("Mean time per stage (ms)", fontsize=10)
axs[1].plot(fr, [r["ms_total"] for r in rows], lw=0.8); axs[1].set_title("Total latency per frame (ms)", fontsize=10); axs[1].set_xlabel("frame")
plt.tight_layout(); plt.savefig("outputs/fig_latency.png", dpi=130); plt.close()

# 4 ablation
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
for nz in noises:
    axs[0].plot(kernels, [abl[(nz, k)]["f1"] for k in kernels], marker="o", label=f"noise sigma={nz}")
    axs[1].plot(kernels, [abl[(nz, k)]["mean_ms"] for k in kernels], marker="o", label=f"noise sigma={nz}")
axs[0].set_title("F1 vs Gaussian kernel size", fontsize=10); axs[1].set_title("Latency (ms) vs kernel size", fontsize=10)
for a in axs: a.set_xlabel("kernel size (1 = no blur)"); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig("outputs/fig_ablation.png", dpi=130); plt.close()
print(json.dumps(main, indent=1)); print(json.dumps(perclass, indent=1))
