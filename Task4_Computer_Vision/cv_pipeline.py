"""Task 4: Real-time shape detector & frame-by-frame segmenter (OpenCV)
Pipeline: grayscale -> Gaussian blur -> adaptive threshold -> morphology -> contours -> shape classification
-> overlay boxes + live count text -> per-frame metrics."""
import cv2, json, time, math, argparse, csv
import numpy as np

COLORS = {"circle": (0, 160, 0), "rectangle": (200, 100, 0), "triangle": (0, 0, 220), "other": (120, 120, 120)}

def classify(cnt):
    """Shape from polygon approximation + circularity."""
    area, peri = cv2.contourArea(cnt), cv2.arcLength(cnt, True)
    if peri == 0: return "other"
    circ = 4 * math.pi * area / peri ** 2
    v = len(cv2.approxPolyDP(cnt, 0.04 * peri, True))
    if circ > 0.82 and v > 5: return "circle"
    if v == 3: return "triangle"
    if v == 4: return "rectangle"
    return "circle" if circ > 0.80 else "other"

def process(frame, blur_k=5, block=31, C=8, min_area=350, max_area=20000):
    """Returns annotated frame, detections, binary mask, stage timings (ms)."""
    t = {}
    t0 = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t["gray"] = time.perf_counter(); 
    g = cv2.GaussianBlur(gray, (blur_k, blur_k), 0) if blur_k > 1 else gray
    t["blur"] = time.perf_counter()
    mask = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, C)
    t["thresh"] = time.perf_counter()
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    t["morph"] = time.perf_counter()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for c in cnts:
        a = cv2.contourArea(c)
        if min_area < a < max_area:
            x, y, w, h = cv2.boundingRect(c)
            M = cv2.moments(c)
            dets.append({"cls": classify(c), "cx": M["m10"] / M["m00"], "cy": M["m01"] / M["m00"],
                         "box": (x, y, w, h), "area": a})
    t["contours"] = time.perf_counter()
    out = frame.copy()
    counts = {k_: 0 for k_ in ["circle", "rectangle", "triangle"]}
    for d in dets:
        x, y, w, h = d["box"]; col = COLORS[d["cls"]]
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        cv2.putText(out, d["cls"], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        if d["cls"] in counts: counts[d["cls"]] += 1
    panel = out.copy(); cv2.rectangle(panel, (0, 0), (250, 70), (255, 255, 255), -1)   # dynamic text-count overlay
    out = cv2.addWeighted(panel, 0.75, out, 0.25, 0)
    for i, (name, n) in enumerate(counts.items()):
        cv2.putText(out, f"{name.capitalize()}s: {n}", (8, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, COLORS[name], 2, cv2.LINE_AA)
    keys = list(t); prev = t0; timing = {}
    for kname in keys:
        timing[kname] = (t[kname] - prev) * 1000; prev = t[kname]
    timing["total"] = (time.perf_counter() - t0) * 1000
    return out, dets, mask, timing

def match(dets, gt, tol=25):
    """Greedy match of detections to ground truth (same class, centroid within tol px)."""
    used, tp = set(), 0
    for d in dets:
        best, bd = None, tol
        for i, g in enumerate(gt):
            if i in used or g["cls"] != d["cls"]: continue
            dist = math.hypot(d["cx"] - g["cx"], d["cy"] - g["cy"])
            if dist < bd: best, bd = i, dist
        if best is not None: used.add(best); tp += 1
    return tp, len(dets) - tp, len(gt) - tp   # TP, FP, FN

def run(source, gt=None, save_video=None, **params):
    cap = cv2.VideoCapture(source)
    rows, writer, f = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        out, dets, mask, tm = process(frame, **params)
        row = {"frame": f, **{f"ms_{k}": round(v, 3) for k, v in tm.items()}, "fps": round(1000 / tm["total"], 1)}
        for c in ["circle", "rectangle", "triangle"]:
            row[f"det_{c}"] = sum(d["cls"] == c for d in dets)
        if gt:
            tp, fp, fn = match(dets, gt[f])
            row.update(tp=tp, fp=fp, fn=fn,
                       count_exact=int(all(row[f"det_{c}"] == sum(g["cls"] == c for g in gt[f])
                                           for c in ["circle", "rectangle", "triangle"])))
        rows.append(row)
        if save_video:
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            writer.write(out)
        if f in (0, 60, 120, 180):
            cv2.imwrite(f"outputs/frames/frame{f}_original.png", frame)
            cv2.imwrite(f"outputs/frames/frame{f}_mask.png", mask)
            cv2.imwrite(f"outputs/frames/frame{f}_annotated.png", out)
        f += 1
    cap.release()
    if writer: writer.release()
    return rows

def summarize(rows):
    n = len(rows); s = {"frames": n}
    s["mean_ms"] = float(np.mean([r["ms_total"] for r in rows]))
    s["p95_ms"] = float(np.percentile([r["ms_total"] for r in rows], 95))
    s["fps"] = 1000 / s["mean_ms"]
    if "tp" in rows[0]:
        TP, FP, FN = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
        s.update(precision=TP / max(TP + FP, 1), recall=TP / max(TP + FN, 1),
                 exact_count_acc=float(np.mean([r["count_exact"] for r in rows])),
                 TP=TP, FP=FP, FN=FN)
        s["f1"] = 2 * s["precision"] * s["recall"] / max(s["precision"] + s["recall"], 1e-9)
    return s

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/test_video.mp4", help="video path or 0 for webcam")
    ap.add_argument("--gt", default="data/ground_truth.json")
    ap.add_argument("--out", default="outputs/annotated_output.mp4")
    a = ap.parse_args()
    src = int(a.source) if a.source.isdigit() else a.source
    try: gt = json.load(open(a.gt))
    except Exception: gt = None
    rows = run(src, gt, a.out)
    with open("outputs/frame_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(json.dumps(summarize(rows), indent=2))
