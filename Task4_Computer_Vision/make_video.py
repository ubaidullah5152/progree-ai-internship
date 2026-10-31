"""Creates a synthetic test video with KNOWN ground truth (so we can measure accuracy)."""
import cv2, json, math, numpy as np

CLASSES = ["circle", "rectangle", "triangle"]

def make_video(path, n_frames=240, noise=12, seed=7, size=(640, 480), gt_path=None):
    rng = np.random.default_rng(seed)
    W, H = size
    # 3 circles, 3 rectangles, 2 triangles
    objs = []
    for cls, k in [("circle", 3), ("rectangle", 3), ("triangle", 2)]:
        for _ in range(k):
            objs.append({"cls": cls, "x": rng.uniform(80, W - 80), "y": rng.uniform(80, H - 80),
                         "vx": rng.uniform(-3, 3), "vy": rng.uniform(-3, 3), "ang": rng.uniform(0, 6.28),
                         "va": rng.uniform(-0.05, 0.05), "s": rng.uniform(26, 38),
                         "col": int(rng.integers(30, 90))})
    # soft background gradient (uneven lighting -> why adaptive threshold is needed)
    xs = np.linspace(185, 235, W)[None, :].repeat(H, 0)
    ys = np.linspace(0, -30, H)[:, None].repeat(W, 1)
    bg = np.clip(xs + ys, 0, 255).astype(np.uint8)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    gt = []
    for f in range(n_frames):
        img = cv2.cvtColor(bg.copy(), cv2.COLOR_GRAY2BGR)
        frame_gt = []
        for o in objs:
            o["x"] += o["vx"]; o["y"] += o["vy"]; o["ang"] += o["va"]
            m = o["s"] + 5
            if not m < o["x"] < W - m: o["vx"] *= -1; o["x"] = min(max(o["x"], m), W - m)
            if not m < o["y"] < H - m: o["vy"] *= -1; o["y"] = min(max(o["y"], m), H - m)
            c = (o["col"], o["col"], o["col"] + 20); cx, cy, s, a = o["x"], o["y"], o["s"], o["ang"]
            if o["cls"] == "circle":
                cv2.circle(img, (int(cx), int(cy)), int(s * 0.85), c, -1)
            elif o["cls"] == "rectangle":
                pts = [(cx + s * math.cos(a + t), cy + s * math.sin(a + t)) for t in (0.6, 2.54, 3.74, 5.68)]
                cv2.fillPoly(img, [np.array(pts, np.int32)], c)
            else:
                pts = [(cx + s * 1.2 * math.cos(a + t), cy + s * 1.2 * math.sin(a + t)) for t in (0, 2.094, 4.189)]
                cv2.fillPoly(img, [np.array(pts, np.int32)], c)
            frame_gt.append({"cls": o["cls"], "cx": cx, "cy": cy})
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
        vw.write(img); gt.append(frame_gt)
    vw.release()
    if gt_path: json.dump(gt, open(gt_path, "w"))
    return gt

if __name__ == "__main__":
    make_video("data/test_video.mp4", gt_path="data/ground_truth.json")
    print("video created")
