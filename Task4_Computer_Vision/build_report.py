import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

R = json.load(open("outputs/results.json")); M, P, A = R["main"], R["perclass"], R["ablation"]
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=14, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#1f3a5f"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=6, spaceAfter=4, textColor=colors.HexColor("#1f3a5f"))
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10.5, leading=15, alignment=4, spaceAfter=6)
CAP = ParagraphStyle("CAP", parent=ss["Italic"], fontSize=9, alignment=1, textColor=colors.grey, spaceAfter=8)
TT = ParagraphStyle("TT", parent=ss["Title"], fontSize=22, leading=27, textColor=colors.HexColor("#1f3a5f"))
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=11, alignment=1, textColor=colors.grey, spaceAfter=4)
CODE = ParagraphStyle("CODE", parent=ss["Code"], fontSize=8.5, leading=11, backColor=colors.HexColor("#f3f3f3"), borderPadding=5, spaceAfter=8)

def tbl(data, widths=None):
    t = Table(data, colWidths=widths, hAlign="CENTER")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t
def img(path, w=16.5): 
    from PIL import Image as PI
    iw, ih = PI.open(path).size
    return Image(path, width=w * cm, height=w * cm * ih / iw)
p = lambda t: Paragraph(t, B)
S = []

# ---- Page 1: title, abstract, introduction ----
S += [Spacer(1, 1.2 * cm), Paragraph("Real-Time Shape Detection and Frame-by-Frame Segmentation Pipeline", TT),
      Paragraph("Model Performance Whitepaper", SUB), Paragraph("Artificial Intelligence Internship, Progree", SUB),
      Paragraph("Prepared by: Ubaid Ullah", SUB), Spacer(1, 0.6 * cm)]
S += [Paragraph("Abstract", H1),
      p(f"This whitepaper presents a classical computer vision pipeline built with OpenCV that detects, classifies and counts geometric shapes "
        f"(circles, rectangles, triangles) in video, frame by frame. The pipeline applies a Gaussian filter and an adaptive Gaussian threshold to "
        f"segment objects under uneven lighting, extracts contours, classifies each contour by polygon approximation and circularity, and overlays "
        f"dynamic per-class counts on every frame. It was evaluated on a 240-frame synthetic video with known ground truth, which allows exact "
        f"measurement of detection quality. The system reached an F1-score of <b>{M['f1']:.3f}</b> (precision {M['precision']:.3f}, recall {M['recall']:.3f}) "
        f"at a mean latency of <b>{M['mean_ms']:.1f} ms per frame</b> (about {M['fps']:.0f} FPS) on CPU. Errors were traced almost entirely to "
        f"touching or overlapping objects that merge into one blob. A noise-versus-blur ablation shows the Gaussian filter is essential at high noise."),
      Paragraph("1. Introduction", H1),
      p("Object detection and segmentation are core computer vision tasks. Deep models such as YOLO are powerful but need trained weights and more compute. "
        "For well-contrasted objects, a classical pipeline built from filtering, thresholding and contour analysis can be fast, transparent and "
        "easy to debug. The objective of this project is to build such a pipeline that works in real time, tracks target shapes through a video stream, "
        "overlays live counts, and is measured rigorously frame by frame."),
      p("The main contributions are: (1) an end-to-end OpenCV pipeline that runs on a video file or a webcam; (2) a synthetic benchmark with exact ground truth; "
        "(3) a frame-by-frame evaluation of accuracy and latency; and (4) an ablation study of noise level against Gaussian kernel size."),
      PageBreak()]

# ---- Page 2-3: method ----
S += [Paragraph("2. Methodology", H1), p("Each frame passes through the following stages. The same code runs on a recorded video or a live webcam stream (<font face='Courier'>--source 0</font>)."),
      Paragraph("2.1 Pipeline stages", H2),
      tbl([["Stage", "OpenCV function", "Purpose"],
           ["1. Grayscale", "cvtColor", "Remove colour, reduce data 3x"],
           ["2. Gaussian filter", "GaussianBlur (5x5)", "Suppress sensor noise before thresholding"],
           ["3. Adaptive threshold", "adaptiveThreshold (block 31, C 8)", "Local threshold, robust to uneven lighting"],
           ["4. Morphology", "morphologyEx open + close (3x3)", "Remove specks, fill small holes"],
           ["5. Contours", "findContours (external)", "Extract shape outlines (pixel bounds)"],
           ["6. Filtering", "contourArea (350 to 20000 px)", "Reject noise blobs and huge regions"],
           ["7. Classification", "approxPolyDP + circularity", "Label circle, rectangle, triangle"],
           ["8. Overlay", "rectangle, putText", "Draw boxes and dynamic count vector"]],
          [3.8 * cm, 7.4 * cm, 5.9 * cm]), Spacer(1, 0.3 * cm),
      Paragraph("2.2 Why adaptive thresholding", H2),
      p("A global threshold uses one brightness cut-off for the whole image, which fails when lighting varies. Adaptive thresholding computes a separate threshold "
        "for every pixel from the Gaussian-weighted mean of its neighbourhood (block size 31 pixels) minus a constant C. The test video has a deliberate brightness "
        "gradient across the background, so this choice is necessary."),
      Paragraph("2.3 Shape classification rule", H2),
      p("For each contour the perimeter and area give the circularity <font face='Courier'>4*pi*area / perimeter^2</font>, which equals 1.0 for a perfect circle. "
        "The contour is simplified with approxPolyDP (epsilon = 4% of the perimeter) and the vertex count v is read: "
        "circularity above 0.82 with more than 5 vertices gives <b>circle</b>; v = 3 gives <b>triangle</b>; v = 4 gives <b>rectangle</b>; anything else is <b>other</b> "
        "(not counted in the class totals). The counts are drawn as a text block in the top-left corner and update every frame."),
      Paragraph("2.4 Core code", H2),
      Paragraph("gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)<br/>g = cv2.GaussianBlur(gray, (5, 5), 0)<br/>"
                "mask = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,<br/>&nbsp;&nbsp;&nbsp;&nbsp;cv2.THRESH_BINARY_INV, 31, 8)<br/>"
                "cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)", CODE),
      img("outputs/fig_stages.png"), Paragraph("Figure 1. Original frame, adaptive-threshold mask, and annotated output with live counts.", CAP),
      PageBreak()]

# ---- Page 3-4: experimental setup + results ----
S += [Paragraph("3. Experimental Setup", H1),
      p("<b>Dataset.</b> A synthetic 640x480, 30 FPS, 240-frame video was generated with fixed random seed 7. It contains 8 moving, rotating, bouncing objects "
        "(3 circles, 3 rectangles, 2 triangles), a brightness gradient background and Gaussian pixel noise (sigma = 12). Because the generator records every object's "
        "position, the ground truth is exact for all 1,920 object instances."),
      p("<b>Metrics.</b> A detection is a true positive (TP) if it has the same class as a ground-truth object and its centroid lies within 25 pixels "
        "(greedy one-to-one matching). Unmatched detections are false positives (FP); unmatched ground-truth objects are false negatives (FN). "
        "Precision = TP/(TP+FP), recall = TP/(TP+FN), F1 is their harmonic mean. <i>Exact-count accuracy</i> is the share of frames where all three class counts equal the truth. "
        "Latency is measured per frame with <font face='Courier'>time.perf_counter</font> on CPU (single thread, no GPU)."),
      Paragraph("4. Results", H1), Paragraph("4.1 Overall detection quality", H2),
      tbl([["Metric", "Value"], ["Frames evaluated", f"{M['frames']}"], ["True positives / FP / FN", f"{M['TP']} / {M['FP']} / {M['FN']}"],
           ["Precision", f"{M['precision']:.3f}"], ["Recall", f"{M['recall']:.3f}"], ["F1-score", f"{M['f1']:.3f}"],
           ["Exact-count accuracy (all 3 classes)", f"{M['exact_count_acc']*100:.1f}%"]], [8 * cm, 4 * cm]), Spacer(1, 0.3 * cm),
      Paragraph("4.2 Per-class results", H2),
      tbl([["Class", "Precision", "Recall", "F1", "TP", "FN"]] + [[c.capitalize(), f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1']:.3f}", v["TP"], v["FN"]] for c, v in P.items()],
          [3.5 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.2 * cm, 2.2 * cm]), Spacer(1, 0.2 * cm),
      p("Per-class precision is 1.000: whenever the system labels an object as a circle, rectangle or triangle at the right place, the label is correct. "
        "All 115 overall false positives are blobs labelled <i>other</i>, which come from merged objects. Circles are easiest (recall 0.926), because their silhouette stays "
        "the same under rotation. Rectangles have the lowest recall (0.811)."),
      img("outputs/fig_counts.png"), Paragraph("Figure 2. Detected count per class versus ground truth for every frame.", CAP),
      PageBreak()]

# ---- Page 4-5: latency, ablation ----
lat = {k: v for k, v in zip(["gray", "blur", "thresh", "morph", "contours"], [0] * 5)}
S += [Paragraph("4.3 Frame-by-frame latency", H2),
      p(f"The mean total time is <b>{M['mean_ms']:.2f} ms</b> per frame and the 95th percentile is {M['p95_ms']:.2f} ms, so even the slow frames are far inside the 33 ms "
        f"budget of a 30 FPS stream. This gives roughly {M['fps']:.0f} FPS of pure processing speed, about 8 times faster than real time, which leaves room for tracking, "
        f"video decoding and display. Figure 3 shows where time is spent. The adaptive threshold is the most expensive processing stage (about 1.9 ms of the 4.1 ms), followed by contour extraction (0.4 ms) and morphology (0.3 ms); grayscale conversion and blur cost under 0.2 ms each. The remaining time (about 1.1 ms) is spent on drawing boxes, labels and the count overlay. Latency is stable across frames; spikes come from operating-system scheduling."),
      img("outputs/fig_latency.png", 15), Paragraph("Figure 3. Mean time per pipeline stage (left) and total latency per frame (right).", CAP),
      Paragraph("4.4 Ablation: noise level and Gaussian kernel size", H2),
      p("The video generator was re-run at three noise levels (sigma 5, 12, 25) with 100 frames each, and the pipeline was evaluated with Gaussian kernel sizes 1 (no filter), 3, 5 and 9."),
      tbl([["Noise sigma", "k=1 (no blur)", "k=3", "k=5", "k=9"]] + [[str(n)] + [f"{A[f'{n}_{k}']['f1']:.3f}" for k in (1, 3, 5, 9)] for n in (5, 12, 25)],
          [3.2 * cm, 3.4 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm]),
      Paragraph("Table 3. F1-score for each noise level and Gaussian kernel size.", CAP),
      img("outputs/fig_ablation.png", 14),
      Paragraph("Figure 4. F1-score and latency against kernel size for three noise levels.", CAP),
      p(f"At low and medium noise the filter makes almost no difference (F1 about 0.85 to 0.86). At sigma = 25, removing the filter drops F1 from 0.860 to <b>{A['25_1']['f1']:.3f}</b>, "
        "because noisy pixels create many small false contours; any kernel of 3 or larger restores performance. A 5x5 kernel is the best choice: it gives the full benefit "
        "with no measurable accuracy loss and negligible extra time. Latency differences between kernels are within timing noise."),
      PageBreak()]

# ---- Page 5-6: discussion ----
S += [Paragraph("5. Discussion", H1), Paragraph("5.1 Error analysis", H2),
      p("Exact-count accuracy is only 45.4% even though F1 is 0.90. The reason is that the metric is strict: a single merged pair in a frame makes that whole frame wrong. "
        "In 180 of the 240 frames at least two objects are within 70 pixels of each other, and in 110 frames the pipeline produced at least one blob labelled <i>other</i>. "
        "When two shapes touch, the external contour of the combined blob has an irregular polygon, so it matches no class, and both objects are lost (false negatives) while one "
        "false positive appears. This explains why recall (0.870) is lower than precision (0.936) and why all false positives are of type <i>other</i>."),
      Paragraph("5.2 Classical pipeline versus deep learning", H2),
      tbl([["Aspect", "This pipeline", "YOLO-type detector"], ["Speed (CPU)", f"{M['mean_ms']:.1f} ms/frame", "tens of ms, usually needs GPU"],
           ["Training data", "None", "Thousands of labelled images"], ["Handles overlap", "Poorly", "Well (learned occlusion)"],
           ["Handles texture / real scenes", "Poorly", "Well"], ["Explainability", "Every step is visible", "Black box"]],
          [4.2 * cm, 5.4 * cm, 6.7 * cm]), Spacer(1, 0.3 * cm),
      p("The classical approach is the right tool for simple, high-contrast objects (for example parts on a conveyor belt, markers, or shape-based games) where speed and "
        "transparency matter. For general real-world objects, a pre-trained YOLO model would be the better choice, and it can be dropped into the same frame loop to replace stages 2 to 7."),
      Paragraph("5.3 Limitations", H2),
      p("(1) Results are from a synthetic video with dark shapes on a light background; real footage has shadows, texture and colour variation and would lower accuracy. "
        "(2) Overlapping shapes are not separated. (3) Classification relies on fixed thresholds (circularity 0.82, 4% polygon epsilon) tuned by hand. "
        "(4) There is no object tracking, so counts are per frame, not unique objects over time. (5) Latency was measured on a single CPU and will differ on other hardware."),
      Paragraph("6. Future Work", H1),
      p("Planned improvements: a distance-transform plus watershed step to split touching objects; centroid tracking to give stable object IDs and cumulative counts; "
        "evaluation on real recorded video; replacing the classifier with a pre-trained YOLO model and comparing mAP and latency on the same video; and automatic threshold tuning."),
      Paragraph("7. Conclusion", H1),
      p(f"A lightweight OpenCV pipeline built from a Gaussian filter, adaptive thresholding and contour analysis detected and classified shapes in real time with an F1-score of "
        f"{M['f1']:.2f} at {M['mean_ms']:.1f} ms per frame. Per-class precision was perfect, the Gaussian filter proved essential under heavy noise, and nearly all remaining errors "
        f"came from touching objects. The project shows that a simple, explainable pipeline is fast and accurate for clean scenes, and it identifies clear next steps for harder ones."),
      Paragraph("8. How to Reproduce", H1),
      Paragraph("pip install -r requirements.txt<br/>python make_video.py&nbsp;&nbsp;&nbsp;# creates data/test_video.mp4 + ground truth<br/>"
                "python cv_pipeline.py&nbsp;&nbsp;&nbsp;# annotated video + per-frame metrics<br/>python experiments.py&nbsp;&nbsp;&nbsp;# evaluation, ablation, figures<br/>"
                "python cv_pipeline.py --source 0&nbsp;&nbsp;&nbsp;# live webcam", CODE),
      Paragraph("References", H1),
      p("[1] OpenCV documentation: Image Thresholding, Contour Features, Structural Analysis (docs.opencv.org).<br/>"
        "[2] Bradski, G. The OpenCV Library. Dr. Dobb's Journal, 2000.<br/>"
        "[3] Redmon, J. et al. You Only Look Once: Unified, Real-Time Object Detection. CVPR, 2016.")]

def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.grey)
    c.drawString(2 * cm, 1.2 * cm, "Shape Detection Pipeline Whitepaper | Progree AI Internship"); c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {d.page}"); c.restoreState()

doc = SimpleDocTemplate("outputs/Task4_Whitepaper.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="Real-Time Shape Detection Whitepaper", author="Ubaid Ullah")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
