#!/usr/bin/env python3
"""
DecodeLabs AI Project 4 - Image / Text Recognition (Basic)
==========================================================
Two paths, one CLI:
  Path 1  OCR              : pytesseract + OpenCV pre-processing
  Path 2  Object detection : OpenCV DNN + YOLOv4-tiny (COCO, auto-downloaded, free)

Validation gates (from the kit):
  1. Library integration     -> pytesseract / cv2.dnn
  2. Pre-processing integrity-> grayscale + blur + deskew + adaptive threshold
  3. Accuracy benchmark      -> confidence >= 80 %
  4. Visual confirmation     -> annotated image written to outputs/

Usage:
  python recognition.py ocr    samples/sample_text.png
  python recognition.py detect samples/street.jpg
  python recognition.py both   samples/image.jpg
"""
import argparse, json, os, sys, urllib.request
from pathlib import Path

import cv2
import numpy as np

CONF_THRESHOLD = 0.80          # kit minimum standard
OUT_DIR = Path("outputs")
MODEL_DIR = Path("models")


# ----------------------------------------------------------------------------
# PATH 1 : OCR
# ----------------------------------------------------------------------------
def preprocess(img_bgr, save_steps=None):
    """Grayscale -> auto-invert -> denoise -> deskew -> adaptive + Otsu threshold."""
    h, w = img_bgr.shape[:2]
    if max(h, w) < 1000:                                   # upscale small images
        k = 1000 / max(h, w)
        img_bgr = cv2.resize(img_bgr, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)       # step 1
    if gray.mean() < 110:                                  # light text on dark bg -> invert
        gray = 255 - gray
        img_bgr = 255 - img_bgr
    clean = cv2.GaussianBlur(cv2.medianBlur(gray, 5), (3, 3), 0)   # step 2: kills salt & pepper
    angle = estimate_skew(clean)                           # step 3
    if abs(angle) > 0.3:
        clean = rotate(clean, angle)
        img_bgr = rotate(img_bgr, angle, border=(255, 255, 255))
    adaptive = cv2.adaptiveThreshold(                      # step 4 (uneven light / shadows)
        clean, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    otsu = cv2.threshold(clean, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if save_steps:
        for name, im in (("1_gray", gray), ("2_blur", clean), ("3_binary", adaptive)):
            cv2.imwrite(str(save_steps / f"{name}.png"), im)
    return img_bgr, {"adaptive": adaptive, "otsu": otsu}, angle


def estimate_skew(gray):
    inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(inv > 0))
    if len(coords) < 50:
        return 0.0
    ang = cv2.minAreaRect(coords.astype(np.float32))[-1]
    ang = -(90 + ang) if ang < -45 else -ang
    if ang > 45:  ang -= 90
    if ang < -45: ang += 90
    return float(ang)


def rotate(im, angle, border=255):
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def configure_tesseract():
    """Find tesseract.exe automatically (PATH, env var TESSERACT_CMD, or common install folders)."""
    import shutil, pytesseract
    cands = [os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"),
             r"C:\Program Files\Tesseract-OCR\tesseract.exe",
             r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
             r"D:\Download\tesseract.exe", r"D:\Tesseract-OCR\tesseract.exe"]
    for c in cands:
        if c and Path(c).exists():
            pytesseract.pytesseract.tesseract_cmd = str(c)
            return str(c)
    sys.exit("Tesseract not found. Install it, or set the TESSERACT_CMD environment variable "
             "to the full path of tesseract.exe.")


def run_ocr(path, psm=3, threshold=CONF_THRESHOLD):
    import pytesseract
    from pytesseract import Output
    configure_tesseract()
    img = cv2.imread(str(path))
    if img is None:
        sys.exit(f"Cannot read image: {path}")

    stem = Path(path).stem
    steps = OUT_DIR / f"{stem}_steps"; steps.mkdir(parents=True, exist_ok=True)
    vis_base, variants, angle = preprocess(img, steps)

    def attempt(binary, p):
        d = pytesseract.image_to_data(binary, config=f"--oem 3 --psm {p}", output_type=Output.DICT)
        keep, drop = [], 0
        for i, txt in enumerate(d["text"]):
            txt, conf = txt.strip(), float(d["conf"][i])
            if not txt or conf < 0:
                continue
            if conf / 100 >= threshold:                     # the 80 % gate
                x, y, w, h = (d[k][i] for k in ("left", "top", "width", "height"))
                keep.append({"text": txt, "confidence": round(conf, 1), "box": [x, y, w, h]})
            else:
                drop += 1
        return keep, drop

    best = None
    for vname, binary in variants.items():                 # auto-pick best binarisation + layout mode
        for p in dict.fromkeys([psm, 6, 11]):
            keep, drop = attempt(binary, p)
            score = sum(k["confidence"] for k in keep) - 3 * drop
            if best is None or score > best[0]:
                best = (score, keep, drop, vname, p)
    _, kept, dropped, used_variant, used_psm = best

    vis = vis_base.copy()
    for k in kept:
        x, y, w, h = k["box"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 160, 0), 2)
        cv2.putText(vis, f"{k['confidence']:.0f}%", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1, cv2.LINE_AA)

    out_img = OUT_DIR / f"{stem}_ocr.png"
    cv2.imwrite(str(out_img), vis)
    mean_conf = float(np.mean([k["confidence"] for k in kept])) if kept else 0.0
    report = {
        "task": "ocr", "input": str(path), "deskew_angle_deg": round(angle, 2),
        "psm": used_psm, "binarisation": used_variant, "threshold": threshold,
        "coverage": round(len(kept) / max(1, len(kept) + dropped), 2),
        "words_kept": len(kept), "words_dropped_below_threshold": dropped,
        "mean_confidence": round(mean_conf, 1),
        "text": " ".join(k["text"] for k in kept),
        "validation_pass": bool(kept) and mean_conf >= threshold * 100
                           and len(kept) / (len(kept) + dropped) >= 0.6,   # most words must pass, not just one
        "annotated_image": str(out_img), "words": kept,
    }
    (OUT_DIR / f"{stem}_ocr.json").write_text(json.dumps(report, indent=2))
    return report


# ----------------------------------------------------------------------------
# PATH 2 : Object detection (YOLOv4-tiny, COCO, 80 classes)
# ----------------------------------------------------------------------------
# Pre-trained, free, no API key. All three files come from the same public repo.
YOLO_FILES = {
    "yolov4-tiny.weights":
        "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
    "yolov4-tiny.cfg":
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "coco.names":
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
}


def ensure_models():
    MODEL_DIR.mkdir(exist_ok=True)
    for name, url in YOLO_FILES.items():
        dest = MODEL_DIR / name
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        print(f"[download] {name}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                f.write(r.read())
        except Exception as e:
            dest.unlink(missing_ok=True)
            sys.exit(f"Download failed for {name}: {e}\n"
                     f"Open this link in your browser, save the file, and put it in the 'models' folder:\n  {url}")
    return MODEL_DIR


def run_detection(path, threshold=CONF_THRESHOLD):
    ensure_models()
    img = cv2.imread(str(path))
    if img is None:
        sys.exit(f"Cannot read image: {path}")
    H, W = img.shape[:2]
    labels = (MODEL_DIR / "coco.names").read_text().strip().split("\n")

    net = cv2.dnn_DetectionModel(str(MODEL_DIR / "yolov4-tiny.weights"),
                                 str(MODEL_DIR / "yolov4-tiny.cfg"))
    net.setInputParams(size=(416, 416), scale=1 / 255.0, swapRB=True)   # blob: resize, scale, BGR->RGB
    ids, confs, boxes = net.detect(img, confThreshold=0.30, nmsThreshold=0.4)  # low raw cut; 80% gate below
    ids, confs, boxes = np.array(ids).flatten(), np.array(confs).flatten(), list(boxes)
    vis, kept, below, dropped = img.copy(), [], [], 0
    for i in range(len(ids)):
        conf = float(confs[i])
        if conf >= threshold:                                  # the 80 % gate
            x, y, w, h = map(int, boxes[i])
            label = labels[int(ids[i])]
            kept.append({"label": label, "confidence": round(conf * 100, 1), "box": [x, y, w, h]})
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 3)
            tag = f"{label} {conf*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(vis, (x, y - th - 10), (x + tw + 6, y), (0, 200, 0), -1)
            cv2.putText(vis, tag, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            dropped += 1
            x, y, w, h = map(int, boxes[i])
            label = labels[int(ids[i])]
            below.append({"label": label, "confidence": round(conf * 100, 1), "box": [x, y, w, h]})
            cv2.rectangle(vis, (x, y), (x + w, y + h), (150, 150, 150), 1)       # grey = rejected by gate
            cv2.putText(vis, f"{label} {conf*100:.0f}% (below gate)", (x, max(14, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1, cv2.LINE_AA)

    stem = Path(path).stem
    out_img = OUT_DIR / f"{stem}_detect.jpg"
    cv2.imwrite(str(out_img), vis)
    report = {
        "task": "detect", "input": str(path), "image_size": [W, H],
        "model": "YOLOv4-tiny (COCO)", "threshold": threshold,
        "detections_kept": len(kept), "detections_dropped_below_threshold": dropped,
        "validation_pass": len(kept) > 0,
        "annotated_image": str(out_img), "detections": kept, "below_gate": below,
    }
    (OUT_DIR / f"{stem}_detect.json").write_text(json.dumps(report, indent=2))
    return report


# ----------------------------------------------------------------------------
def show(report):
    print("=" * 62)
    print(f" {report['task'].upper()}  |  {report['input']}")
    print("=" * 62)
    if report["task"] == "ocr":
        print(f" Deskew angle : {report['deskew_angle_deg']} deg")
        print(f" Words kept   : {report['words_kept']} (dropped {report['words_dropped_below_threshold']} < {int(report['threshold']*100)}%)")
        print(f" Mean conf.   : {report['mean_confidence']} %  (coverage {int(report['coverage']*100)}%, psm {report['psm']}, {report['binarisation']})")
        print(f" TEXT         : {report['text']}")
    else:
        for d in report["detections"]:
            print(f"  - {d['label']:<14} {d['confidence']:>5.1f}%  box={d['box']}")
        print(f" Kept {report['detections_kept']} / dropped {report['detections_dropped_below_threshold']}")
    print(f" Validation   : {'PASS' if report['validation_pass'] else 'FAIL'}")
    print(f" Visual output: {report['annotated_image']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["ocr", "detect", "both"])
    ap.add_argument("image")
    ap.add_argument("--psm", type=int, default=3, help="Tesseract page segmentation mode (3/6/7/11)")
    ap.add_argument("--threshold", type=float, default=CONF_THRESHOLD)
    a = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    if a.mode in ("ocr", "both"):
        show(run_ocr(a.image, a.psm, a.threshold))
    if a.mode in ("detect", "both"):
        show(run_detection(a.image, a.threshold))


if __name__ == "__main__":
    main()
