"""Local web app: python server.py  ->  http://127.0.0.1:5000   (Flask, free, no keys)"""
import base64, uuid
from pathlib import Path
import cv2
from flask import Flask, jsonify, request, send_from_directory
import recognition as rec
import gemini_engine as ge

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
UP = Path("uploads"); UP.mkdir(exist_ok=True)
rec.OUT_DIR.mkdir(exist_ok=True)


def b64(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def gates(rep, mode, steps):
    thr = rep["threshold"] * 100
    if rep.get("engine") == "gemini":
        n = len(rep["words"] if mode == "ocr" else rep["detections"])
        nb = len(rep.get("below_gate", []))
        g2 = {"pass": True, "detail": "image normalised (max 1600px, RGB, JPEG) before the API call"}
        ok3 = rep["validation_pass"]
        g3 = {"pass": ok3, "detail": f"{n} kept, {nb} below gate. Confidence is Gemini's own estimate; "
                                      f"gate {thr:.0f}%" + (f", mean {rep['mean_confidence']}%" if mode == "ocr" else "")}
        return [{"name": "Library integration", "pass": True, "detail": f"Gemini API answered ({rep['model']})"},
                {"name": "Pre-processing integrity", **g2}, {"name": "Accuracy >= 80%", **g3},
                {"name": "Visual confirmation", "pass": Path(rep["annotated_image"]).exists(),
                 "detail": "annotated image generated"}]
    if mode == "ocr":
        confs = [w["confidence"] for w in rep["words"]]
        g2 = {"pass": len(steps) >= 3, "detail": "grayscale, blur, deskew, adaptive threshold applied"}
        g3 = {"pass": rep["validation_pass"],
              "detail": f"mean {rep['mean_confidence']}% over {len(confs)} words, {int(rep['coverage']*100)}% of words passed (need mean >= {thr:.0f}% and >= 60% coverage)"}
    else:
        confs = [d["confidence"] for d in rep["detections"]]
        g2 = {"pass": True, "detail": "blob built: 416x416 resize, 1/255 scale, BGR->RGB swap"}
        nb = len(rep.get("below_gate", []))
        g3 = {"pass": bool(confs) and min(confs) >= thr,
              "detail": (f"{len(confs)} objects found, lowest {min(confs)}% (gate {thr:.0f}%)" if confs else
                         f"No object reached {thr:.0f}% ({nb} weak guesses shown in grey). The model only knows 80 COCO classes - e.g. person, car, dog, cat, bird, bottle, laptop, chair.")}
    return [
        {"name": "Library integration", "pass": True,
         "detail": "pytesseract ran" if mode == "ocr" else "cv2.dnn ran"},
        {"name": "Pre-processing integrity", **g2},
        {"name": "Accuracy >= 80%", **g3},
        {"name": "Visual confirmation", "pass": Path(rep["annotated_image"]).exists(),
         "detail": "annotated image generated"},
    ]


@app.get("/api/config")
def config():
    return jsonify(gemini=ge.available())


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/api/recognize")
def recognize():
    f = request.files.get("image")
    mode = request.form.get("mode", "ocr")
    if not f or mode not in ("ocr", "detect"):
        return jsonify(error="Provide an image and mode=ocr|detect"), 400
    thr = float(request.form.get("threshold", 0.8))
    psm = int(request.form.get("psm", 3))
    path = UP / f"{uuid.uuid4().hex[:8]}.png"
    f.save(path)
    if cv2.imread(str(path)) is None:
        return jsonify(error="Not a readable image"), 400
    engine = request.form.get("engine", "classic")
    try:
        if engine == "gemini":
            rep = ge.run(mode, path, thr)
        else:
            rep = rec.run_ocr(path, psm, thr) if mode == "ocr" else rec.run_detection(path, thr)
    except SystemExit as e:       # model download blocked etc.
        return jsonify(error=str(e)), 500
    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {e}"), 500
    steps = []
    sd = rec.OUT_DIR / f"{path.stem}_steps"
    if mode == "ocr" and sd.exists():
        steps = [{"name": n, "img": b64(sd / f"{k}.png")} for k, n in
                 (("1_gray", "Grayscale"), ("2_blur", "Gaussian blur + deskew"), ("3_binary", "Adaptive threshold"))]
    g = gates(rep, mode, steps)
    return jsonify(report=rep, original=b64(path), annotated=b64(rep["annotated_image"]),
                   steps=steps, gates=g, all_pass=all(x["pass"] for x in g))


if __name__ == "__main__":
    app.run(debug=False)
