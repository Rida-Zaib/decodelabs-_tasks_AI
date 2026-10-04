"""
Advanced engine: Google Gemini vision (free tier, API key from https://aistudio.google.com/apikey).
Open-vocabulary detection (any object, not just 80 COCO classes) + OCR for printed, handwritten,
curved and multi-language text. Uses plain HTTPS (no SDK, no extra installs).

Put your key in a file named  .env  next to this script:   GEMINI_API_KEY=your_key_here
(or set the GEMINI_API_KEY environment variable).

NOTE: Gemini does not return calibrated probabilities. 'confidence' here is the model's own 0-100
estimate, so the 80% gate is a self-reported-confidence gate (the UI says so).
"""
import base64, json, os, re, time, urllib.error, urllib.request
from pathlib import Path

import cv2

OUT_DIR = Path("outputs")
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
LIST_API = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200"
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
_SKIP = ("image", "tts", "live", "audio", "embedding", "imagen", "veo", "lyria", "robotics", "computer-use",
         "deep-research", "learnlm", "gemma", "aqa", "native")


def _version(name):
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)


def discover_models(key):
    """Ask the API which models this key can use, newest Flash first (names change every few months)."""
    forced = _env("GEMINI_MODEL")
    names = []
    try:
        req = urllib.request.Request(LIST_API, headers={"x-goog-api-key": key})
        with urllib.request.urlopen(req, timeout=30) as r:
            for m in json.loads(r.read()).get("models", []):
                n = m["name"].split("/")[-1]
                if ("generateContent" in m.get("supportedGenerationMethods", []) and "flash" in n
                        and not any(x in n for x in _SKIP)):
                    names.append(n)
    except Exception:
        pass
    # newest version first; full Flash before Flash-Lite; stable before preview/exp
    names.sort(key=lambda n: (_version(n), "lite" not in n, "preview" not in n and "exp" not in n), reverse=True)
    out = ([forced] if forced else []) + names + FALLBACK_MODELS
    return list(dict.fromkeys(out))


PROMPTS = {
    "detect": (
        "Detect every distinct object, person, animal, insect, plant, sign, food or product visible in this "
        "image. Be open-vocabulary and specific (e.g. 'monarch butterfly', 'pink zinnia flower', 'red car'), "
        "not limited to common categories. Return ONLY a JSON array (max 25 items). Each item: "
        '{"label": string, "box_2d": [ymin, xmin, ymax, xmax] normalised to 0-1000, '
        '"confidence": integer 0-100 = how sure you are this label is correct}. '
        "Do not list things that are not visible."),
    "ocr": (
        "Read ALL text visible in this image, in reading order: printed, handwritten, curved, rotated or "
        "partly hidden, in any language. Return ONLY a JSON array, one item per line of text. Each item: "
        '{"text": string, "box_2d": [ymin, xmin, ymax, xmax] normalised to 0-1000, '
        '"confidence": integer 0-100 = how sure you are the transcription is exactly right, '
        '"style": "printed" | "handwritten" | "curved"}. '
        "Never invent or guess text that is not there; give low confidence when unsure. "
        "If there is no text, return []."),
}


def _env(*names):
    """Read a setting from environment variables, or from the .env file next to the script."""
    for n in names:
        if os.environ.get(n):
            return os.environ[n].strip()
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() in names and v.strip():
                    return v.strip().strip('"').strip("'")
    return None


def load_key():
    return _env("GEMINI_API_KEY", "GOOGLE_API_KEY")


def available():
    return bool(load_key())


def _call(key, model, jpeg_b64, prompt, think=False):
    body = {
        "contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": jpeg_b64}},
                                {"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }
    if think:                       # try to switch thinking off (faster, better boxes); some models reject it
        body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
    req = urllib.request.Request(API.format(model=model), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())


def _ask(jpeg_b64, prompt):
    key = load_key()
    if not key:
        raise RuntimeError("No Gemini key found. Create a file named .env containing GEMINI_API_KEY=your_key "
                           "(free key: https://aistudio.google.com/apikey) and restart the server.")
    errors = []
    for model in discover_models(key)[:6]:
        for think in (True, False):
            for attempt in range(2):
                try:
                    return model, _call(key, model, jpeg_b64, prompt, think)
                except urllib.error.HTTPError as e:
                    detail = e.read().decode(errors="ignore")
                    low = detail.lower()
                    if e.code in (400, 403) and ("api key" in low or "api_key" in low):
                        raise RuntimeError("Gemini rejected the API key. Check the key in .env.")
                    if e.code == 400 and "think" in low and think:
                        break                                  # retry same model without thinkingConfig
                    errors.append(f"{model}: HTTP {e.code} {detail[:120].strip()}")
                    if e.code in (500, 503) and attempt == 0:
                        time.sleep(2); continue                # busy -> one retry
                    think = False
                    break
                except Exception as e:
                    errors.append(f"{model}: {e}")
                    break
            else:
                continue
            if not think and errors and errors[-1].startswith(model):
                break                                          # this model is unusable -> next model
    raise RuntimeError("Gemini request failed on every model tried:\n" + "\n".join(errors[-4:]))


def _parse(resp):
    cands = resp.get("candidates") or []
    if not cands:
        raise RuntimeError(f"Gemini returned no answer (blocked?): {json.dumps(resp.get('promptFeedback', resp))[:300]}")
    text = "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []) if not p.get("thought"))
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("["), text.rfind("]")
        data = json.loads(text[a:b + 1]) if a != -1 and b > a else []
    return data if isinstance(data, list) else []


def _norm_box(b, W, H):
    try:
        y1, x1, y2, x2 = [max(0.0, min(1000.0, float(v))) for v in b[:4]]
    except Exception:
        return None
    if y2 <= y1 or x2 <= x1:
        return None
    return [int(x1 / 1000 * W), int(y1 / 1000 * H), int((x2 - x1) / 1000 * W), int((y2 - y1) / 1000 * H)]


def _ascii(s):
    return s.encode("ascii", "replace").decode()


def run(mode, path, threshold=0.80):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError("Cannot read image")
    H, W = img.shape[:2]
    k = 1600 / max(H, W)                                   # normalise: cap size, JPEG encode
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])
    model, resp = _ask(base64.b64encode(buf).decode(), PROMPTS[mode])
    items = _parse(resp)

    vis, kept, below = img.copy(), [], []
    thick = max(2, int(max(H, W) / 500))
    fs = max(0.5, max(H, W) / 1600)
    for it in items:
        if not isinstance(it, dict):
            continue
        box = _norm_box(it.get("box_2d", []), W, H)
        try:
            conf = float(it.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        name = str(it.get("label") if mode == "detect" else it.get("text", "")).strip()
        if box is None or not name:
            continue
        rec = {("label" if mode == "detect" else "text"): name, "confidence": round(conf, 1), "box": box}
        if mode == "ocr":
            rec["style"] = it.get("style", "printed")
        x, y, w, h = box
        if conf / 100 >= threshold:
            kept.append(rec)
            col = (0, 200, 0)
            cv2.rectangle(vis, (x, y), (x + w, y + h), col, thick)
            tag = f"{_ascii(name)[:28]} {conf:.0f}%" if mode == "detect" else f"{conf:.0f}%"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            cv2.rectangle(vis, (x, max(0, y - th - 8)), (x + tw + 6, y), col, -1)
            cv2.putText(vis, tag, (x + 3, max(th, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            below.append(rec)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (150, 150, 150), 1)
            cv2.putText(vis, f"{conf:.0f}% (below gate)", (x, max(14, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (110, 110, 110), 1, cv2.LINE_AA)

    OUT_DIR.mkdir(exist_ok=True)
    stem = Path(path).stem
    out = OUT_DIR / f"{stem}_gemini_{mode}.jpg"
    cv2.imwrite(str(out), vis)
    rep = {"task": mode, "engine": "gemini", "model": model, "input": str(path), "threshold": threshold,
           "annotated_image": str(out), "confidence_note": "model-reported (not calibrated)"}
    if mode == "detect":
        rep.update(detections=kept, below_gate=below, detections_kept=len(kept),
                   detections_dropped_below_threshold=len(below), validation_pass=len(kept) > 0)
    else:
        n = len(kept) + len(below)
        mean = sum(w["confidence"] for w in kept) / len(kept) if kept else 0.0
        rep.update(words=kept, below_gate=below, words_kept=len(kept), words_dropped_below_threshold=len(below),
                   mean_confidence=round(mean, 1), coverage=round(len(kept) / n, 2) if n else 0.0,
                   deskew_angle_deg=0.0, psm="n/a", binarisation="gemini",
                   text="\n".join(w["text"] for w in kept),
                   validation_pass=bool(kept) and mean >= threshold * 100 and len(kept) / n >= 0.6)
    (OUT_DIR / f"{stem}_gemini_{mode}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    return rep