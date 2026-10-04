# DecodeLabs — Artificial Intelligence Industrial Training Kit (Batch 2026)

This repository contains all four projects completed for the DecodeLabs AI Internship. Each project lives in its own folder with its own detailed `README.md`, `requirements.txt` (where needed), and source files. This top-level README is the map — read a project's own README for full setup and usage instructions.

> **Folder names below are my best guess from each project's own README — rename the folders to match or adjust the links if yours differ.**

---

## Projects at a glance

| # | Project | What it is | Core tech |
|---|---|---|---|
| 1 | [Rule-Based AI Chatbot](./project-1-chatbot/) | A 20-language chatbot using pure control flow — no AI models | Python, dictionary lookups, Unicode-script detection |
| 2 | [Data Classification Using AI](./project-2-iris-classifier/) | Supervised learning on the Iris dataset (species prediction) | Python, scikit-learn, KNN, StandardScaler |
| 3 | [AI Recommendation Logic — StackMatch](./project-3-stackmatch/) | Content-based career/skill recommendation engine | Python, TF-IDF, cosine similarity, HTML/CSS/JS |
| 4 | [Machine's Optic Nerve](./project-4-optic-nerve/) | Image & text recognition web app (OCR + object detection) | Python, Flask, OpenCV, Tesseract, YOLOv4-tiny, Gemini API |

The four projects form a rough progression: **hand-written rules → trained statistical model → similarity-based recommendation → applied computer vision**, each one trading a bit more hard-coded logic for a bit more learned/derived behavior.

---

## Project 1 — Rule-Based AI Chatbot

A fully rule-based chatbot — no AI models, no APIs — that understands 20 languages, including Roman Urdu, using Unicode-script detection and keyword matching.

- `chatbot.py` — base version, English only
- `chatbot_multilingual.py` — extended version, 20 languages
- `chatbot_frontend.html` — web chat UI, open directly in a browser

**Run it:**
```bash
python chatbot_multilingual.py
```
*(Windows CMD users: run `chcp 65001` first so non-English scripts display correctly.)*

**Concepts practiced:** control flow, `while` loops, dictionaries, the `.get()` lookup-with-fallback pattern, string sanitization, Unicode/regex-based rule design.

[→ Full README](./project-1-chatbot/README.md)

---

## Project 2 — Data Classification Using AI

A K-Nearest Neighbors classifier trained on the classic Iris dataset, predicting a flower's species from sepal and petal measurements. Unlike Project 1, no decision rules are hand-written — the model derives its own decision boundary from 150 labelled examples and is validated against unseen data.

- `iris_classifier.py` — full pipeline, runs end to end
- `elbow_plot.png` — error rate vs. K (why K=7 was chosen over K=1)
- `confusion_matrix.png` — per-class prediction breakdown

**Run it:**
```bash
pip install -r requirements.txt
python iris_classifier.py
```

**Result:** K=7, 96.7% accuracy, macro F1 = 0.967 on a 30-sample test set (reproducible with `random_state=42`).

**Concepts practiced:** supervised learning, feature scaling, train/test split & data leakage, KNN, the elbow method, confusion matrices, precision/recall/F1, the scikit-learn instantiate → fit → predict workflow.

[→ Full README](./project-2-iris-classifier/README.md)

---

## Project 3 — AI Recommendation Logic (StackMatch)

A content-based recommendation engine that matches a person's skills to the tech career that fits them best, using TF-IDF weighting and cosine similarity over a 16-role dataset — no user history required. Ships as both an interactive web frontend and a command-line Python engine sharing the same dataset.

- `index.html` — full interactive frontend (self-contained, no install)
- `recommender.py` — the same engine, command-line version
- `raw_skills.csv` — the 16-role skill dataset
- `tests.py` — automated sanity tests across 5 sample skill profiles

**Run it:**
```bash
# Frontend: just open index.html in a browser
python3 recommender.py   # command-line engine
python3 tests.py         # tests
```

**Concepts practiced:** content-based vs. collaborative filtering, TF-IDF weighting, cosine similarity, the cold-start problem, keeping one dataset in sync across two implementations.

[→ Full README](./project-3-stackmatch/README.md)

---

## Project 4 — Machine's Optic Nerve

A computer-vision web app that reads text and finds objects in a photo, rejects anything below an 80% confidence gate, and draws the surviving results on the image. Combines an offline classic pipeline (OpenCV + Tesseract + YOLOv4-tiny) with an optional free Google Gemini engine for open-ended object recognition, handwriting, and curved text.

- `server.py` — Flask app and the `/api/recognize` endpoint
- `recognition.py` — pre-processing, Tesseract OCR, YOLOv4-tiny detection, the 80% gate
- `gemini_engine.py` — Gemini engine integration
- `static/index.html` — frontend
- `samples/` — synthetic test images; `docs/` — pre-computed GitHub Pages results gallery

**Run it:**
```bash
pip install -r requirements.txt
python server.py
# open http://127.0.0.1:5000
```
*(Requires the separate Tesseract program — see the project's own README for OS-specific install steps. The Gemini engine is optional and needs a free API key.)*

**Concepts practiced:** classical CV pre-processing (deskew, adaptive thresholding), OCR confidence gating, object detection with a pretrained DNN, and combining an offline model with a cloud API behind a single interface.

[→ Full README](./project-4-optic-nerve/README.md)

---

## Suggested repository structure

```
.
├── README.md                      (this file)
├── project-1-chatbot/
│   ├── README.md
│   ├── chatbot.py
│   ├── chatbot_multilingual.py
│   └── chatbot_frontend.html
├── project-2-iris-classifier/
│   ├── README.md
│   ├── iris_classifier.py
│   ├── requirements.txt
│   ├── elbow_plot.png
│   └── confusion_matrix.png
├── project-3-stackmatch/
│   ├── README.md
│   ├── index.html
│   ├── recommender.py
│   ├── raw_skills.csv
│   └── tests.py
└── project-4-optic-nerve/
    ├── README.md
    ├── server.py
    ├── recognition.py
    ├── gemini_engine.py
    ├── requirements.txt
    ├── .env.example
    ├── static/
    ├── samples/
    └── docs/
```

---

## Author

Submitted as part of the **DecodeLabs AI Industrial Training Kit — Batch 2026**.
