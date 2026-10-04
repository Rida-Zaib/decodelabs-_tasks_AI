# StackMatch — AI Recommendation Logic

**DecodeLabs — Artificial Intelligence Industrial Training Kit (Batch 2026) — Project 3**

A content-based recommendation engine that matches a person's skills to the tech career that fits them best — no user history, no collaborative filtering, just TF-IDF weighting and cosine similarity over a 16-role dataset.

---

## What it does

You enter at least 3 skills and rate how well you know each one. StackMatch:

1. Converts your skills into a weighted numeric vector (TF-IDF)
2. Compares that vector against all 16 modeled job roles using cosine similarity
3. Ranks roles by how small the angle is between you and them (smaller angle = closer match)
4. Shows your Top-N matches, with the skills you're missing — ranked by how distinctive they are

It also includes a full interactive dashboard: a side-by-side comparison table, trending-stack cards, an "Industry buzz" news-style section, a 6-chart dataset-insights dashboard (roles per area, skill rarity, a radar comparison of the four career areas, etc.), and an illustrative salary/growth/AI-adoption section modeled on real industry trend reports.

---

## Files

| File | What it is |
|---|---|
| `index.html` | The full web frontend — self-contained, open directly in any browser |
| `recommender.py` | The same recommendation engine as a command-line Python script |
| `raw_skills.csv` | The dataset: 16 roles and the skills each one requires |
| `tests.py` | Automated sanity tests — checks the engine recommends sensible roles for 5 sample skill profiles |

---

## How to run

**Frontend** — no install needed:

```bash
# just open it
open index.html        # macOS
start index.html        # Windows
# or serve it:
python3 -m http.server 8000
# then visit http://localhost:8000/index.html
```

**Command-line engine:**

```bash
python3 recommender.py
```

**Tests:**

```bash
python3 tests.py
```

No `pip install` required anywhere — the Python side uses only the standard library (`csv`, `math`, `difflib`), and the frontend is vanilla HTML/CSS/JS with no build step or external framework.

---

## How the matching works (IPO pipeline)

```
INPUT                      PROCESS                         OUTPUT
-----                      -------                         ------
User skills + 1–5 rating   TF-IDF vector weighting         Ranked Top-N roles
per skill                  Cosine similarity scoring        + missing-skill list
                            Sort by score, filter Top-N      + match percentage
```

- **TF-IDF**: a skill that appears in almost every role (like `git`) is weighted *down*; a skill that appears in only one or two roles (like `pytorch`) is weighted *up*. Your own rating (1–5) acts as the term-frequency component on your side.
- **Cosine similarity**: measures the *angle* between your skill vector and each role's requirement vector, not raw overlap count — so a small, focused profile isn't penalized against a role with a long skill list.
- **Cold start**: a user with no skills yet gets a vector of all zeros, which the frontend handles via a "Trending" fallback section instead of a broken 0% result.

---

## Key concepts practiced

Content-based filtering vs. collaborative filtering, TF-IDF weighting, cosine similarity, the cold-start problem, and building the same recommendation logic twice — once as a reusable Python module with tests, once as a dependency-free interactive frontend — to keep a single source of truth for the dataset.

*Submitted as part of the DecodeLabs AI Internship — Project 3.*
