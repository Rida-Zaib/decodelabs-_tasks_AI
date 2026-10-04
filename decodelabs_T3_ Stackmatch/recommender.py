"""
Project 3 - Tech Stack Recommender (Content-Based Filtering)
Pipeline: Ingestion -> Scoring (TF-IDF + Cosine) -> Sorting -> Filtering (Top-N)
Pure Python (no external libraries), so every step of the math is visible.
"""
import csv
import math
import difflib
from collections import Counter

DATASET = "raw_skills.csv"
TOP_N = 3
MIN_INPUTS = 3

# Shared vocabulary: map user wording onto the dataset's exact skill names.
ALIASES = {
    "cloud computing": "cloud", "aws cloud": "aws", "amazon web services": "aws",
    "k8s": "kubernetes", "js": "javascript", "ml": "machine learning",
    "dl": "deep learning", "natural language processing": "nlp",
    "cicd": "ci/cd", "ci cd": "ci/cd", "node": "nodejs", "node.js": "nodejs",
    "reactjs": "react", "frontend development": "web design",
    "data structure": "data structures", "algorithm": "algorithms",
    "powerbi": "power bi", "scripting": "shell scripting", "tf": "tensorflow",
    "api": "apis", "rest api": "apis", "cyber security": "security",
    "cybersecurity": "security", "data viz": "data visualization",
}


def normalize(skill):
    s = skill.strip().lower()
    return ALIASES.get(s, s)


# ---------- Load dataset (items = job roles) ----------
def load_roles(path=DATASET):
    roles = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            roles[row["role"]] = [normalize(s) for s in row["skills"].split(";") if s.strip()]
    return roles


# ---------- TF-IDF ----------
def compute_idf(roles):
    n_docs = len(roles)
    df = Counter()
    for skills in roles.values():
        df.update(set(skills))
    # IDF = log(total docs / docs containing term)   (as in the slides)
    return {t: math.log(n_docs / d) for t, d in df.items()}


def tfidf_vector(skills, vocab, idf):
    counts = Counter(skills)
    total = len(skills)
    # TF = count of term in doc / total terms in doc
    return [(counts[t] / total) * idf[t] if total else 0.0 for t in vocab]


# ---------- Cosine similarity ----------
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:      # cold-start guard: zero vector -> score 0
        return 0.0
    return dot / (na * nb)


# ---------- Pipeline ----------
def recommend(user_skills, roles, top_n=TOP_N):
    idf = compute_idf(roles)
    vocab = sorted(idf)
    known = [s for s in user_skills if s in idf]
    unknown = [s for s in user_skills if s not in idf]

    user_vec = tfidf_vector(known, vocab, idf)                      # Step 1: ingestion -> vector
    scored = []
    for role, skills in roles.items():                              # Step 2: scoring
        score = cosine(user_vec, tfidf_vector(skills, vocab, idf))
        matched = sorted(set(known) & set(skills))
        scored.append((role, score, matched))
    scored.sort(key=lambda x: x[1], reverse=True)                   # Step 3: sorting
    return scored[:top_n], unknown                                  # Step 4: filtering (Top-N)


def resolve_skill(raw, vocab):
    """Normalize input and suggest a close match ('did you mean') if it's not in vocab."""
    s = normalize(raw)
    if s in vocab:
        return s, None
    guess = difflib.get_close_matches(s, vocab, n=1, cutoff=0.72)
    return None, (guess[0] if guess else None)


def get_user_skills(vocab):
    print("Available skills:", ", ".join(vocab), "\n")
    while True:
        raw = input(f"Enter at least {MIN_INPUTS} skills (comma-separated): ")
        skills, unknown = [], []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            match, guess = resolve_skill(token, vocab)
            if match:
                skills.append(match)
            elif guess:
                ans = input(f'  "{token}" not found — did you mean "{guess}"? [y/N]: ').strip().lower()
                skills.append(guess if ans == "y" else None)
                skills = [s for s in skills if s]
            else:
                unknown.append(token)
        skills = list(dict.fromkeys(skills))
        if unknown:
            print(f"  Ignored (not in dataset): {', '.join(unknown)}")
        if len(skills) >= MIN_INPUTS:
            return skills
        print(f"Please enter at least {MIN_INPUTS} distinct, recognized skills.\n")


def main():
    roles = load_roles()
    vocab = sorted(compute_idf(roles))
    print("=== Tech Stack Recommender ===\n")
    user_skills = get_user_skills(vocab)

    top, unknown = recommend(user_skills, roles)
    if unknown:
        print(f"\nNote: ignored skills not in dataset: {', '.join(unknown)}")

    if not top or top[0][1] == 0:
        print("\nNo matching roles. Try skills from the list above (cold-start fallback).")
        return

    print(f"\nTop {TOP_N} career paths for you:")
    for i, (role, score, matched) in enumerate(top, 1):
        print(f"{i}. {role:<28} match: {score*100:5.1f}%   shared skills: {', '.join(matched) or '-'}")


if __name__ == "__main__":
    main()
