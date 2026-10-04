"""
Sanity tests for the Tech Stack Recommender.
Run with: python3 tests.py
Each persona checks that the engine ranks a sensible role at or near the top.
"""
from recommender import load_roles, recommend

PERSONAS = [
    ("Cloud-leaning skills",
     ["aws", "docker", "kubernetes"],
     {"DevOps Engineer", "Cloud Engineer", "Cloud Architect"}),
    ("Classic backend/CS skills",
     ["java", "algorithms", "data structures"],
     {"Software Engineer", "Backend Developer"}),
    ("Data science skills",
     ["python", "statistics", "machine learning"],
     {"Data Scientist", "Machine Learning Engineer"}),
    ("Frontend/web skills",
     ["javascript", "react", "css"],
     {"Frontend Developer", "Full Stack Developer"}),
    ("Security skills",
     ["security", "networking", "cryptography"],
     {"Cybersecurity Analyst", "System Administrator"}),
]


def run():
    roles = load_roles()
    passed = 0
    for name, skills, acceptable in PERSONAS:
        top, _ = recommend(skills, roles, top_n=1)
        winner = top[0][0] if top else None
        ok = winner in acceptable
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: top pick = {winner!r} (expected one of {sorted(acceptable)})")
    print(f"\n{passed}/{len(PERSONAS)} personas passed.")
    if passed != len(PERSONAS):
        raise SystemExit(1)


if __name__ == "__main__":
    run()
