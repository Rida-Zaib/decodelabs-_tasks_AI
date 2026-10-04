"""
DecodeLabs -- Artificial Intelligence Industrial Training Kit (Batch 2026)
Project 2: Data Classification Using AI

------------------------------------------------------------------
OLD WAY (Project 1): heuristic, hand-written if/elif rules.
NEW WAY (this project): supervised learning -- we do not write the
rules ourselves. We give the machine labelled HISTORY (the Iris
dataset) and let it derive its own decision boundary, then we
measure how well that boundary generalizes to data it has never
seen.
------------------------------------------------------------------

Pipeline (the "Master Blueprint" from the training deck):

    INPUT                  PROCESS                 OUTPUT
    -----                  -------                 ------
    Iris domain            Train-test split         Confusion Matrix
    Feature scaling        KNN algorithm             F1 score

Run it:
    python iris_classifier.py
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)

RANDOM_STATE = 42  # fixed seed -- same shuffle every run, so results are reproducible


# ============================================================
# STEP 1 -- INPUT: Raw Material (The Iris Benchmark)
# ============================================================
def load_and_understand_dataset():
    """Load the Iris dataset and print what's actually in it.
    'Understanding' a dataset before touching it is the first
    requirement -- you should never blindly feed data into a model."""
    iris = load_iris()
    X = iris.data          # shape (150, 4) -- sepal/petal length & width
    y = iris.target        # shape (150,)   -- 0=setosa, 1=versicolor, 2=virginica

    print("=" * 60)
    print("STEP 1 -- INPUT: Raw Material (Iris Benchmark)")
    print("=" * 60)
    print(f"Samples:    {X.shape[0]}  (balanced -- 50 of each class)")
    print(f"Classes:    {len(iris.target_names)}  -> {list(iris.target_names)}")
    print(f"Dimensions: {X.shape[1]}  -> {list(iris.feature_names)}")
    print("\nFirst 3 rows of raw data:")
    for row, label in zip(X[:3], y[:3]):
        print(f"  {row}  -> class {label} ({iris.target_names[label]})")
    print()
    return X, y, iris


# ============================================================
# STEP 2 -- INPUT: The Gatekeeper Rule (Scaling)
# ============================================================
def scale_features(X_train, X_test):
    """KNN measures distance between points, so a feature with a
    bigger numeric range (e.g. 0-1000) would silently dominate one
    with a smaller range (e.g. 0-1), even if it's less important.
    StandardScaler fixes this: every feature ends up with mean = 0
    and variance = 1, so all four measurements count equally.

    IMPORTANT: fit the scaler on the TRAINING data only, then just
    transform the test data with it. If we fit on the test set too,
    the model would be peeking at data it's supposed to be tested
    against -- that's data leakage."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("=" * 60)
    print("STEP 2 -- INPUT: The Gatekeeper Rule (Scaling)")
    print("=" * 60)
    print(f"Before scaling -- feature means: {X_train.mean(axis=0).round(2)}")
    print(f"After  scaling -- feature means: {X_train_scaled.mean(axis=0).round(2)}")
    print(f"After  scaling -- feature std:   {X_train_scaled.std(axis=0).round(2)}")
    print()
    return X_train_scaled, X_test_scaled, scaler


# ============================================================
# STEP 3 -- PROCESS: Structural Integrity (The Split)
# ============================================================
def split_data(X, y, test_size=0.2):
    """80/20 train-test split. shuffle=True (the default) randomizes
    row order before splitting, so the model doesn't accidentally
    train on only the first two classes and get tested on the third."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        shuffle=True,
        stratify=y,             # keeps the 3 classes balanced in both sets
        random_state=RANDOM_STATE,
    )

    print("=" * 60)
    print("STEP 3 -- PROCESS: Structural Integrity (The Split)")
    print("=" * 60)
    print(f"Training set: {len(X_train)} samples ({(1 - test_size) * 100:.0f}%)")
    print(f"Test set:     {len(X_test)} samples ({test_size * 100:.0f}%)")
    print()
    return X_train, X_test, y_train, y_test


# ============================================================
# STEP 4 -- PROCESS: Tuning the Engine (Choosing K)
# ============================================================
def find_best_k(X_train, y_train, X_test, y_test, max_k=20):
    """K=1 memorizes the training data too closely (overfitting --
    noisy, unstable). A very large K blurs every class together
    (underfitting -- too generic). The 'elbow' is the K where test
    error stops dropping sharply -- that's the sweet spot."""
    print("=" * 60)
    print("STEP 4 -- PROCESS: Tuning the Engine (Choosing K)")
    print("=" * 60)

    error_rates = []
    for k in range(1, max_k + 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        error_rates.append(1 - accuracy_score(y_test, preds))

    # K=1 almost always looks great on paper (it just memorizes each
    # test point's single nearest training neighbor) but that's the
    # textbook overfitting trap the deck itself warns against, so we
    # search for the elbow starting at K=3. Among the lowest-error
    # candidates we also prefer an odd K, since an even K can tie
    # during the majority vote.
    search_start = 3 if max_k >= 3 else 1
    candidates = list(range(search_start, max_k + 1))
    min_error = min(error_rates[k - 1] for k in candidates)
    tied = [k for k in candidates if error_rates[k - 1] == min_error]
    odd_tied = [k for k in tied if k % 2 == 1]
    best_k = min(odd_tied) if odd_tied else min(tied)

    print(f"Tried K = 1..{max_k}. Error rate per K:")
    for k, err in enumerate(error_rates, start=1):
        note = ""
        if k == 1:
            note = "  (excluded -- fits noise, classic overfitting)"
        elif k == best_k:
            note = "  <-- chosen (the elbow)"
        print(f"  K={k:2d}: error = {err:.3f}{note}")
    print(f"\nChosen K = {best_k}  (skipping K=1 on purpose -- see note above)")
    print()
    return best_k, error_rates


# ============================================================
# STEP 5 -- PROCESS: The Workflow (scikit-learn: instantiate, fit, predict)
# ============================================================
def train_model(X_train, y_train, k):
    """The three-line scikit-learn pattern from the deck:
    INSTANTIATE -> build the empty model frame
    FIT         -> let it memorize the training map (X_train -> y_train)
    PREDICT     -> apply that learned logic to new points (done by caller)"""
    print("=" * 60)
    print("STEP 5 -- PROCESS: The Workflow (scikit-learn)")
    print("=" * 60)
    model = KNeighborsClassifier(n_neighbors=k)   # INSTANTIATE
    model.fit(X_train, y_train)                   # FIT
    print(f"Model trained: KNeighborsClassifier(n_neighbors={k})")
    print()
    return model


# ============================================================
# STEP 6 -- OUTPUT: Validation (accuracy mirage, confusion matrix, F1)
# ============================================================
def evaluate_model(model, X_test, y_test, target_names):
    predictions = model.predict(X_test)            # PREDICT

    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    f1_macro = f1_score(y_test, predictions, average="macro")

    print("=" * 60)
    print("STEP 6 -- OUTPUT: Validation")
    print("=" * 60)
    print(f"Accuracy: {acc * 100:.2f}%")
    print("(Note: on an imbalanced dataset, a high accuracy number can")
    print(" still hide a model that fails badly on the rare class --")
    print(" that's the 'accuracy mirage'. Iris is balanced, so accuracy")
    print(" is meaningful here, but the confusion matrix below is what")
    print(" actually proves it, class by class.)\n")

    print("Confusion Matrix")
    print("(rows = actual class, columns = predicted class)")
    header = "            " + "  ".join(f"{n[:10]:>10}" for n in target_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>10}" for v in row)
        print(f"{target_names[i][:10]:>10}  {row_str}")
    print()

    print("Classification Report (per-class Precision / Recall / F1)")
    print(classification_report(y_test, predictions, target_names=target_names))

    print(f"Overall F1 Score (macro average): {f1_macro:.3f}")
    print("(F1 is the harmonic mean of Precision and Recall -- it only")
    print(" scores high when BOTH are high, so it can't be gamed by")
    print(" favoring one at the expense of the other.)")
    print()
    return predictions, cm, f1_macro


# ============================================================
# STEP 7 -- Try it yourself: classify a brand-new flower
# ============================================================
def interactive_prediction(model, scaler, target_names, feature_names):
    print("=" * 60)
    print("STEP 7 -- Try it yourself")
    print("=" * 60)
    print("Enter your own flower measurements (in cm) to classify it,")
    print("or press Enter on the first prompt to skip.\n")

    first = input(f"{feature_names[0]}: ").strip()
    if first == "":
        print("Skipped.")
        return

    try:
        vals = [float(first)]
        for name in feature_names[1:]:
            vals.append(float(input(f"{name}: ").strip()))
    except ValueError:
        print("That wasn't a number -- skipping prediction.")
        return

    sample = np.array(vals).reshape(1, -1)
    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)[0]
    proba = model.predict_proba(sample_scaled)[0]

    print(f"\nPredicted class: {target_names[pred]}")
    print("Confidence per class:")
    for name, p in zip(target_names, proba):
        print(f"  {name:12s}: {p * 100:5.1f}%")


def main():
    X, y, iris = load_and_understand_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    best_k, _ = find_best_k(X_train_scaled, y_train, X_test_scaled, y_test)
    model = train_model(X_train_scaled, y_train, best_k)
    evaluate_model(model, X_test_scaled, y_test, iris.target_names)
    interactive_prediction(model, scaler, iris.target_names, iris.feature_names)

    print("\nDone. This is the full IPO pipeline: raw data in one end,")
    print("a validated, trained classifier out the other.")


if __name__ == "__main__":
    main()
