# Data Classification Using AI — Project 2

**DecodeLabs — Artificial Intelligence Industrial Training Kit (Batch 2026)**

A supervised-learning classifier that trains on the classic Iris dataset
and predicts a flower's species (setosa / versicolor / virginica) from
its sepal and petal measurements — built with scikit-learn's K-Nearest
Neighbors algorithm.

Unlike Project 1 (rule-based chatbot, hand-written if/elif logic), this
project doesn't hard-code any decision rules. It's given 150 labelled
examples ("history") and derives its own decision boundary from them,
then that boundary is validated against data it never saw during
training.

## Files

| File | What it is |
|---|---|
| `iris_classifier.py` | Main script — the full pipeline, runs end to end |
| `elbow_plot.png` | Error rate vs K, showing why K=7 was chosen over K=1 |
| `confusion_matrix.png` | Per-class prediction breakdown on the test set |
| `requirements.txt` | Python packages needed |

## How to run

```bash
pip install -r requirements.txt
python iris_classifier.py
```

At the end it asks if you'd like to classify your own flower
measurements — press Enter to skip, or type in sepal/petal length and
width (in cm) to see a live prediction.

## The pipeline (IPO framework)

```
INPUT                    PROCESS                   OUTPUT
-----                    -------                   ------
Iris domain (150         Train-test split (80/20,  Confusion Matrix
samples, 3 classes,      shuffled, stratified)
4 dimensions)                                      F1 Score
                          KNN algorithm
Feature scaling
(StandardScaler)
```

1. **Load & understand the dataset** — 150 samples, 3 balanced
   classes, 4 numeric features (sepal length/width, petal
   length/width).
2. **Split** into 80% training / 20% test, shuffled and stratified so
   both sets keep an even mix of all three species.
3. **Scale** the features with `StandardScaler` (fit on the training
   set only, to avoid leaking test data into the model).
4. **Tune K** — try K = 1 to 20, plot the error rate, and pick the
   elbow. K=1 is deliberately excluded even though it often scores
   lowest error, since it's really just memorizing noise
   (`elbow_plot.png` marks this).
5. **Train** a `KNeighborsClassifier` with the chosen K.
6. **Validate** — accuracy, a full confusion matrix, and the
   per-class classification report (precision / recall / F1).
7. **Try it live** — classify a flower you enter yourself.

## Result summary

With `random_state=42` (reproducible): **K=7**, **96.7% accuracy**,
**F1 (macro) = 0.967** on the 30-sample test set — only 1 misclassified
flower (a virginica predicted as versicolor), visible in
`confusion_matrix.png`.

## Key concepts practiced

Supervised learning, feature scaling, train/test split & data leakage,
the K-Nearest Neighbors algorithm, bias/variance via the elbow method,
confusion matrices, precision/recall/F1, and the scikit-learn
instantiate → fit → predict workflow.

---
*Submitted as part of the DecodeLabs AI Internship — Project 2.*
