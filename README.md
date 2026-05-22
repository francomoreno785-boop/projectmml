# Unsupervised Clustering on the PatternMind Image Dataset

A machine learning course project. We extract image features with a pre-trained ResNet50, reduce them with PCA, and then apply three clustering algorithms - **K-Means**, **Hierarchical (Agglomerative)**, and **DBSCAN**. We compare them with both internal and external metrics and pick the best one.

The dataset is organised into class folders, but we do **not** use those labels during clustering. We only use them at the very end, as ground truth, to check how well each method recovers the real classes.

---

## TL;DR

- **Dataset:** 25,557 images across 233 class folders (`.jpg`), heavily imbalanced.
- **Features:** 2048-dim ResNet50 embeddings, reduced to 50 dimensions with PCA (44.1% variance retained).
- **Best method:** **K-Means with k = 30** — wins on both internal metrics (Silhouette, CH, DB) and the decisive external metric (ARI).
- **Hierarchical Ward** is a close second; **DBSCAN** is unsuitable here (labels 83% of points as noise).
- Fully reproducible with `RANDOM_STATE = 42`. ResNet50 features are cached after the first run.

---

## Authors

- Tara Krstovic
- Franco Moreno
- Maria Nishtelkova

---

## Course

- **Course:** Machine Learning
- **University:** Luiss Guido Carli
- **Academic year:** 2025/2026

---

## Pipeline overview
