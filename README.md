# Unsupervised Clustering on the PatternMind Image Dataset

A machine learning course project. We extract image features with a pre-trained ResNet50 and apply three clustering algorithms — **K-Means**, **Hierarchical (Agglomerative)**, and **DBSCAN** — then compare them.

The dataset is organised in class folders, but we do not use those labels for clustering. We only use them at the end to check how well the clusters match the real classes.

---

## Authors

- Tara Krstovic
- Franco Moreno 
- Maria ------

---

## Course

- **Course:** Machine Learning
- **University:** Luiss Guido Carli
- **Academic year:** 2025/2026

---

## Dataset

We use the **PatternMind** image dataset, provided as a single ZIP archive (`patternmind_dataset.zip`). It is organised with one folder per class.

We do not include the dataset in this repository. To run the notebook, place `patternmind_dataset.zip` in your Google Drive (for Colab) or next to the notebook (for local use).

---

## What the project does

1. **Exploratory analysis** — counts the number of classes and images, checks for empty folders, duplicate filenames, and corrupted images.
2. **Feature extraction** — passes every image through a pre-trained **ResNet50** to get a 2048-dimensional feature vector per image.
3. **Preprocessing** — standardises the features and reduces them to 50 dimensions with **PCA**.
4. **Clustering** — runs **K-Means**, **Hierarchical (Ward / complete / average)**, and **DBSCAN** on the PCA features.
5. **Evaluation** — compares the methods using:
   - Internal metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin.
   - External metrics: ARI and NMI against the folder labels (only for validation, not for choosing the model).
6. **Final comparison** — summary table, side-by-side t-SNE visualisations, and a check of how much the three methods agree with each other.

**Final choice:** K-Means with k = 30, which gave the best internal and external scores.

---

## How to run

### Google Colab (recommended)

1. Upload `main.ipynb` to Google Drive and open it in Colab.
2. Place `patternmind_dataset.zip` in your Drive at `MyDrive/patternmind_dataset.zip`.
3. Click **Runtime → Run all**.

### Local Jupyter

1. Place `patternmind_dataset.zip` next to `main.ipynb`.
2. Open the notebook and run all cells.

The notebook detects whether it is running on Colab and adjusts paths automatically.

---

## Requirements

Python 3.10 or newer. On Colab everything is pre-installed. For local use:

```bash
pip install numpy pandas matplotlib seaborn tqdm pillow scikit-learn scipy tensorflow
```

A GPU helps with feature extraction but is not required.

---

## Files

```
main.ipynb         The full project notebook.
README.md          This file.
images/            Figures generated when the notebook runs.
```

The notebook also saves `.npy` files (features, labels, cluster assignments) so later sections can be re-run without redoing feature extraction.

---

## Results summary

| Method | Best result |
|--------|-------------|
| K-Means (k = 30) | Best Silhouette, best ARI / NMI against folder labels |
| Hierarchical Ward (k = 30) | Close second, agrees strongly with K-Means |
| DBSCAN | Labels most points as noise — not a good fit for high-dimensional ResNet features |

The full numbers are in Section 8 of the notebook.

---

## Notes

- All random seeds are fixed (`RANDOM_STATE = 42`), so the results are reproducible.
- The notebook is written to run top-to-bottom from a fresh kernel.
- **Feature extraction is cached.** ResNet50 inference is the slow step; the notebook saves `X_features.npy`, `y_labels.npy`, `label_names.npy`, and `image_paths.npy` on the first run. On every later run those files are loaded from disk and the slow ResNet50 step is skipped automatically. On Colab the cache is also mirrored to `MyDrive/projectml_colab/` so it survives runtime resets.
