# Unsupervised Clustering on the PatternMind Image Dataset

A machine learning course project. We extract image features with a pre-trained ResNet50, reduce them with PCA, and then apply three clustering algorithms - **K-Means**, **Hierarchical (Agglomerative)**, and **DBSCAN**. We compare them with both internal and external metrics and pick the best one.

The dataset is organised into class folders, but we do **not** use those labels during clustering. We only use them at the very end, as ground truth, to check how well each method recovers the real classes.

---

## Authors

- Tara Krstovic
- Franco Moreno
- Maria Nishtelkova

---

## Course

- **Course:** Machine Leaerning
- **University:** Luiss Guido Carli 
- **Academic year:** 2025/2026

---

## Dataset

We use the **PatternMind** image dataset, provided as a ZIP archive (`patternmind_dataset.zip`). It is organised with one folder per class.

Basic statistics (from the EDA section of the notebook):

| Property | Value |
|----------|-------|
| Number of classes | _[fill in]_ |
| Total images | _[fill in]_ |
| Image formats | `.jpg`, `.jpeg`, `.png` |
| Empty folders | _[fill in]_ |
| Corrupted images | _[fill in]_ |
| Image dimensions | variable — resized to 224 × 224 for ResNet50 |
| Class balance | imbalanced (some classes have many more images than others) |

We do **not** include the dataset in this repository. To run the notebook, place `patternmind_dataset.zip` in your Google Drive (for Colab) or next to the notebook (for local use).

---

## Methodology

The notebook follows a linear pipeline. Each step is one section of the notebook.

### 1. Exploratory Data Analysis (Section 3)

- Counts the number of classes and images.
- Plots the distribution of images per class (class imbalance check).
- Checks for empty folders, duplicate filenames, and corrupted images.
- Samples 200 images to check image dimensions.

### 2. Feature extraction with ResNet50 (Section 4)

- We load **ResNet50 pre-trained on ImageNet** as a fixed feature extractor (`include_top=False`, `pooling='avg'`).
- Each image is resized to 224 × 224 and passed through the network.
- Output: a **2048-dimensional feature vector** per image.
- The features are saved as `.npy` files and cached, so this slow step only runs once.

### 3. Preprocessing (Section 5)

- **Standardisation:** features are scaled with `StandardScaler` so that all dimensions have mean 0 and standard deviation 1.
- **PCA to 50 dimensions:** justified with a scree plot (cumulative explained variance).
- We also compare `StandardScaler` against L2 normalisation at k = 30 to make sure our scaling choice is reasonable.

### 4. K-Means clustering (Section 5)

- **Coarse search** over k ∈ {10, 20, 30, 40, 50}, evaluated with Silhouette, Calinski-Harabasz, and Davies-Bouldin.
- **Finer search** over k ∈ {5, 10, ..., 50} with both an elbow plot (inertia) and a silhouette curve, so we can check that the chosen k is stable.
- The best k is picked by Silhouette.
- The final K-Means model is fitted with `n_init=10` and `random_state=42`.

### 5. Hierarchical (Agglomerative) clustering (Section 6)

- We try three linkage methods at k = 30: **Ward**, **Complete**, **Average**.
- All three are evaluated with the same internal metrics.
- We select **Ward** linkage because:
  - It minimises within-cluster variance, which is the same assumption K-Means makes — so the two methods are directly comparable.
  - It scored best on Calinski-Harabasz with a competitive Silhouette.
- A dendrogram is plotted on a subset of points to visualise the hierarchical structure.

### 6. DBSCAN clustering (Section 7)

- We plot a **k-distance graph** to guide the choice of ε (`eps`).
- We do a **grid search** over ε ∈ {15, 17, 18, 19, 20, 22, 25} and `min_samples` ∈ {5, 10, 15}.
- For each configuration we report the number of clusters, the noise ratio, and the internal metrics (computed only on non-noise points).
- The best configuration is selected by Silhouette.
- Section 7.10 explains why DBSCAN struggles on this kind of data (high-dimensional embedding geometry, curse of dimensionality, variable density).

### 7. Evaluation metrics

We use five metrics in total:

**Internal metrics** (used to choose the model — no labels needed):

- **Silhouette Score** — measures how compact and well-separated the clusters are. Range $[-1, 1]$, **higher is better**.
- **Calinski-Harabasz Index** — ratio of between-cluster dispersion to within-cluster dispersion. **Higher is better.**
- **Davies-Bouldin Index** — average similarity between each cluster and its most similar one. **Lower is better.**

**External metrics** (used only at the end, against the folder labels):

- **Adjusted Rand Index (ARI)** — agreement between two partitions, corrected for chance. Range $[-1, 1]$, **higher is better**.
- **Normalized Mutual Information (NMI)** — normalised mutual information between predicted clusters and true labels. Range $[0, 1]$, **higher is better**.

### 8. Final comparison (Section 8)

- A **unified summary table** showing internal + external metrics for all three methods.
- A **side-by-side t-SNE** with three panels coloured by K-Means / Hierarchical / DBSCAN labels, using the same 2D embedding for all three (so visual differences come purely from the cluster assignments).
- A **method-agreement matrix** showing pairwise ARI between the methods (independent of ground-truth labels) — this checks whether the methods agree with each other.

---

## Results

> The values below are placeholders. Fill them in from the notebook outputs after a clean run.

### K-Means k-search (coarse)

| k | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---|-----------:|------------------:|---------------:|
| 10 | _[ ]_ | _[ ]_ | _[ ]_ |
| 20 | _[ ]_ | _[ ]_ | _[ ]_ |
| 30 | _[ ]_ | _[ ]_ | _[ ]_ |
| 40 | _[ ]_ | _[ ]_ | _[ ]_ |
| 50 | _[ ]_ | _[ ]_ | _[ ]_ |

**Chosen k = 30** (best Silhouette). The finer search and elbow analysis confirmed this choice.

### Hierarchical linkage comparison (k = 30)

| Linkage | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---------|-----------:|------------------:|---------------:|
| Ward | _[ ]_ | _[ ]_ | _[ ]_ |
| Complete | _[ ]_ | _[ ]_ | _[ ]_ |
| Average | _[ ]_ | _[ ]_ | _[ ]_ |

**Chosen linkage: Ward.**

### DBSCAN — best configuration

| eps | min_samples | n_clusters | noise ratio | Silhouette |
|----:|------------:|-----------:|------------:|-----------:|
| _[ ]_ | _[ ]_ | _[ ]_ | _[ ]_ | _[ ]_ |

### Final comparison

| Method | n_clusters | noise_frac | Silhouette | CH | DB | ARI | NMI |
|--------|-----------:|-----------:|-----------:|----:|----:|-----:|-----:|
| K-Means (k = 30) | 30 | 0% | _[ ]_ | _[ ]_ | _[ ]_ | _[ ]_ | _[ ]_ |
| Hierarchical Ward (k = 30) | 30 | 0% | _[ ]_ | _[ ]_ | _[ ]_ | _[ ]_ | _[ ]_ |
| DBSCAN | _[ ]_ | _[ ]_ | _[ ]_ | _[ ]_ | _[ ]_ | _[ ]_ | _[ ]_ |

### Method agreement (pairwise ARI, between methods)

| | K-Means | Hierarchical | DBSCAN (non-noise) |
|---|---:|---:|---:|
| **K-Means** | 1.00 | _[ ]_ | _[ ]_ |
| **Hierarchical** | _[ ]_ | 1.00 | _[ ]_ |
| **DBSCAN (non-noise)** | _[ ]_ | _[ ]_ | 1.00 |

---

## Findings

- **K-Means is the best method** on this dataset. It scores highest on both internal metrics (Silhouette, Davies-Bouldin) and external metrics (ARI, NMI). The qualitative cluster image grid also shows that K-Means clusters are dominated by a single ground-truth class in most cases.
- **Hierarchical Ward is a close second** and agrees strongly with K-Means (high pairwise ARI). Because the two methods optimise different objectives but reach similar partitions, we can be more confident that the chosen partition reflects real structure in the data.
- **DBSCAN does not work well here** — it labels a very large fraction of points as noise. This is not a hyperparameter problem; it is a structural mismatch between density-based clustering and high-dimensional deep features (explained in Section 7.10 of the notebook).
- **Final choice:** **K-Means with k = 30**, on standardised PCA-50 ResNet50 features.

---

## How to run

### Google Colab (recommended)

1. Upload `main.ipynb` to Google Drive and open it in Colab.
2. Place `patternmind_dataset.zip` in your Drive at `MyDrive/patternmind_dataset.zip`.
3. Click **Runtime → Run all**.

### Local Jupyter

1. Place `patternmind_dataset.zip` next to `main.ipynb`.
2. Open the notebook and run all cells.

The notebook automatically detects whether it is running on Colab and adjusts paths accordingly.

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
images/            Figures generated when the notebook runs (PCA, t-SNE, dendrogram, cluster grids, etc.).
```

The notebook also saves `.npy` files (features, labels, cluster assignments) so later sections can be re-run without redoing feature extraction.

---

## Notes

- All random seeds are fixed (`RANDOM_STATE = 42`), so the results are reproducible.
- The notebook is written to run top-to-bottom from a fresh kernel.
- **Feature extraction is cached.** ResNet50 inference is the slow step; the notebook saves `X_features.npy`, `y_labels.npy`, `label_names.npy`, and `image_paths.npy` on the first run. On every later run those files are loaded from disk and the slow ResNet50 step is skipped automatically. On Colab the cache is also mirrored to `MyDrive/projectml_colab/` so it survives runtime resets.
- To force a fresh re-extraction, delete the four `.npy` cache files (and the Drive mirror, on Colab) before running.
