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

## Pipeline overview

The notebook follows a single linear pipeline from raw images to a final clustering decision. There are four stages:

**1. Exploratory Data Analysis (Section 3).** We start from `patternmind_dataset.zip` — 25,557 images across 233 class folders. We count the classes and images, plot the per-class distribution to expose the class imbalance, check for empty folders, duplicate filenames, and corrupted images, and sample 200 images to inspect their dimensions.

**2. Feature extraction with ResNet50 (Section 4).** Every image is resized to 224 × 224 and passed through ResNet50, pre-trained on ImageNet, with the classification head removed (`include_top=False`) and global average pooling at the top (`pooling='avg'`). Each image is reduced to a 2048-dimensional feature vector. The final feature matrix has shape (25557, 2048) and is cached to disk so this slow step only runs once.

**3. Preprocessing (Section 5).** Features are standardised with `StandardScaler` and projected down to 50 dimensions with PCA. The first 50 components retain roughly 44.1% of the variance — enough to preserve the dominant structure while making distance-based clustering far more tractable. The final clustering input has shape (25557, 50).

**4. Clustering (Sections 5–7).** We then apply three clustering algorithms in parallel to the same preprocessed features:

- **K-Means** with k = 30 (Section 5).
- **Hierarchical (Agglomerative) clustering** with Ward linkage and k = 30 (Section 6).
- **DBSCAN** with ε = 15 and min_samples = 15 (Section 7).

**5. Comparison (Section 8).** All three methods are compared on the same internal metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin) and external metrics (ARI, NMI against the folder labels), along with a side-by-side t-SNE visualisation and a method-agreement matrix. **K-Means is selected as the final model.**
  

## Why these three algorithms?

We deliberately chose one method from each of the three main families of clustering, so that the comparison is meaningful rather than redundant:

| Family | Method | Core assumption |
|---|---|---|
| **Partitional** (centroid-based) | K-Means | Clusters are roughly spherical and equally sized; we know `k` in advance. |
| **Hierarchical** (linkage-based) | Agglomerative (Ward) | Clusters can be built bottom-up by merging the most similar pairs; no need to fix `k` before fitting. |
| **Density-based** | DBSCAN | Clusters are dense regions in feature space separated by low-density gaps; no `k`, but needs ε and min_samples. |

Picking three K-Means-like methods would have told us nothing new; picking three with very different assumptions makes the *method-agreement analysis* in Section 8 meaningful.

---

## Dataset

We use the **PatternMind** image dataset, provided as a ZIP archive (`patternmind_dataset.zip`). It is organised with one folder per class.

Basic statistics (from the EDA section of the notebook):

| Property | Value |
|----------|-------|
| Number of classes | 233 |
| Total images | 25,557 |
| Image formats | `.jpg` |
| Empty folders | 0 |
| Corrupted images | 0 |
| Image dimensions | variable - resized to 224 × 224 for ResNet50 |
| Class balance | imbalanced — most classes have ~50–200 images, but a few are much larger (clutter: 761, airplanes: 720, motorbikes: 719) |

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
- **PCA to 50 dimensions:** justified with a scree plot (cumulative explained variance). The first 50 components explain ~44.1% of the total variance.
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

## Final hyperparameters

All hyperparameters in one place, for reproducibility and grading:

| Stage | Hyperparameter | Value | Where it is set |
|---|---|---|---|
| Global | `RANDOM_STATE` | 42 | Section 1.1 |
| Feature extractor | model | ResNet50 (ImageNet weights) | Section 4.3 |
| Feature extractor | `include_top` | `False` | Section 4.3 |
| Feature extractor | `pooling` | `'avg'` | Section 4.3 |
| Feature extractor | input size | 224 × 224 | Section 4.2 |
| Preprocessing | scaler | `StandardScaler` | Section 5 |
| Preprocessing | PCA components | 50 | Section 5 |
| K-Means | `n_clusters` (k) | 30 | Section 5.3 |
| K-Means | `n_init` | 10 | Section 5.4 |
| Hierarchical | linkage | Ward | Section 6.2 |
| Hierarchical | `n_clusters` | 30 | Section 6 |
| DBSCAN | `eps` | 15 | Section 7.4 |
| DBSCAN | `min_samples` | 15 | Section 7.4 |
| t-SNE | `perplexity` | 30 | Sections 5.7, 6.5, 7.8, 8.5 |
| t-SNE | `n_iter` | 1000 | same as above |

---

## Results

### K-Means k-search (coarse)

| k | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---|-----------:|------------------:|---------------:|
| 10 | 0.0607 | 797.06 | 2.8270 |
| 20 | 0.0800 | 586.19 | 2.6637 |
| 30 | **0.0911** | 482.13 | 2.5437 |
| 40 | 0.0888 | 406.90 | **2.5180** |
| 50 | 0.0858 | 355.29 | 2.5715 |

**Chosen k = 30** (best Silhouette). The finer search and elbow analysis confirmed this choice.

### Hierarchical linkage comparison (k = 30)

| Linkage | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---------|-----------:|------------------:|---------------:|
| Ward | 0.0545 | **382.22** | 2.9087 |
| Complete | 0.0192 | 204.23 | 3.0304 |
| Average | **0.0804** | 9.41 | **1.4517** |

**Chosen linkage: Ward.**

### DBSCAN — best configuration

| eps | min_samples | n_clusters | noise ratio | Silhouette |
|----:|------------:|-----------:|------------:|-----------:|
| 15 | 15 | 38 | 82.95% | 0.2381 |

### Final comparison

| Method | n_clusters | noise_frac | Silhouette | CH | DB | ARI | NMI |
|--------|-----------:|-----------:|-----------:|----:|----:|-----:|-----:|
| K-Means (k = 30) | 30 | 0% | **0.0911** | **482.13** | **2.5437** | **0.1719** | 0.5223 |
| Hierarchical Ward (k = 30) | 30 | 0% | 0.0545 | 382.22 | 2.9087 | 0.1565 | **0.5370** |
| DBSCAN | 38 | 82.95% | 0.2381 | 195.95 | 1.3999 | 0.0023 | 0.2282 |

Note: DBSCAN's internal metrics are computed only on non-noise points (4,358 / 25,557). Among methods that cluster the full dataset, K-Means wins on every metric except NMI, where Hierarchical Ward edges it out by ~0.015.

### Method agreement (pairwise ARI, between methods)

| | K-Means | Hierarchical | DBSCAN (non-noise) |
|---|---:|---:|---:|
| **K-Means** | 1.00 | 0.4087 | 0.7479 |
| **Hierarchical** | 0.4087 | 1.00 | 0.6813 |
| **DBSCAN (non-noise)** | 0.7479 | 0.6813 | 1.00 |

---

## Findings

- **K-Means is the best method** on this dataset. Among methods that cluster the full dataset, it scores highest on both internal metrics (Silhouette, Davies-Bouldin) and external metrics (ARI, NMI). The qualitative cluster image grid also shows that K-Means clusters are dominated by a single ground-truth class in most cases.
- **Hierarchical Ward is a close second** and agrees strongly with K-Means (high pairwise ARI). Because the two methods optimise different objectives but reach similar partitions, we can be more confident that the chosen partition reflects real structure in the data.
- **DBSCAN does not work well here** — it labels a very large fraction of points as noise. This is not a hyperparameter problem; it is a structural mismatch between density-based clustering and high-dimensional deep features (explained in Section 7.10 of the notebook).
- **Final choice:** **K-Means with k = 30**, on standardised PCA-50 ResNet50 features.

---

## Limitations

We're aware of the following limitations in the analysis:

- **k = 30 vs. 233 ground-truth classes.** The chosen number of clusters is much smaller than the true number of classes. This is a known property of internal metrics on high-dimensional CNN embeddings: Silhouette tends to peak at small k and decays as k grows, because in 50-dimensional space cluster boundaries become geometrically less clean. Our final partition is therefore a *coarsening* of the true label structure, not a full recovery.
- **DBSCAN was not given alternatives.** We only tried DBSCAN itself. Density-based methods that handle variable density better (HDBSCAN, OPTICS) would be a natural next step, but were out of scope for this project.
- **Feature extractor is fixed.** ResNet50 with ImageNet weights is a strong general-purpose backbone, but a self-supervised model trained directly on this dataset (SimCLR, DINO) would likely produce embeddings whose geometry is better matched to the underlying classes.
- **Class imbalance was not addressed.** We did not sub-sample dominant classes or weight metrics — imbalance is passed through to the clustering step as-is.
- **External validation only against folder labels.** We did not run any human-judged evaluation of cluster quality beyond the qualitative image grids shown in the notebook.

---

## How to run

### Google Colab (recommended)

1. Upload `main.ipynb` to Google Drive and open it in Colab.
2. Place `patternmind_dataset.zip` in your Drive at `MyDrive/patternmind_dataset.zip`.
3. (Optional but recommended) Set the runtime to **GPU**: `Runtime → Change runtime type → GPU`. This only matters for the first ResNet50 inference pass.
4. Click **Runtime → Run all**.

### Local Jupyter

1. Place `patternmind_dataset.zip` next to `main.ipynb` (or set `PATTERNMIND_ZIP=/path/to/zip`).
2. Open the notebook and run all cells.

The notebook automatically detects whether it is running on Colab and adjusts paths accordingly.

### Expected runtime

- ResNet50 feature extraction: ~10–20 min on a Colab GPU. Only happens on the first run; cached afterwards.
- Clustering + hyperparameter sweeps + visualisations: ~15–25 min total, dominated by the two full-dataset t-SNE projections.

---

## Requirements

Python 3.10 or newer. On Colab everything is pre-installed. For local use:

```bash
pip install numpy pandas matplotlib seaborn tqdm pillow scikit-learn scipy tensorflow
```

A GPU helps with feature extraction but is not required.

---

## Repository structure
---

## Key figures (in `images/`)

The notebook saves the following plots to disk when it runs end-to-end. They are the artefacts referenced in the report.

| File / plot | What it shows |
|---|---|
| Class-distribution histogram | The imbalance across the 233 folders. |
| PCA scree plot | Cumulative explained variance vs. number of components. |
| K-Means silhouette curve | Silhouette score as a function of k (fine sweep). |
| K-Means elbow plot | Inertia (within-cluster sum of squares) plotted against k, used to identify the "elbow" where adding more clusters stops meaningfully reducing inertia. |
| K-Means PCA-2D scatter | Final K-Means labels projected onto the first two principal components. |
| K-Means t-SNE | Final K-Means labels in the t-SNE embedding. |
| Cluster image grid (K-Means) | Sample images per cluster, for qualitative inspection. |
| Hierarchical dendrogram | Ward-linkage dendrogram on a random subset. |
| Hierarchical t-SNE | Ward labels in the t-SNE embedding. |
| DBSCAN k-distance plot | Sorted k-th nearest-neighbour distances, used to choose ε. |
| DBSCAN t-SNE | DBSCAN labels (including noise) in the t-SNE embedding. |
| Side-by-side t-SNE comparison | Three panels, same 2D embedding, coloured by each method's labels. |
| Method-agreement heatmap | Pairwise ARI between the three methods. |

---

## Notes

- All random seeds are fixed (`RANDOM_STATE = 42`), so the results are reproducible.
- The notebook is written to run top-to-bottom from a fresh kernel.
- **Feature extraction is cached.** ResNet50 inference is the slow step; the notebook saves `X_features.npy`, `y_labels.npy`, `label_names.npy`, and `image_paths.npy` on the first run. On every later run those files are loaded from disk and the slow ResNet50 step is skipped automatically. On Colab the cache is also mirrored to `MyDrive/projectml_colab/` so it survives runtime resets.
- To force a fresh re-extraction, delete the four `.npy` cache files (and the Drive mirror, on Colab) before running.

---

## Acknowledgements

- Course: **Machine Learning**, Luiss Guido Carli, 2025/2026.
- Dataset: **PatternMind**, provided as part of the course materials.
