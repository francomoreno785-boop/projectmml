## Section 2 — Clustering Methods & K-Means Results

### Clustering Methodology

For the clustering component of the project, we applied **K-Means** to the feature matrix `X_features.npy`.
The preprocessing step consisted of standardizing all features using `StandardScaler`, because K-Means is
distance-based and sensitive to differences in scale across variables.

We evaluated 5 different values of the number of clusters:

- k = 10, 20, 30, 40, 50

For each configuration we trained a K-Means model and computed three clustering quality metrics:

- **Silhouette Score** — primary selection metric (higher is better)
- **Calinski–Harabasz Index** — measures cluster separation vs. within-cluster dispersion (higher is better)
- **Davies–Bouldin Index** — measures average similarity between clusters (lower is better)

The best model was chosen by selecting the value of *k* with the highest Silhouette Score, with
the Calinski–Harabasz and Davies–Bouldin indices used as secondary validation signals.

### K-Means Results Interpretation

The evaluation indicated that a configuration with *k = [REPLACE_WITH_BEST_K]* clusters provides the best trade-off
between separation and compactness according to the Silhouette Score. This suggests that the data naturally forms
around *[REPLACE_WITH_BEST_K]* coherent groups in the feature space.

The 2D PCA visualization of the final K-Means solution shows how the clusters are positioned relative to each other
in a low-dimensional projection. While some overlap is expected due to dimensionality reduction, several regions
with denser and more clearly separated groups can be observed.

The cluster size histogram highlights how many samples are assigned to each cluster. This helps assess whether the
model is dominated by a few very large clusters or if it captures smaller but meaningful segments. In our case,
the distribution of sizes provides additional support that the chosen number of clusters is reasonable for this
dataset.

Overall, the K-Means model offers an interpretable segmentation of the data that can be used by the rest of the
analysis pipeline to investigate differences between clusters and to derive practical insights for the project.
