# Unsupervised Image Clustering via CNN Feature Extraction

Course: Machine Learning
Project Type: Unsupervised Learning (Clustering)
Team Members:


---

## 1. Introduction

The objective of this project is to explore the structure of an unlabeled image dataset using **unsupervised learning techniques**.
Since no ground-truth labels are available, the goal is not classification accuracy, but rather the identification of **meaningful and interpretable clusters** based on visual similarity.

To achieve this, we design a complete machine learning pipeline that combines **deep learning–based feature extraction** with **classical clustering algorithms**. High-level visual features are extracted using a pre-trained Convolutional Neural Network (CNN), and multiple clustering approaches are then applied and compared using appropriate internal evaluation metrics.

The project follows a principled experimental workflow, moving from data exploration and feature extraction to clustering, evaluation, and final model selection.

---

## 2. Methods

### 2.1 Data Exploration

We begin with a light exploratory analysis to understand the dataset structure, the number of images, and their basic properties.
Because the task is unsupervised, the purpose of this step is not label analysis, but rather to identify potential issues (e.g., corrupted images, strong imbalance, or anomalies) that could affect downstream modeling.

---

### 2.2 Feature Extraction with CNNs

Directly clustering raw pixel values is ineffective due to the high dimensionality and lack of semantic structure.
To address this, we use a **pre-trained ResNet50** model as a fixed feature extractor.

Specifically:

* Images are resized and preprocessed according to the ResNet50 requirements.
* The network’s convolutional layers are used to extract high-level feature representations.
* The resulting feature vectors encode semantic visual information such as shapes, textures, and object structure.

This approach allows us to leverage deep representations learned on large-scale image datasets while keeping the clustering task unsupervised.

---

### 2.3 Preprocessing and Dimensionality Reduction

Since clustering algorithms based on distance metrics are sensitive to feature scale, all extracted feature vectors are **standardized**.

To reduce dimensionality and improve computational efficiency, **Principal Component Analysis (PCA)** is applied:

* PCA is used to project the data into a lower-dimensional space while retaining most of the variance.
* The reduced representation is used for clustering and visualization.
* PCA is also employed for 2D visual inspection of cluster structure.

---

### 2.4 Clustering Algorithms

We apply and compare three clustering methods representing different modeling assumptions:

* **K-Means Clustering**
  A centroid-based method that partitions the data into a fixed number of clusters by minimizing intra-cluster variance.
  It is used as the **baseline** due to its simplicity, interpretability, and widespread use.

* **Hierarchical Clustering (Agglomerative)**
  A bottom-up approach that builds clusters incrementally.
  This method provides additional interpretability through hierarchical structure and does not rely on random initialization.

* **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
  A density-based method capable of discovering arbitrarily shaped clusters and explicitly identifying noise points, without requiring the number of clusters in advance.

All methods are applied to the same preprocessed feature representation to ensure a fair comparison.

---

## 3. Experimental Design

### Purpose

The main experiment aims to determine which clustering approach best captures the underlying structure of the image feature space.

### Baseline

K-Means clustering is used as the baseline method due to its efficiency and interpretability.

### Evaluation Metrics

Because no labels are available, we rely on **internal clustering metrics**:

* **Silhouette Score** (higher is better): measures cluster separation and cohesion.
* **Davies–Bouldin Index** (lower is better): evaluates intra-cluster compactness and inter-cluster separation.
* **Calinski–Harabasz Index** (higher is better): measures the ratio of between-cluster to within-cluster dispersion.

These metrics provide complementary perspectives on clustering quality without requiring ground-truth annotations.

---

## 4. Results

### Quantitative Results

Across the evaluated configurations:

* **K-Means** consistently achieves strong internal metric scores, indicating compact and well-separated clusters.
* **Hierarchical Clustering** produces comparable results, with slightly higher computational cost but improved interpretability.
* **DBSCAN** is highly sensitive to hyperparameter choices and often labels a significant fraction of points as noise, leading to less stable clustering outcomes for this dataset.

### Qualitative Analysis

PCA and t-SNE visualizations support the quantitative findings:

* K-Means and Hierarchical Clustering yield visually coherent clusters.
* DBSCAN identifies noise effectively but struggles to form consistent cluster structures given the data density distribution.

Representative visualizations generated by the code are included in the `images/` folder and referenced in this report.

---

## 5. Conclusions

This project demonstrates how **deep feature extraction combined with classical clustering methods** can effectively uncover structure in unlabeled image data.

Based on both quantitative metrics and qualitative inspection, **K-Means clustering** emerges as the most suitable method for this dataset, offering a strong balance between clustering quality, interpretability, and computational efficiency.
Hierarchical Clustering represents a valid alternative when hierarchical relationships are of interest, while DBSCAN is less appropriate for this particular data distribution.

### Limitations and Future Work

The primary limitation of this work is the absence of ground-truth labels, which restricts evaluation to internal metrics.
Future extensions could include:

* Exploring alternative CNN architectures for feature extraction.
* Applying non-linear dimensionality reduction techniques.
* Validating clusters through downstream tasks or domain-specific analysis.
* Investigating hybrid or ensemble clustering approaches.

---

## Reproducibility

All experiments are fully reproducible using the provided `main.ipynb` notebook.
The notebook contains the complete pipeline, from data loading and feature extraction to clustering, evaluation, and visualization.
