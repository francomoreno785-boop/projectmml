# Unsupervised image clustering via CNN feature extraction

**Course:** Machine Learning

**Project type:** Unsupervised learning (clustering)

**Team members:**


## 1. Introduction

The goal of this project is to analyze the structure of an unlabeled image dataset using unsupervised learning techniques.
Since no ground-truth labels are available, the objective is not classification accuracy, but rather the discovery of meaningful and interpretable clusters based on visual similarity.

To achieve this, we design a complete machine learning pipeline that combines deep learning–based feature extraction with classical clustering algorithms. High-level visual features are extracted using a pre-trained convolutional neural network (CNN), after which multiple clustering approaches are applied and compared using internal evaluation metrics.

The project follows a structured workflow from data exploration and feature extraction to clustering, evaluation, and final model selection.

## 2. Methods

### 2.1 Data exploration

We begin with a light exploratory analysis to understand the size and structure of the dataset.
Because the task is unsupervised, the purpose of this step is not label analysis, but rather to identify potential issues such as corrupted images, extreme imbalance, or anomalies that could affect clustering behavior.

### 2.2 Feature extraction with CNNs

Clustering raw pixel values is ineffective due to their high dimensionality and lack of semantic structure.
To address this, we use a pre-trained ResNet50 model as a fixed feature extractor.

Specifically:

* Images are resized and preprocessed according to the ResNet50 requirements.
* The convolutional layers of the network are used to extract high-level feature vectors.
* These feature vectors encode semantic visual information such as shapes, textures, and object structure.

This approach allows us to leverage deep representations learned on large-scale image datasets while keeping the clustering task fully unsupervised.

### 2.3 Preprocessing and dimensionality reduction

Since distance-based clustering algorithms are sensitive to feature scale, all extracted feature vectors are standardized.

To reduce dimensionality and improve computational efficiency, principal component analysis (PCA) is applied:

* PCA projects the data into a lower-dimensional space while retaining most of the variance.
* The reduced representation is used for clustering.
* PCA and t-SNE are also employed for qualitative visualization of cluster structure.

### 2.4 Clustering algorithms

We apply and compare three clustering methods representing different modeling assumptions.

**K-Means clustering**
A centroid-based method that partitions the data into a fixed number of clusters by minimizing intra-cluster variance.
It is used as the baseline due to its simplicity and interpretability.

**Hierarchical clustering (agglomerative)**
A bottom-up approach that incrementally merges samples into clusters.
This method provides additional interpretability through hierarchical structure and does not rely on random initialization.

**DBSCAN (density-based spatial clustering of applications with noise)**
A density-based method capable of discovering arbitrarily shaped clusters and explicitly identifying noise points, without requiring the number of clusters in advance.

All methods are applied to the same preprocessed feature representation to ensure a fair comparison.

## 3. Experimental design

### Purpose

The experiment aims to determine which clustering approach best captures the underlying structure of the image feature space.

### Baseline

K-Means is used as the baseline method due to its efficiency, simplicity, and widespread use.

### Evaluation metrics

Because no labels are available, we rely on internal clustering metrics:

* Silhouette score (higher is better)
* Davies–Bouldin index (lower is better)
* Calinski–Harabasz index (higher is better)

For DBSCAN, we additionally consider the fraction of points labeled as noise, since high metric values may be obtained by clustering only a subset of the data.

## 4. Results

### Quantitative results

The clustering methods exhibit different behaviors:

* K-Means produces stable and interpretable clusters with strong baseline performance.
* Hierarchical clustering achieves comparable results, with additional interpretability at the cost of higher computational complexity.
* DBSCAN often achieves the strongest internal clustering metrics, indicating very well-separated clusters among the points it assigns to clusters.

However, DBSCAN also labels a portion of the dataset as noise, which requires careful interpretation.

### DBSCAN noise interpretation

To avoid selecting a model that clusters only a very small subset of the data, we introduce a noise fraction threshold.
DBSCAN is preferred only if the fraction of points labeled as noise does not exceed 30%.
If this threshold is exceeded, we fall back to the method with the strongest overall clustering metrics.

This criterion provides a balanced trade-off between cluster quality and coverage of the dataset.

### Qualitative analysis

PCA and t-SNE visualizations support the quantitative findings:

* K-Means and hierarchical clustering yield coherent and interpretable clusters.
* DBSCAN forms very compact clusters for core points while explicitly identifying ambiguous or isolated samples as noise.

Representative visualizations generated by the code are included in the `images/` folder.

## 5. Conclusions

This project demonstrates how deep feature extraction combined with classical clustering methods can effectively uncover structure in unlabeled image data.

Based on the internal metric comparison and the imposed noise threshold, DBSCAN is selected as the final method.
It provides the strongest cluster separation among the points it assigns to clusters, while also offering a principled way to handle outliers through noise detection.

K-Means and hierarchical clustering remain strong alternatives, particularly when full data coverage or simpler interpretability is preferred.

## Reproducibility

All experiments are fully reproducible using the provided `main.ipynb` notebook.
The notebook contains the complete pipeline, from data loading and feature extraction to clustering, evaluation, and visualization.
