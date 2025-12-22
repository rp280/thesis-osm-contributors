# Clustering Analysis of OSM Contributor Data

This repository contains a pipeline for clustering analysis of OpenStreetMap (OSM) contributor data.  
The objective is to identify patterns in editing behavior and to group contributors into clusters using KMeans and Gaussian Mixture Models (GMM).

---

## Pipeline Overview

### 1. Data Preparation
- Input: Parquet table of OSM user summary statistics.
- Filtering: Only users with more than 10 edits are considered. Early leavers are excluded.
- Removal of non-relevant or redundant columns (e.g. IDs, timestamps, metadata).

### 2. Feature Scaling and Transformation
- Features are standardized depending on distribution:
  - Standard scaling for near-normal distributions.
  - Log transformation plus standard scaling for highly skewed features.
- Missing values are imputed:
  - `days_to_50`: replaced with `active_duration + 1`.
  - Ratios (e.g. `comment_length_ratio`, `top_feature_ratio`): replaced with 0.
- Only scaled features are used for clustering.

### 3. Principal Component Analysis (PCA)
- Dimensionality reduction is applied to control noise and redundancy.
- Number of components is determined by explained variance thresholds:
  - 80% cumulative variance.
  - 90% cumulative variance (used for the main analysis).
- Outputs:
  - Scree plot (variance explained).
  - PCA loadings (CSV).

### 4. KMeans Clustering
- Tested for k = 2–14.
- Evaluation metrics:
  - Elbow method (distortion/inertia).
  - Silhouette score.
  - Davies-Bouldin index.
  - Calinski-Harabasz index.
- The final k is chosen as a balance between these methods and interpretability.
- Outputs:
  - Elbow and silhouette plots.
  - CSV with evaluation metrics.
  - Cluster profiles (heatmaps of top-variance features, original values per cluster).

### 5. Gaussian Mixture Model (GMM)
- Tested for 2–16 components.
- Model selection based on Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC).
- Final number of components chosen based on AIC, with BIC used for comparison.
- Outputs:
  - BIC/AIC plot.
  - Cluster profiles (heatmaps of top-variance features, original values per cluster).

### 6. Results
- Plots:
  - PCA scree plot.
  - KMeans elbow and silhouette plots.
  - Cluster heatmaps (KMeans and GMM).
  - GMM BIC/AIC curves.
- Tables:
  - PCA loadings.
  - Evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz).
  - Cluster profiles (mean values of original features).
- Data:
  - Cluster assignments (Parquet for KMeans and GMM).
  - Cluster profiles (CSV for KMeans and GMM).

---

## PLOTS

### KMeans

- **PCA Scree plot**  
  ![PCA Scree Plot](plots/KMeans/pca_scree_plot_2025-09-26_15-35-08.png)

- **KMeans Elbow method**  
  ![KMeans Elbow method](plots/KMeans/elbow_kmeans_2025-09-29_19-46-18.png)

- **KMeans Silhouette index**  
  ![KMeans Silhouette index](plots/KMeans/silhouette_kmeans_2025-09-29_19-46-18.png)

- **Davies-Bouldin Score**  
  ![Davies-Bouldin Score](plots/KMeans/davies_bouldin_kmeans_2025-09-29_19-46-18.png)

- **Calinski-Harabasz Score**  
  ![Calinski-Harabasz Score](plots/KMeans/calinski_harabasz_kmeans_2025-09-29_19-46-18.png)

- **Cluster heat map k=5 and top 20 variables**  
  ![Cluster heat map K=5](plots/KMeans/cluster_profiles_kmeans_2025-09-29_19-46-18.png)

- **Cluster heat map k=6 and top 30 variables**  
  ![Cluster heat map K=6](plots/KMeans/important_30_cluster_profiles_with_k_6_2025-09-24_15-27-58.png)


## Gauss Mixture Model (GMM)

- **K selection with BIC/AIC for GMM**  
  ![K selection with BIC/AIC for GMM](plots/Gauss/gmm_bic_aic_2025-09-29_19-46-18.png)

- **Final GMM Cluster profile with top 20 variables**  
  ![Final GMM Cluster profile with top 20 variables](plots/Gauss/cluster_profiles_gmm_2025-09-29_19-46-18.png)

- **GMM Cluster profile with all variables**  
  ![GMM Cluster profile with all variables](plots/Gauss/cluster_profiles_gmm_2025-09-24_15-27-58.png)

## Results
### Clustering Metrics (KMeans for Silhouette, Davies-Bouldin, Calinski-Harabasz different k)

**Silhoutte** measures how well the points in the CLuster fits to the points of other Clusters (-1 to 1).  
**Davies-Bouldin** measures the relation between Cluster similarity to CLuster disparity.  
**Calinski-Harabasz** measures the relation between the variance in clusters to variances in other clusters. 
| K  | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|----|------------|----------------|-------------------|
| 4  | 0.1461     | 1.9274         | 1314.79           |
| 5  | 0.1660     | 1.7710         | 1264.55           |
| 6  | 0.1521     | 1.9757         | 1148.81           |
| 7  | 0.1498     | 1.9446         | 1056.56           |
| 8  | 0.1499     | 1.8720         | 1002.10           |
| 9  | 0.1486     | 1.8558         | 934.11            |
| 10 | 0.1551     | 1.7860         | 901.50            |
| 11 | 0.1380     | 1.8937         | 827.16            |
| 12 | 0.1377     | 1.8969         | 823.97            |
| 13 | 0.1392     | 1.8648         | 785.13            |
| 14 | 0.1272     | 1.8573         | 759.67            |
| 15 | 0.1271     | 1.8376         | 740.13            |



## Script Duration
The script was running for 2463 sec.