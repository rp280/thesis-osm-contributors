# OSM Contributor Analysis - Master's Thesis

## Overview  
This repository contains the code and workflows developed for my Master's thesis in Geography/Geoinformatics.  
The project is guided by two central research questions:

1. **Prediction of Retention**  
   Can it be predicted whether an OpenStreetMap (OSM) contributor remains active or stops contributing within the first six months after their initial edit?

2. **Contributor Categorization**  
   Can OSM contributors be grouped into categories based on a variety of quantitative contribution metrics?

To address these questions, the project integrates large-scale OSM history data, advanced preprocessing workflows in **Python** and **SQL (DuckDB)**, and machine learning approaches.

---

## Data Source  
- This project relies on **preprocessed OpenStreetMap full-history data**.  
- The raw `.pbf` history files were already transformed into **GeoParquet** format using [**ohsome-planet**](https://github.com/GIScience/ohsome-planet), a tool developed at [**HeiGIT (Heidelberg Institute for Geoinformation Technology)**](https://heigit.org).  
- These enriched and extracted GeoParquet datasets (including geometries and changeset metadata) are hosted in a **MinIO storage system**.  
- The workflows in this repository therefore start directly from the **GeoParquet data in MinIO**, without requiring local execution of ohsome-planet.  

---

## Preprocessing Workflow  

The preprocessing pipeline is designed to reduce the massive OSM dataset step by step, while retaining key features for analysis.

1. **Country-level Tables**  
   - For each country, a reduced table is created.  
   - Only the required columns are selected.  
   - Additional derived columns are added (e.g., feature categories, newly added or removed tags).  

2. **Daily User-level Tables**  
   - Aggregation of edits **per user and per day**, across all countries.  
   - Variables are summed or averaged (e.g., number of buildings edited, length of changeset comments, feature ratios).  
   - Result: a more compact representation of user activity per day.  

3. **Final User Table**  
   - Combines daily activity into a single row per contributor.  
   - Contains a wide range of variables (temporal, spatial, feature-type-related, contribution-specific).  
   - Serves as the **fundamental dataset** for both prediction and clustering.  

---

## Analysis  

### 1. Prediction of Contributor Retention  
- **Goal**: Supervised classification (active vs. left early).  
- **Approach**:  
  - Early activity window (first 180 days after first edit).  
  - Feature engineering covering temporal, spatial, and behavioral aspects.  
  - Baseline model: **Random Forest Classifier**; further experiments with linear models.  
- **Output**: Performance metrics, feature importances, insights into predictors of early attrition.  

---

### 2. Contributor Categorization (Behavioral Analysis)  
- **Goal**: Identify typical contributor roles and behavioral clusters.  
- **Pipeline**: Implemented as `OSMClusteringPipeline` (see `src/`).  
  - **Feature Transformation**: Missing values imputed, scaling/log transforms applied.  
  - **Dimensionality Reduction**: PCA until ≥ 90 % variance explained.  
  - **KMeans Clustering**:  
    - Evaluation via Silhouette, Davies-Bouldin, Calinski-Harabasz, Elbow method.  
    - Automatic plots + metrics export.  
  - **Gaussian Mixture Models (GMM)**:  
    - Model selection via AIC/BIC.  
    - Heatmaps of top discriminative features per cluster.  
- **Outputs**:  
  - Cluster labels stored as Parquet.  
  - PCA scree plots, evaluation metrics, and cluster heatmaps stored in `/results` and `/plots`.  

### Reference Data

The file `data/boundaries/world/world_boundaries_overture_iso_a3.parquet` is a static reference dataset
used for country-level joins.  
It is included in the repository for reproducibility and convenience and
should not be regenerated or modified.

**Main Script Example:**  
```bash
 poetry run python 001_preprocessing.py DEU POL
```
