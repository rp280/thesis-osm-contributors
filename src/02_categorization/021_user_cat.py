from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import os


# ----------------------------- CONFIG -------------------------------- #

BASE_DIR = Path.cwd()
print("Base dir:", BASE_DIR)
PROJECT_ROOT = BASE_DIR.parents[1]


DATA_PATH = (
    PROJECT_ROOT 
    / "results" 
    / "00_preprocessing" 
    / "user_summary" 
    / "cat.parquet"
).as_posix()

PLOTS_DIR = PROJECT_ROOT / "results" / "02_categorization" / "plots" 
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = PROJECT_ROOT / "results" / "02_categorization" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------- CLASS AND FUNCTIONS ------------------- #

class OSMClusteringPipeline:
    def __init__(self, parquet_path, output_dir="results", plot_dir="plots"):
        self.parquet_path = parquet_path
        self.output_dir = output_dir
        self.plot_dir = plot_dir
        self.current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        # Container for Data
        self.df_filtered_full = None
        self.df_final = None
        self.X_pca = None
        self.labels_kmeans = None
        self.labels_gmm = None

    # ---------- Helper Functions ----------
    @staticmethod
    def find_elbow(k_values, distortions):
        x1, y1 = k_values[0], distortions[0]
        x2, y2 = k_values[-1], distortions[-1]
        distances = []
        for i in range(len(k_values)):
            x0, y0 = k_values[i], distortions[i]
            num = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
            den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distances.append(num/den)
        return k_values[np.argmax(distances)]

    @staticmethod
    def evaluate_clustering(X, labels, sample_size=None):
        if sample_size and X.shape[0] > sample_size:
            idx = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[idx]
            labels_sample = labels[idx]
        else:
            X_sample = X
            labels_sample = labels
        return {
            'silhouette': float(silhouette_score(X_sample, labels_sample)),
            'davies_bouldin': float(davies_bouldin_score(X_sample, labels_sample)),
            'calinski_harabasz': float(calinski_harabasz_score(X_sample, labels_sample))
        }

    @staticmethod
    def find_optimal_n(data, n_range, threshold=0.01):
        """
        Choose the first n where improvement <= threshold.
        """
        data = np.array(data)
        improvements = (data[:-1] - data[1:]) / np.abs(data[:-1])

        for i, imp in enumerate(improvements):
            print(f"n={n_range[i+1]}, improvement={imp:.4f}")
            if imp <= threshold:
                return n_range[i]  # take the cluster count before it drops below threshold
        return n_range[-1]  # fallback: the last n if never below threshold


    # ---------- Pipeline Steps ----------
    def load_and_filter(self):
        df = pd.read_parquet(self.parquet_path)
        self.df_filtered_full = df[(df["total_edits"] > 10) & (df["left_early"] == False)].copy()

        drop_features = [
            'days_to_100', 'active_week_ratio', 'edits_per_span_day', 'changesets_per_edit_day',
            'entropy_country_distribution', 'max_country_edits', 'social_facility_ratio',
            'body_of_water_ratio', 'financial_service_ratio', 'wash_facility_ratio', 'place_ratio',
            'waterway_ratio', 'activity_ratio_true', 'josm_changeset_ratio',
            'streetccomplete_changeset_ratio', 'user_id', 'first_edit', 'full_last_edit',
            'first_edit_year', 'first_edit_month', 'top_country', 'top_feature_type_name', 'left_early'
        ]

        self.df_filtered_full.drop(columns=drop_features, errors="ignore", inplace=True)
        print("Filtered Data shape:", self.df_filtered_full.shape)

    def transform_features(self):
        df = self.df_filtered_full.copy()
        # Fill missing values
        df["days_to_50"] = df["days_to_50"].fillna(df["active_duration"] + 1)
        df["burstiness_score"] = df["burstiness_score"].fillna(0)
        df["comment_length_ratio"] = df["comment_length_ratio"].fillna(0)
        df["top_feature_ratio"] = df["top_feature_ratio"].fillna(0)

        # Apply scaling
        scaled_cols = []
        for col in df.columns:
            if not np.issubdtype(df[col].dtype, np.number):
                continue
            vals = df[col].dropna()
            if vals.empty:
                continue
            min_val, max_val = vals.min(), vals.max()
            skew = vals.skew()
            if min_val >= 0 and max_val <= 1.5:
                transform = "StandardScaler"
            elif skew > 2:
                transform = "log1p + StandardScaler"
            else:
                transform = "StandardScaler"

            if transform == "log1p + StandardScaler":
                vals = np.log1p(df[col])
            else:
                vals = df[col]
            new_col = f"{col}_scaled"
            df[new_col] = StandardScaler().fit_transform(vals.values.reshape(-1, 1))
            scaled_cols.append(new_col)

        self.df_final = df[scaled_cols]
        print("Transformed features shape:", self.df_final.shape)

    def run_pca(self, var_threshold=0.9):
        pca = PCA(n_components=var_threshold)
        self.X_pca = pca.fit_transform(self.df_final)

        explained = np.cumsum(pca.explained_variance_ratio_)
        plt.figure(figsize=(8,5))
        plt.plot(range(1, len(explained)+1), explained, marker='o')
        plt.axhline(var_threshold, color='r', linestyle='--', label=f"{int(var_threshold*100)}% threshold")
        plt.xlabel("Number of principal components")
        plt.ylabel("Cumulative explained variance")
        plt.title("Scree Plot / Variance explained")
        plt.legend()
        plt.savefig(f"{self.plot_dir}/pca_scree_plot_{self.current_time}.png")
        plt.close()

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)],
            index=self.df_final.columns
        )
        loadings.to_csv(f"{self.output_dir}/pca_loadings_{self.current_time}.csv")
        print("PCA finished, X_pca shape:", self.X_pca.shape)

    def run_kmeans(self, candidate_k=range(2, 16), fixed_n=None, sample_size=10000):
        if fixed_n is None:
            distortions = []
            silhouette_results = {}
            results = []

            for k in candidate_k:
                print(f"Testing KMeans for k={k} ...")
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
                labels = kmeans.fit_predict(self.X_pca)
                distortions.append(kmeans.inertia_)

                # Silhouette (subsample if too large)
                if len(self.X_pca) > sample_size:
                    X_sample, y_sample = resample(self.X_pca, labels,
                                                n_samples=sample_size,
                                                random_state=42)
                    sil = silhouette_score(X_sample, y_sample)
                else:
                    sil = silhouette_score(self.X_pca, labels)
                silhouette_results[k] = sil

                # Other metrics
                metrics = self.evaluate_clustering(self.X_pca, labels, sample_size=sample_size)
                metrics["k"] = k
                metrics["inertia"] = kmeans.inertia_
                results.append(metrics)

            # choose k
            best_k_elbow = self.find_elbow(list(candidate_k), distortions)
            best_k_silhouette = max(silhouette_results, key=silhouette_results.get)
            print("Best k (Elbow):", best_k_elbow)
            print("Best k (Silhouette):", best_k_silhouette)

            # Save results
            pd.DataFrame(results).set_index("k").to_csv(
                f"{self.output_dir}/metrics_kmeans_{self.current_time}.csv"
            )
            # Silhouette plot
            plt.figure(figsize=(6,4))
            plt.plot(list(candidate_k), silhouette_results.values(), marker="o")
            plt.xlabel("Number of clusters")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette Scores for KMeans")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{self.plot_dir}/silhouette_kmeans_{self.current_time}.png")
            plt.close()

            # Elbow plot
            plt.figure(figsize=(6,4))
            plt.plot(list(candidate_k), distortions, marker="o")
            plt.xlabel("Number of clusters (k)")
            plt.ylabel("Distortion (Inertia)")
            plt.title("Elbow Method")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{self.plot_dir}/elbow_kmeans_{self.current_time}.png")
            plt.close()

            # Davies-Bouldin plot
            db_scores = [res['davies_bouldin'] for res in results]
            plt.figure(figsize=(6,4))
            plt.plot(list(candidate_k), db_scores, marker="o")
            plt.xlabel("Number of clusters (k)")
            plt.ylabel("Davies-Bouldin Score")
            plt.title("Davies-Bouldin Scores for KMeans")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{self.plot_dir}/davies_bouldin_kmeans_{self.current_time}.png")
            plt.close()

            # Calinski-Harabasz plot
            ch_scores = [res['calinski_harabasz'] for res in results]   
            plt.figure(figsize=(6,4))
            plt.plot(list(candidate_k), ch_scores, marker="o")
            plt.xlabel("Number of clusters (k)")
            plt.ylabel("Calinski-Harabasz Score")
            plt.title("Calinski-Harabasz Scores for KMeans")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{self.plot_dir}/calinski_harabasz_kmeans_{self.current_time}.png")
            plt.close()

            # pick one of the best (e.g. silhouette)
            best_k = best_k_silhouette
        else:
            best_k = fixed_n
            print(f"KMeans with manually fixed k={best_k}")

        # Final clustering with chosen k
        kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
        self.labels_kmeans = kmeans_final.fit_predict(self.X_pca)

        df_cluster = self.df_final.copy()
        df_cluster["cluster_kmeans"] = self.labels_kmeans
        cluster_means = df_cluster.groupby("cluster_kmeans").mean()

        top_features = cluster_means.var(axis=0).sort_values(ascending=False).head(20).index
        plt.figure(figsize=(20,12))
        sns.heatmap(cluster_means[top_features], cmap="coolwarm", annot=True, fmt=".2f")
        plt.title(f"Cluster Profiles (Top 20 Features) K={best_k}")
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/cluster_profiles_kmeans_{self.current_time}.png")
        plt.close()

        df_cluster.to_parquet(f"{self.output_dir}/kmeans_clusters_{self.current_time}.parquet")
        print("KMeans cluster profiles saved.")

        return best_k, cluster_means


    def run_gmm(self, n_range=range(2, 20), fixed_n=None, criterion="bic"):
        """
        Do the GMM clustering.
        - If fixed_n is set -> use that value.
        - Else -> select best n via AIC or BIC.

        """
        if fixed_n is None:
            bics, aics = [], []
            for n in n_range:
                gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
                gmm.fit(self.X_pca)
                bics.append(gmm.bic(self.X_pca))
                aics.append(gmm.aic(self.X_pca))

            # Plot speichern
            plt.figure(figsize=(8,5))
            plt.plot(n_range, bics, marker='o', label="BIC")
            plt.plot(n_range, aics, marker='s', label="AIC")
            plt.xlabel("Number of components")
            plt.ylabel("Information Criterion")
            plt.title("Model selection with BIC/AIC for GMM")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{self.plot_dir}/gmm_bic_aic_{self.current_time}.png")
            plt.close()

            pd.DataFrame({
                "n_components": list(n_range),
                "BIC": bics,
                "AIC": aics
            }).to_csv(f"{self.output_dir}/gmm_bic_aic_{self.current_time}.csv", index=False)

            if criterion.lower() == "bic":
                best_n = self.find_optimal_n(bics, list(n_range), threshold=0.015)
                print("Best GMM n (BIC-based):", best_n)
            else:
                best_n = self.find_optimal_n(aics, list(n_range), threshold=0.015)
                print("Best GMM n (AIC-based):", best_n)
        else:
            best_n = fixed_n
            print(f"GMM with manually fixed n_components={best_n}")

        # Final Model
        gmm_final = GaussianMixture(n_components=best_n, random_state=42)
        self.labels_gmm = gmm_final.fit_predict(self.X_pca)

        df_cluster = self.df_final.copy()
        df_cluster["cluster_gmm"] = self.labels_gmm
        cluster_means = df_cluster.groupby("cluster_gmm").mean()

        top_features = cluster_means.var(axis=0).sort_values(ascending=False).head(20).index
        plt.figure(figsize=(20,14))
        sns.heatmap(cluster_means[top_features], cmap="coolwarm", annot=True, fmt=".2f")
        plt.title(f"Cluster Profiles (Top 20 Features) with GMM n={best_n} ({criterion.upper()})")
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/cluster_profiles_gmm_{self.current_time}.png")
        plt.close()

        df_cluster.to_parquet(f"{self.output_dir}/gmm_clusters_{self.current_time}.parquet")



# ---------- RUN SCRIPT ----------
if __name__ == "__main__":
    start = time.time()

    pipeline = OSMClusteringPipeline(
        parquet_path=DATA_PATH,
        output_dir=RESULTS_DIR,
        plot_dir=PLOTS_DIR
    )

    pipeline.load_and_filter()
    pipeline.transform_features()
    pipeline.run_pca(var_threshold=0.9)
    pipeline.run_kmeans()
    pipeline.run_gmm()

    print("Total runtime:", round(time.time() - start, 2), "seconds")
