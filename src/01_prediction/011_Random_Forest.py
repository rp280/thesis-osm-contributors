import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    learning_curve,
)


# ----------------------------- CONFIG -------------------------------- #

BASE_DIR = Path.cwd()
print("Base dir:", BASE_DIR)
PROJECT_ROOT = BASE_DIR.parents[1]


DATA_PATH = (
    PROJECT_ROOT 
    / "results" 
    / "00_preprocessing" 
    /"user_summary" 
    / "pred.parquet"
)

PLOTS_DIR = PROJECT_ROOT / "results" / "01_prediction" / "supervised_learning" / "plots" / "Random_Forest" 

METRICS_DIR = PROJECT_ROOT / "results" / "01_prediction" / "supervised_learning" / "metrics" / "Random_Forest"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.20
RANDOM_STATE = 42
TUNE_SAMPLE_SIZE = 500_000  # max rows used for tuning
SHAP_SAMPLE_SIZE = 200      # rows used for SHAP plots
FEATURE_IMPORTANCE_THRESHOLD = 0.01

CURRENT_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ----------------------------- HELPERS -------------------------------- #


def load_and_prepare_data(path: Path):
    """Load parquet data and prepare X, y with frequency encodings."""
    start = time.time()
    df = pd.read_parquet(path)
    print("Data shape:", df.shape)
    print("Class balance:\n", df["left_early"].value_counts())
    print("Data loaded in {:.2f} seconds".format(time.time() - start))

    y = df["left_early"]

    # Drop non-feature columns
    X = df.drop(
        columns=[
            "left_early",
            "user_id",
            "first_edit",
            "full_last_edit",
            "active_duration",
        ],
        errors="ignore",
    )

    # Frequency encoding for categorical columns
    country_freq = df["top_country"].value_counts(normalize=True)
    X["top_country_encoded"] = df["top_country"].map(country_freq)

    feature_type_freq = df["top_feature_type_name"].value_counts(normalize=True)
    X["top_feature_type_encoded"] = df["top_feature_type_name"].map(
        feature_type_freq
    )

    X = X.drop(
        columns=["top_country", "top_feature_type_name"],
        errors="ignore",
    )

    return X, y

def train_test_split_data(X, y, test_size: float, random_state: int):
    """Stratified train-test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    print(f"{test_size:.0%} of data used for testing.")
    return X_train, X_test, y_train, y_test


def tune_random_forest(X_train, y_train, random_state: int):
    """RandomizedSearchCV for RandomForest on a subset of the training data."""
    param_dist = {
        "n_estimators": [100, 200],
        "max_depth": [10, 15, 20],
        "min_samples_leaf": [5, 10, 20],
        "max_features": ["log2", 0.2, 0.3],
        "bootstrap": [True],
        "class_weight": ["balanced", {0: 2, 1: 1}, {0: 3, 1: 1}],
    }

    # Sample subset for tuning to save time
    n_tune = min(TUNE_SAMPLE_SIZE, len(X_train))
    X_tune = X_train.sample(n=n_tune, random_state=random_state)
    y_tune = y_train.loc[X_tune.index]

    clf = RandomForestClassifier(random_state=random_state, n_jobs=10)

    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=15,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=10,
        random_state=random_state,
    )

    print(f"Starting hyperparameter tuning on {n_tune} samples...")
    t0 = time.time()
    random_search.fit(X_tune, y_tune)
    print(
        "Hyperparameter tuning completed in {:.2f} seconds".format(
            time.time() - t0
        )
    )

    print("Best parameters found:")
    print(random_search.best_params_)

    return random_search.best_estimator_, random_search.best_params_


def compute_feature_importance(model, X_train, plot_path: Path):
    """Compute and save feature importances, return selected feature names."""
    feat_imp = (
        pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    # Plot top 20
    plt.figure(figsize=(10, 6))
    top20 = feat_imp.head(20)
    plt.barh(top20["feature"][::-1], top20["importance"][::-1])
    plt.xlabel("Importance")
    plt.title("Top 20 Feature Importances (Random Forest) – user_summary_ohp")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Feature importance plot saved:", plot_path)

    # Select features above threshold
    selected_features = feat_imp.loc[
        feat_imp["importance"] > FEATURE_IMPORTANCE_THRESHOLD, "feature"
    ]
    print(
        f"Selected {len(selected_features)} features with importance > "
        f"{FEATURE_IMPORTANCE_THRESHOLD}"
    )

    return selected_features.tolist(), feat_imp


def train_final_model(X_train, y_train, best_params: dict, random_state: int):
    """Train final RandomForest on reduced features."""
    model = RandomForestClassifier(
        **best_params, random_state=random_state, n_jobs=-1
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(
        "Final model training completed in {:.2f} seconds".format(train_time)
    )
    return model, train_time


def evaluate_model(model, X_test, y_test, base_filename: str):
    """Evaluate model, print metrics and save confusion matrix + report heatmap."""
    # Predictions
    t0 = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - t0
    print("Prediction completed in {:.2f} seconds".format(predict_time))

    # Text metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("F1 Score (binary):", f1_score(y_test, y_pred))

    # Confusion matrix heatmap
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stayed", "Left Early"],
        yticklabels=["Stayed", "Left Early"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix – user_summary_ohp (RandomForest)")
    plt.tight_layout()
    cm_path = PLOTS_DIR / f"{base_filename}_confusion_matrix_{CURRENT_TIME}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Confusion matrix plot saved:", cm_path)

    # Classification report heatmap
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = (
        pd.DataFrame(report_dict)
        .transpose()
        .drop(index=["accuracy"], errors="ignore")
    )
    report_df = report_df[["precision", "recall", "f1-score"]]

    plt.figure(figsize=(6, 3))
    sns.heatmap(
        report_df, annot=True, cmap="YlGnBu", fmt=".2f", cbar=False
    )
    plt.title("Classification Report Heatmap – RandomForest")
    plt.tight_layout()
    cr_path = (
        PLOTS_DIR
        / f"{base_filename}_classification_report_{CURRENT_TIME}.png"
    )
    plt.savefig(cr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Classification report heatmap saved:", cr_path)

    return report_df, predict_time


def plot_shap_beeswarm(model, X_test, base_filename: str):
    """Calculate SHAP values on a sample and save beeswarm plot."""
    n_shap = min(SHAP_SAMPLE_SIZE, len(X_test))
    X_sample = X_test.sample(n=n_shap, random_state=RANDOM_STATE).copy()

    # Ensure same feature order as the model
    X_sample = X_sample.loc[:, model.feature_names_in_]
    X_sample = X_sample.fillna(-999)

    explainer = shap.TreeExplainer(model)
    sv_raw = explainer.shap_values(X_sample, check_additivity=False)

    # Binary classification – handle different SHAP output shapes
    if isinstance(sv_raw, list):
        sv = sv_raw[1]  # class 1 ("Left Early")
    elif sv_raw.ndim == 3:
        sv = sv_raw[:, :, 1]
    else:
        sv = sv_raw

    print("SHAP values shape:", sv.shape)

    plt.close("all")
    shap.summary_plot(
        sv,
        X_sample,
        plot_type="dot",
        show=False,
        max_display=20,
        plot_size=(9, 6),
    )
    shap_path = (
        PLOTS_DIR / f"{base_filename}_shap_beeswarm_{CURRENT_TIME}.png"
    )
    plt.savefig(shap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("SHAP beeswarm plot saved:", shap_path)


def save_model_results(
    train_time: float,
    predict_time: float,
    report_df: pd.DataFrame,
    base_filename: str,
):
    """Save model timings + precision/recall to CSV and plot."""
    results = [
        {
            "Model": "Random Forest",
            "Train_Time_sec": round(train_time, 2),
            "Predict_Time_sec": round(predict_time, 2),
            "Precision_False": round(report_df.loc["False", "precision"], 2),
            "Recall_False": round(report_df.loc["False", "recall"], 2),
            "Precision_True": round(report_df.loc["True", "precision"], 2),
            "Recall_True": round(report_df.loc["True", "recall"], 2),
        }
    ]

    df_results = pd.DataFrame(results)
    csv_path = (
        METRICS_DIR / f"{base_filename}_model_results_{CURRENT_TIME}.csv"
    )
    df_results.to_csv(csv_path, index=False)
    print("Model results CSV saved:", csv_path)


def plot_learning_curve(model, X_train, y_train, base_filename: str):
    """Compute and save learning curve (F1 macro) plot."""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8),
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, "o-", label="Training score")
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
    )

    plt.plot(train_sizes, val_mean, "o-", label="Cross-validation score")
    plt.fill_between(
        train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2
    )

    plt.xlabel("Training examples")
    plt.ylabel("F1 Macro Score")
    plt.title("Learning Curve – Random Forest")
    plt.legend(loc="best")
    plt.tight_layout()
    lc_path = (
        PLOTS_DIR / f"{base_filename}_learning_curve_{CURRENT_TIME}.png"
    )
    plt.savefig(lc_path, dpi=300)
    plt.close()
    print("Learning curve plot saved:", lc_path)



# ----------------------------- MAIN PIPELINE -------------------------- #



def main():
    overall_start = time.time()
    base_filename = "RF"

    # 1) Load & prepare data
    X, y = load_and_prepare_data(DATA_PATH)

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 3) Hyperparameter tuning
    best_model, best_params = tune_random_forest(
        X_train, y_train, random_state=RANDOM_STATE
    )

    # 4) Feature importance & selection
    fi_plot_path = (
        PLOTS_DIR / f"{base_filename}_feature_importance_{CURRENT_TIME}.png"
    )
    selected_features, feat_imp = compute_feature_importance(
        best_model, X_train, fi_plot_path
    )

    X_train_reduced = X_train[selected_features]
    X_test_reduced = X_test[selected_features]
    print("Feature selection done. Reduced feature set size:", len(selected_features))

    # 5) Final training on reduced features
    final_model, train_time = train_final_model(
        X_train_reduced, y_train, best_params, RANDOM_STATE
    )

    # 6) Evaluation (metrics + confusion matrix + report)
    report_df, predict_time = evaluate_model(
        final_model, X_test_reduced, y_test, base_filename
    )

    # 7) SHAP beeswarm
    plot_shap_beeswarm(final_model, X_test_reduced, base_filename)

    # 8) Save result summary and bar plot
    save_model_results(
        train_time=train_time,
        predict_time=predict_time,
        report_df=report_df,
        base_filename=base_filename,
    )

    # 9) Learning curve
    plot_learning_curve(final_model, X_train_reduced, y_train, base_filename)

    print(
        "Total runtime: {:.2f} seconds".format(time.time() - overall_start)
    )

# ----------------------------- SCRIPT -------------------------------- #

if __name__ == "__main__":
    main()

# ----------------------------- END OF FILE ---------------------------- #