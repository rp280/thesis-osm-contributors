import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    learning_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    recall_score,
    precision_score,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils import resample
from scipy.stats import loguniform, uniform


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
)

PLOTS_DIR = PROJECT_ROOT / "results" / "01_prediction" / "supervised_learning" / "plots" / "sgd_classifier"
METRICS_DIR = PROJECT_ROOT / "results" / "01_prediction" / "supervised_learning" / "metrics" / "sgd_classifier"
# Make sure output directories exist
for d in [PLOTS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


TEST_SIZE = 0.20
RANDOM_STATE = 42
MI_SAMPLE_SIZE = 100_000         # rows used for mutual information selection
N_MI_FEATURES = 35               # k in SelectKBest
N_ITER_SEARCH = 50               # RandomizedSearch iterations
CV_SPLITS = 3
THRESHOLD = 0.25                 # decision threshold on SGD probabilities

CURRENT_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ----------------------------- HELPERS -------------------------------- #


def load_and_prepare_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load parquet, handle missing values, encode categoricals, return X, y."""
    start = time.time()
    df = pd.read_parquet(path)
    print("Data shape:", df.shape)
    print("Class balance:\n", df["left_early"].value_counts())
    print("Data loaded in {:.2f} seconds".format(time.time() - start))

    # --- missing value logic variables ---
    nan_logic_cols = [
        "pause_after_first_edit",
        "mean_days_between_edits_true",
        "pause_after_first_editing_day",
        "burstiness_score",
        "days_to_50",
        "days_to_100",
    ]
    imputation_values = {
        "pause_after_first_edit": 9999,
        "mean_days_between_edits_true": 9999,
        "pause_after_first_editing_day": 9999,
        "burstiness_score": 9999,
        "days_to_50": 9999,
        "days_to_100": 9999,
    }

    for col in nan_logic_cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)
            df[col] = df[col].fillna(imputation_values[col])

    for col in ["top_feature_ratio", "comment_length_ratio"]:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)
            df[col] = df[col].fillna(df[col].median())

    # --- target & base features ---
    y = df["left_early"]
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

    # frequency encoding for categoricals
    if "top_country" in df.columns:
        country_freq = df["top_country"].value_counts(normalize=True)
        X["top_country_encoded"] = df["top_country"].map(country_freq)

    if "top_feature_type_name" in df.columns:
        feature_type_freq = df["top_feature_type_name"].value_counts(normalize=True)
        X["top_feature_type_encoded"] = df["top_feature_type_name"].map(
            feature_type_freq
        )

    X = X.drop(
        columns=["top_country", "top_feature_type_name"],
        errors="ignore",
    )

    return X, y


def select_features_mi(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_size: int,
    k_features: int,
    plots_dir: Path,
    base_filename: str,
) -> tuple[np.ndarray, list[str]]:
    """Mutual information feature selection with SelectKBest, returns selector + feature names."""
    # sample subset for MI (for speed)
    n_sample = min(sample_size, len(X_train))
    X_sub, y_sub = resample(
        X_train, y_train, n_samples=n_sample, random_state=RANDOM_STATE
    )

    selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
    selector.fit(X_sub, y_sub)

    X_train_sel = selector.transform(X_train)
    top_features = X_train.columns[selector.get_support()]

    # build MI dataframe for inspection / plot
    scores = selector.scores_
    mi_df = (
        pd.DataFrame({"feature": X_train.columns, "mutual_info": scores})
        .sort_values(by="mutual_info", ascending=False)
        .reset_index(drop=True)
    )

    print("Top 20 features by mutual information:")
    print(mi_df.head(20))

    plt.figure(figsize=(10, 6))
    top_20 = mi_df.head(20)
    plt.barh(top_20["feature"][::-1], top_20["mutual_info"][::-1])
    plt.xlabel("Mutual Information Score")
    plt.title("Top 20 Features – Mutual Information (SGD)")
    plt.tight_layout()
    mi_path = plots_dir / f"{base_filename}_MI_top20_{CURRENT_TIME}.png"
    plt.savefig(mi_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("MI feature selection plot saved:", mi_path)

    return selector, list(top_features)


def build_sgd_pipeline() -> any:
    """Create the SGD pipeline with StandardScaler."""
    pipeline = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            random_state=RANDOM_STATE,
            max_iter=5000,
            tol=1e-4,
        ),
    )
    return pipeline


def tune_sgd(
    pipeline,
    X_train_sel: np.ndarray,
    y_train: pd.Series,
) -> tuple[any, dict, float]:
    """Hyperparameter tuning for SGDClassifier using RandomizedSearchCV."""
    param_distributions = {
        "sgdclassifier__loss": ["modified_huber", "log_loss"],
        "sgdclassifier__alpha": loguniform(1e-6, 1e-3),
        "sgdclassifier__penalty": ["l2", "l1", "elasticnet"],
        "sgdclassifier__l1_ratio": uniform(0.1, 0.9),
        "sgdclassifier__class_weight": [
            {False: 4, True: 1},
            {False: 5, True: 1},
            {False: 6, True: 1},
        ],
        "sgdclassifier__early_stopping": [True, False],
        "sgdclassifier__tol": [1e-3, 1e-4],
    }

    scoring = {
        "recall_false": make_scorer(recall_score, pos_label=False),
        "precision_false": make_scorer(
            lambda y_true, y_pred: precision_score(
                y_true, y_pred, pos_label=False
            )
        ),
        "f1_false": make_scorer(
            lambda y_true, y_pred: f1_score(y_true, y_pred, pos_label=False)
        ),
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=N_ITER_SEARCH,
        cv=CV_SPLITS,
        scoring=scoring,
        refit="f1_false",
        verbose=10,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    print(
        f"Starting SGD hyperparameter tuning "
        f"(n_iter={N_ITER_SEARCH}, cv={CV_SPLITS})..."
    )
    t0 = time.time()
    search.fit(X_train_sel, y_train)
    elapsed = time.time() - t0
    print("Hyperparameter tuning completed in {:.2f} seconds".format(elapsed))
    print("Best parameters:")
    print(search.best_params_)

    return search.best_estimator_, search.best_params_, elapsed


def evaluate_model(
    model,
    X_test_sel: np.ndarray,
    y_test: pd.Series,
    base_filename: str,
) -> tuple[pd.DataFrame, float, np.ndarray]:
    """Evaluate model: metrics, confusion matrix, classification report heatmap."""
    t0 = time.time()
    probas = model.predict_proba(X_test_sel)[:, 1]
    y_pred = (probas >= THRESHOLD).astype(int)
    pred_time = time.time() - t0
    print("Prediction completed in {:.2f} seconds".format(pred_time))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["False (stayed)", "True (left early)"],
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    plt.title("Confusion Matrix – SGD Classifier")
    cm_path = PLOTS_DIR / f"{base_filename}_confusion_matrix_{CURRENT_TIME}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Confusion matrix plot saved:", cm_path)

    # classification report heatmap
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df_filtered = report_df.loc[
        ["False", "True"], ["precision", "recall", "f1-score"]
    ]

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        report_df_filtered, annot=True, cmap="YlGnBu", fmt=".2f", cbar=False
    )
    plt.title("Classification Report – SGD Classifier")
    cr_path = (
        PLOTS_DIR
        / f"{base_filename}_classification_report_{CURRENT_TIME}.png"
    )
    plt.savefig(cr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Classification report heatmap saved:", cr_path)

    return report_df, pred_time, probas


def plot_precision_recall_thresholds(
    y_test: pd.Series, probas: np.ndarray, base_filename: str
) -> None:
    """Plot precision, recall, F1 vs decision threshold."""
    precision, recall, thresholds = precision_recall_curve(y_test, probas)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.plot(thresholds, f1_scores[:-1], label="F1 Score")
    plt.axvline(
        x=THRESHOLD,
        color="grey",
        linestyle="--",
        label=f"Threshold = {THRESHOLD}",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision–Recall–F1 vs Threshold – SGD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pr_path = (
        PLOTS_DIR
        / f"{base_filename}_precision_recall_thresholds_{CURRENT_TIME}.png"
    )
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Precision–Recall–Threshold plot saved:", pr_path)


def plot_coefficients(
    model,
    feature_names: list[str],
    base_filename: str,
) -> None:
    """Plot top 20 absolute SGD coefficients."""
    coefs = model.named_steps["sgdclassifier"].coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})

    # sort by absolute importance
    coef_df = coef_df.reindex(
        coef_df["coefficient"].abs().sort_values(ascending=False).index
    )
    coef_df_top20 = coef_df.head(20)

    plt.figure(figsize=(8, 6))
    plt.barh(
        coef_df_top20["feature"],
        coef_df_top20["coefficient"],
    )
    plt.xlabel("Coefficient (positive = left early, negative = stayed)")
    plt.title("Top 20 Feature Coefficients – SGD")
    plt.tight_layout()
    coef_path = (
        PLOTS_DIR
        / f"{base_filename}_coefficients_top20_{CURRENT_TIME}.png"
    )
    plt.savefig(coef_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Coefficient plot saved:", coef_path)


def save_model_results(
    train_time: float,
    predict_time: float,
    report_df: pd.DataFrame,
    base_filename: str,
) -> None:
    """Save model timings + precision/recall to CSV and plot."""
    results = [
        {
            "Model": "SGD Classifier",
            "Train_Time_sec": round(train_time, 2),
            "Predict_Time_sec": round(predict_time, 2),
            "Precision_False": round(report_df.loc["False", "precision"], 2),
            "Recall_False": round(report_df.loc["False", "recall"], 2),
            "Precision_True": round(report_df.loc["True", "precision"], 2),
            "Recall_True": round(report_df.loc["True", "recall"], 2),
        }
    ]

    df_results = pd.DataFrame(results)
    csv_path = METRICS_DIR / f"{base_filename}_model_results_{CURRENT_TIME}.csv"
    df_results.to_csv(csv_path, index=False)
    print("Model results CSV saved:", csv_path)


def plot_learning_curve_sgd(
    model,
    X_train_sel: np.ndarray,
    y_train: pd.Series,
    base_filename: str,
) -> None:
    """Compute and save learning curve (F1 macro) for SGDClassifier."""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X_train_sel,
        y=y_train,
        cv=CV_SPLITS,
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

    plt.plot(train_sizes, val_mean, "o-", label="CV score")
    plt.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.2,
    )

    plt.xlabel("Training examples")
    plt.ylabel("F1 Macro Score")
    plt.title("Learning Curve – SGDClassifier")
    plt.legend(loc="best")
    plt.tight_layout()
    lc_path = (
        PLOTS_DIR / f"{base_filename}_learning_curve_{CURRENT_TIME}.png"
    )
    plt.savefig(lc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Learning curve plot saved:", lc_path)


# ----------------------------- MAIN PIPELINE -------------------------- #


def main() -> None:
    overall_start = time.time()
    base_filename = "SGD_user_summary_ohp"

    # 1) Load & prepare data
    X, y = load_and_prepare_data(DATA_PATH)

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"{TEST_SIZE:.0%} of data used for testing.")

    # 3) Mutual information feature selection
    selector, top_features = select_features_mi(
        X_train=X_train,
        y_train=y_train,
        sample_size=MI_SAMPLE_SIZE,
        k_features=N_MI_FEATURES,
        plots_dir=PLOTS_DIR,
        base_filename=base_filename,
    )

    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)
    print("Selected features:", len(top_features))

    # 4) Build and tune SGD pipeline
    pipeline = build_sgd_pipeline()
    best_model, best_params, train_time = tune_sgd(
        pipeline=pipeline,
        X_train_sel=X_train_sel,
        y_train=y_train,
    )

    # 5) Evaluation
    report_df, predict_time, probas = evaluate_model(
        model=best_model,
        X_test_sel=X_test_sel,
        y_test=y_test,
        base_filename=base_filename,
    )

    # 6) Precision–Recall vs Threshold
    plot_precision_recall_thresholds(
        y_test=y_test,
        probas=probas,
        base_filename=base_filename,
    )

    # 7) Coefficient-based feature importance
    plot_coefficients(
        model=best_model,
        feature_names=top_features,
        base_filename=base_filename,
    )

    # 8) Save results (CSV + bar plot)
    save_model_results(
        train_time=train_time,
        predict_time=predict_time,
        report_df=report_df,
        base_filename=base_filename,
    )

    # 9) Learning curve
    plot_learning_curve_sgd(
        model=best_model,
        X_train_sel=X_train_sel,
        y_train=y_train,
        base_filename=base_filename,
    )

    print("Total runtime: {:.2f} seconds".format(time.time() - overall_start))


# ----------------------------- SCRIPT ENTRY POINT --------------------- #
if __name__ == "__main__":
    main()

# ----------------------------- END OF FILE ---------------------------- #