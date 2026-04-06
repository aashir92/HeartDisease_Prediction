import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(script_dir: str) -> pd.DataFrame:
    """Load the heart disease dataset from a set of expected locations."""
    candidate_files = [
        "heart.csv",
        "heart_disease.csv",
        "heart_disease_uci.csv",
        os.path.join("archive", "heart_disease_uci.csv"),
    ]

    for candidate in candidate_files:
        candidate_path = os.path.join(script_dir, candidate)
        if os.path.isfile(candidate_path):
            print(f"Loading dataset from: {candidate_path}")
            return pd.read_csv(candidate_path)

    raise FileNotFoundError(
        "No heart disease dataset found. "
        "Expected one of: heart.csv, heart_disease.csv, heart_disease_uci.csv, "
        "archive/heart_disease_uci.csv"
    )


def normalize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the target column name and binary encoding."""
    possible_targets = [
        "target",
        "condition",
        "heart_disease",
        "diagnosis",
        "output",
        "num",
    ]
    target_column = None

    for name in possible_targets:
        if name in df.columns:
            target_column = name
            break

    if target_column is None:
        lowers = [col.lower() for col in df.columns]
        for name in ["target", "condition", "diagnosis", "heart", "disease"]:
            if name in lowers:
                target_column = df.columns[lowers.index(name)]
                break

    if target_column is None:
        raise ValueError(
            "Unable to locate a target column. "
            "Please ensure the dataset has 'target', 'condition', or similar."
        )

    df = df.rename(columns={target_column: "target"})
    df["target"] = df["target"].astype(int)
    unique_values = sorted(df["target"].unique())
    if any(val not in {0, 1} for val in unique_values):
        df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
        print(
            "Normalized target values to binary labels: "
            f"{sorted(df['target'].unique())}"
        )

    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values using mean for numeric and mode for categorical."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    for col in object_cols:
        if df[col].isna().any():
            mode_value = df[col].mode()
            fill_value = mode_value.iloc[0] if not mode_value.empty else ""
            df[col] = df[col].fillna(fill_value)

    remaining_missing = df.isna().sum().sum()
    if remaining_missing > 0:
        print(
            f"Warning: {remaining_missing} missing values remain after imputation. "
            "Dropping rows with missing values."
        )
        df = df.dropna(axis=0)

    return df


def plot_correlation_heatmap(df: pd.DataFrame, output_path: str) -> None:
    """Create and save a correlation heatmap for numeric variables only."""
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if "id" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["id"])

    plt.figure(figsize=(12, 10))
    correlation = numeric_df.corr()
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=True,
    )
    plt.title("Correlation Heatmap for Heart Disease Dataset")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved correlation heatmap to: {output_path}")


def preprocess_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Encode categorical data, scale numerical features, and return X/y."""
    df = normalize_target(df)
    target = df["target"]

    categorical_candidates = [
        "cp",
        "restecg",
        "slope",
        "thal",
        "ca",
        "sex",
        "fbs",
        "exang",
    ]
    categorical_cols = [col for col in categorical_candidates if col in df.columns]
    if categorical_cols:
        categorical_data = pd.get_dummies(df[categorical_cols], drop_first=True)
    else:
        categorical_data = pd.DataFrame(index=df.index)

    non_feature_cols = set(categorical_cols + ["target", "id", "dataset"])
    numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in non_feature_cols
    ]
    numeric_data = df[numeric_cols].copy()
    numeric_data[numeric_cols] = StandardScaler().fit_transform(numeric_data)

    X = pd.concat([numeric_data, categorical_data], axis=1)
    if X.empty:
        raise ValueError("No features were created during preprocessing.")

    return X, target


def plot_confusion_matrix(y_test, y_pred, output_path: str) -> None:
    """Render and save a confusion matrix heatmap."""
    matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved confusion matrix to: {output_path}")


def plot_roc_curve(y_test, y_proba, output_path: str) -> None:
    """Create and save the ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_value = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_value:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved ROC curve to: {output_path}")


def plot_feature_importance(
    model: LogisticRegression, feature_names: list[str], output_path: str
) -> None:
    """Plot the top 10 feature importance values for the logistic model."""
    coefficients = model.coef_[0]
    feature_importances = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients}
    )
    feature_importances["abs_coefficient"] = feature_importances["coefficient"].abs()
    top_features = feature_importances.sort_values(
        by="abs_coefficient", ascending=False
    ).head(10)

    plt.figure(figsize=(10, 7))
    sns.barplot(
        data=top_features,
        x="coefficient",
        y="feature",
        palette="coolwarm",
    )
    plt.title("Top 10 Features Driving Heart Disease Prediction")
    plt.xlabel("Logistic Regression Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved feature importance chart to: {output_path}")


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        df = load_dataset(script_dir)
    except FileNotFoundError as error:
        print(error)
        return 1

    print(f"Dataset shape: {df.shape}")
    print("First 5 rows of the dataset:")
    print(df.head(5).to_string(index=False))

    df = impute_missing_values(df)
    plot_correlation_heatmap(df, os.path.join(script_dir, "correlation_heatmap.png"))

    X, y = preprocess_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    plot_confusion_matrix(y_test, y_pred, os.path.join(script_dir, "confusion_matrix.png"))
    plot_roc_curve(y_test, y_proba, os.path.join(script_dir, "roc_curve.png"))
    plot_feature_importance(model, X.columns.tolist(), os.path.join(script_dir, "feature_importance.png"))

    print("\nCompleted training and evaluation. Generated plots are saved in the task3 folder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
