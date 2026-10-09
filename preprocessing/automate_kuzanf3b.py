from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automated preprocessing for Telco Customer Churn dataset "
            "(Eksperimen_SML_kuzanf3b)."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to repository root. Default is parent directory of this file.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="CSV input path. Default: <project-root>telco_churn_raw.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for processed train/test CSV files.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Output directory for preprocessor and metadata artifacts.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio. Default: 0.2",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility. Default: 42",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.project_root.resolve()
    input_path = (
        args.input_path.resolve()
        if args.input_path is not None
        else root / "telco_churn_raw.csv"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "preprocessing" / "telco_customer_churn_preprocessing"
    )
    artifacts_dir = (
        args.artifacts_dir.resolve()
        if args.artifacts_dir is not None
        else root / "preprocessing" / "artifacts"
    )
    return {
        "root": root,
        "input_path": input_path,
        "output_dir": output_dir,
        "artifacts_dir": artifacts_dir,
    }


def build_preprocessor(features: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    missing_numeric = [column for column in numeric_features if column not in features.columns]
    if missing_numeric:
        raise ValueError(f"Missing expected numeric columns: {missing_numeric}")

    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def preprocess_dataframe(
    raw_df: pd.DataFrame, test_size: float, random_state: int
) -> dict[str, Any]:
    prepared = raw_df.copy(deep=True)
    total_charges_numeric = pd.to_numeric(
        prepared["TotalCharges"].astype(str).str.strip(), errors="coerce"
    )
    prepared.loc[:, "TotalCharges"] = total_charges_numeric
    prepared = prepared.drop_duplicates(ignore_index=True)
    prepared = prepared.drop(columns=["customerID"]).copy(deep=True)
    prepared.loc[:, "Churn"] = prepared["Churn"].map({"No": 0, "Yes": 1}).astype("int64")

    features = prepared.drop(columns=["Churn"])
    target = prepared["Churn"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(x_train)
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out().tolist()

    x_train_df = pd.DataFrame(x_train_processed, columns=feature_names, index=x_train.index)
    x_test_df = pd.DataFrame(x_test_processed, columns=feature_names, index=x_test.index)

    train_output = x_train_df.copy()
    train_output["Churn"] = y_train
    test_output = x_test_df.copy()
    test_output["Churn"] = y_test

    metadata: dict[str, Any] = {
        "dataset_rows_raw": int(raw_df.shape[0]),
        "dataset_rows_prepared": int(prepared.shape[0]),
        "duplicates_removed": int(raw_df.duplicated().sum()),
        "total_charges_blank_as_missing": int(
            (raw_df["TotalCharges"].astype(str).str.strip() == "").sum()
        ),
        "test_size": test_size,
        "random_state": random_state,
        "train_shape": [int(train_output.shape[0]), int(train_output.shape[1])],
        "test_shape": [int(test_output.shape[0]), int(test_output.shape[1])],
        "target_distribution": {
            "train": float(y_train.mean()),
            "test": float(y_test.mean()),
        },
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "processed_feature_count": int(len(feature_names)),
        "processed_feature_names_file": "feature_names.json",
    }

    return {
        "train_output": train_output,
        "test_output": test_output,
        "preprocessor": preprocessor,
        "metadata": metadata,
        "feature_names": feature_names,
    }


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)

    input_path = paths["input_path"]
    output_dir = paths["output_dir"]
    artifacts_dir = paths["artifacts_dir"]

    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(input_path)
    results = preprocess_dataframe(
        raw_df=raw_df, test_size=args.test_size, random_state=args.random_state
    )

    train_output_path = output_dir / "telco_churn_train.csv"
    test_output_path = output_dir / "telco_churn_test.csv"
    preprocessor_path = artifacts_dir / "preprocessor.joblib"
    metadata_path = artifacts_dir / "preprocessing_metadata.json"
    feature_names_path = artifacts_dir / "feature_names.json"

    results["train_output"].to_csv(train_output_path, index=False)
    results["test_output"].to_csv(test_output_path, index=False)
    joblib.dump(results["preprocessor"], preprocessor_path)
    feature_names_path.write_text(json.dumps(results["feature_names"], indent=2))
    metadata_path.write_text(json.dumps(results["metadata"], indent=2))

    print(f"Input: {input_path}")
    print(f"Train output: {train_output_path}")
    print(f"Test output: {test_output_path}")
    print(f"Preprocessor artifact: {preprocessor_path}")
    print(f"Metadata artifact: {metadata_path}")
    print(f"Feature names artifact: {feature_names_path}")
    print(f"Train shape: {results['metadata']['train_shape']}")
    print(f"Test shape: {results['metadata']['test_shape']}")


if __name__ == "__main__":
    main()
