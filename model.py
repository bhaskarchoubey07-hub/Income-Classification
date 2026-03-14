from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REQUIRED_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

DEFAULT_PREDICTION_FIELDS = [
    "age",
    "education-num",
    "hours-per-week",
    "capital-gain",
    "capital-loss",
]


def validate_dataset(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing_text}")

    if df.empty or len(df) < 10:
        raise ValueError("Dataset must contain at least 10 rows to train the model.")


def get_data_profile(df: pd.DataFrame) -> Dict[str, object]:
    missing_summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_percent": (df.isna().mean() * 100).round(2).values,
        }
    )
    column_types = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
        }
    )

    return {
        "shape": df.shape,
        "missing_summary": missing_summary,
        "column_types": column_types,
        "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),
    }


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )


def run_training_pipeline(df: pd.DataFrame, target_column: str) -> Dict[str, object]:
    X = df.drop(columns=[target_column])
    y = df[target_column].astype(str).str.strip()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(X)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=12,
                    random_state=42,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    report_df = pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).transpose()
    report_df = report_df.reset_index().rename(columns={"index": "label"})
    accuracy = accuracy_score(y_test, predictions)
    class_labels = list(model.named_steps["classifier"].classes_)
    conf_matrix = confusion_matrix(y_test, predictions, labels=class_labels)

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    numeric_defaults = X_train.select_dtypes(include=np.number).median().to_dict()

    return {
        "model": model,
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report": report_df.round(3),
        "class_labels": class_labels,
        "feature_names": feature_names,
        "feature_columns": X.columns.tolist(),
        "numeric_defaults": numeric_defaults,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
    }


def get_feature_importance(model: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    importances = model.named_steps["classifier"].feature_importances_
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )
    importance_df["feature"] = importance_df["feature"].str.replace("num__", "", regex=False)
    importance_df["feature"] = importance_df["feature"].str.replace("cat__", "", regex=False)
    return importance_df.sort_values("importance", ascending=False).head(15)


def build_prediction_row(
    reference_df: pd.DataFrame, feature_columns: List[str], user_inputs: Dict[str, float]
) -> pd.DataFrame:
    baseline = {}
    for column in feature_columns:
        if column in user_inputs:
            baseline[column] = user_inputs[column]
        else:
            series = reference_df[column]
            if pd.api.types.is_numeric_dtype(series):
                baseline[column] = float(series.median())
            else:
                baseline[column] = series.mode(dropna=True).iloc[0]
    return pd.DataFrame([baseline], columns=feature_columns)
