import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "census-bureau.data"
COLUMNS_PATH = ROOT_DIR / "census-bureau.columns"

CATEGORICAL_CODES = [
    "detailed industry recode",
    "detailed occupation recode",
    "own business or self employed",
    "veterans benefits",
    "year",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train CatBoost, TabPFN v2.5, and TabICL on the census-bureau dataset.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def load_dataset(target_column: str):
    columns = [line.strip() for line in COLUMNS_PATH.read_text().splitlines() if line.strip()]
    df = pd.read_csv(DATA_PATH, header=None, names=columns, na_values="?", skipinitialspace=True)

    object_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        df[column] = df[column].map(lambda value: value.strip() if isinstance(value, str) else value).astype(object)

    x = df.drop(columns=[target_column]).copy()
    x_object_columns = x.select_dtypes(include=["object", "string"]).columns
    for column in x_object_columns:
        x[column] = x[column].where(pd.notna(x[column]), np.nan).astype(object)

    y = df[target_column].map(lambda value: value.strip() if isinstance(value, str) else value).astype(str)

    categorical_columns = list(x.select_dtypes(include=["object", "string"]).columns)
    categorical_columns.extend(column for column in CATEGORICAL_CODES if column in x.columns)
    categorical_columns = list(dict.fromkeys(categorical_columns))
    numeric_columns = [column for column in x.columns if column not in categorical_columns]
    x = fill_missing_values(x, numeric_columns, categorical_columns)
    return x, y, numeric_columns, categorical_columns


def fill_missing_values(x, numeric_columns, categorical_columns):
    filled = x.copy()

    for column in numeric_columns:
        if filled[column].isna().any():
            median_value = filled[column].median()
            filled[column] = filled[column].fillna(median_value)

    for column in categorical_columns:
        if filled[column].isna().any():
            modes = filled[column].mode(dropna=True)
            fill_value = modes.iloc[0] if not modes.empty else "Missing"
            filled[column] = filled[column].fillna(fill_value).astype(object)

    return filled


def build_dense_preprocessor(numeric_columns, categorical_columns):
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
    )


def build_foundation_preprocessor(numeric_columns, categorical_columns):
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_columns),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                categorical_columns,
            ),
        ],
    )


def prepare_catboost_frames(x_train, x_test, categorical_columns):
    train_frame = x_train.copy()
    test_frame = x_test.copy()

    for column in categorical_columns:
        train_frame[column] = train_frame[column].fillna("Missing").astype(str)
        test_frame[column] = test_frame[column].fillna("Missing").astype(str)

    return train_frame, test_frame


def subsample_training_rows(x_train, y_train, limit, seed):
    if limit is None or len(x_train) <= limit:
        return x_train, y_train

    class_counts = y_train.value_counts().sort_index()
    class_limits = {}
    remaining = limit

    for class_value, class_count in class_counts.items():
        allocated = max(1, int(round(limit * class_count / len(y_train))))
        allocated = min(allocated, int(class_count))
        class_limits[class_value] = allocated
        remaining -= allocated

    class_order = list(class_counts.sort_values(ascending=False).index)
    cursor = 0
    while remaining != 0 and class_order:
        class_value = class_order[cursor % len(class_order)]
        max_for_class = int(class_counts[class_value])

        if remaining > 0 and class_limits[class_value] < max_for_class:
            class_limits[class_value] += 1
            remaining -= 1
        elif remaining < 0 and class_limits[class_value] > 1:
            class_limits[class_value] -= 1
            remaining += 1

        cursor += 1

    grouped_indices = []
    for class_value, class_index in y_train.groupby(y_train).groups.items():
        class_rows = x_train.loc[class_index]
        sampled_rows = class_rows.sample(
            n=class_limits[int(class_value)],
            random_state=seed,
        )
        grouped_indices.extend(sampled_rows.index.tolist())

    sampled_index = pd.Index(grouped_indices).to_series().sample(frac=1.0, random_state=seed).to_list()
    sampled_x = x_train.loc[sampled_index].reset_index(drop=True)
    sampled_y = y_train.loc[sampled_index].reset_index(drop=True).astype(int)
    return sampled_x, sampled_y


def predict_scores(model, x_test, batch_size):
    if hasattr(model, "predict_proba"):
        chunks = []
        for start in range(0, len(x_test), batch_size):
            stop = start + batch_size
            probabilities = model.predict_proba(x_test[start:stop])
            probabilities = np.asarray(probabilities)
            if probabilities.ndim == 1:
                chunks.append(probabilities.astype(float))
            else:
                chunks.append(probabilities[:, 1].astype(float))
        return np.concatenate(chunks)

    predictions = model.predict(x_test)
    return np.asarray(predictions, dtype=float)


def summarize_metrics(frame):
    summary = (
        frame.groupby("model")[["roc_auc", "average_precision", "accuracy", "f1", "fit_seconds", "predict_seconds"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["model"] + [f"{metric}_{stat}" for metric, stat in summary.columns.tolist()[1:]]
    return summary


def build_model(model_name, model_params, seed):
    if model_name == "catboost":
        params = {
            "random_seed": seed,
            "allow_writing_files": False,
            **model_params,
        }
        return CatBoostClassifier(**params)

    if model_name == "tabpfn_v2_5":
        return TabPFNClassifier.create_default_for_version(ModelVersion.V2_5, **model_params)

    if model_name == "tabicl":
        return TabICLClassifier(**model_params)

    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_models(config):
    x, y_labels, numeric_columns, categorical_columns = load_dataset(config["target_column"])
    y = y_labels.eq(config["positive_label"]).astype(int)

    splitter = StratifiedKFold(
        n_splits=config["n_splits"],
        shuffle=True,
        random_state=config["seed"],
    )

    dense_preprocessor = build_dense_preprocessor(numeric_columns, categorical_columns)
    foundation_preprocessor = build_foundation_preprocessor(numeric_columns, categorical_columns)
    rows = []

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
        x_train_raw = x.iloc[train_idx].reset_index(drop=True)
        x_test_raw = x.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        x_train_dense = dense_preprocessor.fit_transform(x_train_raw)
        x_test_dense = dense_preprocessor.transform(x_test_raw)
        x_train_foundation = foundation_preprocessor.fit_transform(x_train_raw)
        x_test_foundation = foundation_preprocessor.transform(x_test_raw)

        for model_config in config["models"]:
            model_name = model_config["name"]
            train_limit = model_config.get("train_row_limit")
            predict_batch_size = model_config.get("predict_batch_size", len(x_test_raw))

            if model_name == "catboost":
                x_train_model, y_train_model = subsample_training_rows(
                    x_train_raw,
                    y_train,
                    train_limit,
                    seed=config["seed"] + fold_index,
                )
                x_test_model = x_test_raw
                x_train_model, x_test_model = prepare_catboost_frames(
                    x_train_model,
                    x_test_model,
                    categorical_columns,
                )
                cat_features = [x_train_model.columns.get_loc(column) for column in categorical_columns]
            else:
                if model_name == "tabpfn_v2_5":
                    train_frame = pd.DataFrame(x_train_foundation)
                    test_frame = pd.DataFrame(x_test_foundation)
                else:
                    train_frame = pd.DataFrame(x_train_dense)
                    test_frame = pd.DataFrame(x_test_dense)

                x_train_model, y_train_model = subsample_training_rows(
                    train_frame,
                    y_train,
                    train_limit,
                    seed=config["seed"] + fold_index,
                )
                x_test_model = test_frame
                cat_features = None

            model = build_model(model_name, model_config.get("params", {}), seed=config["seed"] + fold_index)

            fit_started = time.perf_counter()
            if model_name == "catboost":
                model.fit(x_train_model, y_train_model, cat_features=cat_features)
            else:
                model.fit(x_train_model, y_train_model)
            fit_seconds = time.perf_counter() - fit_started

            predict_started = time.perf_counter()
            scores = predict_scores(model, x_test_model, predict_batch_size)
            predict_seconds = time.perf_counter() - predict_started
            predictions = (scores >= 0.5).astype(int)

            rows.append(
                {
                    "fold": fold_index,
                    "model": model_name,
                    "train_rows": len(x_train_model),
                    "test_rows": len(x_test_model),
                    "roc_auc": roc_auc_score(y_test, scores),
                    "average_precision": average_precision_score(y_test, scores),
                    "accuracy": accuracy_score(y_test, predictions),
                    "f1": f1_score(y_test, predictions),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                }
            )

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = ROOT_DIR / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_models(config)
    summary = summarize_metrics(results)

    results.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2))

    print(f"Saved {output_dir / 'fold_metrics.csv'}")
    print(f"Saved {output_dir / 'summary_metrics.csv'}")
    print(f"Saved {output_dir / 'resolved_config.json'}")


if __name__ == "__main__":
    main()
