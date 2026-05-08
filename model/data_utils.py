import numpy as np
import pandas as pd

from .paths import COLUMNS_PATH, DATA_PATH


CATEGORICAL_CODES = [
    "detailed industry recode",
    "detailed occupation recode",
    "own business or self employed",
    "veterans benefits",
    "year",
]

MONEY_COLS = [
    "wage per hour",
    "capital gains",
    "capital losses",
    "dividends from stocks",
]


def load_dataset():
    cols = [x.strip() for x in COLUMNS_PATH.read_text().splitlines() if x.strip()]
    df = pd.read_csv(DATA_PATH, header=None, names=cols, na_values="?")
    target_col = cols[-1]
    return df, target_col


def prepare_features(df, target_col):
    x = df.drop(columns=target_col).copy()
    cat_cols = list(x.select_dtypes(include="object").columns) + CATEGORICAL_CODES
    cat_cols = list(dict.fromkeys(cat_cols))
    num_cols = [c for c in x.columns if c not in cat_cols]
    x = x.assign(**{c: np.log1p(x[c].astype("float32")) for c in MONEY_COLS})
    return x, num_cols, cat_cols


def binary_target(df, target_col, positive_label="50000+."):
    return df[target_col].eq(positive_label).astype("float32").to_numpy()
