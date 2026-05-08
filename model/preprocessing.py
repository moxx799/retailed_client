from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(num_cols, cat_cols):
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        sparse_threshold=1.0,
    )


def make_dense_train_val_features(x_train, x_val, num_cols, cat_cols, n_components=256, random_state=42):
    pre = build_preprocessor(num_cols, cat_cols)
    z_train = pre.fit_transform(x_train)
    z_val = pre.transform(x_val)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    train_dense = svd.fit_transform(z_train)
    val_dense = svd.transform(z_val)
    scaler = StandardScaler()
    return (
        scaler.fit_transform(train_dense).astype("float32"),
        scaler.transform(val_dense).astype("float32"),
    )


def make_clustering_embedding(x, num_cols, cat_cols, n_components=64, random_state=42):
    pre = build_preprocessor(num_cols, cat_cols)
    z = pre.fit_transform(x)
    embed = TruncatedSVD(n_components=n_components, random_state=random_state).fit_transform(z)
    return StandardScaler().fit_transform(embed)
