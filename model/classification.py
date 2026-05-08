import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from .preprocessing import make_dense_train_val_features

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False

DEVICE = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"


if TORCH_AVAILABLE:
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden=(512, 256, 64), dropout=(0.25, 0.20, 0.10)):
            super().__init__()
            layers = []
            in_dim = input_dim
            for h, p in zip(hidden, dropout):
                layers.extend(
                    [
                        nn.Linear(in_dim, h),
                        nn.BatchNorm1d(h),
                        nn.GELU(),
                        nn.Dropout(p),
                    ]
                )
                in_dim = h
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(1)


def set_torch_seed(seed):
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def run_fold(
    x_train,
    x_val,
    y_train,
    y_val,
    num_cols,
    cat_cols,
    n_components=256,
    epochs=20,
    batch_size=4096 * 4,
    lr=1e-3,
    seed=42,
):
    set_torch_seed(seed)
    x_train_dense, x_val_dense = make_dense_train_val_features(
        x_train,
        x_val,
        num_cols=num_cols,
        cat_cols=cat_cols,
        n_components=n_components,
        random_state=seed,
    )
    if TORCH_AVAILABLE:
        train_ds = TensorDataset(torch.tensor(x_train_dense), torch.tensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        x_val_t = torch.tensor(x_val_dense, device=DEVICE)

        model = MLP(x_train_dense.shape[1]).to(DEVICE)
        pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()], device=DEVICE)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)

        for _ in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(x_val_t)).cpu().numpy()
    else:
        model = MLPClassifier(
            hidden_layer_sizes=(512, 256, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=min(batch_size, len(x_train_dense)),
            learning_rate_init=lr,
            max_iter=epochs,
            random_state=seed,
        )
        model.fit(x_train_dense, y_train.astype(int))
        prob = model.predict_proba(x_val_dense)[:, 1]

    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val.astype(int), pred).ravel()
    return {
        "accuracy": accuracy_score(y_val, pred),
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred, zero_division=0),
        "f1": f1_score(y_val, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, prob),
        "pr_auc": average_precision_score(y_val, prob),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def run_classification_experiment(
    x,
    y,
    num_cols,
    cat_cols,
    n_components=256,
    epochs=20,
    batch_size=4096 * 4,
    lr=1e-3,
):
    splitters = {
        "cv_5_seed_42": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        "cv_5_seed_7": StratifiedKFold(n_splits=5, shuffle=True, random_state=7),
        "cv_10_seed_42": StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    }

    rows = []
    for name, cv in splitters.items():
        for fold, (tr, va) in enumerate(cv.split(x, y), 1):
            scores = run_fold(
                x.iloc[tr].copy(),
                x.iloc[va].copy(),
                y[tr],
                y[va],
                num_cols=num_cols,
                cat_cols=cat_cols,
                n_components=n_components,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                seed=42 + fold,
            )
            rows.append({"splitter": name, "fold": fold, **scores})

    cv_results = pd.DataFrame(rows)
    summary = cv_results.groupby("splitter", as_index=False).mean(numeric_only=True).sort_values("f1", ascending=False)
    return cv_results, summary
