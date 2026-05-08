import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from model.classification import run_classification_experiment
    from model.clustering import run_clustering_experiment
    from model.data_utils import binary_target, load_dataset, prepare_features
    from model.paths import RESULTS_DIR
else:
    from .classification import run_classification_experiment
    from .clustering import run_clustering_experiment
    from .data_utils import binary_target, load_dataset, prepare_features
    from .paths import RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate classification and clustering experiments.")
    parser.add_argument(
        "--task",
        choices=["all", "classification", "clustering"],
        default="all",
        help="Which experiment to run.",
    )
    parser.add_argument("--classification-epochs", type=int, default=20)
    parser.add_argument("--classification-components", type=int, default=256)
    parser.add_argument("--classification-batch-size", type=int, default=4096 * 4)
    parser.add_argument("--classification-lr", type=float, default=1e-3)
    parser.add_argument("--clustering-components", type=int, default=64)
    parser.add_argument("--clustering-sample-size", type=int, default=20000)
    return parser.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    df, target_col = load_dataset()
    x, num_cols, cat_cols = prepare_features(df, target_col)

    if args.task in {"all", "classification"}:
        y = binary_target(df, target_col)
        cv_results, summary = run_classification_experiment(
            x,
            y,
            num_cols=num_cols,
            cat_cols=cat_cols,
            n_components=args.classification_components,
            epochs=args.classification_epochs,
            batch_size=args.classification_batch_size,
            lr=args.classification_lr,
        )
        cv_results.to_csv(RESULTS_DIR / "classification_cv_results.csv", index=False)
        summary.to_csv(RESULTS_DIR / "classification_summary.csv", index=False)
        print(f"Saved {RESULTS_DIR / 'classification_cv_results.csv'}")
        print(f"Saved {RESULTS_DIR / 'classification_summary.csv'}")

    if args.task in {"all", "clustering"}:
        summary, projection, label_classes = run_clustering_experiment(
            x,
            labels=df[target_col],
            num_cols=num_cols,
            cat_cols=cat_cols,
            embed_components=args.clustering_components,
            sample_n=args.clustering_sample_size,
        )
        summary.to_csv(RESULTS_DIR / "clustering_summary.csv", index=False)
        projection.to_csv(RESULTS_DIR / "clustering_sample_projection.csv", index=False)
        with open(RESULTS_DIR / "clustering_label_classes.json", "w") as f:
            json.dump(label_classes, f, indent=2)
        print(f"Saved {RESULTS_DIR / 'clustering_summary.csv'}")
        print(f"Saved {RESULTS_DIR / 'clustering_sample_projection.csv'}")
        print(f"Saved {RESULTS_DIR / 'clustering_label_classes.json'}")


if __name__ == "__main__":
    main()
