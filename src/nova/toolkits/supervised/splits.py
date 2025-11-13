import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from nova.toolkits.supervised.splits_viz import plot_cv_indices, plot_manual_cv_assignments


def make_kfold_splits(df, n_folds, X, y, path, random_seed) -> tuple[str, list[str], str]:
    split_method = "kfold"
    fold_columns = []
    split_obj = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    for fold, (train_idx, val_idx) in enumerate(split_obj.split(X)):
        colname = f"fold_{fold + 1}"
        df[colname] = "train"
        df.loc[df["slide_id"].isin(X[val_idx]), colname] = "test"
        fold_columns.append(colname)

    fig, ax = plt.subplots()
    ax = plot_cv_indices(split_obj, X, y, ax, n_folds)
    plot_path = path.parent / f"{split_method}splits_{path.stem}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    split_metadata_path = path.parent / f"{split_method}splits_{path.stem}.csv"
    df.to_csv(split_metadata_path, index=False)

    return str(split_metadata_path), fold_columns, str(plot_path)


def make_stratified_kfold_splits(
    df,
    n_folds,
    X,
    y,
    path,
    cat_col,
    random_seed,
) -> tuple[str, list[str], str]:
    split_method = "stratifiedkfold"
    fold_columns = []
    split_obj = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    for fold, (train_idx, val_idx) in enumerate(split_obj.split(X, df[cat_col])):
        colname = f"fold_{fold + 1}"
        df[colname] = "train"
        df.loc[df["slide_id"].isin(X[val_idx]), colname] = "test"
        fold_columns.append(colname)

    fig, ax = plt.subplots()
    ax = plot_cv_indices(split_obj, X, y, ax, n_folds)
    plot_path = path.parent / f"{split_method}splits_{path.stem}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    split_metadata_path = path.parent / f"{split_method}splits_{path.stem}.csv"
    df.to_csv(split_metadata_path, index=False)

    return str(split_metadata_path), fold_columns, str(plot_path)


def make_montecarlo_splits(
    df,
    split_ratios,
    n_folds,
    X,
    y,
    path,
    random_seed,
    cat_col,
) -> tuple[str, list[str], str]:
    split_method = "montecarlo"
    fold_columns = []

    if split_ratios is None:
        split_ratios = [0.8, 0.2]  # Default to 80% train, 20% test
        warnings.warn("split_ratios not provided. Defaulting to [0.8, 0.2] for train/test split.", UserWarning)

    if sum(split_ratios) != 1.0:
        raise ValueError("split_ratios must sum to 1.0.")

    # if there are any classes in _cat column whose count is less than n_folds, raise an informative error
    value_counts = df[cat_col].value_counts()
    low_sample_classes = value_counts[value_counts < n_folds]
    if not low_sample_classes.empty:
        raise ValueError(
            f"Cannot perform stratified split with n_folds={n_folds}: "
            f"The following classes in '{cat_col}' have fewer samples than the number of folds:\n"
            f"{low_sample_classes.to_string()}\n"
            "You must merge/remove these classes or lower the number of folds."
        )

    montecarlo_split_assignments = []  # NEW: To store "test" assignments for each fold

    for fold in range(n_folds):
        colname = f"fold_{fold + 1}"
        df[colname] = "train"
        train_cases, test_cases = train_test_split(
            X,
            test_size=split_ratios[1],
            stratify=y,
            random_state=random_seed + fold,  # ensure diff split per fold!
        )
        df.loc[df["slide_id"].isin(test_cases), colname] = "test"
        df.loc[df["slide_id"].isin(train_cases), colname] = "train"
        fold_columns.append(colname)

        # Store for plotting: a mask/array with 0=train, 1=test, np.nan=should not happen
        assignment = np.full(len(X), np.nan)
        assignment[np.isin(X, test_cases)] = 1
        assignment[np.isin(X, train_cases)] = 0
        montecarlo_split_assignments.append(assignment)

    fig, ax = plot_manual_cv_assignments(montecarlo_split_assignments, y)
    plot_path = path.parent / f"{split_method}splits_{path.stem}.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    split_metadata_path = path.parent / f"{split_method}splits_{path.stem}.csv"
    df.to_csv(split_metadata_path, index=False)

    return str(split_metadata_path), fold_columns, str(plot_path)
