import warnings
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from smolagents import tool
from trident.Converter import OPENSLIDE_EXTENSIONS, PIL_EXTENSIONS

from nova.toolkits.supervised.dataprep import (
    detect_case_id_column,
    drop_rare_categories,
    encode_label_column,
    get_missing_slide_samples,
    get_unopenable_slides,
    preview_columns,
    reorder_columns,
    validate_slide_ids,
)
from nova.toolkits.supervised.splits import (
    make_kfold_splits,
    make_montecarlo_splits,
    make_stratified_kfold_splits,
)
from nova.toolkits.supervised.training import WSIClassificationExperiment
from nova.utils.deterministic import _set_deterministic
from nova.utils.summarize import print_log

# Slide extensions
VALID_SLIDE_EXTS = OPENSLIDE_EXTENSIONS.union(PIL_EXTENSIONS)


# --> tools
@tool
def train_test_wsi_classification_mil_model(
    splits_path: str,
    patch_features_path: str,
    job_dir: str,
    label_column: str,
    model_name: str = 'ABMIL',
    input_feature_dim: int | None = None,
    n_heads: int = 1,
    head_dim: int = 512,
    dropout: float = 0.0,
    gated: bool = True,
    hidden_dim: int = 256,
    num_classes: int | None = None,
    num_epochs: int = 10,
    batch_size: int = 1,
    learning_rate: float = 0.001,
    optim: str = 'adamw',
    num_folds: int = 5,
    criterion: str = 'cross_entropy',
    sample_during_training: bool = True,
    num_patches_to_sample: int | None = None,
) -> dict:
    """
    Trains and evaluates an Attention-Based Multiple Instance Learning (ABMIL) model for whole slide image classification using cross-validation.

    This function manages the complete experimental pipeline for weakly-supervised learning on WSIs.
    It trains an attention-based MIL model for each fold, saves model checkpoints and metrics, predicts on held-out data,
    saves attention scores, and aggregates results across all folds. This tool should never be used to classify patches.

    Notes on using this tool:
        - Loads patch features and metadata from provided paths
        - Trains ABMIL model for specified number of folds using cross-validation
        - Evaluates model performance on test splits
        - Saves attention scores for interpretability analysis
        - Aggregates metrics across all folds
        - The tool creates the following output directories and files:
            - {job_dir}/{label_column}_classification/{model_name}/ --> this the root directory for the experiment and all results are stored here
            - {job_dir}/{label_column}_classification/{model_name}/experiment_config.json --> config for the experiment with `data_args`, `model_args`, and `experiment_args`
            - {job_dir}/{label_column}_classification/{model_name}/final_metrics.json --> final per fold metrics (ex: `accuracy`) and `attention_save_dirs` (where attention scores on eval set of each fold are stored)
        - For each fold going from 1 to num_folds, the tool creates:
            - {job_dir}/{label_column}_classification/{model_name}/fold_{fold_idx}/final_val_metrics.json --> metrics for the validation set of the fold
            - {job_dir}/{label_column}_classification/{model_name}/fold_{fold_idx}/metrics_per_epoch.json --> per epoch train and val loss and metrics for the fold
            - {job_dir}/{label_column}_classification/{model_name}/fold_{fold_idx}/model_last.pt --> model checkpoint from the last epoch of training
            - {job_dir}/{label_column}_classification/{model_name}/fold_{fold_idx}/attention_scores/{slide_id}.h5 --> attention scores for each slide in the test set. Saved in h5 dataset called `attention_scores` with shape (1,1,number_of_patches)
            - {job_dir}/{label_column}_classification/{model_name}/fold_{fold_idx}/train_val_metrics.png --> visualization of train and validation metrics over epochs

    Prerequisites to run this tool:
        - Patch features must be extracted and saved as .h5 files (one per slide) at `patch_features_path`
        - Metadata CSV must have the following columns at the least (`slide_id`, {label_column}_cats which is the categorical version of labels, fold_{idx} where idx goes from 0 to {num_folds - 1})
        - the tool internally uses {label_column}_cats for labels.

    The tool returns:
        dict: Summary of experiment results containing:
            - "all_fold_results": List of dictionaries with test metrics for each fold (index corresponds to fold). Each dictionary has keys:
                - 'y_true': list of ground truth labels in test set for fold
                - 'y_pred': predicted labels in test set for fold
                - 'y_prob': predicted probabilities (list of tuples) for each sample in test set for fold
                - 'loss': total test loss
                - 'accuracy': test set accuracy for fold
                - 'bacc': test set balanced accuracy for fold
                - 'auc': area under the curve in test set for fold
                - 'f1': F1 score in test set for fold
                - 'n_samples': number of samples in test set for fold
                - 'attention_save_dir': directory where attention scores are saved for each slide_id in fold
                - 'slide_ids' : List of slide_ids in the fold's test set in the EXACT order in which they were processed by the forward loop. They match the order of `y_true`, `y_pred`, and `y_prob`
            - "root_save_dir": Absolute path to experiment root directory
            - "experiment_config_json_path": Absolute path to saved experiment configuration
            - "final_metrics_json_path": Absolute path to aggregated metrics across all folds
            - "data_args": Dictionary with data-related arguments used in the experiment
            - "model_args": Dictionary with model-related arguments used in the experiment
            - "experiment_args": Dictionary with experiment-related arguments used in the experiment
            - "operations_log" : A log of operations performed during the experiment

    Args:
        splits_path: Path to CSV file with metadata for all slides. Must include `slide_id`,
            `case_id`, `{label_column}_cats` (categorical labels), and fold columns (`fold_0`..`fold_{num_folds-1}`).
        patch_features_path: Directory containing .h5 files for each slide, named `{slide_id}.h5`. Each file must
            contain dataset `features` with shape (num_patches, input_feature_dim).
        job_dir: Directory to save all results, configs, logs, models, and metrics.
        label_column: Base column name for labels in the metadata CSV. Internally, `{label_column}_cats` is used.

        model_name: Optional. MIL architecture to use. Currently only `ABMIL` is supported. Default: 'ABMIL'.
        input_feature_dim: Optional. Dimension of each patch feature vector. If None, inferred from the first .h5 file. Default: None.

        n_heads: Optional. Number of attention heads in ABMIL (>=1). Default: 1.
        head_dim: Optional. Dimension per attention head (the attention projection size). Default: 512.
        dropout: Optional. Dropout rate applied in the attention/classifier modules (0.0–1.0). Default: 0.0.
        gated: Optional. If True, use gated attention; if False, use standard (ungated) attention. Default: True.
        hidden_dim: Optional. Hidden layer size of the slide-level classifier head. Default: 256.
        num_classes: Optional. Number of target classes (>=2). If None, inferred from `{label_column}_cats`. Default: None.
        num_epochs: Optional. Number of training epochs per fold (>=1). Default: 10.
        batch_size: Optional. Mini-batch size for training (bags per step). MIL commonly uses 1. Test time always uses 1. Default: 1.
        learning_rate: Optional. Optimizer learning rate (>0). Default: 1e-3.
        optim: Optional. Optimizer to use. Only 'adamw' is supported. Default: 'adamw'.
        num_folds: Optional. Number of cross-validation folds (>=2). Folds iterated from 0 to `num_folds-1`. Default: 5.
        criterion: Optional. Loss function. Only 'cross_entropy' is supported. Default: 'cross_entropy'.
        sample_during_training: Optional. If True, randomly sample a subset of patches per slide at each epoch/step
                                to speed training. Sampling is not used for validation/test. Default: True.
        num_patches_to_sample: Optional. Number of patches to sample per slide when `sample_during_training=True`
                               (must be a positive integer). Required if `sample_during_training=True`. Default: None.
    """
    _set_deterministic()

    data_args = {
        "splits_path": splits_path,
        "patch_features_path": patch_features_path,
        "job_dir": job_dir,
        "label_column": label_column,
    }
    model_args = {
        "model_name": model_name,
        "input_feature_dim": input_feature_dim,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "dropout": dropout,
        "gated": gated,
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
    }
    experiment_args = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optim": optim,
        "num_folds": num_folds,
        "criterion": criterion,
        "sample_during_training": sample_during_training,
        "num_patches_to_sample": num_patches_to_sample,
    }

    # define the experiment
    experiment = WSIClassificationExperiment(
        data_args=data_args,
        model_args=model_args,
        experiment_args=experiment_args,
    )

    all_fold_results = []
    for curr_fold_idx in range(num_folds):
        # Fit returns a dict with the last_model_path
        fit_results = experiment.fit(
            curr_fold_idx=curr_fold_idx,
        )

        # Run predict with the loaded last model
        val_metrics = experiment.predict(
            curr_fold_idx=curr_fold_idx,
            model_path=fit_results["last_model_path"],
            return_probs=True,
            return_raw_attention=True,  # save the attention scores for each slide in the test set
        )

        # save val_metrics
        fold_dir = experiment.get_root_save_dir() / f"fold_{curr_fold_idx}"
        experiment.save_json(val_metrics, fold_dir / "final_val_metrics.json")
        all_fold_results.append(val_metrics)

    # gather and save results for all folds
    metrics_path = experiment.gather_exp_metrics(all_fold_results)

    asset_dict = {
        "all_fold_results": all_fold_results,
        "root_save_dir": str(experiment.get_root_save_dir()),
        "experiment_config_json_path": str(experiment.get_exp_config_dir()),
        "final_metrics_json_path": str(metrics_path),
        "data_args": data_args,
        "model_args": model_args,
        "experiment_args": experiment_args,
        "operations_log": experiment.get_exp_log(),
    }

    return asset_dict


@tool
def create_wsi_classification_splits(
    cleaned_metadata_path: str,
    label_column: str,
    split_method: str,
    n_folds: int = 5,
    split_ratios: Optional[List[float]] = None,
    random_seed: int = 42,
) -> dict:
    """
    Creates split assignments (train and val only) for WSI classification datasets with support for multiple splitting strategies (kfold, stratified kfold, montecarlo).

    This tool modifies a cleaned metadata CSV by adding fold assignment columns. A visualization of split assignments is produced.
    Splits are assigned at the case level to prevent data leakage.

    Notes on using this tool:
        - Loads cleaned metadata CSV with encoded labels
        - Creates cross-validation train-val splits using specified strategy (k-fold, stratified k-fold, or Monte Carlo)
        - Generates visualization of split assignments
        - Saves updated metadata with fold columns
        - The tool creates the following output directories and files:
            - path = Path(cleaned_metadata_path)
            - {path.parent}/f"{split_method}splits_{path.stem}.csv" --> splits metadata with fold assignments (case_id, slide_id, label_column, label_column_cat, fold_1, fold_2, ..., fold_{n_folds})
            - {path.parent}/f"{split_method}splits_{path.stem}.png" --> visualization of split assignments

    Prerequisites to run this tool:
        - Input metadata must be cleaned to contain case_id, slide_id, label_column, and {label_column}_cats columns
        - For stratified methods, all classes must have sufficient samples. Adjust method and fraction if necessary.

    The tool returns:
        dict: Dictionary containing keys:
            - "split_metadata_path": Path to the updated metadata CSV with fold assignments
            - "n_folds": Number of folds used in the split
            - "fold_columns": List of fold assignment column names (e.g., fold_1, fold_2, ...)
            - "splits_visualization_path": Path to the generated split assignments visualization image
            - "Operations_log": Log of operations performed during the split creation

    Args:
        cleaned_metadata_path: Path to input metadata CSV from prepare_wsi_classification_metadata
        label_column: Name of the label column (without '_cats'). '{label_column}_cats' is used for stratification
        split_method: Splitting method to use. Must be one of: 'kfold', 'stratifiedkfold', 'montecarlo'
        n_folds: Number of folds/splits. For kfold/stratifiedkfold: number of folds. For montecarlo: number of repeated splits
        split_ratios: Only used for 'montecarlo'. List of floats for train/test fractions. Must sum to 1.0
        random_seed: Random seed for reproducible splits
    """

    log = []

    _set_deterministic()

    path = Path(cleaned_metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"File '{cleaned_metadata_path}' not found. Operations log: {print_log(log)}")
    df = pd.read_csv(path)
    col_names, preview = preview_columns(df)
    log.append(f"\n Loaded metadata from {path}. Columns: {col_names}\nPreview:\n{preview}")

    for col in ["case_id", "slide_id"]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' missing from metadata.\nColumns: {col_names}\nPreview:\n{preview}. Operations log: {print_log(log)}"
            )
    log.append("\n Found required columns: case_id, slide_id in metadata.")

    cat_col = f"{label_column}_cats"
    if cat_col not in df.columns:
        raise ValueError(
            f"Required categorical label column '{cat_col}' not found in metadata.\n"
            f"Columns: {col_names}\nPreview:\n{preview}"
        )
    log.append(f"\n Found required categorical label column: {cat_col} in metadata.")

    if split_method.lower() not in ["kfold", "stratifiedkfold", "montecarlo"]:
        raise ValueError(
            f"Invalid split_method '{split_method}'. Must be one of: kfold, stratifiedkfold, montecarlo. Operations log: {print_log(log)}"
        )
    log.append(f"\n Using split method: {split_method} with n_folds={n_folds}.")

    if split_method.lower() in ["kfold", "stratifiedkfold"] and split_ratios is not None:
        warnings.warn(
            f"split_ratios is provided but will be ignored for split_method='{split_method}'. "
            "kfold and stratifiedkfold splits always use fixed proportions by design."
            f"Since number of folds is {n_folds}, train-test split will be {1 - round(1 / n_folds, 2)}-{round(1 / n_folds, 2)}",
            UserWarning,
        )
        log.append(f"\n Ignored split_ratios for split_method='{split_method}'.")

    X = np.array(df["slide_id"])
    y = np.array(df[cat_col])

    if split_method.lower() == "kfold":
        split_metadata_path, fold_columns, plot_path = make_kfold_splits(df, n_folds, X, y, path, random_seed)
        log.append(f"\n Created kfold splits with {n_folds} folds. Metadata saved to {split_metadata_path}.")
    elif split_method.lower() == "stratifiedkfold":
        split_metadata_path, fold_columns, plot_path = make_stratified_kfold_splits(
            df, n_folds, X, y, path, cat_col, random_seed
        )
        log.append(f"\n Created stratified kfold splits with {n_folds} folds. Metadata saved to {split_metadata_path}.")
    elif split_method.lower() == "montecarlo":
        split_metadata_path, fold_columns, plot_path = make_montecarlo_splits(
            df, split_ratios, n_folds, X, y, path, random_seed, cat_col
        )
        log.append(
            f"\n Created Monte Carlo splits with {n_folds} folds and ratios {split_ratios}. Metadata saved to {split_metadata_path}."
        )
    else:
        raise ValueError(
            f"Unsupported split method: {split_method}. Must be one of: kfold, stratifiedkfold, montecarlo. Operations log: {print_log(log)}"
        )

    return {
        "split_metadata_path": str(split_metadata_path),
        "n_folds": int(n_folds),
        "fold_columns": fold_columns,
        'splits_visualization_path': str(plot_path),
        'Operations_log': "\n".join(log),
    }


@tool
def prepare_wsi_classification_metadata(
    metadata_path: str,
    label_column: str,
    drop_threshold: int = 5,
) -> dict[str, Any]:
    """
    Cleans, validates, and encodes WSI metadata for supervised learning pipelines.

    This tool performs comprehensive data cleaning and validation on pathology slide metadata to ensure it is ready for
    machine learning workflows. It handles missing slides, rare class removal, label encoding, and column standardization.

    Notes on using this tool:
        - Validates slide paths and removes missing or unreadable slides
        - Drops rare classes below specified threshold
        - Encodes categorical labels to integers
        - Standardizes column names and order
        - Saves cleaned metadata for downstream use
        - Saves the cleaned metadata to a new CSV file with the prefix "cleaned_". Saves in the same directory as the input file.
        - The tool returns samples it dropped due to low class count, missing slides, or unreadable slides in the 'dropped_samples' key of the returned dict, where keys are:
            - "low_class_count": List of dict, each dict contains a row that was dropped due to low class count
            - "missing_slide": List of dict, each dict contains a row that was dropped due to missing slide files
            - "cannot_load_slide": List of dict, each dict contains a row that was dropped due to unreadable slide files

    Prerequisites to run this tool:
        - Input metadata file must exist and be in CSV or TSV format. Must contain a "case_id" column, "slide_id" column, "slide_path" column, and the specified "label_column".
        - Must contain columns: slide_id, slide_path, case ID column, and specified label_column
        - Slide paths must point to valid slide files with supported extensions. Must be openslide readable files.

    Return:
        - dict: Dictionary containing:
            - "cleaned_metadata_path": Path to the cleaned metadata CSV file
            - "dropped_samples": Dictionary with counts of dropped samples due to low class count, missing slides, or unreadable slides
            - "label_map": Mapping of original labels to encoded integer labels
            - "Operations_log": A log of operations performed during the metadata preparation

    Args:
        metadata_path: Path to input metadata file (.csv or .tsv)
        label_column: Name of the label column in the file to be used for training and encoding
        drop_threshold: Minimum number of samples required per class. Classes with fewer samples are dropped
    """

    log = []

    _set_deterministic()

    # --- File checks and load ---
    path_to_metadata = Path(metadata_path)
    if not path_to_metadata.exists():
        raise FileNotFoundError(f"Metadata file '{path_to_metadata}' does not exist. Operations log: {print_log(log)}")

    if path_to_metadata.suffix not in {'.csv', '.tsv'}:
        raise ValueError(
            f"Metadata file must be .csv or .tsv, got '{path_to_metadata.suffix}'. Operations log: {print_log(log)}"
        )

    if path_to_metadata.suffix == '.csv':
        df = pd.read_csv(path_to_metadata)
    else:
        df = pd.read_csv(path_to_metadata, sep='\t')
    col_names, preview = preview_columns(df)
    log.append(f"\n Loaded metadata from {path_to_metadata}. Columns: {col_names}\nPreview:\n{preview}")

    # --- Slide ID detection and validation ---
    if "slide_id" not in col_names:
        raise ValueError(
            "No column named 'slide_id' found in metadata.\n"
            f"Available columns: {col_names}\nFirst five entries in each column:\n{preview}"
            f"Operations log: {print_log(log)}"
        )
    log.append("\n Found required column: slide_id in metadata.")

    # ensure that slide_path column is present
    if "slide_path" not in col_names:
        raise ValueError(
            "No column named 'slide_path' found in metadata.\n"
            f"Available columns: {col_names}\nFirst five entries in each column:\n{preview}"
            f"Operations log: {print_log(log)}"
        )
    validate_slide_ids(df, "slide_path")
    log.append("\n Found required column: slide_path in metadata.")

    # --- Case ID detection ---
    case_id_col = detect_case_id_column(df)
    if not case_id_col:
        raise ValueError(
            "No column with case IDs found in metadata.\n"
            f"Available columns: {col_names}\nFirst five entries in each column:\n{preview}"
            f"Operations log: {print_log(log)}"
        )
    if case_id_col != "case_id":
        df = df.rename(columns={case_id_col: "case_id"})
    log.append(f"\n Detected case ID column: {case_id_col}. Renamed to 'case_id'.")

    # check label_column is present
    if label_column not in col_names:
        raise ValueError(
            f"Label column '{label_column}' does not exist in the metadata.\n"
            f"Available columns: {col_names}\nFirst five entries in each column:\n{preview}"
            f"Operations log: {print_log(log)}"
        )
    log.append(f"\n Found required label column: {label_column} in metadata.")

    # --- Drop rare categories ---
    df, dropped_low_count = drop_rare_categories(df, label_column, drop_threshold)
    if len(dropped_low_count) > 0:
        log.append(
            f"\n Dropped {dropped_low_count} samples with low class count (< {drop_threshold}) from '{label_column}'."
        )

    # --- Drop missing slide files ---
    df, dropped_missing_slide = get_missing_slide_samples(df, "slide_path")
    if len(dropped_missing_slide) > 0:
        log.append(f"\n Dropped {len(dropped_missing_slide)} samples with missing slide files from 'slide_path'.")

    # --- Drop slides that cannot be opened ---
    df, dropped_unopenable_slide = get_unopenable_slides(df, "slide_path")
    if len(dropped_unopenable_slide) > 0:
        log.append(f"\n Dropped {len(dropped_unopenable_slide)} samples with unreadable slide files from 'slide_path'.")

    # --- Collect all drop info to return ---
    dropped_samples = {
        "low_class_count": dropped_low_count,
        "missing_slide": dropped_missing_slide,
        "cannot_load_slide": dropped_unopenable_slide,
    }

    # --- Label encoding --- Creates _cat column
    df, label_map = encode_label_column(df, label_column)
    log.append(f"\n Encoded label column '{label_column}' to categorical integers. Mapping: {label_map}")

    # --- Column order ---
    df = reorder_columns(df, label_column)

    # --- Final index reset ---
    df.reset_index(drop=True, inplace=True)
    log.append(f"\n Reset index of the DataFrame after all operations. Len of df is now: {len(df)}")

    # --- Save cleaned metadata ---
    cleaned_metadata_path = path_to_metadata.parent / f"cleaned_{path_to_metadata.stem}.csv"
    df.to_csv(cleaned_metadata_path, index=False)
    log.append(f"\n Saved cleaned metadata to {cleaned_metadata_path}.")

    return {
        'cleaned_metadata_path': str(cleaned_metadata_path),
        'dropped_samples': dropped_samples,
        'label_map': label_map,
        'Operations_log': print_log(log),
    }
