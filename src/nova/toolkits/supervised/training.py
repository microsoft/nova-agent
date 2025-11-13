import json
import warnings
from pathlib import Path
from typing import Any, List, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from trident.Converter import OPENSLIDE_EXTENSIONS, PIL_EXTENSIONS

from nova.toolkits.supervised.data import H5Dataset
from nova.toolkits.supervised.mil import ABMILClassificationModel
from nova.utils.deterministic import _set_deterministic

# Constants
VALID_SLIDE_EXTS = OPENSLIDE_EXTENSIONS.union(PIL_EXTENSIONS)

_set_deterministic()


__all__ = [
    "WSIClassificationExperiment",
]


class WSIClassificationExperiment:
    def __init__(
        self,
        data_args: dict[str, Any],
        model_args: dict[str, Any],
        experiment_args: dict[str, Any],
    ) -> None:
        """
        Initialize a WSIClassificationExperiment for slide-level classification.

        Args:
            data_args (dict): Required data configuration. Must include:
                - "splits_path" (str): Path to CSV or TSV with metadata, folds, and labels.
                - "patch_features_path" (str): Directory with .h5 files (named by slide_id).
                - "job_dir" (str): Directory to save all experiment results and configs.
                - "label_column" (str): Name of column in splits file with class labels.
            model_args (dict): Model architecture and input configuration, such as:
                - "model_name" (str): Model type (e.g., "ABMIL").
                - "input_feature_dim" (int|None): Patch feature size; inferred from .h5 if None.
                - "n_heads" (int): Number of attention heads for ABMIL (default: 1).
                - "head_dim" (int): Hidden dim per attention head (default: 512).
                - "dropout" (float): Dropout rate (default: 0.0).
                - "gated" (bool): Whether to use gated attention (default: True).
                - "hidden_dim" (int): Hidden size for classifier (default: 256).
                - "num_classes" (int|None): Number of classes; inferred from labels if None.
            experiment_args (dict): Training configuration, such as:
                - "num_epochs" (int): Number of training epochs.
                - "batch_size" (int): Batch size for DataLoader.
                - "learning_rate" (float): Optimizer learning rate.
                - "optim" (str): Optimizer name, e.g., "adamw".
                - "num_folds" (int): Number of CV folds.
                - "criterion" (str): Loss function, e.g., "cross_entropy".
                - "sample_during_training" (bool): If True, sample patches per slide during training.
                - "num_patches_to_sample" (int|None): Number of patches to sample if sampling enabled.

        Returns:
            None

        Raises:
            FileNotFoundError: If data paths are missing or .h5 files cannot be found for inferring input size.
            ValueError: If label_column is missing from splits file, or num_patches_to_sample is not set
                when required.

        Warnings:
            - If batch_size > 1 and patch sampling is not enabled, sampling will be auto-enabled with a warning.
            - If input_feature_dim is None, it will be inferred from the first .h5 file (with a warning).
            - If num_classes is None, it will be inferred from the label_column in splits file (with a warning).

        Notes:
            - The experiment root directory is created at: {job_dir}/{label_column}_classification/{model_name}
            - All configuration (data, model, experiment) is saved as experiment_config.json in the root save directory.
            - No model training or output is performed at this stage—only environment/configuration setup.
            - Ensures all downstream training code can safely rely on validated, reproducible experiment configuration.

        """

        self.data_args = data_args
        self.model_args = model_args
        self.experiment_args = experiment_args
        self.log = []

        self._validate_paths()
        self._validate_sampling_config()
        self._infer_model_params()
        self._setup_save_directory()
        self._save_config()
        self._set_device()

    def _set_device(self) -> None:
        self.model_args["device"] = "cuda:0"

    def get_exp_log(self) -> str:
        """
        Returns the experiment log as a string.
        """
        return "\n".join(self.log)

    def _validate_paths(self) -> None:
        """Validate that required data paths exist."""
        if not Path(self.data_args["splits_path"]).exists():
            raise FileNotFoundError(
                f'Failed to find splits at {self.data_args["splits_path"]}. '
                f'Please provide a valid path to the splits CSV file. Operations log: {self.get_exp_log()}'
            )
        self.log.append(f"\n Validated splits file at {self.data_args['splits_path']}")

        if not Path(self.data_args["patch_features_path"]).exists():
            raise FileNotFoundError(
                f'Failed to find patch features at {self.data_args["patch_features_path"]}. '
                f'Please provide a valid path to the directory containing HDF5 files for patch features. Operations log: {self.get_exp_log()}'
            )
        self.log.append(f"\n Validated patch features directory at {self.data_args['patch_features_path']}")

    def _validate_sampling_config(self) -> None:
        """Validate and fix sampling configuration."""
        if self.experiment_args["batch_size"] > 1 and not self.experiment_args["sample_during_training"]:
            warnings.warn(
                "Batch size > 1 requires sample_during_training to be True. Setting sample_during_training to True."
            )
            self.experiment_args["sample_during_training"] = True
            self.log.append(
                "\n Batch size > 1 requires sample_during_training to be True. Setting sample_during_training to True."
            )

        if self.experiment_args["sample_during_training"] and self.experiment_args.get("num_patches_to_sample") is None:
            raise ValueError(
                f"When sample_during_training is True, num_patches_to_sample must be specified in experiment_args. Operations log: {self.get_exp_log()}"
            )
        self.log.append(
            f"\n Sample during training is set to {self.experiment_args['sample_during_training']} "
            f"with num_patches_to_sample={self.experiment_args.get('num_patches_to_sample', 'not set')}"
        )

    def _infer_model_params(self) -> None:
        """Infer missing model parameters from data."""

        self._infer_input_feature_dim()

        self._infer_num_classes()

    def _infer_input_feature_dim(self) -> None:
        """Infer input feature dimension from first H5 file if not provided."""
        if self.model_args['input_feature_dim'] is not None:
            self.log.append(f"\n input_feature_dim is set to {self.model_args['input_feature_dim']}.")
            return

        h5_files = list(Path(self.data_args["patch_features_path"]).glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(
                f"Failed to find any patch feature files in {self.data_args['patch_features_path']}. Operations log: {self.get_exp_log()}"
            )

        path_to_infer_from = h5_files[0]
        with h5py.File(path_to_infer_from, "r") as f:
            features_dataset = f['features']
            if hasattr(features_dataset, 'shape'):
                self.model_args['input_feature_dim'] = features_dataset.shape[1]  # type: ignore
            else:
                raise ValueError(
                    f"Cannot infer feature dimension from {path_to_infer_from}. Operations log: {self.get_exp_log()}"
                )

        warnings.warn(
            f"input_feature_dim not set. Inferred {self.model_args['input_feature_dim']} "
            f"from {path_to_infer_from} (key='features')."
        )
        self.log.append(
            f"\n Inferred input_feature_dim={self.model_args['input_feature_dim']} "
            f"from {path_to_infer_from} (key='features')."
        )

    def _infer_num_classes(self) -> None:
        """Infer number of classes from splits file if not provided."""
        if self.model_args['num_classes'] is not None:
            self.log.append(f"\n num_classes is set to {self.model_args['num_classes']}.")
            return

        df = pd.read_csv(self.data_args["splits_path"])
        if self.data_args["label_column"] not in df.columns:
            raise ValueError(
                f"Label column '{self.data_args['label_column']}' not found in splits file. Operations log: {self.get_exp_log()}"
            )

        unique_labels = df[self.data_args["label_column"]].unique()
        self.model_args['num_classes'] = len(unique_labels)
        warnings.warn(
            f"num_classes not set. Inferred {self.model_args['num_classes']} "
            f"from splits file at {self.data_args['splits_path']}."
        )
        self.log.append(
            f"\n Inferred num_classes={self.model_args['num_classes']} "
            f"from splits file at {self.data_args['splits_path']}."
        )

    def _setup_save_directory(self) -> None:
        """Set up the root save directory for experiment outputs."""
        self.root_save_dir = (
            Path(self.data_args["job_dir"])
            / f"{self.data_args['label_column']}_classification"
            / self.model_args['model_name']
        )
        self.root_save_dir.mkdir(parents=True, exist_ok=True)
        self.log.append(f"\n Created root save directory at {self.root_save_dir}")

    def _save_config(self) -> None:
        """Save experiment configuration to JSON file."""
        combined_args = {
            "data_args": self.data_args,
            "model_args": self.model_args,
            "experiment_args": self.experiment_args,
        }
        config_path = self.root_save_dir / "experiment_config.json"
        with open(config_path, "w") as f:
            json.dump(combined_args, f, indent=2)

        self.log.append(f"\n Saved experiment configuration to {config_path}")

    def gather_exp_metrics(self, all_fold_results: List[dict]) -> Path:
        """
        Aggregates metrics and attention directories from all folds and saves them to a JSON file.

        Args:
            all_fold_results (list of dict): List where each dict contains per-fold metrics and attention_save_dir.

        Returns:
            Path: Path to the saved final_metrics.json file.
        """
        metrics = {'accuracy': [], 'bacc': [], 'auc': [], 'f1': []}
        attn_save_dirs = []
        for fold in all_fold_results:
            for metric in metrics:
                metrics[metric].append(fold[metric])
            attn_save_dirs.append(fold['attention_save_dir'])

        # save metrics and attention save dirs into one json file.
        metrics_path = self.get_root_save_dir() / "final_metrics.json"
        self.save_json({"metrics": metrics, "attention_save_dirs": attn_save_dirs}, metrics_path)
        return metrics_path

    def get_root_save_dir(self) -> Path:
        """
        Returns the root save directory for this experiment.
        """
        return self.root_save_dir

    def get_exp_config_dir(self) -> Path:
        """
        Returns the path to the experiment configuration JSON file.
        """
        return self.root_save_dir / "experiment_config.json"

    def _define_model(self, weights_path: Optional[str] = None) -> nn.Module:
        """
        Construct and return an ABMILClassificationModel.

        Args:
            weights_path: Optional path to saved model weights.

        Returns:
            Initialized ABMILClassificationModel.
        """
        if self.model_args["model_name"] != 'ABMIL':
            raise ValueError(
                f"Model {self.model_args['model_name']} not supported. Only 'ABMIL' is implemented. Operations log: {self.get_exp_log()}"
            )

        model = ABMILClassificationModel(
            input_feature_dim=self.model_args["input_feature_dim"],
            n_heads=self.model_args["n_heads"],
            head_dim=self.model_args["head_dim"],
            dropout=self.model_args["dropout"],
            gated=self.model_args["gated"],
            hidden_dim=self.model_args["hidden_dim"],
            num_classes=self.model_args["num_classes"],
        )

        if weights_path:
            if not Path(weights_path).exists():
                raise FileNotFoundError(
                    f"Model weights file '{weights_path}' does not exist. Operations log: {self.get_exp_log()}"
                )
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))

        return model

    def _get_criterion(self) -> nn.Module:
        """Get loss criterion based on experiment configuration."""
        if self.experiment_args["criterion"] == 'cross_entropy':
            return torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(
                f"Criterion {self.experiment_args['criterion']} not supported. Only 'cross_entropy' implemented. Operations log: {self.get_exp_log()}"
            )

    def _get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Get optimizer based on experiment configuration."""
        if self.experiment_args["optim"] == 'adamw':
            return torch.optim.AdamW(model.parameters(), lr=self.experiment_args["learning_rate"])
        else:
            raise ValueError(
                f"Optimizer {self.experiment_args['optim']} not supported. Only 'adamw' implemented. Operations log: {self.get_exp_log()}"
            )

    def _get_loader(
        self,
        split: str,
        curr_fold_idx: int,
        batch_size: int = 1,
        shuffle: bool = False,
        sample: bool = False,
        num_features: Optional[int] = None,
        seed: int = 42,
    ) -> DataLoader:
        """
        Constructs a DataLoader for slide-level patch features for a given split and fold.
        """
        dataset = H5Dataset(
            feats_path=self.data_args["patch_features_path"],
            df_path=self.data_args["splits_path"],
            split=split,
            fold_idx=curr_fold_idx,
            sample=sample,
            label_column=self.data_args["label_column"],
            num_features=num_features,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=lambda _: np.random.seed(seed),
        )
        return loader

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, num_classes: int
    ) -> dict[str, Any]:
        """
        Compute standard classification metrics.

        Args:
            y_true: Ground truth class labels.
            y_pred: Predicted class labels.
            y_prob: Predicted class probabilities.
            num_classes: Number of target classes.

        Returns:
            dictionary with accuracy, balanced accuracy, AUC, and F1 scores.
        """
        metrics = {}
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["bacc"] = balanced_accuracy_score(y_true, y_pred)
        if num_classes == 2:
            metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])  # y_prob shape [n, 2]
        else:
            try:
                metrics["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
            except Exception:
                metrics["auc"] = None
        metrics["f1"] = f1_score(y_true, y_pred, average="macro")
        return metrics

    @staticmethod
    def save_json(data: dict[str, Any], path: Path) -> None:
        """Save dictionary to JSON file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _plot_metrics(metrics_dict: dict[str, List[float]], fold_save_dir: Path) -> None:
        """
        Plot training and validation metrics over epochs.

        Args:
            metrics_dict: dictionary containing metric lists for each epoch.
            fold_save_dir: Directory to save the plot.
        """
        epochs = range(1, len(metrics_dict["train_loss"]) + 1)
        plt.figure(figsize=(10, 6))

        plt.plot(epochs, metrics_dict["train_loss"], label="Train Loss", marker='o')
        plt.plot(epochs, metrics_dict["val_loss"], label="Val Loss", marker='o')
        plt.plot(epochs, metrics_dict["train_bacc"], label="Train BAcc", marker='o')
        plt.plot(epochs, metrics_dict["val_bacc"], label="Val BAcc", marker='o')
        plt.plot(epochs, metrics_dict["train_f1"], label="Train F1", marker='o')
        plt.plot(epochs, metrics_dict["val_f1"], label="Val F1", marker='o')

        plt.xlabel("Epoch")
        plt.ylabel("Loss / Metric")
        plt.legend()
        plt.title(f"Training & Validation Loss/Metrics [Fold = {fold_save_dir.stem}]")
        plt.tight_layout()
        plt.savefig(str(fold_save_dir / "train_val_metrics.png"))
        plt.close()

    @staticmethod
    def save_attention_h5(
        slide_id: str, attention: torch.Tensor, save_dir: Path | str, dataset_name: str = "attention_scores"
    ) -> str:
        """
        Save attention tensor for a slide as a h5 file.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{slide_id}.h5"

        # Convert to numpy if needed
        attention_np = attention.cpu().numpy() if isinstance(attention, torch.Tensor) else attention
        with h5py.File(out_path, "w") as f:
            f.create_dataset(dataset_name, data=attention_np)
        return str(out_path)

    def fit(self, curr_fold_idx: int) -> dict[str, Any]:
        """
        Train a model for a single cross-validation fold.
        """

        print(f"\nStarting training for fold {curr_fold_idx}")
        self.log.append(f"\n\n\n Starting training for fold {curr_fold_idx}")

        # Setup results folder for the current fold
        fold_save_dir = self.root_save_dir / f"fold_{curr_fold_idx}"
        fold_save_dir.mkdir(parents=True, exist_ok=True)
        self.log.append(f"\n [FOLD:{curr_fold_idx}] Created fold save directory at {fold_save_dir}")

        # basic info
        num_epochs = self.experiment_args["num_epochs"]
        num_classes = self.model_args["num_classes"]
        self.log.append(
            f"\n [FOLD:{curr_fold_idx}] Training with {num_epochs} epochs, "
            f"{num_classes} classes, and batch size {self.experiment_args['batch_size']}"
        )

        # Define train loader
        train_loader = self._get_loader(
            split="train",
            curr_fold_idx=curr_fold_idx,
            batch_size=self.experiment_args["batch_size"],
            shuffle=True,
            sample=self.experiment_args["sample_during_training"],
            num_features=self.experiment_args.get("num_patches_to_sample", None),
        )
        self.log.append(
            f"\n [FOLD:{curr_fold_idx}] Created train DataLoader with batch size {self.experiment_args['batch_size']} "
            f"and sample_during_training={self.experiment_args['sample_during_training']}"
        )

        # Define model, loss, and optimizer
        model = self._define_model().to(self.model_args["device"])
        self.log.append(
            f"\n [FOLD:{curr_fold_idx}] Defined model {self.model_args['model_name']} with input feature dim {self.model_args['input_feature_dim']}"
        )

        criterion = self._get_criterion()
        self.log.append(f"\n [FOLD:{curr_fold_idx}] Using loss criterion: {self.experiment_args['criterion']}")

        optimizer = self._get_optimizer(model)
        self.log.append(
            f"\n [FOLD:{curr_fold_idx}] Using optimizer: {self.experiment_args['optim']} with learning rate {self.experiment_args['learning_rate']}"
        )

        train_metrics_log = {
            "train_loss": [],
            "train_bacc": [],
            "train_f1": [],
            "train_auc": [],
            "val_loss": [],
            "val_bacc": [],
            "val_f1": [],
            "val_auc": [],
        }
        val_results_all_epochs = []

        # Main training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            train_y_true = []
            train_y_pred = []
            train_y_prob = []

            with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True) as pbar:
                for batch_idx, (features, labels, slide_id) in enumerate(train_loader):
                    features = features.to(self.model_args["device"])
                    labels = labels.to(self.model_args["device"])

                    optimizer.zero_grad()
                    logits = model(features)
                    loss = criterion(logits, labels.long())
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    y_pred = torch.argmax(logits, dim=1).cpu().numpy()
                    y_true = labels.cpu().numpy()
                    y_prob = torch.softmax(logits, dim=1).detach().cpu().numpy()

                    train_y_true.extend(list(y_true))
                    train_y_pred.extend(list(y_pred))
                    train_y_prob.extend(list(y_prob))

                    pbar.set_postfix({'loss': loss.item(), 'batch_idx': batch_idx})
                    pbar.update(1)

            avg_loss = running_loss / len(train_loader)
            train_metrics = self._compute_metrics(
                np.array(train_y_true), np.array(train_y_pred), np.array(train_y_prob), num_classes
            )
            train_metrics_log["train_loss"].append(avg_loss)
            train_metrics_log["train_bacc"].append(train_metrics["bacc"])
            train_metrics_log["train_f1"].append(train_metrics["f1"])
            train_metrics_log["train_auc"].append(train_metrics["auc"])
            self.log.append(
                f"\n [FOLD:{curr_fold_idx}] Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {avg_loss:.4f}, BAcc: {train_metrics['bacc']:.4f}, "
                f"F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}"
            )

            # Validation phase
            val_results = self.predict(curr_fold_idx, model=model, return_probs=True, return_raw_attention=False)
            val_results_all_epochs.append(val_results)
            train_metrics_log["val_loss"].append(val_results["loss"])
            train_metrics_log["val_bacc"].append(val_results["bacc"])
            train_metrics_log["val_f1"].append(val_results["f1"])
            train_metrics_log["val_auc"].append(val_results["auc"])
            self.log.append(
                f"\n [FOLD:{curr_fold_idx}] Epoch {epoch + 1}/{num_epochs} - "
                f"Val Loss: {val_results['loss']:.4f}, BAcc: {val_results['bacc']:.4f}, "
                f"F1: {val_results['f1']:.4f}, AUC: {val_results['auc']:.4f}"
            )

            # Save metrics/logs
            self.save_json(train_metrics_log, fold_save_dir / "metrics_per_epoch.json")
            self._plot_metrics(train_metrics_log, fold_save_dir)
            self.log.append(
                f"\n [FOLD:{curr_fold_idx}] Saved metrics and plots for epoch {epoch + 1} at {fold_save_dir}. Saved plot with metrics at {fold_save_dir / 'train_val_metrics.png'}"
            )
            self.log.append(f"\n [FOLD:{curr_fold_idx}] Completed epoch {epoch + 1}/{num_epochs}")

        # Save last model checkpoint
        last_model_path = fold_save_dir / "model_last.pt"
        torch.save(model.cpu().state_dict(), last_model_path)
        self.log.append(f"\n [FOLD:{curr_fold_idx}] Saved last model checkpoint at {last_model_path}")

        print("\033[92mDone\033[0m")
        print()

        return {
            'last_model_path': str(last_model_path),
            'train_metrics_log': train_metrics_log,
            'val_results_last_epoch': val_results_all_epochs[-1],
        }

    @torch.no_grad()
    def predict(
        self,
        curr_fold_idx: int,
        model: nn.Module | None = None,
        model_path: str | None = None,
        return_probs: bool = True,
        return_raw_attention: bool = False,
    ) -> dict[str, Any]:
        """
        Run inference on the val split for a given fold.
        """
        self.log.append(f"\n\n\n [FOLD:{curr_fold_idx}] Starting inference for fold {curr_fold_idx}")

        if return_raw_attention:
            attention_save_dir = self.get_root_save_dir() / f"fold_{curr_fold_idx}" / "attention_scores"
            attention_save_dir.mkdir(parents=True, exist_ok=True)
        else:
            attention_save_dir = ''
        self.log.append(
            f"\n [FOLD:{curr_fold_idx}] Running inference on val split. Attention will be saved to {attention_save_dir if return_raw_attention else 'not saved'}"
        )

        # DataLoader
        test_loader = self._get_loader(
            split="test",
            curr_fold_idx=curr_fold_idx,
            batch_size=1,
            shuffle=False,
            sample=False,
            num_features=None,
        )
        self.log.append(f"\n [FOLD:{curr_fold_idx}] Created test DataLoader with batch size 1 and no sampling")

        # Initialize model and set to eval mode
        if model is None:
            if model_path is not None:
                model = self._define_model(weights_path=model_path).to(self.model_args["device"])
                self.log.append(f"\n [FOLD:{curr_fold_idx}] Loaded model weights from {model_path}")
            else:
                model = self._define_model().to(self.model_args["device"])
                warnings.warn(
                    "No model path or instance provided. Using randomly initialized weights.",
                    UserWarning,
                )
                self.log.append(
                    f"\n [FOLD:{curr_fold_idx}] No model path provided. Using randomly initialized weights."
                )

        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        all_y_true, all_y_pred, all_y_prob = [], [], []
        running_loss = 0.0
        slide_ids = []

        with torch.no_grad():
            for batch_idx, (features, labels, slide_id) in tqdm(
                enumerate(test_loader), total=len(test_loader), desc="Running eval..."
            ):
                features = features.to(self.model_args["device"])
                labels = labels.to(self.model_args["device"])
                slide_ids.append(slide_id[0])

                if return_raw_attention:
                    logits, attn = model(features, return_raw_attention=return_raw_attention)
                    attn_np = attn.squeeze(0).cpu().numpy()
                    self.save_attention_h5(slide_id[0], attn_np, save_dir=attention_save_dir)
                    self.log.append(
                        f"\n [FOLD:{curr_fold_idx}] Saved attention for slide {slide_id[0]} to {Path(attention_save_dir) / f'{slide_id[0]}.h5'} in dataset attention_scores"
                    )
                else:
                    logits = model(features)
                    self.log.append(
                        f"\n [FOLD:{curr_fold_idx}] Inference on slide {slide_id[0]} completed without saving attention scores."
                    )

                loss = criterion(logits, labels.long())
                running_loss += loss.item()

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()

                all_y_true.extend(labels_np)
                all_y_pred.extend(preds)
                all_y_prob.extend(probs)
        self.log.append(f"\n [FOLD:{curr_fold_idx}] Inference completed for fold {curr_fold_idx}")

        avg_loss = running_loss / len(test_loader)
        num_classes = self.model_args["num_classes"]
        metrics = self._compute_metrics(np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob), num_classes)
        asset_dict = {
            "y_true": np.array(all_y_true).tolist(),
            "y_pred": np.array(all_y_pred).tolist(),
            "y_prob": np.array(all_y_prob).tolist() if return_probs else None,
            "slide_ids": slide_ids,
            "loss": avg_loss,
            "accuracy": metrics["accuracy"],
            "bacc": metrics["bacc"],
            "auc": metrics["auc"],
            "f1": metrics["f1"],
            "n_samples": len(all_y_true),
            'attention_save_dir': str(attention_save_dir) if return_raw_attention else '',
        }
        return asset_dict
