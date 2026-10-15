"""Leakage-safe training for tree, MLP, and hybrid temporal models."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .calibration import ProbabilityCalibrator
from .feature_extraction import model_input_names
from .feature_selection import StableFeatureSelector
from .models import build_sklearn_model
from .utils import ensure_dirs, get_logger

logger = get_logger()


class Trainer:
    """Train using train-only preprocessing and a held-out calibration split."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_cfg = config["model"]
        self.train_cfg = config["training"]
        self.seed = int(config.get("seed", 42))
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        selection_cfg = config.get("features", {}).get("selection", {})
        self.selector = StableFeatureSelector(
            enabled=selection_cfg.get("enabled", True),
            variance_threshold=selection_cfg.get("variance_threshold", 1e-12),
            correlation_threshold=selection_cfg.get("correlation_threshold", 0.995),
            minimum_features=selection_cfg.get("minimum_features", 20),
        )
        self.scaler = _make_scaler(config.get("features", {}).get("normalization", "standard"))
        self.all_class_names = list(config.get("classes", {}).get("astro_names", []))
        self.class_names: list[str] = []
        self.active_class_ids: list[int] = []
        self.n_classes = 0
        self.calibrator: ProbabilityCalibrator | None = None
        self.input_dim: int | None = None
        self.training_history: list[dict[str, float]] = []
        self.planet_threshold = 0.5
        self.threshold_diagnostics: dict[str, float | str] = {}
        self.checkpoint_signature = ""
        ensure_dirs(config["paths"]["models_dir"])

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        flux: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ):
        """Split by host group when supplied, then train and return the test split."""
        X, y, flux = self._validate_arrays(X, y, flux)
        train_idx, val_idx, test_idx = self._split_indices(y, groups)
        return self._fit_splits(
            X[train_idx],
            y[train_idx],
            X[val_idx],
            y[val_idx],
            X[test_idx],
            y[test_idx],
            flux[train_idx] if flux is not None else None,
            flux[val_idx] if flux is not None else None,
            flux[test_idx] if flux is not None else None,
        )

    def train_pre_split(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        flux_train: np.ndarray | None = None,
        flux_val: np.ndarray | None = None,
        flux_test: np.ndarray | None = None,
    ):
        """Train from already host-grouped train/validation/test arrays."""
        X_train, y_train, flux_train = self._validate_arrays(X_train, y_train, flux_train)
        X_val, y_val, flux_val = self._validate_arrays(X_val, y_val, flux_val)
        X_test, y_test, flux_test = self._validate_arrays(X_test, y_test, flux_test)
        return self._fit_splits(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            flux_train,
            flux_val,
            flux_test,
        )

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted training imputer and scaler."""
        imputed = self.imputer.transform(np.asarray(X, dtype=float))
        return self.scaler.transform(self.selector.transform(imputed))

    def _impute(self, X: np.ndarray) -> np.ndarray:
        """Backward-compatible transform; fit only when no training fit exists."""
        values = np.asarray(X, dtype=float)
        if hasattr(self.imputer, "statistics_"):
            return self.imputer.transform(values)
        logger.warning("_impute fitted outside Trainer.train; use only for legacy tooling")
        return self.imputer.fit_transform(values)

    def _fit_splits(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        flux_train,
        flux_val,
        flux_test,
    ):
        if min(len(y_train), len(y_val), len(y_test)) == 0:
            raise ValueError("Train, validation, and test splits must all be non-empty")
        self._configure_active_classes(y_train, y_val, y_test)
        y_train_encoded = self._encode_labels(y_train)
        y_val_encoded = self._encode_labels(y_val)
        y_test_encoded = self._encode_labels(y_test)

        X_train_imputed = self.imputer.fit_transform(X_train)
        X_val_imputed = self.imputer.transform(X_val)
        X_test_imputed = self.imputer.transform(X_test)
        use_masks = bool(self.config.get("features", {}).get("use_masks", True))
        raw_names = model_input_names(use_masks)
        if len(raw_names) != X_train_imputed.shape[1]:
            raw_names = [
                f"feature_{index}" for index in range(X_train_imputed.shape[1])
            ]
        X_train_selected = self.selector.fit_transform(X_train_imputed, raw_names)
        X_val_selected = self.selector.transform(X_val_imputed)
        X_test_selected = self.selector.transform(X_test_imputed)
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_val_scaled = self.scaler.transform(X_val_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        self.input_dim = int(X_train_scaled.shape[1])
        self.checkpoint_signature = self._training_signature(X_train_scaled, y_train_encoded)

        started = time.perf_counter()
        model_type = self.model_cfg["type"]
        if model_type == "hybrid_tcn":
            if flux_train is None or flux_val is None or flux_test is None:
                raise ValueError("Hybrid TCN requires flux for every split")
            model = self._train_hybrid(
                X_train_scaled,
                y_train_encoded,
                X_val_scaled,
                y_val_encoded,
                flux_train,
                flux_val,
            )
        elif model_type == "neural_net":
            model = self._train_nn(
                X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded
            )
        else:
            model = self._train_sklearn(
                X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded
            )

        self.calibrator = self._fit_calibrator(
            model, X_val_scaled, y_val_encoded, flux_val
        )
        self.planet_threshold, self.threshold_diagnostics = self._select_planet_threshold(
            model, X_val_scaled, y_val_encoded, flux_val
        )
        self._save(model, time.perf_counter() - started, y_train_encoded)
        if flux_test is not None:
            return model, X_test_scaled, y_test_encoded, flux_test
        return model, X_test_scaled, y_test_encoded

    def _configure_active_classes(self, y_train, y_val, y_test) -> None:
        active = sorted(int(value) for value in np.unique(y_train))
        if len(active) < 2:
            raise ValueError("Training requires at least two classes in the training split")
        unseen = sorted(set(int(value) for value in np.concatenate((y_val, y_test))) - set(active))
        if unseen:
            raise ValueError(f"Validation/test contain classes absent from training: {unseen}")
        self.active_class_ids = active
        self.class_names = [
            self.all_class_names[class_id]
            if 0 <= class_id < len(self.all_class_names)
            else f"class_{class_id}"
            for class_id in active
        ]
        self.n_classes = len(active)

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        mapping = {class_id: index for index, class_id in enumerate(self.active_class_ids)}
        try:
            return np.asarray([mapping[int(value)] for value in y], dtype=int)
        except KeyError as exc:
            raise ValueError(f"Label {exc.args[0]} is not an active training class") from exc

    def _split_indices(
        self, y: np.ndarray, groups: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.arange(len(y))
        if groups is not None:
            import pandas as pd

            from .datasets.splits import DatasetSplitter

            groups = np.asarray(groups).astype(str)
            if len(groups) != len(y):
                raise ValueError("groups must have one value per sample")
            frame = pd.DataFrame(
                {"row_id": indices, "label": y, "group": groups}
            )
            parts = DatasetSplitter(
                self.config.get("dataset", {}), self.seed
            ).split(frame, group_col="group")
            split_indices = tuple(
                part["row_id"].to_numpy(dtype=int)
                for part in (parts["train"], parts["val"], parts["test"])
            )
            if all(len(part) for part in split_indices):
                return split_indices
            raise ValueError("Grouped split produced an empty partition")

        stratify = y if _can_stratify(y) else None
        idx_tmp, idx_test = train_test_split(
            indices,
            test_size=float(self.train_cfg.get("test_size", 0.15)),
            stratify=stratify,
            random_state=self.seed,
        )
        validation_relative = float(self.train_cfg.get("val_size", 0.15)) / (
            1.0 - float(self.train_cfg.get("test_size", 0.15))
        )
        tmp_stratify = y[idx_tmp] if stratify is not None and _can_stratify(y[idx_tmp]) else None
        idx_train, idx_val = train_test_split(
            idx_tmp,
            test_size=validation_relative,
            stratify=tmp_stratify,
            random_state=self.seed,
        )
        return idx_train, idx_val, idx_test

    def _fit_calibrator(
        self, model, X_val, y_val, flux_val=None
    ) -> ProbabilityCalibrator | None:
        cfg = self.config.get("calibration", {})
        if not cfg.get("enabled", True):
            return None
        try:
            from .evaluator import Evaluator

            probabilities = Evaluator(self._evaluation_config()).predict_proba(
                model, X_val, flux_val
            )
            calibrator = ProbabilityCalibrator(
                cfg.get("method", "temperature"),
                cfg.get("min_samples_per_class", 10),
            ).fit(probabilities, y_val)
            logger.info(
                "Fitted %s calibration on held-out validation data",
                calibrator.method,
            )
            return calibrator
        except (ValueError, RuntimeError) as exc:
            logger.warning("Calibration skipped: %s", exc)
            return None

    def _train_sklearn(self, X_train, y_train, X_val, y_val):
        cfg = dict(self.model_cfg, seed=self.seed)
        model = build_sklearn_model(cfg)
        fit_kwargs = {}
        if (
            self.model_cfg["type"] == "xgboost"
            and self.train_cfg.get("class_balancing") == "weighted_loss"
        ):
            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, **fit_kwargs)
        prediction = model.predict(X_val)
        logger.info(
            "%s validation balanced_accuracy=%.4f",
            self.model_cfg["type"],
            balanced_accuracy_score(y_val, prediction),
        )
        return model

    def _select_planet_threshold(self, model, X_val, y_val, flux_val):
        planet_indices = [
            index
            for index, name in enumerate(self.class_names)
            if name in ("planet", "planet_candidate")
        ]
        if not planet_indices:
            return 0.5, {"status": "no active planet class"}
        from .evaluator import Evaluator

        probabilities = Evaluator(self._evaluation_config()).predict_proba(
            model, X_val, flux_val
        )
        if self.calibrator is not None:
            probabilities = self.calibrator.calibrate(probabilities)
        score = np.sum(probabilities[:, planet_indices], axis=1)
        truth = np.isin(y_val, planet_indices).astype(int)
        counts = np.bincount(truth, minlength=2)
        minimum = int(self.train_cfg.get("minimum_threshold_samples_per_class", 10))
        if np.any(counts < minimum):
            return 0.5, {
                "status": "under-supported validation labels",
                "negative_samples": int(counts[0]),
                "positive_samples": int(counts[1]),
            }
        beta = float(self.train_cfg.get("threshold_fbeta", 2.0))
        target_recall = float(self.train_cfg.get("threshold_target_recall", 0.9))
        candidates = []
        for threshold in np.linspace(0.05, 0.95, 181):
            prediction = (score >= threshold).astype(int)
            recall = recall_score(truth, prediction, zero_division=0)
            precision = precision_score(truth, prediction, zero_division=0)
            fbeta = fbeta_score(
                truth, prediction, beta=beta, zero_division=0
            )
            candidates.append((recall >= target_recall, fbeta, precision, threshold, recall))
        eligible = [candidate for candidate in candidates if candidate[0]]
        selected = max(eligible or candidates, key=lambda item: (item[1], item[2], item[3]))
        return float(selected[3]), {
            "status": "optimized on validation set",
            "fbeta": float(selected[1]),
            "precision": float(selected[2]),
            "recall": float(selected[4]),
            "target_recall": target_recall,
        }

    def _train_nn(self, X_train, y_train, X_val, y_val):
        from .models import TransitNet

        cfg = self.model_cfg["neural_net"]
        model = TransitNet(
            X_train.shape[1],
            list(cfg["hidden_layers"]),
            float(cfg["dropout"]),
            self.n_classes,
        )
        return self._fit_torch_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            None,
            None,
            cfg,
            "nn_checkpoint.pt",
        )

    def _train_hybrid(
        self, X_train, y_train, X_val, y_val, flux_train, flux_val
    ):
        from .models import HybridTCNClassifier

        cfg = self.model_cfg["hybrid_tcn"]
        model = HybridTCNClassifier(
            X_train.shape[1], self.n_classes, self.model_cfg
        )
        return self._fit_torch_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            flux_train,
            flux_val,
            cfg,
            "hybrid_tcn_checkpoint.pt",
        )

    def _fit_torch_model(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        flux_train,
        flux_val,
        cfg,
        checkpoint_name,
    ):
        import torch
        from torch import optim
        from torch.utils.data import DataLoader, TensorDataset

        device = self._device()
        model = model.to(device)
        tensors = [torch.as_tensor(X_train, dtype=torch.float32)]
        if flux_train is not None:
            tensors.append(torch.as_tensor(flux_train, dtype=torch.float32))
        tensors.append(torch.as_tensor(y_train, dtype=torch.long))
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(*tensors),
            batch_size=int(cfg["batch_size"]),
            shuffle=True,
            generator=generator,
            num_workers=int(self.train_cfg.get("data_loader_workers", 0)),
            pin_memory=device.type == "cuda",
            persistent_workers=int(self.train_cfg.get("data_loader_workers", 0)) > 0,
        )
        X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32, device=device)
        y_val_tensor = torch.as_tensor(y_val, dtype=torch.long, device=device)
        flux_val_tensor = (
            torch.as_tensor(flux_val, dtype=torch.float32, device=device)
            if flux_val is not None
            else None
        )
        criterion = self._criterion(torch, y_train, device)
        optimizer = self._optimizer(optim, model.parameters(), cfg)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=float(cfg["lr_scheduler_factor"]),
            patience=int(cfg["lr_scheduler_patience"]),
        )
        start_epoch = self._restore_torch_checkpoint(
            torch, checkpoint_name, model, optimizer, scheduler
        )
        best_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        stale = 0
        use_amp = bool(
            self.train_cfg.get("mixed_precision", False) and device.type == "cuda"
        )
        grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        accumulation = max(int(self.train_cfg.get("gradient_accumulation", 1)), 1)

        for epoch in range(start_epoch, int(cfg["epochs"])):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            for step, batch in enumerate(loader, start=1):
                batch = [tensor.to(device, non_blocking=True) for tensor in batch]
                xb, yb = batch[0], batch[-1]
                fb = batch[1] if flux_train is not None else None
                with torch.amp.autocast(
                    device_type=device.type, enabled=use_amp
                ):
                    logits = model(xb, fb) if fb is not None else model(xb)
                    loss = criterion(logits, yb) / accumulation
                grad_scaler.scale(loss).backward()
                running_loss += float(loss.detach().cpu()) * accumulation
                if step % accumulation == 0 or step == len(loader):
                    self._clip_gradients(
                        torch, model, grad_scaler, optimizer, use_amp
                    )
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            model.eval()
            with torch.no_grad(), torch.amp.autocast(
                device_type=device.type, enabled=use_amp
            ):
                logits = (
                    model(X_val_tensor, flux_val_tensor)
                    if flux_val_tensor is not None
                    else model(X_val_tensor)
                )
                validation_loss = float(criterion(logits, y_val_tensor).cpu())
            scheduler.step(validation_loss)
            self.training_history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": running_loss / max(len(loader), 1),
                    "val_loss": validation_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if validation_loss < best_loss - float(cfg.get("min_delta", 1e-5)):
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            self._save_torch_checkpoint(
                torch,
                checkpoint_name,
                model,
                optimizer,
                scheduler,
                epoch,
                validation_loss,
            )
            if epoch % 10 == 0:
                logger.info("epoch %d val_loss=%.6f", epoch, validation_loss)
            if stale >= int(cfg["early_stopping_patience"]):
                logger.info("Early stopping at epoch %d", epoch)
                break

        model.load_state_dict(best_state)
        model.eval()
        return model

    def _criterion(self, torch, y_train: np.ndarray, device):
        weights = self._class_weights(y_train, device)
        label_smoothing = float(self.train_cfg.get("label_smoothing", 0.0))
        loss_name = str(self.train_cfg.get("loss", "cross_entropy")).lower()
        if loss_name == "focal":
            gamma = float(self.train_cfg.get("focal_gamma", 2.0))

            def focal_loss(logits, targets):
                ce = torch.nn.functional.cross_entropy(
                    logits,
                    targets,
                    reduction="none",
                    label_smoothing=label_smoothing,
                )
                probability = torch.softmax(logits, dim=-1).gather(
                    1, targets[:, None]
                ).squeeze(1)
                loss = (1.0 - probability) ** gamma * ce
                if weights is not None:
                    loss = loss * weights[targets]
                return torch.mean(loss)

            return focal_loss
        return torch.nn.CrossEntropyLoss(
            weight=weights, label_smoothing=label_smoothing
        )

    def _device(self):
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _class_weights(self, y: np.ndarray, device):
        if self.train_cfg.get("class_balancing") != "weighted_loss":
            return None
        import torch

        counts = np.bincount(np.asarray(y, dtype=int), minlength=self.n_classes)
        weights = len(y) / (self.n_classes * np.maximum(counts, 1))
        weights /= max(float(np.mean(weights)), 1e-12)
        weights = np.clip(
            weights,
            1.0 / float(self.train_cfg.get("max_class_weight", 10.0)),
            float(self.train_cfg.get("max_class_weight", 10.0)),
        )
        return torch.as_tensor(weights, dtype=torch.float32, device=device)

    def _optimizer(self, optim, parameters, cfg: dict):
        name = str(self.train_cfg.get("optimizer", "adamw")).lower()
        learning_rate = float(cfg["learning_rate"])
        weight_decay = float(cfg.get("weight_decay", 0.0))
        if name == "adam":
            return optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
        if name == "sgd":
            return optim.SGD(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9,
            )
        return optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )

    def _clip_gradients(
        self, torch, model, grad_scaler, optimizer, use_amp: bool
    ) -> None:
        maximum = float(self.train_cfg.get("gradient_clip_norm", 0.0) or 0.0)
        if maximum <= 0:
            return
        if use_amp:
            grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), maximum)

    def _checkpoint_path(self, checkpoint_name: str) -> Path:
        return Path(self.config["paths"]["models_dir"]) / checkpoint_name

    def _restore_torch_checkpoint(
        self, torch, checkpoint_name, model, optimizer, scheduler
    ) -> int:
        if not self.train_cfg.get("checkpoint_resume", True):
            return 0
        path = self._checkpoint_path(checkpoint_name)
        if not path.exists():
            return 0
        try:
            checkpoint = torch.load(path, map_location=self._device(), weights_only=False)
            if checkpoint.get("signature") != self.checkpoint_signature:
                logger.warning("Ignoring incompatible checkpoint %s", path)
                return 0
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            epoch = int(checkpoint.get("epoch", -1)) + 1
            logger.info("Resumed compatible checkpoint %s at epoch %d", path, epoch)
            return epoch
        except (OSError, KeyError, RuntimeError, ValueError) as exc:
            logger.warning("Could not resume checkpoint %s: %s", path, exc)
            return 0

    def _save_torch_checkpoint(
        self,
        torch,
        checkpoint_name,
        model,
        optimizer,
        scheduler,
        epoch,
        validation_loss,
    ) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": int(epoch),
                "val_loss": float(validation_loss),
                "signature": self.checkpoint_signature,
                "training_config": self.train_cfg,
                "model_config": self.model_cfg,
                "git_commit": _git_commit(),
            },
            self._checkpoint_path(checkpoint_name),
        )

    def _save(self, model, runtime_seconds: float, y_train: np.ndarray) -> None:
        directory = Path(self.config["paths"]["models_dir"])
        joblib.dump(self.imputer, directory / "imputer.joblib")
        joblib.dump(self.selector, directory / "selector.joblib")
        joblib.dump(self.scaler, directory / "scaler.joblib")
        model_type = self.model_cfg["type"]
        if model_type in ("neural_net", "hybrid_tcn"):
            import torch

            filename = "hybrid_tcn.pt" if model_type == "hybrid_tcn" else "nn_model.pt"
            torch.save(model.state_dict(), directory / filename)
        else:
            joblib.dump(model, directory / f"{model_type}.joblib")
        calibrator_path = directory / "calibrator.joblib"
        if self.calibrator is not None:
            joblib.dump(self.calibrator, calibrator_path)
        elif calibrator_path.exists():
            calibrator_path.unlink()

        use_masks = bool(self.config.get("features", {}).get("use_masks", True))
        metadata = {
            "artifact_version": 2,
            "feature_version": self.config.get("dataset", {}).get("version"),
            "n_classes": self.n_classes,
            "type": model_type,
            "input_dim": self.input_dim,
            "use_masks": use_masks,
            "raw_input_names": list(self.selector.input_names_),
            "input_names": list(self.selector.selected_names_),
            "class_names": self.class_names,
            "active_global_class_ids": self.active_class_ids,
            "calibration": self.calibrator.method if self.calibrator else None,
            "checkpoint_signature": self.checkpoint_signature,
            "model_config": copy.deepcopy(self.model_cfg),
            "planet_threshold": self.planet_threshold,
            "threshold_diagnostics": self.threshold_diagnostics,
        }
        joblib.dump(metadata, directory / "model_meta.joblib")
        run = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
            "dataset_version": self.config.get("dataset", {}).get("version"),
            "class_distribution_train": np.bincount(
                y_train, minlength=self.n_classes
            ).tolist(),
            "runtime_seconds": float(runtime_seconds),
            "training_history": self.training_history,
            "git_commit": _git_commit(),
            "training_config": self.train_cfg,
            "model_config": self.model_cfg,
        }
        (directory / "training_run.json").write_text(
            json.dumps(run, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Saved model, imputer, scaler, and metadata to %s", directory)

    def _evaluation_config(self) -> dict:
        config = copy.deepcopy(self.config)
        config.setdefault("classes", {})["astro_names"] = list(self.class_names)
        return config

    def _training_signature(self, X_train: np.ndarray, y_train: np.ndarray) -> str:
        hasher = hashlib.sha256()
        hasher.update(
            json.dumps(
                {
                    "model": self.model_cfg,
                    "training": self.train_cfg,
                    "classes": self.active_class_ids,
                    "shape": X_train.shape,
                    "feature_version": self.config.get("dataset", {}).get("version"),
                },
                sort_keys=True,
                default=str,
            ).encode()
        )
        hasher.update(np.ascontiguousarray(y_train).tobytes())
        sample = np.ascontiguousarray(X_train[: min(len(X_train), 256)], dtype=np.float32)
        hasher.update(sample.tobytes())
        return hasher.hexdigest()[:20]

    @staticmethod
    def _validate_arrays(X, y, flux):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2-D and y must be an aligned 1-D array")
        if len(y) < 1:
            raise ValueError("Training split is empty")
        if np.any(~np.isfinite(y)) or np.any(y < 0):
            raise ValueError("Labels must be finite non-negative integers")
        if flux is not None:
            flux = np.asarray(flux, dtype=np.float32)
            if flux.ndim != 2 or len(flux) != len(y):
                raise ValueError("flux must be a 2-D array aligned with y")
            if np.any(~np.isfinite(flux)):
                raise ValueError("flux input contains NaN or infinity")
        return X, y, flux


def _make_scaler(name: str):
    normalized = str(name).lower()
    if normalized in ("standard", "zscore", "z-score"):
        return StandardScaler()
    if normalized == "robust":
        return RobustScaler(quantile_range=(10.0, 90.0))
    if normalized in ("minmax", "min-max"):
        return MinMaxScaler()
    if normalized in ("none", "identity"):
        from sklearn.preprocessing import FunctionTransformer

        return FunctionTransformer(validate=True)
    raise ValueError(f"Unsupported feature normalization: {name}")


def _can_stratify(y: np.ndarray) -> bool:
    _, counts = np.unique(y, return_counts=True)
    return len(counts) > 1 and bool(np.all(counts >= 2))


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
