from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, Union, Any

import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from sklearn.utils.validation import check_is_fitted
import shap
import scipy.stats


ArrayLike = Union[np.ndarray, Sequence[float]]
Dataset = Tuple[ArrayLike, ArrayLike]


@dataclass
class ScoreHistogramPlotResult:
    fitted_model: object
    fig: plt.Figure
    axes: np.ndarray
    train_scores: np.ndarray
    val_scores: np.ndarray
    best_threshold: float
    best_metric_value: float
    val_thresholds: np.ndarray
    val_metric_values: np.ndarray


@dataclass
class ScoreFeatureContourPlotResult:
    fitted_model: object
    fig: plt.Figure
    train_scores: np.ndarray
    val_scores: np.ndarray
    train_feature_values: np.ndarray
    val_feature_values: np.ndarray


@dataclass
class ShapBeeswarmResult:
    explainer: object
    explanation: object
    fig: plt.Figure


class BinaryClassifierInterpreter:
    def __init__(
        self,
        train: Dataset,
        val: Dataset,
        model,
        *,
        enable_early_stopping: bool = False,
        use_best_model: bool = True,
        early_stopping_rounds: int = 100,
    ) -> None:
        self.X_train, self.y_train = train
        self.X_val, self.y_val = val

        self.y_train = self._validate_binary_target(self.y_train, "y_train")
        self.y_val = self._validate_binary_target(self.y_val, "y_val")

        self.model = self._fit_model_on_train(
            model,
            self.X_train,
            self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            enable_early_stopping=enable_early_stopping,
            use_best_model=use_best_model,
            early_stopping_rounds=early_stopping_rounds,
        )

        self.train_scores = self._extract_raw_scores(self.model, self.X_train)
        self.val_scores = self._extract_raw_scores(self.model, self.X_val)

    # ============================================================
    # Generic helpers
    # ============================================================
    @staticmethod
    def _build_metric_registry() -> Dict[str, Callable]:
        return {
            "accuracy": accuracy_score,
            "balanced_accuracy": balanced_accuracy_score,
            "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
            "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
            "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef,
        }

    @staticmethod
    def _to_numpy_1d(x: ArrayLike, name: str) -> np.ndarray:
        arr = np.asarray(x)
        if arr.ndim != 1:
            arr = np.ravel(arr)
        if arr.ndim != 1:
            raise ValueError(
                f"{name} must be 1D after ravel, got shape={arr.shape}")
        return arr

    @classmethod
    def _validate_binary_target(cls, y: ArrayLike, name: str) -> np.ndarray:
        y = cls._to_numpy_1d(y, name).astype(int)
        uniq = np.unique(y)
        if not np.array_equal(uniq, np.array([0, 1])):
            raise ValueError(
                f"{name} must contain exactly binary labels {{0,1}}, got {uniq}")
        return y

    @staticmethod
    def _safe_clone_model(model):
        try:
            return clone(model)
        except Exception:
            return model

    @staticmethod
    def _is_model_fitted(model) -> bool:
        class_name = model.__class__.__name__.lower()
        module_name = model.__class__.__module__.lower()

        if "catboost" in module_name or "catboost" in class_name:
            if hasattr(model, "is_fitted"):
                try:
                    return bool(model.is_fitted())
                except Exception:
                    pass
            try:
                _ = model.tree_count_
                return True
            except Exception:
                return False

        if "xgboost" in module_name or "xgb" in class_name:
            if hasattr(model, "get_booster"):
                try:
                    booster = model.get_booster()
                    return booster is not None
                except Exception:
                    return False

        if "lightgbm" in module_name or "lgbm" in class_name:
            if hasattr(model, "booster_"):
                try:
                    return model.booster_ is not None
                except Exception:
                    return False
            return False

        try:
            check_is_fitted(model)
            return True
        except Exception:
            return False

    @staticmethod
    def _supports_fit_kwarg(model, kwarg_name: str) -> bool:
        try:
            sig = inspect.signature(model.fit)
            return kwarg_name in sig.parameters
        except Exception:
            return False

    @staticmethod
    def _set_model_param_if_supported(model, param_name: str, value):
        if not hasattr(model, "get_params") or not hasattr(model, "set_params"):
            return model
        try:
            params = model.get_params(deep=False)
            if param_name in params:
                model.set_params(**{param_name: value})
        except Exception:
            pass
        return model

    @classmethod
    def _fit_model_on_train(
        cls,
        model,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        *,
        enable_early_stopping: bool = False,
        use_best_model: bool = True,
        early_stopping_rounds: int = 100,
    ):
        if cls._is_model_fitted(model):
            return model

        fitted_model = cls._safe_clone_model(model)
        class_name = fitted_model.__class__.__name__.lower()
        module_name = fitted_model.__class__.__module__.lower()

        if not enable_early_stopping:
            fitted_model.fit(X_train, y_train)
            return fitted_model

        if X_val is None or y_val is None:
            raise ValueError(
                "Validation data must be provided when enable_early_stopping=True.")

        if "catboost" in module_name or "catboost" in class_name:
            fitted_model = cls._set_model_param_if_supported(
                fitted_model, "use_best_model", use_best_model)
            fit_kwargs = {}
            if cls._supports_fit_kwarg(fitted_model, "eval_set"):
                fit_kwargs["eval_set"] = (X_val, y_val)
            if cls._supports_fit_kwarg(fitted_model, "early_stopping_rounds"):
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            elif cls._supports_fit_kwarg(fitted_model, "od_wait"):
                fit_kwargs["od_wait"] = early_stopping_rounds
            fitted_model.fit(X_train, y_train, **fit_kwargs)
            return fitted_model

        if "xgboost" in module_name or "xgb" in class_name:
            fit_kwargs = {}
            if cls._supports_fit_kwarg(fitted_model, "eval_set"):
                fit_kwargs["eval_set"] = [(X_val, y_val)]
            if cls._supports_fit_kwarg(fitted_model, "early_stopping_rounds"):
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            if cls._supports_fit_kwarg(fitted_model, "verbose"):
                fit_kwargs["verbose"] = False
            fitted_model.fit(X_train, y_train, **fit_kwargs)
            return fitted_model

        if "lightgbm" in module_name or "lgbm" in class_name:
            fit_kwargs = {}
            if cls._supports_fit_kwarg(fitted_model, "eval_set"):
                fit_kwargs["eval_set"] = [(X_val, y_val)]
            callbacks = []
            try:
                import lightgbm as lgb  # type: ignore
                callbacks.append(lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds, verbose=False))
            except Exception:
                pass
            if callbacks and cls._supports_fit_kwarg(fitted_model, "callbacks"):
                fit_kwargs["callbacks"] = callbacks
            elif cls._supports_fit_kwarg(fitted_model, "early_stopping_rounds"):
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            fitted_model.fit(X_train, y_train, **fit_kwargs)
            return fitted_model

        fitted_model.fit(X_train, y_train)
        return fitted_model

    @staticmethod
    def _extract_raw_scores(model, X) -> np.ndarray:
        class_name = model.__class__.__name__.lower()
        module_name = model.__class__.__module__.lower()

        if "catboost" in module_name or "catboost" in class_name:
            scores = model.predict(X, prediction_type="RawFormulaVal")
            return np.ravel(np.asarray(scores)).astype(float)

        if "xgboost" in module_name or "xgb" in class_name:
            scores = model.predict(X, output_margin=True)
            return np.ravel(np.asarray(scores)).astype(float)

        if "lightgbm" in module_name or "lgbm" in class_name:
            scores = np.asarray(model.predict(X, raw_score=True))
            if scores.ndim == 2:
                if scores.shape[1] != 1:
                    raise ValueError(
                        f"Only binary classification is supported, got shape={scores.shape}")
                scores = scores[:, 0]
            return scores.astype(float)

        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(X))
            if scores.ndim == 2:
                if scores.shape[1] != 1:
                    raise ValueError(
                        f"Only binary classification is supported, got shape={scores.shape}")
                scores = scores[:, 0]
            return scores.astype(float)

        raise TypeError("Unsupported model type for raw score extraction.")

    # ============================================================
    # Score histogram with validation metric
    # ============================================================
    @classmethod
    def _quantile_clip_mask(cls, scores: np.ndarray, lower_q: float = 0.002, upper_q: float = 0.998) -> np.ndarray:
        scores = cls._to_numpy_1d(scores, "scores")
        finite_mask = np.isfinite(scores)
        s = scores[finite_mask]
        if len(s) < 10:
            return finite_mask
        lo, hi = np.quantile(s, [lower_q, upper_q])
        return finite_mask & (scores >= lo) & (scores <= hi)

    @classmethod
    def _adaptive_bins_with_min_width(
        cls,
        scores: np.ndarray,
        min_bins: int = 20,
        max_bins: int = 80,
        min_bin_width: float = 0.08,
    ) -> Tuple[int, np.ndarray]:
        scores = cls._to_numpy_1d(scores, "scores")
        scores = scores[np.isfinite(scores)]

        if len(scores) == 0:
            return min_bins, np.linspace(-1.0, 1.0, min_bins + 1)

        lo = scores.min()
        hi = scores.max()
        data_range = hi - lo
        if data_range <= 0:
            eps = 1e-6 if lo == 0 else abs(lo) * 1e-6
            return min_bins, np.linspace(lo - eps, hi + eps, min_bins + 1)

        q1, q3 = np.percentile(scores, [25, 75])
        iqr = q3 - q1
        n = len(scores)
        fd_width = 2.0 * iqr / \
            np.cbrt(n) if iqr > 0 else data_range / np.sqrt(n)
        bin_width = max(fd_width, min_bin_width)
        n_bins = int(np.ceil(data_range / bin_width))
        n_bins = int(np.clip(n_bins, min_bins, max_bins))
        edges = np.linspace(lo, hi, n_bins + 1)
        return n_bins, edges

    @classmethod
    def _build_threshold_grid(cls, scores: np.ndarray, max_thresholds: int = 400) -> np.ndarray:
        scores = cls._to_numpy_1d(scores, "scores")
        scores = scores[np.isfinite(scores)]
        uniq = np.unique(scores)
        if len(uniq) <= max_thresholds:
            return uniq.astype(float)
        q = np.linspace(0.0, 1.0, max_thresholds)
        return np.unique(np.quantile(scores, q)).astype(float)

    @classmethod
    def _compute_threshold_metric_curve(
        cls,
        y_true: np.ndarray,
        scores: np.ndarray,
        metric_name: str,
        *,
        positive_if_score_ge_threshold: bool = True,
        max_thresholds: int = 400,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if metric_name == "roc_auc":
            raise ValueError(
                "roc_auc is threshold-independent. It should not be plotted as a function of threshold. "
                "Use one of: accuracy, balanced_accuracy, f1, precision, recall, mcc."
            )
        registry = cls._build_metric_registry()
        if metric_name not in registry:
            raise ValueError(
                f"Unsupported threshold metric: {metric_name}. Available: {sorted(registry)}")

        metric_fn = registry[metric_name]
        thresholds = cls._build_threshold_grid(
            scores, max_thresholds=max_thresholds)
        values = np.empty_like(thresholds, dtype=float)

        for i, thr in enumerate(thresholds):
            y_pred = (scores >= thr).astype(
                int) if positive_if_score_ge_threshold else (scores <= thr).astype(int)
            values[i] = metric_fn(y_true, y_pred)
        return thresholds, values

    def plot_score_histograms_with_val_metric(
        self,
        val_metric_name: str = "f1",
        *,
        positive_if_score_ge_threshold: bool = True,
        alpha: float = 0.45,
        figsize: Tuple[int, int] = (15, 6),
        title_prefix: Optional[str] = None,
        class_0_label: str = "class 0",
        class_1_label: str = "class 1",
        min_bins: int = 20,
        max_bins: int = 80,
        min_bin_width: float = 0.08,
        clip_quantiles_for_hist: Tuple[float, float] = (0.002, 0.998),
        max_thresholds: int = 400,
        show_means: bool = True,
        use_log_density_y: bool = False,
    ) -> ScoreHistogramPlotResult:
        train_mask = self._quantile_clip_mask(
            self.train_scores,
            lower_q=clip_quantiles_for_hist[0],
            upper_q=clip_quantiles_for_hist[1],
        )
        val_mask = self._quantile_clip_mask(
            self.val_scores,
            lower_q=clip_quantiles_for_hist[0],
            upper_q=clip_quantiles_for_hist[1],
        )

        train_scores_hist = self.train_scores[train_mask]
        y_train_hist = self.y_train[train_mask]
        val_scores_hist = self.val_scores[val_mask]
        y_val_hist = self.y_val[val_mask]

        _, train_edges = self._adaptive_bins_with_min_width(
            train_scores_hist,
            min_bins=min_bins,
            max_bins=max_bins,
            min_bin_width=min_bin_width,
        )
        _, val_edges = self._adaptive_bins_with_min_width(
            val_scores_hist,
            min_bins=min_bins,
            max_bins=max_bins,
            min_bin_width=min_bin_width,
        )

        val_thresholds, val_metric_values = self._compute_threshold_metric_curve(
            y_true=self.y_val,
            scores=self.val_scores,
            metric_name=val_metric_name,
            positive_if_score_ge_threshold=positive_if_score_ge_threshold,
            max_thresholds=max_thresholds,
        )

        best_idx = int(np.nanargmax(val_metric_values))
        best_threshold = float(val_thresholds[best_idx])
        best_metric_value = float(val_metric_values[best_idx])

        fig, axes = plt.subplots(
            1, 2, figsize=figsize, constrained_layout=True)

        ax_train = axes[0]
        s0_train = train_scores_hist[y_train_hist == 0]
        s1_train = train_scores_hist[y_train_hist == 1]
        ax_train.hist(s0_train, bins=train_edges, density=True,
                      alpha=alpha, label=f"{class_0_label} (n={len(s0_train)})")
        ax_train.hist(s1_train, bins=train_edges, density=True,
                      alpha=alpha, label=f"{class_1_label} (n={len(s1_train)})")
        if show_means and len(s0_train):
            ax_train.axvline(np.mean(s0_train), linestyle="--", linewidth=1.3)
        if show_means and len(s1_train):
            ax_train.axvline(np.mean(s1_train), linestyle="--", linewidth=1.3)
        if use_log_density_y:
            ax_train.set_yscale("log")
        ax_train.set_title(
            "Train score distribution" if title_prefix is None else f"{title_prefix} — Train score distribution")
        ax_train.set_xlabel("Raw model score")
        ax_train.set_ylabel("Density")
        ax_train.grid(True, alpha=0.25)
        ax_train.legend()

        ax_val = axes[1]
        s0_val = val_scores_hist[y_val_hist == 0]
        s1_val = val_scores_hist[y_val_hist == 1]
        h0 = ax_val.hist(s0_val, bins=val_edges, density=True,
                         alpha=alpha, label=f"{class_0_label} (n={len(s0_val)})")
        h1 = ax_val.hist(s1_val, bins=val_edges, density=True,
                         alpha=alpha, label=f"{class_1_label} (n={len(s1_val)})")
        if show_means and len(s0_val):
            ax_val.axvline(np.mean(s0_val), linestyle="--", linewidth=1.3)
        if show_means and len(s1_val):
            ax_val.axvline(np.mean(s1_val), linestyle="--", linewidth=1.3)
        if use_log_density_y:
            ax_val.set_yscale("log")
        ax_val.set_title(
            "Validation score distribution" if title_prefix is None else f"{title_prefix} — Validation score distribution")
        ax_val.set_xlabel("Raw model score")
        ax_val.set_ylabel("Density")
        ax_val.grid(True, alpha=0.25)

        ax_metric = ax_val.twinx()
        metric_line = ax_metric.plot(
            val_thresholds, val_metric_values, linestyle="--", linewidth=2.0, label=val_metric_name)[0]
        thr_line = ax_metric.axvline(
            best_threshold, linestyle=":", linewidth=1.5, label=f"best thr = {best_threshold:.4f}")
        ax_metric.set_ylabel(val_metric_name)
        ax_val.legend(
            [h0[2][0], h1[2][0], metric_line, thr_line],
            [f"{class_0_label} (n={len(s0_val)})", f"{class_1_label} (n={len(s1_val)})",
             val_metric_name, f"best thr = {best_threshold:.4f}"],
            loc="best",
        )

        return ScoreHistogramPlotResult(
            fitted_model=self.model,
            fig=fig,
            axes=axes,
            train_scores=self.train_scores,
            val_scores=self.val_scores,
            best_threshold=best_threshold,
            best_metric_value=best_metric_value,
            val_thresholds=val_thresholds,
            val_metric_values=val_metric_values,
        )

    # ============================================================
    # Score-feature contours
    # ============================================================
    @classmethod
    def _extract_feature_column(cls, X, feature_name: Union[str, int]) -> np.ndarray:
        if hasattr(X, "columns"):
            if isinstance(feature_name, str):
                if feature_name not in X.columns:
                    raise KeyError(
                        f"Feature '{feature_name}' not found in X.columns")
                values = X[feature_name].to_numpy()
                return cls._to_numpy_1d(values, f"feature '{feature_name}'").astype(float)
            if isinstance(feature_name, int):
                values = X.iloc[:, feature_name].to_numpy()
                return cls._to_numpy_1d(values, f"feature index {feature_name}").astype(float)
            raise TypeError(
                "For DataFrame input feature_name must be str or int")

        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X_arr.shape}")
        if not isinstance(feature_name, int):
            raise TypeError(
                "When X is not a DataFrame, feature_name must be an integer column index")
        return cls._to_numpy_1d(X_arr[:, feature_name], f"feature index {feature_name}").astype(float)

    @classmethod
    def _joint_visual_clip_mask(
        cls,
        scores: np.ndarray,
        feature_values: np.ndarray,
        *,
        score_q: Tuple[float, float] = (0.002, 0.998),
        feature_q: Tuple[float, float] = (0.002, 0.998),
    ) -> np.ndarray:
        scores = cls._to_numpy_1d(scores, "scores")
        feature_values = cls._to_numpy_1d(feature_values, "feature_values")
        finite_mask = np.isfinite(scores) & np.isfinite(feature_values)
        s = scores[finite_mask]
        f = feature_values[finite_mask]
        if len(s) < 10:
            return finite_mask
        s_lo, s_hi = np.quantile(s, score_q)
        f_lo, f_hi = np.quantile(f, feature_q)
        return finite_mask & (scores >= s_lo) & (scores <= s_hi) & (feature_values >= f_lo) & (feature_values <= f_hi)

    @classmethod
    def _adaptive_2d_bin_counts(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        *,
        min_bins: int = 25,
        max_bins: int = 80,
    ) -> Tuple[int, int]:
        x = cls._to_numpy_1d(x, "x")
        y = cls._to_numpy_1d(y, "y")
        n = max(len(x), 1)
        base = int(np.sqrt(n) / 3.0)
        base = int(np.clip(base, min_bins, max_bins))
        x_unique = len(np.unique(np.round(x, 8)))
        y_unique = len(np.unique(np.round(y, 8)))
        x_bins = int(
            np.clip(min(base, max(10, x_unique // 3)), min_bins, max_bins))
        y_bins = int(
            np.clip(min(base, max(10, y_unique // 3)), min_bins, max_bins))
        return x_bins, y_bins

    @staticmethod
    def _compute_hist2d_density(
        x: np.ndarray,
        y: np.ndarray,
        *,
        bins: Tuple[int, int],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        H, x_edges, y_edges = np.histogram2d(
            x, y, bins=bins, range=[x_range, y_range], density=True)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        return H.T, x_centers, y_centers

    @staticmethod
    def _positive_contour_levels(
        H: np.ndarray,
        *,
        n_levels: int = 8,
        low_q: float = 0.55,
        high_q: float = 0.98,
    ) -> Optional[np.ndarray]:
        positive = H[H > 0]
        if positive.size < 2:
            return None
        lo = np.quantile(positive, low_q)
        hi = np.quantile(positive, high_q)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= 0:
            return None
        if hi <= lo:
            return np.array([lo], dtype=float)
        return np.linspace(lo, hi, n_levels)

    @classmethod
    def _draw_joint_hist_contours(
        cls,
        fig: plt.Figure,
        subspec,
        scores: np.ndarray,
        feature_values: np.ndarray,
        y_true: np.ndarray,
        *,
        title: str,
        feature_label: str,
        class_0_label: str = "False",
        class_1_label: str = "True",
        class_0_color: str = "tab:blue",
        class_1_color: str = "tab:orange",
        alpha_marginal: float = 0.45,
        contour_linewidth: float = 1.6,
        marginal_bins: int = 40,
        hist2d_min_bins: int = 25,
        hist2d_max_bins: int = 80,
        score_q: Tuple[float, float] = (0.002, 0.998),
        feature_q: Tuple[float, float] = (0.002, 0.998),
    ):
        inner = GridSpecFromSubplotSpec(
            2,
            2,
            subplot_spec=subspec,
            width_ratios=[4.5, 1.0],
            height_ratios=[1.0, 4.5],
            wspace=0.0,
            hspace=0.0,
        )
        ax_top = fig.add_subplot(inner[0, 0])
        ax_main = fig.add_subplot(inner[1, 0], sharex=ax_top)
        ax_right = fig.add_subplot(inner[1, 1], sharey=ax_main)

        mask = cls._joint_visual_clip_mask(
            scores, feature_values, score_q=score_q, feature_q=feature_q)
        s = scores[mask]
        f = feature_values[mask]
        y = y_true[mask]

        s0 = s[y == 0]
        s1 = s[y == 1]
        f0 = f[y == 0]
        f1 = f[y == 1]

        if len(s) == 0:
            raise ValueError(
                f"No points left after clipping for panel '{title}'")

        x_range = (float(np.min(s)), float(np.max(s)))
        y_range = (float(np.min(f)), float(np.max(f)))
        if x_range[0] == x_range[1]:
            eps = 1e-6 if x_range[0] == 0 else abs(x_range[0]) * 1e-6
            x_range = (x_range[0] - eps, x_range[1] + eps)
        if y_range[0] == y_range[1]:
            eps = 1e-6 if y_range[0] == 0 else abs(y_range[0]) * 1e-6
            y_range = (y_range[0] - eps, y_range[1] + eps)

        bins = cls._adaptive_2d_bin_counts(
            s, f, min_bins=hist2d_min_bins, max_bins=hist2d_max_bins)
        score_edges = np.linspace(x_range[0], x_range[1], marginal_bins + 1)
        feature_edges = np.linspace(y_range[0], y_range[1], marginal_bins + 1)

        ax_top.hist(s0, bins=score_edges, density=True,
                    alpha=alpha_marginal, label=class_0_label, color=class_0_color)
        ax_top.hist(s1, bins=score_edges, density=True,
                    alpha=alpha_marginal, label=class_1_label, color=class_1_color)
        ax_top.legend(loc="upper left")
        ax_top.set_title(title)
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.grid(True, alpha=0.2)

        ax_right.hist(f0, bins=feature_edges, density=True,
                      alpha=alpha_marginal, orientation="horizontal", color=class_0_color)
        ax_right.hist(f1, bins=feature_edges, density=True,
                      alpha=alpha_marginal, orientation="horizontal", color=class_1_color)
        ax_right.tick_params(axis="y", labelleft=False)
        ax_right.grid(True, alpha=0.2)

        rng = np.random.default_rng(42)
        max_scatter = 5000
        if len(s0) > max_scatter:
            idx0 = rng.choice(len(s0), size=max_scatter, replace=False)
            s0_sc, f0_sc = s0[idx0], f0[idx0]
        else:
            s0_sc, f0_sc = s0, f0
        if len(s1) > max_scatter:
            idx1 = rng.choice(len(s1), size=max_scatter, replace=False)
            s1_sc, f1_sc = s1[idx1], f1[idx1]
        else:
            s1_sc, f1_sc = s1, f1

        ax_main.scatter(s0_sc, f0_sc, s=4, alpha=0.25, color=class_0_color)
        ax_main.scatter(s1_sc, f1_sc, s=4, alpha=0.25, color=class_1_color)

        if len(s0) >= 10:
            H0, x_centers0, y_centers0 = cls._compute_hist2d_density(
                s0, f0, bins=bins, x_range=x_range, y_range=y_range)
            levels0 = cls._positive_contour_levels(H0)
            if levels0 is not None:
                X0, Y0 = np.meshgrid(x_centers0, y_centers0)
                ax_main.contour(X0, Y0, H0, levels=levels0,
                                linewidths=contour_linewidth, colors=[class_0_color])

        if len(s1) >= 10:
            H1, x_centers1, y_centers1 = cls._compute_hist2d_density(
                s1, f1, bins=bins, x_range=x_range, y_range=y_range)
            levels1 = cls._positive_contour_levels(H1)
            if levels1 is not None:
                X1, Y1 = np.meshgrid(x_centers1, y_centers1)
                ax_main.contour(X1, Y1, H1, levels=levels1,
                                linewidths=contour_linewidth, colors=[class_1_color])

        ax_main.set_xlabel("model_score")
        ax_main.set_ylabel(feature_label)
        ax_main.grid(True, alpha=0.2)
        return ax_top, ax_main, ax_right

    def plot_score_feature_contours(
        self,
        feature_name: Union[str, int],
        *,
        figsize: Tuple[int, int] = (16, 7),
        title_prefix: Optional[str] = None,
        class_0_label: str = "False",
        class_1_label: str = "True",
        class_0_color: str = "tab:blue",
        class_1_color: str = "tab:orange",
        alpha_marginal: float = 0.45,
        contour_linewidth: float = 1.6,
        marginal_bins: int = 40,
        hist2d_min_bins: int = 25,
        hist2d_max_bins: int = 80,
        score_q: Tuple[float, float] = (0.002, 0.998),
        feature_q: Tuple[float, float] = (0.002, 0.998),
    ) -> ScoreFeatureContourPlotResult:
        train_feature_values = self._extract_feature_column(
            self.X_train, feature_name)
        val_feature_values = self._extract_feature_column(
            self.X_val, feature_name)
        feature_label = str(feature_name)

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        outer = GridSpec(1, 2, figure=fig, wspace=0.12)
        train_title = "Train" if title_prefix is None else f"{title_prefix} — Train"
        val_title = "Validation" if title_prefix is None else f"{title_prefix} — Validation"

        self._draw_joint_hist_contours(
            fig,
            outer[0, 0],
            self.train_scores,
            train_feature_values,
            self.y_train,
            title=train_title,
            feature_label=feature_label,
            class_0_label=class_0_label,
            class_1_label=class_1_label,
            class_0_color=class_0_color,
            class_1_color=class_1_color,
            alpha_marginal=alpha_marginal,
            contour_linewidth=contour_linewidth,
            marginal_bins=marginal_bins,
            hist2d_min_bins=hist2d_min_bins,
            hist2d_max_bins=hist2d_max_bins,
            score_q=score_q,
            feature_q=feature_q,
        )
        self._draw_joint_hist_contours(
            fig,
            outer[0, 1],
            self.val_scores,
            val_feature_values,
            self.y_val,
            title=val_title,
            feature_label=feature_label,
            class_0_label=class_0_label,
            class_1_label=class_1_label,
            class_0_color=class_0_color,
            class_1_color=class_1_color,
            alpha_marginal=alpha_marginal,
            contour_linewidth=contour_linewidth,
            marginal_bins=marginal_bins,
            hist2d_min_bins=hist2d_min_bins,
            hist2d_max_bins=hist2d_max_bins,
            score_q=score_q,
            feature_q=feature_q,
        )

        return ScoreFeatureContourPlotResult(
            fitted_model=self.model,
            fig=fig,
            train_scores=self.train_scores,
            val_scores=self.val_scores,
            train_feature_values=train_feature_values,
            val_feature_values=val_feature_values,
        )

    # ============================================================
    # Float feature histograms
    # ============================================================
    @staticmethod
    def _get_float_feature_names(X) -> list:
        if hasattr(X, "dtypes") and hasattr(X, "columns"):
            float_cols = []
            for col in X.columns:
                if np.issubdtype(X[col].dtype, np.floating):
                    float_cols.append(col)
            return float_cols

        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X_arr.shape}")
        if np.issubdtype(X_arr.dtype, np.floating):
            return list(range(X_arr.shape[1]))
        return []

    @classmethod
    def _extract_feature_values(cls, X, feature_name: Any) -> np.ndarray:
        if hasattr(X, "columns"):
            if isinstance(feature_name, str):
                if feature_name not in X.columns:
                    raise KeyError(
                        f"Feature '{feature_name}' not found in DataFrame")
                values = X[feature_name].to_numpy()
                return cls._to_numpy_1d(values, f"feature '{feature_name}'").astype(float)
            if isinstance(feature_name, int):
                values = X.iloc[:, feature_name].to_numpy()
                return cls._to_numpy_1d(values, f"feature index {feature_name}").astype(float)
            raise TypeError(
                "For DataFrame input, feature_name must be str or int")

        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X_arr.shape}")
        if not isinstance(feature_name, int):
            raise TypeError(
                "For ndarray input, feature_name must be an integer column index")
        return cls._to_numpy_1d(X_arr[:, feature_name], f"feature index {feature_name}").astype(float)

    @classmethod
    def _feature_visual_clip_mask(
        cls,
        values: np.ndarray,
        *,
        lower_q: float = 0.002,
        upper_q: float = 0.998,
        iqr_k: float = 3.0,
    ) -> np.ndarray:
        values = cls._to_numpy_1d(values, "values")
        finite_mask = np.isfinite(values)
        v = values[finite_mask]
        if len(v) < 10:
            return finite_mask

        q_lo, q_hi = np.quantile(v, [lower_q, upper_q])
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        if iqr > 0 and np.isfinite(iqr):
            iqr_lo = q1 - iqr_k * iqr
            iqr_hi = q3 + iqr_k * iqr
        else:
            iqr_lo, iqr_hi = q_lo, q_hi
        lo = max(q_lo, iqr_lo)
        hi = min(q_hi, iqr_hi)
        return finite_mask & (values >= lo) & (values <= hi)

    @classmethod
    def _adaptive_feature_bins(
        cls,
        values: np.ndarray,
        *,
        min_bins: int = 15,
        max_bins: int = 80,
        min_bin_width: Optional[float] = None,
    ) -> Tuple[int, np.ndarray]:
        values = cls._to_numpy_1d(values, "values")
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return min_bins, np.linspace(-1.0, 1.0, min_bins + 1)

        lo = float(np.min(values))
        hi = float(np.max(values))
        data_range = hi - lo
        if data_range <= 0:
            eps = 1e-6 if lo == 0 else abs(lo) * 1e-6
            return min_bins, np.linspace(lo - eps, hi + eps, min_bins + 1)

        n = len(values)
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            bin_width = 2.0 * iqr / np.cbrt(n)
        else:
            bin_width = data_range / max(np.sqrt(n), 1.0)
        if not np.isfinite(bin_width) or bin_width <= 0:
            bin_width = data_range / max(np.sqrt(n), 1.0)
        if min_bin_width is not None:
            bin_width = max(bin_width, float(min_bin_width))
        n_bins = int(np.ceil(data_range / bin_width))
        n_bins = int(np.clip(n_bins, min_bins, max_bins))
        edges = np.linspace(lo, hi, n_bins + 1)
        return n_bins, edges

    def plot_float_feature_histograms(
        self,
        *,
        split: str = "train",
        features: Optional[Sequence[Any]] = None,
        class_0_label: str = "class 0",
        class_1_label: str = "class 1",
        alpha: float = 0.45,
        figsize_per_feature: Tuple[float, float] = (12.0, 3.6),
        min_bins: int = 15,
        max_bins: int = 80,
        min_bin_width: Optional[float] = None,
        lower_q: float = 0.002,
        upper_q: float = 0.998,
        iqr_k: float = 3.0,
        density: bool = True,
        max_features: Optional[int] = None,
    ):
        if split == "train":
            X, y = self.X_train, self.y_train
        elif split == "val":
            X, y = self.X_val, self.y_val
        else:
            raise ValueError("split must be either 'train' or 'val'")

        if features is None:
            features = self._get_float_feature_names(X)
        else:
            features = list(features)
        if max_features is not None:
            features = list(features)[:max_features]
        if len(features) == 0:
            raise ValueError("No float features found to plot.")

        n_features = len(features)
        fig_width, fig_height_per_row = figsize_per_feature
        fig, axes = plt.subplots(
            n_features,
            2,
            figsize=(fig_width, fig_height_per_row * n_features),
            constrained_layout=True,
            squeeze=False,
        )

        for row_idx, feature_name in enumerate(features):
            values = self._extract_feature_values(X, feature_name)
            if len(values) != len(y):
                raise ValueError(
                    f"Feature '{feature_name}' length mismatch: len(values)={len(values)} vs len(y)={len(y)}")

            mask = self._feature_visual_clip_mask(
                values, lower_q=lower_q, upper_q=upper_q, iqr_k=iqr_k)
            v = values[mask]
            y_masked = y[mask]
            if len(v) == 0:
                raise ValueError(
                    f"All values were removed by clipping for feature '{feature_name}'")

            _, edges = self._adaptive_feature_bins(
                v, min_bins=min_bins, max_bins=max_bins, min_bin_width=min_bin_width)
            v0 = v[y_masked == 0]
            v1 = v[y_masked == 1]

            ax_left = axes[row_idx, 0]
            ax_left.hist(v, bins=edges, density=density)
            ax_left.set_title(f"{feature_name} — overall ({split})")
            ax_left.set_xlabel(str(feature_name))
            ax_left.set_ylabel("Density" if density else "Count")
            ax_left.grid(True, alpha=0.25)

            ax_right = axes[row_idx, 1]
            ax_right.hist(v0, bins=edges, density=density,
                          alpha=alpha, label=f"{class_0_label} (n={len(v0)})")
            ax_right.hist(v1, bins=edges, density=density,
                          alpha=alpha, label=f"{class_1_label} (n={len(v1)})")
            ax_right.set_title(f"{feature_name} — by class ({split})")
            ax_right.set_xlabel(str(feature_name))
            ax_right.set_ylabel("Density" if density else "Count")
            ax_right.grid(True, alpha=0.25)
            ax_right.legend()

        return fig, axes

    # ============================================================
    # SHAP
    # ============================================================
    def plot_shap_beeswarm(
        self,
        *,
        max_display: int = 20,
        figsize: Tuple[float, float] = (10.0, 8.0),
        title: Optional[str] = None,
        sample_frac: Optional[float] = None,
        sample_size: Optional[int] = None,
        random_state: int = 42,
    ) -> ShapBeeswarmResult:
        if sample_frac is not None and sample_size is not None:
            raise ValueError("Specify only one of sample_frac or sample_size")

        X_val = self.X_val
        n = len(X_val)

        if sample_frac is not None:
            if not (0 < sample_frac <= 1):
                raise ValueError("sample_frac must be in (0, 1]")
            k = max(1, int(round(n * sample_frac)))
        elif sample_size is not None:
            if sample_size <= 0:
                raise ValueError("sample_size must be positive")
            k = min(int(sample_size), n)
        else:
            k = n

        if k < n:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(n, size=k, replace=False)
            if hasattr(X_val, "iloc"):
                X_used = X_val.iloc[idx]
            else:
                X_used = np.asarray(X_val)[idx]
        else:
            X_used = X_val

        explainer = shap.Explainer(self.model, X_used)
        explanation = explainer(X_used)

        fig = plt.figure(figsize=figsize)
        shap.plots.beeswarm(explanation, max_display=max_display, show=False)

        if title is not None:
            plt.title(title)

        return ShapBeeswarmResult(
            explainer=explainer,
            explanation=explanation,
            fig=fig,
        )


@dataclass
class FeatureStats:
    nan_ratio: float
    zero_ratio: float
    skew: float
    max_ratio: float
    min_ratio: float
    max_val: float
    min_val: float


class AutoFeatureStandardizer(BaseEstimator, TransformerMixin):
    """
    Automatic feature transformation for tabular data.

    Properties:
    - does NOT impute NaNs
    - learns transformation per feature
    - adds auxiliary features when needed
    """

    def __init__(
        self,
        skew_threshold: float = 1.5,
        zero_threshold: float = 0.5,
        clip_quantile: float = 0.999,
        add_indicators: bool = True,
    ):
        self.skew_threshold = skew_threshold
        self.zero_threshold = zero_threshold
        self.clip_quantile = clip_quantile
        self.add_indicators = add_indicators

        self.feature_stats_: Dict[str, FeatureStats] = {}
        self.transforms_: Dict[str, str] = {}

    # ============================================================
    # Core stats
    # ============================================================

    def _analyze_feature(self, x: np.ndarray) -> FeatureStats:
        mask = ~np.isnan(x)
        x_clean = x[mask]

        if len(x_clean) < 10:
            return FeatureStats(0, 0, 0, 0, 0, np.nan, np.nan)

        return FeatureStats(
            nan_ratio=np.mean(~mask),
            zero_ratio=np.mean(x_clean == 0),
            skew=scipy.stats.skew(x_clean),
            max_ratio=np.mean(x_clean == np.max(x_clean)),
            min_ratio=np.mean(x_clean == np.min(x_clean)),
            max_val=np.max(x_clean),
            min_val=np.min(x_clean),
        )

    # ============================================================
    # Fit
    # ============================================================

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_stats_ = {}
        self.transforms_ = {}

        for col in X.columns:
            if not np.issubdtype(X[col].dtype, np.floating):
                continue

            x = X[col].values.astype(float)
            stats = self._analyze_feature(x)
            self.feature_stats_[col] = stats

            # -----------------------------
            # Decision logic
            # -----------------------------
            if stats.skew > self.skew_threshold:
                self.transforms_[col] = "log"

            elif stats.max_ratio > 0.05:
                self.transforms_[col] = "cap"

            elif stats.zero_ratio > self.zero_threshold:
                self.transforms_[col] = "zero"

            else:
                self.transforms_[col] = "none"

        return self

    # ============================================================
    # Transform
    # ============================================================

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        for col, transform_type in self.transforms_.items():
            if col not in X_out.columns:
                continue

            x = X_out[col].values.astype(float)
            stats = self.feature_stats_[col]

            mask = ~np.isnan(x)

            # -------- LOG TRANSFORM --------
            if transform_type == "log":
                x_new = x.copy()
                x_new[mask] = np.log1p(x[mask])
                X_out[col] = x_new

            # -------- CAP FIX --------
            elif transform_type == "cap":
                cap = stats.max_val

                if self.add_indicators:
                    X_out[f"{col}_is_capped"] = (x == cap).astype(float)

                x_new = x.copy()
                x_new[mask] = np.minimum(x[mask], cap - 1)
                X_out[col] = x_new

            # -------- ZERO SPLIT --------
            elif transform_type == "zero":
                if self.add_indicators:
                    X_out[f"{col}_is_zero"] = (x == 0).astype(float)

                x_new = x.copy()
                x_new[mask] = np.log1p(x[mask])
                X_out[col] = x_new

            # -------- NONE --------
            else:
                pass

            # -------- NaN indicator (optional) --------
            if self.add_indicators:
                X_out[f"{col}_is_nan"] = np.isnan(x).astype(float)

        return X_out
