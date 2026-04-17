from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, Union, Any

import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas.api.types import is_float_dtype
from ipywidgets import widgets

from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted
import shap
import scipy.stats
from tqdm import tqdm


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
    fig: object
    train_scores: np.ndarray
    val_scores: np.ndarray
    train_feature_values: np.ndarray
    val_feature_values: np.ndarray


@dataclass
class ShapBeeswarmResult:
    explainer: object
    explanation: object
    fig: plt.Figure


@dataclass
class ShapInteractionHeatmapResult:
    explainer: object
    interaction_values: np.ndarray
    mean_abs_interactions: np.ndarray
    feature_names: list[str]
    fig: plt.Figure
    ax: plt.Axes


@dataclass
class FeatureAblationResult:
    baseline_roc_auc: float
    baseline_pr_auc: float
    baseline_logloss: float
    table: pd.DataFrame


@dataclass
class FeatureSelectionSuggestion:
    constant_features: list[str]
    duplicate_features: list[tuple[str, str]]
    high_corr_drop_candidates: list[str]
    low_importance_features: list[str]
    harmful_features_by_ablation: list[str]


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
    def _infer_catboost_feature_types(X) -> dict[str, list]:
        """
        Infer CatBoost special feature lists from pandas DataFrame.

        Returns a dict with keys:
        - cat_features
        - text_features
        - embedding_features

        Uses column names for DataFrame input, which CatBoost supports.
        """
        result = {
            "cat_features": [],
            "text_features": [],
            "embedding_features": [],
        }

        if not isinstance(X, pd.DataFrame):
            return result

        for col in X.columns:
            s = X[col]
            dtype_str = str(s.dtype).lower()

            # categorical / object -> categorical feature
            if pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s):
                # distinguish text vs ordinary categorical object
                non_null = s.dropna()
                if len(non_null) > 0:
                    sample = non_null.iloc[0]
                    if isinstance(sample, str):
                        # heuristic:
                        # long/free-form text -> text feature
                        # short label-like strings -> categorical feature
                        avg_len = non_null.astype(str).str.len().mean()
                        nunique = non_null.nunique()
                        if avg_len > 20 and nunique > 20:
                            result["text_features"].append(col)
                        else:
                            result["cat_features"].append(col)
                    else:
                        result["cat_features"].append(col)
                else:
                    result["cat_features"].append(col)
                continue

            # list/array-like columns -> embedding feature
            if dtype_str == "object":
                non_null = s.dropna()
                if len(non_null) > 0:
                    sample = non_null.iloc[0]
                    if isinstance(sample, (list, tuple, np.ndarray)):
                        result["embedding_features"].append(col)

        # remove overlaps just in case
        text_set = set(result["text_features"])
        emb_set = set(result["embedding_features"])

        result["cat_features"] = [
            c for c in result["cat_features"]
            if c not in text_set and c not in emb_set
        ]

        return result

    @classmethod
    def _build_catboost_fit_kwargs(
        cls,
        X_train,
        X_val=None,
        y_val=None,
        *,
        enable_early_stopping: bool = False,
        early_stopping_rounds: int = 100,
        use_best_model: bool = True,
    ) -> dict:
        fit_kwargs = {}

        feature_types = cls._infer_catboost_feature_types(X_train)

        if feature_types["cat_features"]:
            fit_kwargs["cat_features"] = feature_types["cat_features"]
        if feature_types["text_features"]:
            fit_kwargs["text_features"] = feature_types["text_features"]
        if feature_types["embedding_features"]:
            fit_kwargs["embedding_features"] = feature_types["embedding_features"]

        # IMPORTANT:
        # CatBoost cannot train with use_best_model=True without eval_set.
        if enable_early_stopping and X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = (X_val, y_val)
            fit_kwargs["use_best_model"] = use_best_model
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
        else:
            fit_kwargs["use_best_model"] = False

        return fit_kwargs

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

        if "catboost" in module_name or "catboost" in class_name:
            # make sure cloned model itself does not keep an incompatible setting
            fitted_model = cls._set_model_param_if_supported(
                fitted_model,
                "use_best_model",
                bool(use_best_model) if enable_early_stopping else False,
            )

            fit_kwargs = cls._build_catboost_fit_kwargs(
                X_train,
                X_val=X_val,
                y_val=y_val,
                enable_early_stopping=enable_early_stopping,
                early_stopping_rounds=early_stopping_rounds,
                use_best_model=use_best_model,
            )

            fitted_model.fit(X_train, y_train, **fit_kwargs, verbose=False)
            return fitted_model

        if not enable_early_stopping:
            fitted_model.fit(X_train, y_train)
            return fitted_model

        if X_val is None or y_val is None:
            raise ValueError(
                "Validation data must be provided when enable_early_stopping=True."
            )

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
                callbacks.append(
                    lgb.early_stopping(
                        stopping_rounds=early_stopping_rounds,
                        verbose=False,
                    )
                )
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

    @classmethod
    def _get_feature_options(cls, X) -> list:
        if hasattr(X, "columns"):
            return list(X.columns)
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X_arr.shape}")
        return list(range(X_arr.shape[1]))

    @staticmethod
    def _get_float_feature_names(X) -> list:
        if hasattr(X, "dtypes") and hasattr(X, "columns"):
            float_cols = []
            for col in X.columns:
                if is_float_dtype(X[col]):
                    float_cols.append(col)
            return float_cols

        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X_arr.shape}")

        if np.issubdtype(X_arr.dtype, np.floating):
            return list(range(X_arr.shape[1]))

        return []

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
    def _extract_feature_values(cls, X, feature_name: Any) -> np.ndarray:
        return cls._extract_feature_column(X, feature_name)

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

    @staticmethod
    def _predict_proba_binary(model, X) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(X))
            if proba.ndim != 2 or proba.shape[1] < 2:
                raise ValueError(f"predict_proba returned shape={proba.shape}")
            return proba[:, 1]

        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(X)).ravel()
            return 1.0 / (1.0 + np.exp(-scores))

        class_name = model.__class__.__name__.lower()
        module_name = model.__class__.__module__.lower()

        if "catboost" in module_name or "catboost" in class_name:
            raw = np.asarray(model.predict(X, prediction_type="Probability"))
            if raw.ndim == 2 and raw.shape[1] >= 2:
                return raw[:, 1]
            return np.ravel(raw).astype(float)

        raise TypeError(
            "Model must support predict_proba or decision_function.")

    @staticmethod
    def _ensure_dataframe(X, prefix: str = "feature") -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X_arr.shape}")
        cols = [f"{prefix}_{i}" for i in range(X_arr.shape[1])]
        return pd.DataFrame(X_arr, columns=cols)

    @staticmethod
    def _is_numeric_series(s: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(s)

    def _evaluate_current_model_on_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[object, float, float, float]:
        model = self._safe_clone_model(self.model)
        if self._is_model_fitted(model):
            model = self._safe_clone_model(self.model)

        model = self._fit_model_on_train(
            model,
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            enable_early_stopping=False,
        )

        val_pred = self._predict_proba_binary(model, X_val)
        roc = roc_auc_score(y_val, val_pred)
        pr = average_precision_score(y_val, val_pred)
        ll = log_loss(y_val, val_pred, labels=[0, 1])
        return model, roc, pr, ll

    @staticmethod
    def _normalize_importance_series(
        importance_values: np.ndarray,
        feature_names: list[str],
        name: str,
    ) -> pd.DataFrame:
        imp = np.asarray(importance_values).ravel()
        if len(imp) != len(feature_names):
            raise ValueError(
                f"Importance length mismatch: len(values)={len(imp)} vs len(features)={len(feature_names)}"
            )
        return (
            pd.DataFrame({"feature": feature_names, name: imp})
            .sort_values(name, ascending=False)
            .reset_index(drop=True)
        )

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
    # Interactive score-feature contours (Jupyter-safe)
    # ============================================================

    def plot_score_feature_contours(
        self,
        *,
        features: Optional[Sequence[Any]] = None,
        title_prefix: Optional[str] = None,
        class_0_label: str = "False",
        class_1_label: str = "True",
        class_0_color: str = "royalblue",
        class_1_color: str = "darkorange",
        score_q: Tuple[float, float] = (0.002, 0.998),
        feature_q: Tuple[float, float] = (0.002, 0.998),
        hist2d_min_bins: int = 25,
        hist2d_max_bins: int = 80,
        max_scatter: int = 4000,
        width: int = 1300,
        height: int = 650,
    ):
        """
        Jupyter-friendly interactive contour explorer.

        Returns
        -------
        ipywidgets.VBox
            Dropdown + FigureWidget. This avoids Plotly updatemenus JSON spam in notebook
            frontends and updates traces in-place instead of stacking them.
        """
        if features is None:
            features = self._get_float_feature_names(self.X_train)
            if len(features) == 0:
                features = self._get_feature_options(self.X_train)
        else:
            features = list(features)

        if len(features) == 0:
            raise ValueError(
                "No features available for interactive contour plot.")

        feature_options = [str(f) for f in features]
        feature_map = {str(f): f for f in features}
        rng = np.random.default_rng(42)

        base_fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Train" if title_prefix is None else f"{title_prefix} — Train",
                "Validation" if title_prefix is None else f"{title_prefix} — Validation",
            ),
            horizontal_spacing=0.10,
        )
        fig = go.FigureWidget(base_fig)

        # Precreate 8 traces: (train scatter0, train contour0, train scatter1, train contour1,
        #                     val scatter0, val contour0, val scatter1, val contour1)
        trace_specs = [
            (1, 1, class_0_label, class_0_color, "train", 0, True),
            (1, 1, class_0_label, class_0_color, "train", 0, False),
            (1, 1, class_1_label, class_1_color, "train", 1, True),
            (1, 1, class_1_label, class_1_color, "train", 1, False),
            (1, 2, class_0_label, class_0_color, "val", 0, True),
            (1, 2, class_0_label, class_0_color, "val", 0, False),
            (1, 2, class_1_label, class_1_color, "val", 1, True),
            (1, 2, class_1_label, class_1_color, "val", 1, False),
        ]

        for row, col, cls_label, cls_color, split_name, cls_value, is_scatter in trace_specs:
            if is_scatter:
                fig.add_trace(
                    go.Scattergl(
                        x=[],
                        y=[],
                        mode="markers",
                        marker=dict(size=4, color=cls_color, opacity=0.22),
                        name=f"{split_name} {cls_label} points",
                        legendgroup=f"{split_name}_{cls_label}",
                        showlegend=(split_name == "train"),
                        hovertemplate="score=%{x:.4f}<br>feature=%{y:.4f}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
            else:
                fig.add_trace(
                    go.Histogram2dContour(
                        x=[],
                        y=[],
                        nbinsx=30,
                        nbinsy=30,
                        histnorm="probability density",
                        contours=dict(coloring="lines", showlabels=False),
                        line=dict(color=cls_color, width=2),
                        showscale=False,
                        name=f"{split_name} {cls_label} contour",
                        legendgroup=f"{split_name}_{cls_label}",
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

        fig.update_layout(
            width=width,
            height=height,
            title="Score-feature contours",
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.02, xanchor="left", x=0.0),
            template="plotly_white",
        )
        fig.update_xaxes(title_text="model_score", row=1, col=1)
        fig.update_xaxes(title_text="model_score", row=1, col=2)

        dropdown = widgets.Dropdown(
            options=feature_options,
            value=feature_options[0],
            description="Feature:",
            layout=widgets.Layout(width="420px"),
        )

        def _prepare_split(scores: np.ndarray, feature_values: np.ndarray, y_true: np.ndarray):
            mask = self._joint_visual_clip_mask(
                scores,
                feature_values,
                score_q=score_q,
                feature_q=feature_q,
            )
            s = scores[mask]
            f = feature_values[mask]
            y = y_true[mask]

            if len(s) == 0:
                return None

            x_bins, y_bins = self._adaptive_2d_bin_counts(
                s,
                f,
                min_bins=hist2d_min_bins,
                max_bins=hist2d_max_bins,
            )

            payload = {}
            for cls_value in [0, 1]:
                s_cls = s[y == cls_value]
                f_cls = f[y == cls_value]

                if len(s_cls) > max_scatter:
                    idx = rng.choice(
                        len(s_cls), size=max_scatter, replace=False)
                    s_sc = s_cls[idx]
                    f_sc = f_cls[idx]
                else:
                    s_sc = s_cls
                    f_sc = f_cls

                payload[cls_value] = {
                    "s_sc": s_sc,
                    "f_sc": f_sc,
                    "s_cls": s_cls,
                    "f_cls": f_cls,
                    "x_bins": x_bins,
                    "y_bins": y_bins,
                }
            return payload

        def _update(feature_key: str):
            feature_name = feature_map[feature_key]
            train_feature_values = self._extract_feature_column(
                self.X_train, feature_name)
            val_feature_values = self._extract_feature_column(
                self.X_val, feature_name)

            train_payload = _prepare_split(
                self.train_scores, train_feature_values, self.y_train)
            val_payload = _prepare_split(
                self.val_scores, val_feature_values, self.y_val)

            if train_payload is None or val_payload is None:
                raise ValueError(
                    f"No points left after clipping for feature '{feature_name}'")

            with fig.batch_update():
                # train class 0
                fig.data[0].x = train_payload[0]["s_sc"]
                fig.data[0].y = train_payload[0]["f_sc"]
                fig.data[1].x = train_payload[0]["s_cls"]
                fig.data[1].y = train_payload[0]["f_cls"]
                fig.data[1].nbinsx = train_payload[0]["x_bins"]
                fig.data[1].nbinsy = train_payload[0]["y_bins"]

                # train class 1
                fig.data[2].x = train_payload[1]["s_sc"]
                fig.data[2].y = train_payload[1]["f_sc"]
                fig.data[3].x = train_payload[1]["s_cls"]
                fig.data[3].y = train_payload[1]["f_cls"]
                fig.data[3].nbinsx = train_payload[1]["x_bins"]
                fig.data[3].nbinsy = train_payload[1]["y_bins"]

                # val class 0
                fig.data[4].x = val_payload[0]["s_sc"]
                fig.data[4].y = val_payload[0]["f_sc"]
                fig.data[5].x = val_payload[0]["s_cls"]
                fig.data[5].y = val_payload[0]["f_cls"]
                fig.data[5].nbinsx = val_payload[0]["x_bins"]
                fig.data[5].nbinsy = val_payload[0]["y_bins"]

                # val class 1
                fig.data[6].x = val_payload[1]["s_sc"]
                fig.data[6].y = val_payload[1]["f_sc"]
                fig.data[7].x = val_payload[1]["s_cls"]
                fig.data[7].y = val_payload[1]["f_cls"]
                fig.data[7].nbinsx = val_payload[1]["x_bins"]
                fig.data[7].nbinsy = val_payload[1]["y_bins"]

                title = f"Score-feature contours — {feature_name}"
                if title_prefix is not None:
                    title = f"{title_prefix} — {feature_name}"
                fig.layout.title = title
                fig.layout.yaxis.title = str(feature_name)
                fig.layout.yaxis2.title = str(feature_name)

            return train_feature_values, val_feature_values

        initial_train_feature_values, initial_val_feature_values = _update(
            dropdown.value)

        def _on_change(change):
            if change["name"] == "value" and change["new"] is not None:
                _update(change["new"])

        dropdown.observe(_on_change, names="value")

        box = widgets.VBox([dropdown, fig])
        box._score_feature_result = ScoreFeatureContourPlotResult(
            fitted_model=self.model,
            fig=fig,
            train_scores=self.train_scores,
            val_scores=self.val_scores,
            train_feature_values=initial_train_feature_values,
            val_feature_values=initial_val_feature_values,
        )
        return box

    # ============================================================
    # Interactive float feature histograms (Jupyter-safe)
    # ============================================================

    def plot_float_feature_histograms(
        self,
        *,
        split: str = "train",
        features: Optional[Sequence[Any]] = None,
        class_0_label: str = "class 0",
        class_1_label: str = "class 1",
        class_0_color: str = "royalblue",
        class_1_color: str = "darkorange",
        alpha: float = 0.55,
        min_bins: int = 15,
        max_bins: int = 80,
        min_bin_width: Optional[float] = None,
        lower_q: float = 0.002,
        upper_q: float = 0.998,
        iqr_k: float = 3.0,
        density: bool = True,
        width: int = 1200,
        height: int = 550,
    ):
        """
        Jupyter-friendly interactive histogram explorer.

        Returns
        -------
        ipywidgets.VBox
            Dropdown + FigureWidget. Traces are updated in-place, so histograms do not
            accumulate on top of each other after feature switches.
        """
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

        if len(features) == 0:
            raise ValueError("No float features found to plot.")

        feature_options = [str(f) for f in features]
        feature_map = {str(f): f for f in features}

        base_fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(f"overall ({split})", f"by class ({split})"),
            horizontal_spacing=0.12,
        )
        fig = go.FigureWidget(base_fig)

        histnorm = "probability density" if density else None

        fig.add_trace(
            go.Histogram(
                x=[],
                histnorm=histnorm,
                marker=dict(color="steelblue"),
                opacity=0.90,
                name="overall",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=[],
                histnorm=histnorm,
                marker=dict(color=class_0_color),
                opacity=alpha,
                name=class_0_label,
                legendgroup="class0",
                showlegend=True,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Histogram(
                x=[],
                histnorm=histnorm,
                marker=dict(color=class_1_color),
                opacity=alpha,
                name=class_1_label,
                legendgroup="class1",
                showlegend=True,
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            width=width,
            height=height,
            barmode="overlay",
            title="Feature histograms",
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.02, xanchor="left", x=0.0),
            template="plotly_white",
        )
        fig.update_yaxes(
            title_text="Density" if density else "Count", row=1, col=1)
        fig.update_yaxes(
            title_text="Density" if density else "Count", row=1, col=2)

        dropdown = widgets.Dropdown(
            options=feature_options,
            value=feature_options[0],
            description="Feature:",
            layout=widgets.Layout(width="420px"),
        )

        def _update(feature_key: str):
            feature_name = feature_map[feature_key]
            values = self._extract_feature_values(X, feature_name)

            if len(values) != len(y):
                raise ValueError(
                    f"Feature '{feature_name}' length mismatch: len(values)={len(values)} vs len(y)={len(y)}"
                )

            mask = self._feature_visual_clip_mask(
                values,
                lower_q=lower_q,
                upper_q=upper_q,
                iqr_k=iqr_k,
            )
            v = values[mask]
            y_masked = y[mask]

            if len(v) == 0:
                raise ValueError(
                    f"All values were removed by clipping for feature '{feature_name}'")

            _, edges = self._adaptive_feature_bins(
                v,
                min_bins=min_bins,
                max_bins=max_bins,
                min_bin_width=min_bin_width,
            )

            v0 = v[y_masked == 0]
            v1 = v[y_masked == 1]
            bin_size = float(edges[1] - edges[0]) if len(edges) > 1 else None

            with fig.batch_update():
                fig.data[0].x = v
                fig.data[0].xbins.start = float(edges[0])
                fig.data[0].xbins.end = float(edges[-1])
                fig.data[0].xbins.size = bin_size

                fig.data[1].x = v0
                fig.data[1].xbins.start = float(edges[0])
                fig.data[1].xbins.end = float(edges[-1])
                fig.data[1].xbins.size = bin_size

                fig.data[2].x = v1
                fig.data[2].xbins.start = float(edges[0])
                fig.data[2].xbins.end = float(edges[-1])
                fig.data[2].xbins.size = bin_size

                fig.layout.title = f"Feature histograms — {feature_name} ({split})"
                fig.layout.xaxis.title = str(feature_name)
                fig.layout.xaxis2.title = str(feature_name)

        _update(dropdown.value)

        def _on_change(change):
            if change["name"] == "value" and change["new"] is not None:
                _update(change["new"])

        dropdown.observe(_on_change, names="value")
        return widgets.VBox([dropdown, fig])

    @staticmethod
    def _is_catboost_model(model) -> bool:
        class_name = model.__class__.__name__.lower()
        module_name = model.__class__.__module__.lower()
        return "catboost" in class_name or "catboost" in module_name

    @staticmethod
    def _get_feature_names_for_shap(X) -> list[str]:
        if hasattr(X, "columns"):
            return [str(c) for c in X.columns]
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X_arr.shape}")
        return [f"feature_{i}" for i in range(X_arr.shape[1])]

    @staticmethod
    def _sample_rows(X, k: int, random_state: int):
        n = len(X)
        if k >= n:
            return X

        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=k, replace=False)

        if hasattr(X, "iloc"):
            return X.iloc[idx]
        return np.asarray(X)[idx]

    def _resolve_shap_sample(self, X, sample_frac, sample_size, random_state):
        if sample_frac is not None and sample_size is not None:
            raise ValueError("Specify only one of sample_frac or sample_size")

        n = len(X)
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

        return self._sample_rows(X, k=k, random_state=random_state)

    def _make_shap_explainer(self):
        """
        Build a SHAP explainer compatible with the fitted model.

        Important:
        - For CatBoost with categorical splits, do NOT pass background data.
        - Use tree_path_dependent perturbation so string categorical values are accepted.
        """
        if self._is_catboost_model(self.model):
            return shap.TreeExplainer(
                self.model,
                feature_perturbation="tree_path_dependent",
                model_output="raw",
            )

        return shap.Explainer(self.model)

    @staticmethod
    def _normalize_shap_values_for_binary(shap_values):
        """
        Normalize SHAP outputs to a 2D array of shape (n_samples, n_features).

        Handles:
        - Explanation.values already 2D
        - Explanation.values 3D with trailing class axis
        - list outputs from older SHAP APIs
        """
        values = shap_values

        if isinstance(values, list):
            if len(values) == 1:
                values = np.asarray(values[0])
            elif len(values) == 2:
                values = np.asarray(values[1])
            else:
                raise ValueError(
                    f"Unsupported SHAP list output with length={len(values)} for binary classification."
                )

        values = np.asarray(values)

        if values.ndim == 2:
            return values

        if values.ndim == 3:
            # common shape: (n_samples, n_features, n_outputs)
            if values.shape[2] == 1:
                return values[:, :, 0]
            if values.shape[2] == 2:
                return values[:, :, 1]
            raise ValueError(
                f"Unsupported 3D SHAP values shape {values.shape} for binary classification."
            )

        raise ValueError(
            f"Unsupported SHAP values ndim={values.ndim}. Expected 2D or 3D."
        )

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
        X_used = self._resolve_shap_sample(
            self.X_val,
            sample_frac=sample_frac,
            sample_size=sample_size,
            random_state=random_state,
        )

        explainer = self._make_shap_explainer()

        # For CatBoost + categorical features this works because:
        # - no background data is passed to TreeExplainer
        # - feature_perturbation="tree_path_dependent"
        explanation = explainer(X_used)

        fig = plt.figure(figsize=figsize)
        shap.plots.beeswarm(
            explanation,
            max_display=max_display,
            show=False,
        )

        if title is not None:
            plt.title(title)

        return ShapBeeswarmResult(
            explainer=explainer,
            explanation=explanation,
            fig=fig,
        )

    def plot_shap_interaction_heatmap(
        self,
        *,
        max_display: int = 20,
        figsize: Tuple[float, float] = (11.0, 9.0),
        title: Optional[str] = None,
        sample_frac: Optional[float] = None,
        sample_size: Optional[int] = 1000,
        random_state: int = 42,
        sort_by_strength: bool = True,
        annotate: bool = False,
    ) -> ShapInteractionHeatmapResult:
        """
        Plot mean absolute SHAP interaction strengths as a heatmap.

        Notes
        -----
        - This uses TreeExplainer.shap_interaction_values, so it is intended for tree models.
        - For CatBoost with categorical splits, TreeExplainer must be constructed without
          background data and with feature_perturbation="tree_path_dependent".
        - Interaction computation can be expensive; default sample_size is capped.
        """
        X_used = self._resolve_shap_sample(
            self.X_val,
            sample_frac=sample_frac,
            sample_size=sample_size,
            random_state=random_state,
        )

        explainer = self._make_shap_explainer()

        try:
            interaction_values = explainer.shap_interaction_values(X_used)
        except Exception as e:
            raise RuntimeError(
                "Failed to compute SHAP interaction values for this model/input. "
                "For CatBoost, make sure the model is tree-based and the input format "
                "matches the model training format."
            ) from e

        interaction_values = np.asarray(interaction_values)

        # Normalize possible binary-class outputs:
        # - (n_samples, n_features, n_features)
        # - (2, n_samples, n_features, n_features)
        # - list length 2 in older SHAP
        if isinstance(interaction_values, list):
            if len(interaction_values) == 1:
                interaction_values = np.asarray(interaction_values[0])
            elif len(interaction_values) == 2:
                interaction_values = np.asarray(interaction_values[1])
            else:
                raise ValueError(
                    f"Unsupported interaction output list length={len(interaction_values)}."
                )

        if interaction_values.ndim == 4:
            # assume leading axis is class/output axis
            if interaction_values.shape[0] == 1:
                interaction_values = interaction_values[0]
            elif interaction_values.shape[0] == 2:
                interaction_values = interaction_values[1]
            else:
                raise ValueError(
                    f"Unsupported 4D interaction tensor shape={interaction_values.shape}."
                )

        if interaction_values.ndim != 3:
            raise ValueError(
                f"Expected interaction tensor of shape (n_samples, n_features, n_features), "
                f"got shape={interaction_values.shape}."
            )

        mean_abs_interactions = np.mean(np.abs(interaction_values), axis=0)

        feature_names = self._get_feature_names_for_shap(X_used)

        if mean_abs_interactions.shape[0] != len(feature_names):
            raise ValueError(
                "Mismatch between interaction matrix size and number of feature names: "
                f"{mean_abs_interactions.shape[0]} vs {len(feature_names)}."
            )

        if sort_by_strength:
            strength = mean_abs_interactions.sum(axis=1)
            order = np.argsort(-strength)
        else:
            order = np.arange(len(feature_names))

        if max_display is not None and max_display > 0:
            order = order[: min(max_display, len(order))]

        heatmap = mean_abs_interactions[np.ix_(order, order)]
        shown_feature_names = [feature_names[i] for i in order]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(heatmap, aspect="auto")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("mean(|SHAP interaction|)")

        ax.set_xticks(np.arange(len(shown_feature_names)))
        ax.set_yticks(np.arange(len(shown_feature_names)))
        ax.set_xticklabels(shown_feature_names, rotation=90)
        ax.set_yticklabels(shown_feature_names)

        if annotate:
            for i in range(heatmap.shape[0]):
                for j in range(heatmap.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{heatmap[i, j]:.3g}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        if title is None:
            title = "SHAP feature interaction heatmap"
        ax.set_title(title)
        ax.set_xlabel("feature")
        ax.set_ylabel("feature")
        fig.tight_layout()

        return ShapInteractionHeatmapResult(
            explainer=explainer,
            interaction_values=interaction_values,
            mean_abs_interactions=mean_abs_interactions,
            feature_names=feature_names,
            fig=fig,
            ax=ax,
        )

    def find_constant_features(
        self,
        *,
        split: str = "train",
        min_unique: int = 1,
        include_nan_as_value: bool = True,
    ) -> list[str]:
        X = self.X_train if split == "train" else self.X_val
        X_df = self._ensure_dataframe(X)

        constant_cols = []
        for col in X_df.columns:
            nunique = X_df[col].nunique(dropna=not include_nan_as_value)
            if nunique <= min_unique:
                constant_cols.append(col)
        return constant_cols

    def find_duplicate_features(
        self,
        *,
        split: str = "train",
    ) -> list[tuple[str, str]]:
        X = self.X_train if split == "train" else self.X_val
        X_df = self._ensure_dataframe(X)

        hashes: dict[str, str] = {}
        duplicates: list[tuple[str, str]] = []

        for col in X_df.columns:
            h = pd.util.hash_pandas_object(X_df[col], index=False).sum()
            key = str(h)
            if key in hashes:
                other = hashes[key]
                if X_df[col].equals(X_df[other]):
                    duplicates.append((other, col))
            else:
                hashes[key] = col

        return duplicates

    def find_high_corr_features(
        self,
        *,
        split: str = "train",
        threshold: float = 0.995,
        method: str = "spearman",
    ) -> list[str]:
        X = self.X_train if split == "train" else self.X_val
        X_df = self._ensure_dataframe(X)

        num_df = X_df.select_dtypes(include=[np.number]).copy()
        if num_df.shape[1] == 0:
            return []

        corr = num_df.corr(method=method).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = [
            col for col in upper.columns
            if (upper[col] >= threshold).any()
        ]
        return to_drop

    def get_model_feature_importance(
        self,
        *,
        refit: bool = True,
    ) -> pd.DataFrame:
        X_train = self._ensure_dataframe(self.X_train)
        X_val = self._ensure_dataframe(self.X_val)

        if refit:
            model, _, _, _ = self._evaluate_current_model_on_features(
                X_train, X_val, self.y_train, self.y_val
            )
        else:
            model = self.model

        feature_names = list(X_train.columns)

        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_)
        elif hasattr(model, "get_feature_importance"):
            imp = np.asarray(model.get_feature_importance())
        else:
            raise TypeError(
                "Model does not expose built-in feature importances.")

        return self._normalize_importance_series(imp, feature_names, "model_importance")

    def get_permutation_importance(
        self,
        *,
        n_repeats: int = 5,
        random_state: int = 42,
        scoring: str = "roc_auc",
        refit: bool = True,
    ) -> pd.DataFrame:
        X_train = self._ensure_dataframe(self.X_train)
        X_val = self._ensure_dataframe(self.X_val)

        if refit:
            model, _, _, _ = self._evaluate_current_model_on_features(
                X_train, X_val, self.y_train, self.y_val
            )
        else:
            model = self.model

        result = permutation_importance(
            model,
            X_val,
            self.y_val,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )

        return (
            pd.DataFrame({
                "feature": list(X_val.columns),
                "perm_importance_mean": result.importances_mean,
                "perm_importance_std": result.importances_std,
            })
            .sort_values("perm_importance_mean", ascending=False)
            .reset_index(drop=True)
        )

    def analyze_feature_ablation(
        self,
        *,
        features: Optional[Sequence[str]] = None,
        max_features: Optional[int] = None,
        sort_by: str = "delta_roc_auc",
    ) -> FeatureAblationResult:
        X_train = self._ensure_dataframe(self.X_train)
        X_val = self._ensure_dataframe(self.X_val)

        _, baseline_roc, baseline_pr, baseline_ll = self._evaluate_current_model_on_features(
            X_train, X_val, self.y_train, self.y_val
        )

        if features is None:
            features = list(X_train.columns)
        else:
            features = [f for f in features if f in X_train.columns]

        if max_features is not None:
            features = list(features)[:max_features]

        rows = []
        all_cols = list(X_train.columns)

        for feature in tqdm(features):
            kept = [c for c in all_cols if c != feature]
            if len(kept) == 0:
                continue

            _, roc, pr, ll = self._evaluate_current_model_on_features(
                X_train[kept], X_val[kept], self.y_train, self.y_val
            )

            rows.append({
                "feature": feature,
                "roc_auc": roc,
                "pr_auc": pr,
                "logloss": ll,
                "delta_roc_auc": roc - baseline_roc,
                "delta_pr_auc": pr - baseline_pr,
                "delta_logloss": ll - baseline_ll,
            })

        table = pd.DataFrame(rows).sort_values(
            sort_by, ascending=False).reset_index(drop=True)

        return FeatureAblationResult(
            baseline_roc_auc=baseline_roc,
            baseline_pr_auc=baseline_pr,
            baseline_logloss=baseline_ll,
            table=table,
        )

    def greedy_drop_harmful_features(
        self,
        *,
        min_delta_roc_auc: float = 1e-4,
        max_rounds: int = 20,
        verbose: bool = True,
    ) -> pd.DataFrame:
        X_train = self._ensure_dataframe(self.X_train)
        X_val = self._ensure_dataframe(self.X_val)

        current_cols = list(X_train.columns)
        history = []

        _, current_roc, current_pr, current_ll = self._evaluate_current_model_on_features(
            X_train[current_cols], X_val[current_cols], self.y_train, self.y_val
        )

        for round_idx in range(max_rounds):
            best_feature = None
            best_roc = current_roc
            best_pr = current_pr
            best_ll = current_ll
            best_delta = 0.0

            for feature in current_cols:
                kept = [c for c in current_cols if c != feature]
                if len(kept) == 0:
                    continue

                _, roc, pr, ll = self._evaluate_current_model_on_features(
                    X_train[kept], X_val[kept], self.y_train, self.y_val
                )
                delta = roc - current_roc

                if delta > best_delta:
                    best_delta = delta
                    best_feature = feature
                    best_roc = roc
                    best_pr = pr
                    best_ll = ll

            if best_feature is None or best_delta < min_delta_roc_auc:
                break

            current_cols.remove(best_feature)
            history.append({
                "round": round_idx + 1,
                "dropped_feature": best_feature,
                "roc_auc": best_roc,
                "pr_auc": best_pr,
                "logloss": best_ll,
                "delta_roc_auc": best_delta,
                "n_features_left": len(current_cols),
            })

            current_roc, current_pr, current_ll = best_roc, best_pr, best_ll

            if verbose:
                print(
                    f"[round {round_idx + 1}] drop '{best_feature}' | "
                    f"ROC AUC={best_roc:.6f} | Δ={best_delta:.6f} | features left={len(current_cols)}"
                )

        return pd.DataFrame(history)

    def suggest_features_to_drop(
        self,
        *,
        corr_threshold: float = 0.995,
        perm_threshold: float = 0.0,
        ablation_min_delta: float = 0.0,
        n_perm_repeats: int = 5,
        max_ablation_features: Optional[int] = 100,
    ) -> FeatureSelectionSuggestion:
        constant_features = self.find_constant_features(split="train")
        duplicate_pairs = self.find_duplicate_features(split="train")
        duplicate_features = [b for _, b in duplicate_pairs]
        high_corr_drop_candidates = self.find_high_corr_features(
            split="train",
            threshold=corr_threshold,
            method="spearman",
        )

        perm_df = self.get_permutation_importance(
            n_repeats=n_perm_repeats,
            scoring="roc_auc",
            refit=True,
        )
        low_importance_features = (
            perm_df.loc[perm_df["perm_importance_mean"]
                        <= perm_threshold, "feature"]
            .tolist()
        )

        candidate_features = list(dict.fromkeys(
            low_importance_features + high_corr_drop_candidates + duplicate_features
        ))

        ablation_features = candidate_features
        if max_ablation_features is not None:
            ablation_features = ablation_features[:max_ablation_features]

        ablation_res = self.analyze_feature_ablation(
            features=ablation_features,
            sort_by="delta_roc_auc",
        )

        harmful_features = (
            ablation_res.table.loc[
                ablation_res.table["delta_roc_auc"] >= ablation_min_delta,
                "feature"
            ]
            .tolist()
        )

        return FeatureSelectionSuggestion(
            constant_features=constant_features,
            duplicate_features=duplicate_pairs,
            high_corr_drop_candidates=high_corr_drop_candidates,
            low_importance_features=low_importance_features,
            harmful_features_by_ablation=harmful_features,
        )

    @staticmethod
    def plot_feature_importance_table(
        importance_df: pd.DataFrame,
        *,
        feature_col: str = "feature",
        value_col: str = "model_importance",
        top_n: int = 25,
        figsize: Tuple[int, int] = (10, 8),
        title: Optional[str] = None,
    ) -> plt.Figure:
        plot_df = importance_df.head(top_n).iloc[::-1]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(plot_df[feature_col], plot_df[value_col])
        ax.set_xlabel(value_col)
        ax.set_ylabel(feature_col)
        ax.set_title(title or f"Top {top_n} features by {value_col}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig


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
        to_drop: list[str] = []
    ):
        self.skew_threshold = skew_threshold
        self.zero_threshold = zero_threshold
        self.clip_quantile = clip_quantile
        self.add_indicators = add_indicators

        self.feature_stats_: Dict[str, FeatureStats] = {}
        self.transforms_: Dict[str, str] = {}
        self.to_drop = to_drop

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

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_stats_ = {}
        self.transforms_ = {}

        for col in X.columns:
            if not is_float_dtype(X[col]):
                continue

            x = X[col].values.astype(float)
            stats = self._analyze_feature(x)
            self.feature_stats_[col] = stats

            if stats.skew > self.skew_threshold:
                self.transforms_[col] = "log"
            elif stats.max_ratio > 0.05:
                self.transforms_[col] = "cap"
            elif stats.zero_ratio > self.zero_threshold:
                self.transforms_[col] = "zero"
            else:
                self.transforms_[col] = "none"

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        for col, transform_type in self.transforms_.items():
            if col not in X_out.columns:
                continue

            x = X_out[col].values.astype(float)
            stats = self.feature_stats_[col]
            mask = ~np.isnan(x)

            if transform_type == "log":
                x_new = x.copy()
                x_new[mask] = np.log1p(x[mask])
                X_out[col] = x_new

            elif transform_type == "cap":
                cap = stats.max_val
                if self.add_indicators:
                    X_out[f"{col}_is_capped"] = (x == cap).astype(float)
                x_new = x.copy()
                x_new[mask] = np.minimum(x[mask], cap - 1)
                X_out[col] = x_new

            elif transform_type == "zero":
                if self.add_indicators:
                    X_out[f"{col}_is_zero"] = (x == 0).astype(float)
                x_new = x.copy()
                x_new[mask] = np.log1p(x[mask])
                X_out[col] = x_new

            if self.add_indicators:
                X_out[f"{col}_is_nan"] = np.isnan(x).astype(float)

        X_out = X_out.copy()

        X_out.drop(columns=self.to_drop, inplace=True)

        return X_out
