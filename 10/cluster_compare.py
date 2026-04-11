from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import panel as pn
import umap

from bokeh.layouts import column as bk_column
from bokeh.layouts import row as bk_row
from bokeh.models import (
    ColumnDataSource,
    Div,
    Select,
    TextInput,
    Button,
    ColorPicker,
    Spinner,
)
from bokeh.plotting import figure
from bokeh.palettes import Category10, Category20

from catboost import CatBoostClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


pn.extension()


ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, List[float]]


@dataclass
class ClusterComparisonResult:
    cluster_1: str
    cluster_2: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    best_threshold: float
    thresholds: np.ndarray
    f1_values: np.ndarray
    scores_cluster_1: np.ndarray
    scores_cluster_2: np.ndarray
    fitted_model: CatBoostClassifier
    cluster_1_label_distribution: Dict[Any, float]
    cluster_2_label_distribution: Dict[Any, float]


class NotebookBinaryClusterInterpreter:
    """
    Jupyter-native interactive tool based on Panel + Bokeh.

    Workflow:
    1. Pass X and y into constructor.
    2. UMAP maps X to 2D.
    3. On the scatter plot, select points via lasso.
    4. Assign them a cluster label and color.
    5. Compare any two labeled clusters via CatBoost trained on the ORIGINAL
       high-dimensional features for those two clusters only.
    6. Inspect metrics + score histograms + F1 vs threshold curve.
    """

    def __init__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[ArrayLike] = None,
        *,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_metric: str = "euclidean",
        umap_random_state: int = 42,
        point_size: int = 6,
        point_alpha: float = 0.65,
        compare_test_fraction: float = 0.35,
        threshold_grid_size: int = 300,
    ) -> None:
        self._validate_inputs(X, y)

        self.X_original = X.copy() if hasattr(X, "copy") else np.array(X, copy=True)
        self.y = None if y is None else np.asarray(y)

        self.X_numeric, self.feature_names = self._prepare_numeric_matrix(X)
        self.n_samples = self.X_numeric.shape[0]
        self.n_features = self.X_numeric.shape[1]

        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_metric = umap_metric
        self.umap_random_state = umap_random_state
        self.point_size = point_size
        self.point_alpha = point_alpha
        self.compare_test_fraction = compare_test_fraction
        self.threshold_grid_size = threshold_grid_size

        self.embedding = self._compute_umap_embedding()

        self.cluster_to_indices: Dict[str, set[int]] = {}
        self.cluster_to_color: Dict[str, str] = {}
        self.selection_info_text = "Selected points: 0"
        self.last_comparison_result: Optional[ClusterComparisonResult] = None

        self._init_sources()
        self._build_widgets()
        self._build_figures()
        self._wire_callbacks()

    # ============================================================
    # Validation / preprocessing
    # ============================================================

    @staticmethod
    def _validate_inputs(X: Union[pd.DataFrame, np.ndarray], y: Optional[ArrayLike]) -> None:
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a pandas.DataFrame or numpy.ndarray")

        n = len(X)
        if n == 0:
            raise ValueError("X must not be empty")

        if y is not None and len(y) != n:
            raise ValueError(
                f"len(y) must equal len(X), got len(y)={len(y)} and len(X)={n}")

    @staticmethod
    def _prepare_numeric_matrix(
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        UMAP and CatBoost in this notebook tool use a numeric matrix.
        NaNs are preserved for CatBoost compatibility and replaced only for UMAP.
        """
        if isinstance(X, pd.DataFrame):
            numeric_cols = [
                c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
            if len(numeric_cols) == 0:
                raise ValueError("X must contain at least one numeric column")

            X_num = X[numeric_cols].to_numpy(dtype=float)
            feature_names = [str(c) for c in numeric_cols]
            return X_num, feature_names

        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={X_arr.shape}")
        if not np.issubdtype(X_arr.dtype, np.number):
            raise ValueError("For ndarray input, X must be numeric")

        feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]
        return X_arr.astype(float), feature_names

    def _compute_umap_embedding(self) -> np.ndarray:
        """
        UMAP cannot handle NaN in all modes reliably, so for embedding only
        we replace NaN with column medians. This does NOT affect CatBoost
        training later, which uses original high-dimensional data slices.
        """
        X_umap = self.X_numeric.copy()

        if np.isnan(X_umap).any():
            col_medians = np.nanmedian(X_umap, axis=0)
            col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
            inds = np.where(np.isnan(X_umap))
            X_umap[inds] = np.take(col_medians, inds[1])

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            metric=self.umap_metric,
            # random_state=self.umap_random_state,
        )
        return reducer.fit_transform(X_umap)

    # ============================================================
    # Source / widgets / figures
    # ============================================================

    def _init_sources(self) -> None:
        default_color = "#9e9e9e"
        default_cluster = "unassigned"

        data = {
            "x": self.embedding[:, 0],
            "y": self.embedding[:, 1],
            "index": np.arange(self.n_samples),
            "cluster": np.array([default_cluster] * self.n_samples, dtype=object),
            "color": np.array([default_color] * self.n_samples, dtype=object),
        }

        if self.y is not None:
            data["y_true"] = self.y

        self.scatter_source = ColumnDataSource(data=data)

        self.hist_source_left = ColumnDataSource(
            data={"top": [], "left": [], "right": []})
        self.hist_source_right = ColumnDataSource(
            data={"top": [], "left": [], "right": []})
        self.threshold_curve_source = ColumnDataSource(data={"x": [], "y": []})

    def _build_widgets(self) -> None:
        palette = (
            list(Category10[10]) +
            list(Category20[20])
        )

        self.cluster_name_input = TextInput(title="Cluster label", value="")
        self.cluster_color_picker = ColorPicker(
            title="Cluster color", color=palette[0])
        self.assign_cluster_button = Button(
            label="Assign selected points to cluster", button_type="success")
        self.remove_cluster_button = Button(
            label="Remove selected points from cluster", button_type="warning")

        self.cluster_a_select = Select(title="Cluster A", value="", options=[])
        self.cluster_b_select = Select(title="Cluster B", value="", options=[])

        self.catboost_iterations_spinner = Spinner(
            title="CatBoost iterations", low=20, step=10, value=300, width=140)
        self.catboost_depth_spinner = Spinner(
            title="Depth", low=2, step=1, value=6, width=140)
        self.catboost_lr_spinner = Spinner(
            title="Learning rate", low=0.005, step=0.005, value=0.05, width=140)
        self.compare_button = Button(
            label="Compare selected clusters", button_type="primary")

        self.selection_div = Div(
            text=f"<b>{self.selection_info_text}</b>", width=300, height=30)
        self.cluster_summary_div = Div(
            text=self._cluster_summary_html(), width=350, height=220)
        self.metrics_div = Div(
            text=self._default_metrics_html(), width=400, height=180)

    def _build_figures(self) -> None:
        self.scatter_fig = figure(
            width=760,
            height=620,
            title="UMAP projection of high-dimensional data",
            tools="pan,wheel_zoom,box_zoom,reset,save,lasso_select,tap",
            active_drag="lasso_select",
        )
        self.scatter_fig.scatter(
            x="x",
            y="y",
            source=self.scatter_source,
            color="color",
            size=self.point_size,
            alpha=self.point_alpha,
        )
        self.scatter_fig.xaxis.axis_label = "UMAP-1"
        self.scatter_fig.yaxis.axis_label = "UMAP-2"

        self.threshold_fig = figure(
            width=520,
            height=260,
            title="F1 vs threshold",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        self.threshold_fig.line(
            "x", "y", source=self.threshold_curve_source, line_width=2)
        self.threshold_fig.xaxis.axis_label = "Raw score threshold"
        self.threshold_fig.yaxis.axis_label = "F1"

        self.hist_left_fig = figure(
            width=520,
            height=260,
            title="Score histogram: Cluster A",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        self.hist_left_fig.quad(
            source=self.hist_source_left,
            top="top",
            bottom=0,
            left="left",
            right="right",
            alpha=0.55,
            color="#1f77b4",
            line_color="#1f77b4",
        )
        self.hist_left_fig.xaxis.axis_label = "Raw model score"
        self.hist_left_fig.yaxis.axis_label = "Density"

        self.hist_right_fig = figure(
            width=520,
            height=260,
            title="Score histogram: Cluster B",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        self.hist_right_fig.quad(
            source=self.hist_source_right,
            top="top",
            bottom=0,
            left="left",
            right="right",
            alpha=0.55,
            color="#ff7f0e",
            line_color="#ff7f0e",
        )
        self.hist_right_fig.xaxis.axis_label = "Raw model score"
        self.hist_right_fig.yaxis.axis_label = "Density"

    def _wire_callbacks(self) -> None:
        self.scatter_source.selected.on_change(
            "indices", self._on_selection_change)
        self.assign_cluster_button.on_click(self._on_assign_cluster)
        self.remove_cluster_button.on_click(self._on_remove_from_cluster)
        self.compare_button.on_click(self._on_compare_clusters)

    # ============================================================
    # UI helpers
    # ============================================================

    def _cluster_summary_html(self) -> str:
        if len(self.cluster_to_indices) == 0:
            return "<b>Clusters:</b><br>No clusters assigned yet."

        lines = ["<b>Clusters:</b><br>"]
        for name in sorted(self.cluster_to_indices.keys()):
            color = self.cluster_to_color.get(name, "#9e9e9e")
            size = len(self.cluster_to_indices[name])
            lines.append(
                f'<span style="color:{color};font-weight:bold;">■</span> '
                f"{name}: {size} points"
            )
        return "<br>".join(lines)

    @staticmethod
    def _default_metrics_html() -> str:
        return (
            "<b>Comparison metrics</b><br>"
            "Select two clusters and click <i>Compare selected clusters</i>.<br><br>"
            "<b>Cluster label distribution</b><br>"
            "Available only if original y was provided."
        )

    def _refresh_cluster_widgets(self) -> None:
        options = sorted(self.cluster_to_indices.keys())
        select_options = [""] + options

        self.cluster_a_select.options = select_options
        self.cluster_b_select.options = select_options

        if self.cluster_a_select.value not in select_options:
            self.cluster_a_select.value = ""
        if self.cluster_b_select.value not in select_options:
            self.cluster_b_select.value = ""

        self.cluster_summary_div.text = self._cluster_summary_html()

    def _selected_indices(self) -> List[int]:
        return list(self.scatter_source.selected.indices)

    # ============================================================
    # Cluster editing
    # ============================================================

    def _on_selection_change(self, attr: str, old: Any, new: Any) -> None:
        n_selected = len(new)
        self.selection_info_text = f"Selected points: {n_selected}"
        self.selection_div.text = f"<b>{self.selection_info_text}</b>"

    def _on_assign_cluster(self) -> None:
        selected = self._selected_indices()
        if len(selected) == 0:
            self.selection_div.text = "<b>Selected points: 0</b><br>No points selected."
            return

        cluster_name = self.cluster_name_input.value.strip()
        if cluster_name == "":
            self.selection_div.text = (
                f"<b>{self.selection_info_text}</b><br>"
                "Cluster label must not be empty."
            )
            return

        cluster_color = self.cluster_color_picker.color

        cluster_col = np.array(
            self.scatter_source.data["cluster"], dtype=object)
        color_col = np.array(self.scatter_source.data["color"], dtype=object)

        for idx in selected:
            cluster_col[idx] = cluster_name
            color_col[idx] = cluster_color

        self.scatter_source.data["cluster"] = cluster_col
        self.scatter_source.data["color"] = color_col

        if cluster_name not in self.cluster_to_indices:
            self.cluster_to_indices[cluster_name] = set()
        self.cluster_to_indices[cluster_name].update(selected)
        self.cluster_to_color[cluster_name] = cluster_color

        # Ensure points do not remain registered in conflicting old clusters.
        for other_name, idx_set in self.cluster_to_indices.items():
            if other_name != cluster_name:
                idx_set.difference_update(selected)

        # Drop empty clusters.
        empty_clusters = [
            k for k, v in self.cluster_to_indices.items() if len(v) == 0]
        for k in empty_clusters:
            self.cluster_to_indices.pop(k, None)
            self.cluster_to_color.pop(k, None)

        self._refresh_cluster_widgets()
        self.selection_div.text = (
            f"<b>{self.selection_info_text}</b><br>"
            f"Assigned {len(selected)} points to cluster <b>{cluster_name}</b>."
        )

    def _on_remove_from_cluster(self) -> None:
        selected = self._selected_indices()
        if len(selected) == 0:
            self.selection_div.text = "<b>Selected points: 0</b><br>No points selected."
            return

        cluster_col = np.array(
            self.scatter_source.data["cluster"], dtype=object)
        color_col = np.array(self.scatter_source.data["color"], dtype=object)

        for idx in selected:
            cluster_col[idx] = "unassigned"
            color_col[idx] = "#9e9e9e"

        self.scatter_source.data["cluster"] = cluster_col
        self.scatter_source.data["color"] = color_col

        for idx_set in self.cluster_to_indices.values():
            idx_set.difference_update(selected)

        empty_clusters = [
            k for k, v in self.cluster_to_indices.items() if len(v) == 0]
        for k in empty_clusters:
            self.cluster_to_indices.pop(k, None)
            self.cluster_to_color.pop(k, None)

        self._refresh_cluster_widgets()
        self.selection_div.text = (
            f"<b>{self.selection_info_text}</b><br>"
            f"Removed {len(selected)} points from assigned clusters."
        )

    # ============================================================
    # Cluster comparison
    # ============================================================

    @staticmethod
    def _split_train_test_indices(n: int, test_fraction: float, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(n)
        n_test = max(1, int(round(n * test_fraction)))
        n_test = min(n_test, n - 1)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        return train_idx, test_idx

    @staticmethod
    def _raw_scores_catboost(model: CatBoostClassifier, X: np.ndarray) -> np.ndarray:
        scores = model.predict(X, prediction_type="RawFormulaVal")
        return np.asarray(scores).reshape(-1).astype(float)

    @staticmethod
    def _threshold_grid(scores: np.ndarray, max_thresholds: int = 300) -> np.ndarray:
        scores = np.asarray(scores).reshape(-1)
        unique_scores = np.unique(scores)
        if len(unique_scores) <= max_thresholds:
            return unique_scores
        qs = np.linspace(0.0, 1.0, max_thresholds)
        return np.unique(np.quantile(scores, qs))

    @classmethod
    def _find_best_threshold(
        cls,
        y_true: np.ndarray,
        scores: np.ndarray,
        max_thresholds: int = 300,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        thresholds = cls._threshold_grid(scores, max_thresholds=max_thresholds)
        f1_values = np.zeros_like(thresholds, dtype=float)

        for i, thr in enumerate(thresholds):
            pred = (scores >= thr).astype(int)
            f1_values[i] = f1_score(y_true, pred, zero_division=0)

        best_idx = int(np.nanargmax(f1_values))
        best_threshold = float(thresholds[best_idx])
        return best_threshold, thresholds, f1_values

    @staticmethod
    def _density_hist(scores: np.ndarray, bins: int = 40) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores = np.asarray(scores).reshape(-1)
        hist, edges = np.histogram(scores, bins=bins, density=True)
        left = edges[:-1]
        right = edges[1:]
        return hist, left, right

    def _cluster_matrix(self, cluster_name: str) -> np.ndarray:
        idx = sorted(self.cluster_to_indices[cluster_name])
        return self.X_numeric[idx]

    def _cluster_label_distribution(self, indices: np.ndarray) -> Dict[Any, float]:
        """
        Returns percentage distribution of original y labels inside a cluster.
        Example:
            {0: 73.5, 1: 26.5}
        """
        if self.y is None:
            return {}

        labels = np.asarray(self.y)[indices]
        if len(labels) == 0:
            return {}

        unique, counts = np.unique(labels, return_counts=True)
        total = counts.sum()

        return {
            unique_label.item() if hasattr(unique_label, "item") else unique_label: float(count / total * 100.0)
            for unique_label, count in zip(unique, counts)
        }

    @staticmethod
    def _format_label_distribution_html(
        distribution: Dict[Any, float],
        cluster_name: str,
    ) -> str:
        if not distribution:
            return f"<b>{cluster_name} label distribution:</b><br>Unavailable"

        parts = [f"<b>{cluster_name} label distribution:</b>"]
        for label, pct in sorted(distribution.items(), key=lambda x: x[0]):
            parts.append(f"class {label}: {pct:.2f}%")
        return "<br>".join(parts)

    def _on_compare_clusters(self) -> None:
        cluster_a = self.cluster_a_select.value
        cluster_b = self.cluster_b_select.value

        if cluster_a == "" or cluster_b == "":
            self.metrics_div.text = "<b>Comparison metrics</b><br>Please choose two clusters."
            return

        if cluster_a == cluster_b:
            self.metrics_div.text = "<b>Comparison metrics</b><br>Please choose two different clusters."
            return

        if cluster_a not in self.cluster_to_indices or cluster_b not in self.cluster_to_indices:
            self.metrics_div.text = "<b>Comparison metrics</b><br>One of the chosen clusters does not exist."
            return

        idx_a = np.array(sorted(self.cluster_to_indices[cluster_a]), dtype=int)
        idx_b = np.array(sorted(self.cluster_to_indices[cluster_b]), dtype=int)

        cluster_a_label_distribution = self._cluster_label_distribution(idx_a)
        cluster_b_label_distribution = self._cluster_label_distribution(idx_b)

        if len(idx_a) < 5 or len(idx_b) < 5:
            self.metrics_div.text = (
                "<b>Comparison metrics</b><br>"
                "Each cluster must contain at least 5 points."
            )
            return

        X_a = self.X_numeric[idx_a]
        X_b = self.X_numeric[idx_b]

        X_all = np.vstack([X_a, X_b])
        y_all = np.concatenate([
            np.zeros(len(X_a), dtype=int),
            np.ones(len(X_b), dtype=int),
        ])

        train_idx, test_idx = self._split_train_test_indices(
            len(X_all),
            test_fraction=self.compare_test_fraction,
            random_state=42,
        )

        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]

        model = CatBoostClassifier(
            iterations=int(self.catboost_iterations_spinner.value),
            depth=int(self.catboost_depth_spinner.value),
            learning_rate=float(self.catboost_lr_spinner.value),
            loss_function="Logloss",
            verbose=False,
            random_seed=42,
        )
        model.fit(X_train, y_train)

        scores_test = self._raw_scores_catboost(model, X_test)
        best_threshold, thresholds, f1_values = self._find_best_threshold(
            y_test,
            scores_test,
            max_thresholds=self.threshold_grid_size,
        )

        y_pred = (scores_test >= best_threshold).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, scores_test)

        scores_cluster_a = scores_test[y_test == 0]
        scores_cluster_b = scores_test[y_test == 1]

        hist_a, left_a, right_a = self._density_hist(scores_cluster_a, bins=40)
        hist_b, left_b, right_b = self._density_hist(scores_cluster_b, bins=40)

        self.hist_source_left.data = {
            "top": hist_a,
            "left": left_a,
            "right": right_a,
        }
        self.hist_source_right.data = {
            "top": hist_b,
            "left": left_b,
            "right": right_b,
        }
        self.threshold_curve_source.data = {
            "x": thresholds,
            "y": f1_values,
        }

        self.hist_left_fig.title.text = f"Score histogram: {cluster_a}"
        self.hist_right_fig.title.text = f"Score histogram: {cluster_b}"

        color_a = self.cluster_to_color.get(cluster_a, "#1f77b4")
        color_b = self.cluster_to_color.get(cluster_b, "#ff7f0e")

        self.hist_left_fig.renderers[-1].glyph.fill_color = color_a
        self.hist_left_fig.renderers[-1].glyph.line_color = color_a
        self.hist_right_fig.renderers[-1].glyph.fill_color = color_b
        self.hist_right_fig.renderers[-1].glyph.line_color = color_b

        cluster_a_dist_html = self._format_label_distribution_html(
            cluster_a_label_distribution,
            cluster_a,
        )
        cluster_b_dist_html = self._format_label_distribution_html(
            cluster_b_label_distribution,
            cluster_b,
        )

        self.metrics_div.text = (
            f"<b>Comparison metrics: {cluster_a} vs {cluster_b}</b><br>"
            f"Precision: {precision:.4f}<br>"
            f"Recall: {recall:.4f}<br>"
            f"F1: {f1:.4f}<br>"
            f"ROC AUC: {roc_auc:.4f}<br>"
            f"Optimal threshold: {best_threshold:.6f}<br>"
            f"Train size: {len(train_idx)}<br>"
            f"Test size: {len(test_idx)}<br><br>"
            f"{cluster_a_dist_html}<br><br>"
            f"{cluster_b_dist_html}"
        )

        self.last_comparison_result = ClusterComparisonResult(
            cluster_1=cluster_a,
            cluster_2=cluster_b,
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            roc_auc=float(roc_auc),
            best_threshold=float(best_threshold),
            thresholds=thresholds,
            f1_values=f1_values,
            scores_cluster_1=scores_cluster_a,
            scores_cluster_2=scores_cluster_b,
            fitted_model=model,
            cluster_1_label_distribution=cluster_a_label_distribution,
            cluster_2_label_distribution=cluster_b_label_distribution,
        )

    # ============================================================
    # Public rendering API
    # ============================================================

    def panel(self) -> pn.layout.Panel:
        left_controls = pn.Column(
            "## Cluster labeling",
            pn.pane.Bokeh(self.selection_div),
            pn.pane.Bokeh(self.cluster_name_input),
            pn.pane.Bokeh(self.cluster_color_picker),
            pn.Row(
                pn.pane.Bokeh(self.assign_cluster_button),
                pn.pane.Bokeh(self.remove_cluster_button),
            ),
            pn.pane.Bokeh(self.cluster_summary_div),
            sizing_mode="stretch_width",
        )

        right_controls = pn.Column(
            "## Cluster comparison",
            pn.Row(
                pn.pane.Bokeh(self.cluster_a_select),
                pn.pane.Bokeh(self.cluster_b_select),
            ),
            pn.Row(
                pn.pane.Bokeh(self.catboost_iterations_spinner),
                pn.pane.Bokeh(self.catboost_depth_spinner),
                pn.pane.Bokeh(self.catboost_lr_spinner),
            ),
            pn.pane.Bokeh(self.compare_button),
            pn.pane.Bokeh(self.metrics_div),
            sizing_mode="stretch_width",
        )

        controls = pn.Row(left_controls, right_controls,
                          sizing_mode="stretch_width")

        plots = pn.Column(
            pn.pane.Bokeh(self.scatter_fig),
            pn.Row(
                pn.pane.Bokeh(self.threshold_fig),
                pn.Column(
                    pn.pane.Bokeh(self.hist_left_fig),
                    pn.pane.Bokeh(self.hist_right_fig),
                ),
            ),
            sizing_mode="stretch_width",
        )

        return pn.Column(controls, plots, sizing_mode="stretch_width")

    def show(self):
        return self.panel()
