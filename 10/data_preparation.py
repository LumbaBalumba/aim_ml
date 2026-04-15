import os
from datetime import date, timedelta
import re

import polars as pl
import polars.selectors as cs

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# text hyperparams
N_SVD_COMPONENTS = 8
MIN_DF = 5
MAX_DF = 0.8
MAX_FEATURES = 5000

# product hyperparams
TOP_K_CATEGORIES = 10
PRODUCT_ACTIONS = ["view", "click", "to_cart", "favorite", "order"]
PRICE_ACTIONS = ["view", "click", "to_cart", "order"]

SESSION_GAP_SECONDS = 40 * 60


class OzonDataFormer:
    def __init__(self, data_path: str = "./predict-user-fresh-order/"):
        self.actions_history = (
            pl.scan_parquet(os.path.join(data_path, "actions_history"))
            .join(
                pl.scan_csv(os.path.join(data_path, "action_type_info.csv")),
                on="action_type_id",
            )
            .drop("action_type_id")
            .with_columns(pl.col("action_type").cast(pl.Categorical))
        )

        self.search_history = (
            pl.scan_parquet(os.path.join(data_path, "search_history"))
            .join(
                pl.scan_csv(os.path.join(data_path, "action_type_info.csv")),
                on="action_type_id",
            )
            .drop("action_type_id")
            .with_columns(pl.col("action_type").cast(pl.Categorical))
        )

        self.product_information = pl.scan_csv(
            os.path.join(data_path, "product_information.csv")
        )

        self.test_users_submission = pl.scan_csv(
            os.path.join(data_path, "test_users.csv")
        )

        self.vectorizer = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=MIN_DF,
                        max_df=MAX_DF,
                        max_features=MAX_FEATURES,
                        sublinear_tf=True,
                    ),
                ),
                ("dim_red", TruncatedSVD(n_components=N_SVD_COMPONENTS)),
            ]
        )

        self._vectorizer_is_fitted = False

    def _make_target(self, target_start: date, target_end: date) -> pl.LazyFrame:
        return (
            self.actions_history.filter(
                pl.col("timestamp").dt.date() >= target_start)
            .filter(pl.col("timestamp").dt.date() <= target_end)
            .group_by("user_id")
            .agg((pl.col("action_type") == "order").max().cast(pl.Int8).alias("target"))
        )

    def fit_text_vectorizer(self, feature_end_date: date) -> None:
        search_df = (
            self.search_history.with_columns(
                [
                    pl.col("search_query")
                    .cast(pl.String)
                    .str.to_lowercase()
                    .str.replace_all(r"ё", "е")
                    .str.replace_all(r"[^a-zа-я0-9\s]+", " ")
                    .str.replace_all(r"\s+", " ")
                    .str.strip_chars()
                    .alias("search_query_norm"),
                ]
            )
            .filter(pl.col("timestamp").dt.date() <= feature_end_date)
            .filter(
                pl.col("timestamp").dt.date() >= feature_end_date -
                timedelta(days=30)
            )
            .group_by("user_id")
            .agg(
                [
                    pl.col("search_query_norm")
                    .str.concat(" ")
                    .alias("all_queries_text_30d")
                ]
            )
            .collect(engine="streaming")
        )

        texts = (
            search_df["all_queries_text_30d"].to_list() if len(
                search_df) > 0 else [""]
        )
        self.vectorizer.fit(texts)
        self._vectorizer_is_fitted = True

    def _get_basic_features(
        self, feature_end_date: date, anchor_date: date
    ) -> dict[str, pl.LazyFrame]:
        actions_aggs = {}
        actions_to_suffix = {
            "click": "click",
            "favorite": "favorite",
            "order": "order",
            "to_cart": "to_cart",
        }

        for action_name, suf in actions_to_suffix.items():
            aggs = (
                self.actions_history.filter(
                    pl.col("timestamp").dt.date() <= feature_end_date
                )
                .filter(
                    pl.col("timestamp").dt.date()
                    >= feature_end_date - timedelta(days=120)
                )
                .filter(pl.col("action_type") == action_name)
                .join(
                    self.product_information.select(
                        "product_id", "discount_price"),
                    on="product_id",
                    how="left",
                )
                .group_by("user_id")
                .agg(
                    pl.count("product_id").
                    cast(pl.Int32).alias(f"num_products_{suf}"),
                    pl.sum("discount_price")
                    .cast(pl.Float32)
                    .alias(f"sum_discount_price_{suf}"),
                    pl.max("discount_price")
                    .cast(pl.Float32)
                    .alias(f"max_discount_price_{suf}"),
                    pl.max("timestamp").alias(f"last_{suf}_time"),
                    pl.min("timestamp").alias(f"first_{suf}_time"),
                )
                .with_columns(
                    [
                        (pl.lit(anchor_date) -
                         pl.col(f"last_{suf}_time").dt.date())
                        .dt.total_days()
                        .cast(pl.Int32)
                        .alias(f"days_since_last_{suf}"),
                        (pl.lit(anchor_date) -
                         pl.col(f"first_{suf}_time").dt.date())
                        .dt.total_days()
                        .cast(pl.Int32)
                        .alias(f"days_since_first_{suf}"),
                    ]
                )
                .select(
                    "user_id",
                    f"num_products_{suf}",
                    f"sum_discount_price_{suf}",
                    f"max_discount_price_{suf}",
                    f"days_since_last_{suf}",
                    f"days_since_first_{suf}",
                )
            )
            actions_aggs[suf] = aggs

        search_aggs = (
            self.search_history.filter(pl.col("action_type") == "search")
            .filter(pl.col("timestamp").dt.date() <= feature_end_date)
            .filter(
                pl.col("timestamp").dt.date() >= feature_end_date -
                timedelta(days=120)
            )
            .group_by("user_id")
            .agg(
                pl.count("search_query").cast(pl.Int32).alias("num_search"),
                pl.max("timestamp").alias("last_search_time"),
                pl.min("timestamp").alias("first_search_time"),
            )
            .with_columns(
                [
                    (pl.lit(anchor_date) - pl.col("last_search_time").dt.date())
                    .dt.total_days()
                    .cast(pl.Int32)
                    .alias("days_since_last_search"),
                    (pl.lit(anchor_date) - pl.col("first_search_time").dt.date())
                    .dt.total_days()
                    .cast(pl.Int32)
                    .alias("days_since_first_search"),
                ]
            )
            .select(
                "user_id",
                "num_search",
                "days_since_last_search",
                "days_since_first_search",
            )
        )

        actions_aggs["search"] = search_aggs
        return actions_aggs

    def _get_intent_features(self, feature_end_date: date) -> pl.LazyFrame:
        intent_df = (
            self.actions_history.with_columns(
                pl.col("timestamp").cast(pl.Datetime))
            .filter(pl.col("timestamp").dt.date() <= feature_end_date)
            .sort(["user_id", "timestamp"])
        )

        intent_df_7d = intent_df.filter(
            pl.col("timestamp").dt.date() >= feature_end_date -
            timedelta(days=7)
        )
        intent_df_30d = intent_df.filter(
            pl.col("timestamp").dt.date() >= feature_end_date -
            timedelta(days=30)
        )

        agg_7d = intent_df_7d.group_by("user_id").agg(
            [
                (pl.col("action_type") == "to_cart")
                .sum()
                .alias("add_to_cart_count_7d"),
                (pl.col("action_type") == "click").sum().alias(
                    "product_clicks_7d"),
            ]
        )

        agg_30d = intent_df_30d.group_by("user_id").agg(
            [
                (pl.col("action_type") == "to_cart")
                .sum()
                .alias("add_to_cart_count_30d"),
                (pl.col("action_type") == "view").sum().alias("view_count_30d"),
                (pl.col("action_type") == "search").sum().alias("search_count_30d"),
                (pl.col("action_type") == "click").sum().alias("click_count_30d"),
            ]
        )

        ratios = agg_30d.with_columns(
            [
                (
                    (pl.col("add_to_cart_count_30d") + 1)
                    / (pl.col("click_count_30d") + 1)
                ).alias("cart_to_click_ratio")
            ]
        ).select("user_id", "cart_to_click_ratio")

        df_sessions = (
            intent_df.with_columns(
                [
                    pl.col("timestamp")
                    .diff()
                    .over("user_id")
                    .dt.total_seconds()
                    .alias("time_diff_sec")
                ]
            )
            .with_columns(
                [
                    (
                        (pl.col("time_diff_sec") > SESSION_GAP_SECONDS)
                        .fill_null(True)
                        .cast(pl.Int8)
                    ).alias("new_session_flag")
                ]
            )
            .with_columns(
                [
                    pl.col("new_session_flag")
                    .cum_sum()
                    .over("user_id")
                    .alias("session_id")
                ]
            )
        )

        session_stats = df_sessions.group_by(["user_id", "session_id"]).agg(
            pl.len().alias("session_len")
        )

        session_agg = session_stats.group_by("user_id").agg(
            [
                pl.col("session_len").mean().alias("avg_actions_per_session"),
                pl.col("session_len").max().alias("max_actions_per_session"),
            ]
        )

        last_actions = (
            intent_df.group_by("user_id")
            .agg(pl.col("action_type").tail(3).alias("last_3_actions"))
            .with_columns(
                [
                    pl.col("last_3_actions").list.len().alias(
                        "last_actions_count"),
                    pl.col("last_3_actions")
                    .list.get(-1, null_on_oob=True)
                    .fill_null("NONE")
                    .cast(pl.Categorical)
                    .alias("last_action"),
                    pl.col("last_3_actions")
                    .list.get(-2, null_on_oob=True)
                    .fill_null("NONE")
                    .cast(pl.Categorical)
                    .alias("last_2_action"),
                    pl.col("last_3_actions")
                    .list.get(-3, null_on_oob=True)
                    .fill_null("NONE")
                    .cast(pl.Categorical)
                    .alias("last_3_action"),
                ]
            )
            .with_columns(
                [
                    pl.concat_str(
                        ["last_3_action", "last_2_action", "last_action"],
                        separator="__",
                    ).cast(pl.Categorical).alias("last_3_actions_str")
                ]
            )
            .drop("last_3_actions")
        )

        return (
            agg_7d.join(agg_30d, on="user_id", how="full", coalesce=True)
            .join(ratios, on="user_id", how="full", coalesce=True)
            .join(session_agg, on="user_id", how="full", coalesce=True)
            .join(last_actions, on="user_id", how="full", coalesce=True)
        )

    def _get_search_features(self, feature_end_date: date) -> pl.LazyFrame:
        PRODUCT_KEYWORDS = [
            "молоко",
            "сыр",
            "творог",
            "йогурт",
            "кефир",
            "батон",
            "хлеб",
            "яйца",
            "курица",
            "банан",
            "яблок",
            "кофе",
            "чай",
            "шоколад",
            "вода",
            "колбас",
        ]

        def normalize_query_expr(col_name: str = "search_query") -> pl.Expr:
            return (
                pl.col(col_name)
                .cast(pl.String)
                .str.to_lowercase()
                .str.replace_all(r"ё", "е")
                .str.replace_all(r"[^a-zа-я0-9\s]+", " ")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
            )

        def build_keyword_flag_exprs(
            query_col: str, prefixes: list[str], prefix_name: str
        ) -> list[pl.Expr]:
            exprs = []
            for kw in prefixes:
                safe = re.sub(r"[^a-zA-Zа-яА-Я0-9]+",
                              "_", kw.lower()).strip("_")
                exprs.append(
                    pl.col(query_col)
                    .str.contains(re.escape(kw), literal=False)
                    .fill_null(False)
                    .cast(pl.Int8)
                    .alias(f"{prefix_name}_{safe}_flag")
                )
            return exprs

        search_df = (
            self.search_history.with_columns(
                [
                    pl.col("timestamp").cast(pl.Datetime),
                    normalize_query_expr("search_query").alias(
                        "search_query_norm"),
                ]
            )
            .filter(pl.col("timestamp").dt.date() <= feature_end_date)
            .sort(["user_id", "timestamp"])
        )

        query_level = search_df.with_columns(
            [
                pl.col("search_query_norm").str.len_chars().alias(
                    "query_len_chars"),
                pl.col("search_query_norm")
                .str.split(" ")
                .list.len()
                .alias("query_len_words"),
            ]
            + build_keyword_flag_exprs("search_query_norm",
                                       PRODUCT_KEYWORDS, "product")
        )

        query_level_7d = query_level.filter(
            pl.col("timestamp").dt.date() >= feature_end_date -
            timedelta(days=7)
        )
        query_level_30d = query_level.filter(
            pl.col("timestamp").dt.date() >= feature_end_date -
            timedelta(days=30)
        )

        basic_30d = query_level_30d.group_by("user_id").agg(
            [
                pl.len().alias("search_count_30d"),
                pl.col("search_query_norm")
                .n_unique().alias(
                    "unique_queries_30d"),
                pl.col("query_len_chars")
                .mean().alias("avg_query_len_chars_30d"),
                pl.col("query_len_chars")
                .median().alias("median_query_len_chars_30d"),
                pl.col("query_len_words")
                .mean()
                .alias("avg_query_len_words_30d"),
                pl.col("query_len_words")
                .max()
                .alias("max_query_len_words_30d"),
            ]
        )

        basic_7d = query_level_7d.group_by("user_id").agg(
            [
                pl.len().alias("search_count_7d"),
                pl.col("search_query_norm")
                .n_unique()
                .alias("unique_queries_7d"),
            ]
        )

        recency = search_df.group_by("user_id").agg(
            [
                (pl.lit(feature_end_date) - pl.col("timestamp").max().dt.date())
                .dt.total_days()
                .alias("days_since_last_search")
            ]
        )

        product_flag_cols = [
            f"product_{re.sub(r'[^a-zA-Zа-яА-Я0-9]+', '_', kw.lower()).strip('_')}_flag"
            for kw in PRODUCT_KEYWORDS
        ]

        keyword_aggs = []
        for col in product_flag_cols:
            keyword_aggs.extend(
                [
                    pl.col(col).sum().alias(f"{col}_count_30d"),
                    pl.col(col).mean().alias(f"{col}_rate_30d"),
                ]
            )

        keyword_features = query_level_30d.group_by(
            "user_id").agg(keyword_aggs)

        keyword_summary = query_level_30d.group_by("user_id").agg(
            [
                pl.sum_horizontal(product_flag_cols)
                .sum()
                .alias("product_keyword_hits_30d"),
                (pl.sum_horizontal(product_flag_cols) > 0)
                .sum()
                .alias("queries_with_product_keywords_30d"),
            ]
        )

        user_docs = (
            query_level_30d.group_by("user_id")
            .agg(
                [
                    pl.col("search_query_norm")
                    .str.concat(" ")
                    .alias("all_queries_text_30d")
                ]
            )
            .collect(engine="streaming")
        )

        if not self._vectorizer_is_fitted:
            raise RuntimeError(
                "Text vectorizer is not fitted. Call fit_text_vectorizer(...) first."
            )

        if len(user_docs) > 0:
            user_docs_pd = user_docs.to_pandas()
            X_vec = self.vectorizer.transform(
                user_docs_pd["all_queries_text_30d"].fillna("")
            )

            svd_cols = [
                f"search_tfidf_svd_{i:02d}" for i in range(X_vec.shape[1])]
            tfidf_svd_features = (
                pl.DataFrame(
                    {
                        "user_id": user_docs_pd["user_id"].tolist(),
                        **{col: X_vec[:, i].tolist() for i, col in enumerate(svd_cols)},
                    }
                )
                .with_columns(pl.col("user_id").cast(pl.Int32))
                .lazy()
            )
        else:
            tfidf_svd_features = pl.LazyFrame({"user_id": []})

        last_query_features = (
            query_level.group_by("user_id").agg(
                [
                    pl.col("search_query_norm")
                    .last()
                    .alias("last_search_query"),
                    pl.col("query_len_chars")
                    .last()
                    .alias("last_search_len_chars"),
                    pl.col("query_len_words")
                    .last()
                    .alias("last_search_len_words"),
                ]
            )
        ).with_columns(
            [
                *[
                    pl.col("last_search_query")
                    .str.contains(re.escape(kw), literal=False)
                    .fill_null(False)
                    .cast(pl.Int8)
                    .alias(
                        f"last_query_product_{re.sub(r'[^a-zA-Zа-яА-Я0-9]+', '_', kw.lower()).strip('_')}"
                    )
                    for kw in PRODUCT_KEYWORDS
                ]
            ]
        ).drop('last_search_query')

        return (
            basic_30d.join(recency, on="user_id", how="full", coalesce=True)
            .join(keyword_features, on="user_id", how="full", coalesce=True)
            .join(keyword_summary, on="user_id", how="full", coalesce=True)
            .join(tfidf_svd_features, on="user_id", how="full", coalesce=True)
            .join(last_query_features, on="user_id", how="full", coalesce=True)
            .join(basic_7d, on="user_id", how="full", coalesce=True)
        )

    def _get_price_features(self, feature_end_date: date) -> pl.LazyFrame:
        actions_df = (
            self.actions_history.filter(
                pl.col("timestamp").dt.date() <= feature_end_date
            )
            .filter(
                pl.col("timestamp").dt.date() >= feature_end_date -
                timedelta(days=120)
            )
            .filter(pl.col("product_id").is_not_null())
        )

        products_df = self.product_information.with_columns(
            [
                pl.col("price").cast(pl.Float64),
                pl.col("discount_price").cast(pl.Float64),
                pl.col("category_id").cast(pl.Int64),
                pl.col("category_name").cast(pl.Categorical),
                pl.col("brand").cast(pl.Categorical),
                pl.col("type").cast(pl.Categorical),
            ]
        ).select(
            [
                "product_id",
                "brand",
                "type",
                "category_id",
                "category_name",
                "price",
                "discount_price",
            ]
        )

        events = actions_df.join(products_df, on="product_id", how="left").with_columns(
            [
                pl.when(
                    pl.col("discount_price").is_not_null()
                    & (pl.col("discount_price") > 0)
                )
                .then(pl.col("discount_price"))
                .otherwise(pl.col("price"))
                .alias("effective_price"),
                pl.when(
                    pl.col("price").is_not_null()
                    & (pl.col("price") > 0)
                    & pl.col("discount_price").is_not_null()
                    & (pl.col("discount_price") >= 0)
                    & (pl.col("discount_price") <= pl.col("price"))
                )
                .then((pl.col("price") - pl.col("discount_price")) / pl.col("price"))
                .otherwise(0.0)
                .alias("discount_ratio"),
                pl.when(
                    pl.col("price").is_not_null()
                    & pl.col("discount_price").is_not_null()
                    & (pl.col("price") > pl.col("discount_price"))
                )
                .then(1)
                .otherwise(0)
                .cast(pl.Int8)
                .alias("is_discounted"),
            ]
        )

        product_events = events.filter(
            pl.col("action_type").is_in(PRODUCT_ACTIONS))
        price_events = events.filter(
            pl.col("action_type").is_in(PRICE_ACTIONS))

        price_features = price_events.group_by("user_id").agg(
            [
                pl.col("effective_price")
                .mean()
                .alias("avg_viewed_price"),
                pl.col("effective_price")
                .median()
                .alias("median_viewed_price"),
                pl.col("effective_price")
                .max()
                .alias("max_viewed_price"),
                pl.col("discount_ratio")
                .mean()
                .alias("discount_ratio_mean"),
                pl.col("is_discounted")
                .mean().alias("fraction_discounted_products"),
            ]
        )

        user_category_counts = (
            product_events.filter(pl.col("category_id").is_not_null())
            .group_by(["user_id", "category_id", "category_name"])
            .agg(pl.len().alias("category_event_count"))
        )

        user_category_totals = (
            user_category_counts.group_by("user_id")
            .agg(
                [
                    pl.col("category_event_count").sum().alias(
                        "total_category_events"),
                    pl.col("category_event_count")
                    .max()
                    .alias("favorite_category_count"),
                    pl.len().alias("unique_categories"),
                ]
            )
            .with_columns(
                [
                    (
                        pl.col("favorite_category_count")
                        / pl.col("total_category_events")
                    ).alias("favorite_category_share")
                ]
            )
        )

        category_entropy = (
            user_category_counts.join(
                user_category_totals.select(
                    ["user_id", "total_category_events"]),
                on="user_id",
                how="left",
            )
            .with_columns(
                [
                    (
                        pl.col("category_event_count") /
                        pl.col("total_category_events")
                    ).alias("p_cat")
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("p_cat") > 0)
                    .then(-pl.col("p_cat") * pl.col("p_cat").log())
                    .otherwise(0.0)
                    .alias("entropy_term")
                ]
            )
            .group_by("user_id")
            .agg(pl.col("entropy_term").sum().alias("category_entropy"))
        )

        top_categories = (
            product_events.filter(pl.col("category_id").is_not_null())
            .group_by(["category_id", "category_name"])
            .agg(pl.len().alias("global_category_count"))
            .sort("global_category_count", descending=True)
            .head(TOP_K_CATEGORIES)
            .collect(engine="streaming")
        )

        top_category_ids = top_categories["category_id"].to_list()

        user_top_category_features = (
            product_events.filter(
                pl.col("category_id").is_in(top_category_ids))
            .group_by(["user_id", "category_id"])
            .agg(pl.len().alias("cnt"))
            .join(
                user_category_totals.select(
                    ["user_id", "total_category_events"]),
                on="user_id",
                how="left",
            )
            .with_columns(
                [(pl.col("cnt") / pl.col("total_category_events")).alias("share")]
            )
            .collect(engine="streaming")
            .pivot(
                values="share",
                index="user_id",
                on="category_id",
                aggregate_function="first",
            )
            # .fill_null(0)
        )

        rename_map = {
            col: f"top_category_share_{col}"
            for col in user_top_category_features.columns
            if col != "user_id"
        }

        user_top_category_features = user_top_category_features.rename(
            rename_map
        ).lazy()

        category_features = user_category_totals.join(
            category_entropy, on="user_id", how="full", coalesce=True
        ).join(user_top_category_features, on="user_id", how="full", coalesce=True)

        return price_features.join(
            category_features, on="user_id", how="full", coalesce=True
        )

    def _get_trend_features(self, feature_end_date: date) -> pl.LazyFrame:
        actions_df = (
            self.actions_history
            .filter(pl.col("timestamp").dt.date() <= feature_end_date)
            .filter(pl.col("timestamp").dt.date() >= feature_end_date - timedelta(days=60))
        )

        def _agg_for_window(start_date: date, end_date: date, suffix: str) -> pl.LazyFrame:
            window_df = actions_df.filter(
                (pl.col("timestamp").dt.date() >= start_date) &
                (pl.col("timestamp").dt.date() <= end_date)
            )

            return window_df.group_by("user_id").agg([
                pl.len().alias(f"actions_count_{suffix}"),
                (pl.col("action_type") == "click").sum().alias(
                    f"click_count_{suffix}"),
                (pl.col("action_type") == "to_cart").sum().alias(
                    f"to_cart_count_{suffix}"),
                (pl.col("action_type") == "favorite").sum().alias(
                    f"favorite_count_{suffix}"),
                (pl.col("action_type") == "order").sum().alias(
                    f"order_count_{suffix}"),
                pl.col("product_id").n_unique().alias(
                    f"unique_products_{suffix}"),
            ])

        last_7_start = feature_end_date - timedelta(days=6)
        prev_7_start = feature_end_date - timedelta(days=13)
        prev_7_end = feature_end_date - timedelta(days=7)

        last_30_start = feature_end_date - timedelta(days=29)
        prev_30_start = feature_end_date - timedelta(days=59)
        prev_30_end = feature_end_date - timedelta(days=30)

        last_7 = _agg_for_window(last_7_start, feature_end_date, "last_7d")
        prev_7 = _agg_for_window(prev_7_start, prev_7_end, "prev_7d")
        last_30 = _agg_for_window(last_30_start, feature_end_date, "last_30d")
        prev_30 = _agg_for_window(prev_30_start, prev_30_end, "prev_30d")

        trend = (
            last_7
            .join(prev_7, on="user_id", how="full", coalesce=True)
            .join(last_30, on="user_id", how="full", coalesce=True)
            .join(prev_30, on="user_id", how="full", coalesce=True)
            .with_columns([
                (pl.col("actions_count_last_7d") -
                 pl.col("actions_count_prev_7d")).alias("actions_trend_delta_7d"),
                ((pl.col("actions_count_last_7d") + 1) /
                 (pl.col("actions_count_prev_7d") + 1)).alias("actions_trend_ratio_7d"),

                (pl.col("click_count_last_7d") -
                 pl.col("click_count_prev_7d")).alias("click_trend_delta_7d"),
                ((pl.col("click_count_last_7d") + 1) /
                 (pl.col("click_count_prev_7d") + 1)).alias("click_trend_ratio_7d"),

                (pl.col("to_cart_count_last_7d") -
                 pl.col("to_cart_count_prev_7d")).alias("to_cart_trend_delta_7d"),
                ((pl.col("to_cart_count_last_7d") + 1) /
                 (pl.col("to_cart_count_prev_7d") + 1)).alias("to_cart_trend_ratio_7d"),

                (pl.col("order_count_last_30d") -
                 pl.col("order_count_prev_30d")).alias("order_trend_delta_30d"),
                ((pl.col("order_count_last_30d") + 1) /
                 (pl.col("order_count_prev_30d") + 1)).alias("order_trend_ratio_30d"),

                (pl.col("unique_products_last_7d") - pl.col("unique_products_prev_7d")
                 ).alias("unique_products_trend_delta_7d"),
                ((pl.col("unique_products_last_7d") + 1) /
                 (pl.col("unique_products_prev_7d") + 1)).alias("unique_products_trend_ratio_7d"),
            ])
        )

        search_df = (
            self.search_history
            .filter(pl.col("timestamp").dt.date() <= feature_end_date)
            .filter(pl.col("timestamp").dt.date() >= feature_end_date - timedelta(days=60))
        )

        def _search_agg_for_window(start_date: date, end_date: date, suffix: str) -> pl.LazyFrame:
            window_df = search_df.filter(
                (pl.col("timestamp").dt.date() >= start_date) &
                (pl.col("timestamp").dt.date() <= end_date)
            )
            return window_df.group_by("user_id").agg([
                pl.len().alias(f"search_count_{suffix}"),
                pl.col("search_query").n_unique().alias(
                    f"unique_searches_{suffix}"),
            ])

        search_last_7 = _search_agg_for_window(
            last_7_start, feature_end_date, "last_7d")
        search_prev_7 = _search_agg_for_window(
            prev_7_start, prev_7_end, "prev_7d")

        search_trend = (
            search_last_7
            .join(search_prev_7, on="user_id", how="full", coalesce=True)
            .with_columns([
                (pl.col("search_count_last_7d") -
                 pl.col("search_count_prev_7d")).alias("search_trend_delta_7d"),
                ((pl.col("search_count_last_7d") + 1) /
                 (pl.col("search_count_prev_7d") + 1)).alias("search_trend_ratio_7d"),
                (pl.col("unique_searches_last_7d") - pl.col("unique_searches_prev_7d")
                 ).alias("unique_searches_trend_delta_7d"),
            ])
        )

        return trend.join(search_trend, on="user_id", how="full", coalesce=True)

    def _get_search_action_conversion_features(self, feature_end_date: date) -> pl.LazyFrame:
        search_events = (
            self.search_history
            .filter(pl.col("timestamp").dt.date() <= feature_end_date)
            .filter(pl.col("timestamp").dt.date() >= feature_end_date - timedelta(days=30))
            .select([
                "user_id",
                pl.col("timestamp").cast(pl.Datetime),
                pl.lit("search").alias("event_kind"),
                pl.lit(None).cast(pl.String).alias("action_kind"),
            ])
        )

        action_events = (
            self.actions_history
            .filter(pl.col("timestamp").dt.date() <= feature_end_date)
            .filter(pl.col("timestamp").dt.date() >= feature_end_date - timedelta(days=30))
            .filter(pl.col("action_type").is_in(["click", "to_cart", "favorite", "order"]))
            .select([
                "user_id",
                pl.col("timestamp").cast(pl.Datetime),
                pl.lit("action").alias("event_kind"),
                pl.col("action_type").cast(pl.String).alias("action_kind"),
            ])
        )

        union_events = (
            pl.concat([search_events, action_events], how="vertical")
            .sort(["user_id", "timestamp"])
            .with_columns([
                pl.when(pl.col("event_kind") == "search")
                .then(pl.col("timestamp"))
                .otherwise(None)
                .alias("search_ts_raw")
            ])
            .with_columns([
                pl.col("search_ts_raw").forward_fill().over(
                    "user_id").alias("last_search_ts")
            ])
            .with_columns([
                (pl.col("timestamp") - pl.col("last_search_ts")
                 ).dt.total_seconds().alias("secs_since_last_search")
            ])
        )

        action_after_search = (
            union_events
            .filter(pl.col("event_kind") == "action")
            .filter(pl.col("last_search_ts").is_not_null())
            .filter(pl.col("secs_since_last_search") >= 0)
            .with_columns([
                (pl.col("secs_since_last_search") <= 300).cast(
                    pl.Int8).alias("within_5m"),
                (pl.col("secs_since_last_search") <= 1800).cast(
                    pl.Int8).alias("within_30m"),
                (pl.col("secs_since_last_search") <= 3600).cast(
                    pl.Int8).alias("within_60m"),
            ])
        )

        search_counts = search_events.group_by("user_id").agg([
            pl.len().alias("search_events_30d")
        ])

        conv = action_after_search.group_by("user_id").agg([
            ((pl.col("action_kind") == "click") & (pl.col("within_5m") == 1)
             ).sum().alias("search_to_click_5m_count"),
            ((pl.col("action_kind") == "click") & (pl.col("within_30m") == 1)
             ).sum().alias("search_to_click_30m_count"),
            ((pl.col("action_kind") == "click") & (pl.col("within_60m") == 1)
             ).sum().alias("search_to_click_60m_count"),

            ((pl.col("action_kind") == "to_cart") & (pl.col("within_5m") == 1)
             ).sum().alias("search_to_cart_5m_count"),
            ((pl.col("action_kind") == "to_cart") & (pl.col("within_30m") == 1)
             ).sum().alias("search_to_cart_30m_count"),
            ((pl.col("action_kind") == "to_cart") & (pl.col("within_60m") == 1)
             ).sum().alias("search_to_cart_60m_count"),

            pl.col("secs_since_last_search").filter(pl.col("action_kind")
                                                    == "click").mean().alias("mean_secs_search_to_click"),
            pl.col("secs_since_last_search").filter(pl.col("action_kind")
                                                    == "to_cart").mean().alias("mean_secs_search_to_cart"),
        ])

        return (
            search_counts
            .join(conv, on="user_id", how="full", coalesce=True)
            .with_columns([
                ((pl.col("search_to_click_30m_count") + 1) /
                 (pl.col("search_events_30d") + 1)).alias("search_to_click_30m_rate"),
                ((pl.col("search_to_cart_30m_count") + 1) /
                 (pl.col("search_events_30d") + 1)).alias("search_to_cart_30m_rate"),
            ])
        )

    def _get_repeat_loyalty_features(self, feature_end_date: date) -> pl.LazyFrame:
        actions_df = (
            self.actions_history
            .filter(pl.col("timestamp").dt.date() <= feature_end_date)
            .filter(pl.col("timestamp").dt.date() >= feature_end_date - timedelta(days=120))
            .filter(pl.col("product_id").is_not_null())
            .join(
                self.product_information.select(
                    ["product_id", "brand", "category_id"]),
                on="product_id",
                how="left",
            )
            .select(["user_id", "product_id", "brand", "category_id", "action_type"])
        )

        product_counts = (
            actions_df
            .group_by(["user_id", "product_id"])
            .agg(pl.len().alias("product_cnt"))
        )

        product_loyalty = product_counts.group_by("user_id").agg([
            pl.len().alias("unique_products_120d"),
            pl.col("product_cnt").sum().alias("total_product_events_120d"),
            (pl.col("product_cnt") > 1).sum().alias("repeat_products_120d"),
            pl.col("product_cnt").max().alias("favorite_product_cnt"),
        ]).with_columns([
            (pl.col("repeat_products_120d") /
             (pl.col("unique_products_120d") + 1)).alias("repeat_product_rate"),
            (pl.col("favorite_product_cnt") / pl.col("total_product_events_120d")
             ).alias("favorite_product_share"),
        ])

        category_counts = (
            actions_df
            .filter(pl.col("category_id").is_not_null())
            .group_by(["user_id", "category_id"])
            .agg(pl.len().alias("category_cnt"))
        )

        category_loyalty = category_counts.group_by("user_id").agg([
            pl.len().alias("unique_categories_120d"),
            pl.col("category_cnt").sum().alias("total_category_events_120d"),
            (pl.col("category_cnt") > 1).sum().alias("repeat_categories_120d"),
            pl.col("category_cnt").max().alias("favorite_category_cnt_120d"),
        ]).with_columns([
            (pl.col("repeat_categories_120d") /
             (pl.col("unique_categories_120d") + 1)).alias("repeat_category_rate"),
            (pl.col("favorite_category_cnt_120d") / pl.col("total_category_events_120d")
             ).alias("favorite_category_share_120d"),
        ])

        brand_counts = (
            actions_df
            .filter(pl.col("brand").is_not_null() & (pl.col("brand") != ""))
            .group_by(["user_id", "brand"])
            .agg(pl.len().alias("brand_cnt"))
        )

        brand_loyalty = brand_counts.group_by("user_id").agg([
            pl.len().alias("unique_brands_120d"),
            pl.col("brand_cnt").sum().alias("total_brand_events_120d"),
            (pl.col("brand_cnt") > 1).sum().alias("repeat_brands_120d"),
            pl.col("brand_cnt").max().alias("favorite_brand_cnt"),
        ]).with_columns([
            (pl.col("repeat_brands_120d") /
             (pl.col("unique_brands_120d") + 1)).alias("repeat_brand_rate"),
            (pl.col("favorite_brand_cnt") / pl.col("total_brand_events_120d")
             ).alias("favorite_brand_share"),
        ])

        per_action = (
            actions_df
            .group_by(["user_id", "action_type"])
            .agg([
                pl.col("product_id").n_unique().alias(
                    "unique_products_by_action"),
                pl.len().alias("events_by_action"),
            ])
            .with_columns([
                ((pl.col("events_by_action") - pl.col("unique_products_by_action")
                  ) / (pl.col("events_by_action") + 1))
                .alias("repeat_share_by_action")
            ])
            .collect(engine="streaming")
            .pivot(
                values="repeat_share_by_action",
                index="user_id",
                on="action_type",
                aggregate_function="first",
            )
            # .fill_null(0)
        )

        per_action = per_action.rename({
            c: f"repeat_share_action_{c}"
            for c in per_action.columns if c != "user_id"
        }).lazy()

        return (
            product_loyalty
            .join(category_loyalty, on="user_id", how="full", coalesce=True)
            .join(brand_loyalty, on="user_id", how="full", coalesce=True)
            .join(per_action, on="user_id", how="full", coalesce=True)
        )

    def _get_widget_features(self, feature_end_date: date, top_k_widgets: int = 20) -> pl.LazyFrame:
        actions_df = (
            self.actions_history
            .filter(pl.col("timestamp").dt.date() <= feature_end_date)
            .filter(pl.col("timestamp").dt.date() >= feature_end_date - timedelta(days=30))
            .select(["user_id", "widget_name_id", "timestamp", "action_type"])
        )

        basic = actions_df.group_by("user_id").agg([
            pl.len().alias("widget_events_30d"),
            pl.col("widget_name_id").n_unique().alias("unique_widgets_30d"),
            pl.col("widget_name_id").drop_nulls().mode(
            ).first().alias("favorite_widget_name_id"),
            pl.col("widget_name_id").drop_nulls(
            ).last().alias("last_widget_name_id"),
        ])

        widget_counts = (
            actions_df
            .filter(pl.col("widget_name_id").is_not_null())
            .group_by(["user_id", "widget_name_id"])
            .agg(pl.len().alias("widget_cnt"))
        )

        widget_totals = widget_counts.group_by("user_id").agg([
            pl.col("widget_cnt").sum().alias("total_widget_cnt"),
            pl.col("widget_cnt").max().alias("favorite_widget_cnt"),
        ]).with_columns([
            (pl.col("favorite_widget_cnt") /
             pl.col("total_widget_cnt")).alias("favorite_widget_share")
        ])

        widget_entropy = (
            widget_counts
            .join(widget_totals.select(["user_id", "total_widget_cnt"]), on="user_id", how="left")
            .with_columns([
                (pl.col("widget_cnt") / pl.col("total_widget_cnt")).alias("p_widget")
            ])
            .with_columns([
                pl.when(pl.col("p_widget") > 0)
                .then(-pl.col("p_widget") * pl.col("p_widget").log())
                .otherwise(0.0)
                .alias("entropy_term")
            ])
            .group_by("user_id")
            .agg(pl.col("entropy_term").sum().alias("widget_entropy"))
        )

        top_widgets = (
            actions_df
            .filter(pl.col("widget_name_id").is_not_null())
            .group_by("widget_name_id")
            .agg(pl.len().alias("global_widget_cnt"))
            .sort("global_widget_cnt", descending=True)
            .head(top_k_widgets)
            .collect(engine="streaming")
        )

        top_widget_ids = top_widgets["widget_name_id"].to_list()

        widget_top_shares_df = (
            widget_counts
            .filter(pl.col("widget_name_id").is_in(top_widget_ids))
            .join(widget_totals.select(["user_id", "total_widget_cnt"]), on="user_id", how="left")
            .with_columns([
                (pl.col("widget_cnt") / pl.col("total_widget_cnt")).alias("share")
            ])
            .collect(engine="streaming")
            .pivot(
                values="share",
                index="user_id",
                on="widget_name_id",
                aggregate_function="first",
            )
            # .fill_null(0)
        )

        widget_top_shares = widget_top_shares_df.rename({
            c: f"top_widget_share_{c}"
            for c in widget_top_shares_df.columns if c != "user_id"
        }).lazy()

        return (
            basic
            .join(widget_totals, on="user_id", how="full", coalesce=True)
            .join(widget_entropy, on="user_id", how="full", coalesce=True)
            .join(widget_top_shares, on="user_id", how="full", coalesce=True)
        )

    def construct_dataset(
        self,
        feature_end_date: date,
        target_start_date: date | None = None,
        target_end_date: date | None = None,
        users_df: pl.LazyFrame | None = None,
        include_actions: bool = False,
        include_search: bool = False,
        include_price: bool = False,
        include_trend: bool = False,
        include_action_conversion: bool = False,
        include_repeat_loyalty: bool = False,
        include_widget: bool = False,
        group_embedding: bool = False
    ) -> pl.DataFrame:
        if target_start_date is not None and target_end_date is not None:
            df = self._make_target(target_start_date, target_end_date)
        elif users_df is not None:
            df = users_df
        else:
            raise ValueError(
                "Either target dates or users_df must be provided.")

        aggs = self._get_basic_features(
            feature_end_date=feature_end_date, anchor_date=feature_end_date
        )

        for _, actions_aggs_df in aggs.items():
            df = df.join(actions_aggs_df, on="user_id", how="left")

        if include_actions:
            df = df.join(
                self._get_intent_features(feature_end_date), on="user_id", how="left"
            )

        if include_search:
            df = df.join(
                self._get_search_features(feature_end_date), on="user_id", how="left"
            )

        if include_price:
            df = df.join(
                self._get_price_features(feature_end_date), on="user_id", how="left"
            )

        if include_trend:
            df = df.join(
                self._get_trend_features(feature_end_date), on="user_id", how="left"
            )

        if include_action_conversion:
            df = df.join(
                self._get_search_action_conversion_features(feature_end_date), on="user_id", how="left"
            )

        if include_repeat_loyalty:
            df = df.join(
                self._get_repeat_loyalty_features(feature_end_date), on="user_id", how="left"
            )

        if include_widget:
            df = df.join(
                self._get_widget_features(feature_end_date), on="user_id", how="left"
            )

        df = df.with_columns(
            # cs.numeric().fill_null(strategy="zero"),
            cs.string().fill_null(""),
            cs.categorical().fill_null("NONE"),
        )

        if group_embedding:
            df = df.with_columns(
                pl.concat_list([f"search_tfidf_svd_{i:02d}"
                                for i in range(N_SVD_COMPONENTS)]).alias('search_emb')
            )

        return df.collect(engine="streaming").to_pandas().sort_index(axis=1)
