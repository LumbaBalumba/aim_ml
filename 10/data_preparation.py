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
            self.actions_history.filter(pl.col("timestamp").dt.date() >= target_start)
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
                pl.col("timestamp").dt.date() >= feature_end_date - timedelta(days=30)
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
            search_df["all_queries_text_30d"].to_list() if len(search_df) > 0 else [""]
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
                    self.product_information.select("product_id", "discount_price"),
                    on="product_id",
                    how="left",
                )
                .group_by("user_id")
                .agg(
                    pl.count("product_id").cast(pl.Int32).alias(f"num_products_{suf}"),
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
                        (pl.lit(anchor_date) - pl.col(f"last_{suf}_time").dt.date())
                        .dt.total_days()
                        .cast(pl.Int32)
                        .alias(f"days_since_last_{suf}"),
                        (pl.lit(anchor_date) - pl.col(f"first_{suf}_time").dt.date())
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
                pl.col("timestamp").dt.date() >= feature_end_date - timedelta(days=120)
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
            self.actions_history.with_columns(pl.col("timestamp").cast(pl.Datetime))
            .filter(pl.col("timestamp").dt.date() <= feature_end_date)
            .sort(["user_id", "timestamp"])
        )

        intent_df_7d = intent_df.filter(
            pl.col("timestamp").dt.date() >= feature_end_date - timedelta(days=7)
        )
        intent_df_30d = intent_df.filter(
            pl.col("timestamp").dt.date() >= feature_end_date - timedelta(days=30)
        )

        agg_7d = intent_df_7d.group_by("user_id").agg(
            [
                (pl.col("action_type") == "to_cart")
                .sum()
                .alias("add_to_cart_count_7d"),
                (pl.col("action_type") == "click").sum().alias("product_clicks_7d"),
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
                    pl.col("last_3_actions").list.len().alias("last_actions_count"),
                    pl.col("last_3_actions")
                    .list.get(-1, null_on_oob=True)
                    .fill_null("NONE")
                    .alias("last_action"),
                    pl.col("last_3_actions")
                    .list.get(-2, null_on_oob=True)
                    .fill_null("NONE")
                    .alias("last_2_action"),
                    pl.col("last_3_actions")
                    .list.get(-3, null_on_oob=True)
                    .fill_null("NONE")
                    .alias("last_3_action"),
                ]
            )
            .with_columns(
                [
                    pl.concat_str(
                        ["last_3_action", "last_2_action", "last_action"],
                        separator="__",
                    ).alias("last_3_actions_str")
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
                safe = re.sub(r"[^a-zA-Zа-яА-Я0-9]+", "_", kw.lower()).strip("_")
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
                    normalize_query_expr("search_query").alias("search_query_norm"),
                ]
            )
            .filter(pl.col("timestamp").dt.date() <= feature_end_date)
            .sort(["user_id", "timestamp"])
        )

        query_level = search_df.with_columns(
            [
                pl.col("search_query_norm").str.len_chars().alias("query_len_chars"),
                pl.col("search_query_norm")
                .str.split(" ")
                .list.len()
                .alias("query_len_words"),
            ]
            + build_keyword_flag_exprs("search_query_norm", PRODUCT_KEYWORDS, "product")
        )

        query_level_7d = query_level.filter(
            pl.col("timestamp").dt.date() >= feature_end_date - timedelta(days=7)
        )
        query_level_30d = query_level.filter(
            pl.col("timestamp").dt.date() >= feature_end_date - timedelta(days=30)
        )

        basic_30d = query_level_30d.group_by("user_id").agg(
            [
                pl.len().alias("search_count_30d"),
                pl.col("search_query_norm").n_unique().alias("unique_queries_30d"),
                pl.col("query_len_chars").mean().alias("avg_query_len_chars_30d"),
                pl.col("query_len_chars").median().alias("median_query_len_chars_30d"),
                pl.col("query_len_words").mean().alias("avg_query_len_words_30d"),
                pl.col("query_len_words").max().alias("max_query_len_words_30d"),
            ]
        )

        basic_7d = query_level_7d.group_by("user_id").agg(
            [
                pl.len().alias("search_count_7d"),
                pl.col("search_query_norm").n_unique().alias("unique_queries_7d"),
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

        keyword_features = query_level_30d.group_by("user_id").agg(keyword_aggs)

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

            svd_cols = [f"search_tfidf_svd_{i:02d}" for i in range(X_vec.shape[1])]
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
                    pl.col("search_query_norm").last().alias("last_search_query"),
                    pl.col("query_len_chars").last().alias("last_search_len_chars"),
                    pl.col("query_len_words").last().alias("last_search_len_words"),
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
        )

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
                pl.col("timestamp").dt.date() >= feature_end_date - timedelta(days=120)
            )
            .filter(pl.col("product_id").is_not_null())
        )

        products_df = self.product_information.with_columns(
            [
                pl.col("price").cast(pl.Float64),
                pl.col("discount_price").cast(pl.Float64),
                pl.col("category_id").cast(pl.Int64),
                pl.col("category_name").cast(pl.String),
                pl.col("brand").cast(pl.String),
                pl.col("type").cast(pl.String),
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

        product_events = events.filter(pl.col("action_type").is_in(PRODUCT_ACTIONS))
        price_events = events.filter(pl.col("action_type").is_in(PRICE_ACTIONS))

        price_features = price_events.group_by("user_id").agg(
            [
                pl.col("effective_price").mean().alias("avg_viewed_price"),
                pl.col("effective_price").median().alias("median_viewed_price"),
                pl.col("effective_price").max().alias("max_viewed_price"),
                pl.col("discount_ratio").mean().alias("discount_ratio_mean"),
                pl.col("is_discounted").mean().alias("fraction_discounted_products"),
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
                    pl.col("category_event_count").sum().alias("total_category_events"),
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
                user_category_totals.select(["user_id", "total_category_events"]),
                on="user_id",
                how="left",
            )
            .with_columns(
                [
                    (
                        pl.col("category_event_count") / pl.col("total_category_events")
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
            product_events.filter(pl.col("category_id").is_in(top_category_ids))
            .group_by(["user_id", "category_id"])
            .agg(pl.len().alias("cnt"))
            .join(
                user_category_totals.select(["user_id", "total_category_events"]),
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
            .fill_null(0)
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

    def construct_dataset(
        self,
        feature_end_date: date,
        target_start_date: date | None = None,
        target_end_date: date | None = None,
        users_df: pl.LazyFrame | None = None,
        include_actions: bool = True,
        include_search: bool = True,
        include_price: bool = True,
    ) -> pl.DataFrame:
        if target_start_date is not None and target_end_date is not None:
            df = self._make_target(target_start_date, target_end_date)
        elif users_df is not None:
            df = users_df
        else:
            raise ValueError("Either target dates or users_df must be provided.")

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

        return df.with_columns(
            cs.numeric().fill_null(strategy="zero"),
            cs.string().fill_null(""),
            cs.categorical().fill_null("NONE"),
        ).collect(engine="streaming").to_pandas().sort_index(axis=1)
