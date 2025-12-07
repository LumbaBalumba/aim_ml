import polars as pl
from pathlib import Path

def normalize_russian_name(s):
    return (
        s.str.strip_chars()
         .str.replace_all(r"\s+", " ")
         .str.to_lowercase()
         .str.replace_all("ё", "е")
    )

def preprocess_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    # Fill nulls in all relevant fields
    text_columns = ["surname", "name", "fathername", "passport", "inn", "gender"]
    for col in text_columns:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).fill_null("").cast(pl.Utf8)
            )
    
    # Normalize name fields
    for col in ["surname", "name", "fathername"]:
        if col in df.columns:
            df = df.with_columns(
                normalize_russian_name(pl.col(col)).alias(col)
            )
    
    # Normalize birthdate
    if "birthdate" in df.columns:
        df = df.with_columns(
            pl.col("birthdate")
            .str.strip_chars()
            .str.replace_all(r"\s+", "")
            .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
        )
    
    # Fill missing birthdates with a sentinel (optional)
    df = df.with_columns(
        pl.col("birthdate").fill_null(pl.date(1900, 1, 1))
    )

    # Deduplicate based on identifying fields
    dedup_keys = ["surname", "name", "fathername", "birthdate", "passport", "inn"]
    dedup_keys_present = [col for col in dedup_keys if col in df.columns]
    df = df.unique(subset=dedup_keys_present)

    return df

def main():
    # Paths to your Parquet files
    df1_path = Path("employees.parquet")
    df2_path = Path("orcs.parquet")

    # Load data using Polars
    df1 = pl.read_parquet(df1_path)
    df2 = pl.read_parquet(df2_path)

    # Preprocess both datasets
    df1_clean = preprocess_dataframe(df1)
    df2_clean = preprocess_dataframe(df2)

    # Save cleaned versions if needed
    df1_clean.write_parquet("df1_cleaned.parquet")
    df2_clean.write_parquet("df2_cleaned.parquet")

if __name__ == "__main__":
    main()
