#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2: Correct errors in Russian/Cyrillic names, surnames, fathernames.

Key ideas:
1) No hard-coded name lists.
2) Build dictionaries from reliable open sources:
   - Zenodo russiannames_db_jsonl.zip
   - sorokinpf/russian_names text/jsonl lists
3) Add strong in-domain frequencies from df1+df2 to avoid overcorrecting rare but valid names.
4) Correct ONLY unique values per column for speed.
5) Multi-core parallel correction via joblib.

Expected columns (string):
    name, surname, fathername, birthdate, gender, passport, inn
Other columns are preserved.

Input:  parquet files
Output: parquet files with corrected FIO columns

Usage:
    python step2_correct_fio.py \
        --df1 df1_cleaned.parquet \
        --df2 df2_cleaned.parquet \
        --out1 df1_fio_corrected.parquet \
        --out2 df2_fio_corrected.parquet \
        --cache-dir .cache_names \
        --threads 32
"""

import argparse
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import polars as pl
import requests
from joblib import Parallel, delayed
from symspellpy.symspellpy import SymSpell, Verbosity

# -----------------------------
# Config: external sources
# -----------------------------

ZENODO_RUSSIANNAMES_JSONL_ZIP = (
    "https://zenodo.org/records/2747011/files/russiannames_db_jsonl.zip?download=1"
)

# sorokinpf/russian_names raw files
SOROKIN_BASE = "https://raw.githubusercontent.com/sorokinpf/russian_names/master/"
SOROKIN_FILES = {
    "names_txt": SOROKIN_BASE + "russian_names.txt",
    "surnames_txt": SOROKIN_BASE + "russian_surnames.txt",
    "midnames_jsonl": SOROKIN_BASE + "midnames.jsonl",
    "names_jsonl": SOROKIN_BASE + "names.jsonl",
    "surnames_jsonl": SOROKIN_BASE + "surnames.jsonl",
}

# -----------------------------
# Text normalization utilities
# -----------------------------

_CYR_LETTERS_RE = re.compile(r"[А-Яа-яЁёІіЇїЄєҐґҚқЎўҲҳӘәӨөҮүҢңҺһІі]")
_NON_NAME_CHARS_RE = re.compile(r"[^A-Za-zА-Яа-яЁёІіЇїЄєҐґҚқЎўҲҳӘәӨөҮүҢңҺһІі\- ]+")
_MULTI_SPACE_RE = re.compile(r"\s+")

def is_blank(x: Optional[str]) -> bool:
    if x is None:
        return True
    s = str(x).strip()
    if s == "":
        return True
    low = s.lower()
    return low in {"nan", "none", "null", "na", "n/a", "-", "_", "?"}


def normalize_whitespace(s: str) -> str:
    return _MULTI_SPACE_RE.sub(" ", s).strip()


def normalize_name_surface(s: str, replace_yo: bool = True) -> str:
    """
    Normalizes a raw name-like field:
    - strips
    - removes obvious junk chars
    - normalizes spaces
    - keeps hyphens
    - optionally 'ё' -> 'е' to reduce variation
    """
    s = str(s)
    s = s.strip()
    s = _NON_NAME_CHARS_RE.sub(" ", s)
    s = normalize_whitespace(s)
    if replace_yo:
        s = s.replace("Ё", "Е").replace("ё", "е")
    # Normalize hyphen spacing
    s = s.replace(" - ", "-").replace("- ", "-").replace(" -", "-")
    return s


def to_title_cyr(s: str) -> str:
    """
    Title-case each hyphen/space-separated part.
    """
    if is_blank(s):
        return ""
    s = normalize_name_surface(s, replace_yo=True)
    if s == "":
        return ""

    def cap_part(p: str) -> str:
        if p == "":
            return p
        return p[0].upper() + p[1:].lower()

    # Keep compound names intact
    parts = []
    for chunk in s.split(" "):
        sub = [cap_part(x) for x in chunk.split("-")]
        parts.append("-".join(sub))
    return " ".join(parts)


def is_cyrillicish(s: str) -> bool:
    if is_blank(s):
        return False
    return _CYR_LETTERS_RE.search(s) is not None


# -----------------------------
# Download & cache
# -----------------------------

def download_to_cache(url: str, path: Path, timeout: int = 60) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    return path


# -----------------------------
# External dictionary loaders
# -----------------------------

def _extract_text_from_json(obj: dict) -> Optional[str]:
    # Try common keys first
    for key in ("name", "surname", "midname", "text", "value", "word"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Fallback: first non-empty string value
    for v in obj.values():
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def load_zenodo_russiannames(cache_dir: Path) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Attempts to load:
      - given names
      - surnames
      - midnames (patronymics)
    from Zenodo JSONL zip.

    The zip content naming may vary; we use robust heuristics.

    Returns sets (names, surnames, midnames).
    """
    zip_path = cache_dir / "zenodo_russiannames_db_jsonl.zip"
    try:
        download_to_cache(ZENODO_RUSSIANNAMES_JSONL_ZIP, zip_path)
    except Exception as e:
        print(f"[WARN] Failed to download Zenodo russiannames zip: {e}", file=sys.stderr)
        return set(), set(), set()

    names, surnames, midnames = set(), set(), set()

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            jsonl_files = [n for n in zf.namelist() if n.lower().endswith(".jsonl")]
            for fname in jsonl_files:
                lower = fname.lower()
                bucket = None
                if "midname" in lower or "midnames" in lower:
                    bucket = "mid"
                elif "surname" in lower or "surnames" in lower:
                    bucket = "sur"
                elif re.search(r"(^|/|_)names(\.|_|/)", lower) or lower.endswith("names.jsonl"):
                    # may match first names collection
                    bucket = "name"
                else:
                    # unknown file; we will still parse but only add if content looks plausible
                    bucket = "unknown"

                with zf.open(fname, "r") as f:
                    for raw in f:
                        try:
                            obj = json.loads(raw.decode("utf-8", errors="ignore"))
                        except Exception:
                            continue
                        text = _extract_text_from_json(obj)
                        if not text:
                            continue
                        text = normalize_name_surface(text, replace_yo=True)
                        if not text or not is_cyrillicish(text):
                            continue

                        if bucket == "name":
                            names.add(text)
                        elif bucket == "sur":
                            surnames.add(text)
                        elif bucket == "mid":
                            midnames.add(text)
                        else:
                            # heuristic routing by common patronymic suffixes
                            lowt = text.lower()
                            if lowt.endswith(("вич", "вна", "ична", "оглы", "кызы")):
                                midnames.add(text)
                            else:
                                # ambiguous: could be name or surname;
                                # keep as name to broaden coverage
                                names.add(text)
    except Exception as e:
        print(f"[WARN] Failed to parse Zenodo zip content: {e}", file=sys.stderr)

    return names, surnames, midnames


def load_sorokin_lists(cache_dir: Path) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Loads fallback lists from sorokinpf/russian_names repo.
    Returns sets (names, surnames, midnames).
    """
    names, surnames, midnames = set(), set(), set()

    def load_txt(url: str, local_name: str) -> List[str]:
        p = cache_dir / local_name
        try:
            download_to_cache(url, p)
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            return [normalize_name_surface(x, replace_yo=True) for x in lines if x.strip()]
        except Exception as e:
            print(f"[WARN] Failed to load {url}: {e}", file=sys.stderr)
            return []

    def load_jsonl(url: str, local_name: str) -> List[str]:
        p = cache_dir / local_name
        try:
            download_to_cache(url, p)
            out = []
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    text = _extract_text_from_json(obj)
                    if not text:
                        continue
                    text = normalize_name_surface(text, replace_yo=True)
                    if text:
                        out.append(text)
            return out
        except Exception as e:
            print(f"[WARN] Failed to load {url}: {e}", file=sys.stderr)
            return []

    # Prefer simple txt lists if present
    for x in load_txt(SOROKIN_FILES["names_txt"], "sorokin_russian_names.txt"):
        if x and is_cyrillicish(x):
            names.add(x)
    for x in load_txt(SOROKIN_FILES["surnames_txt"], "sorokin_russian_surnames.txt"):
        if x and is_cyrillicish(x):
            surnames.add(x)

    # Add jsonl where available
    for x in load_jsonl(SOROKIN_FILES["midnames_jsonl"], "sorokin_midnames.jsonl"):
        if x and is_cyrillicish(x):
            midnames.add(x)
    for x in load_jsonl(SOROKIN_FILES["names_jsonl"], "sorokin_names.jsonl"):
        if x and is_cyrillicish(x):
            names.add(x)
    for x in load_jsonl(SOROKIN_FILES["surnames_jsonl"], "sorokin_surnames.jsonl"):
        if x and is_cyrillicish(x):
            surnames.add(x)

    return names, surnames, midnames


# -----------------------------
# In-domain frequency extraction
# -----------------------------

def get_value_counts_union(df1: pl.DataFrame, df2: pl.DataFrame, col: str) -> Counter:
    """
    Returns Counter of normalized values for a column across both dataframes.
    Works fast via Polars value_counts.
    """
    s1 = df1.get_column(col) if col in df1.columns else pl.Series(name=col, values=[])
    s2 = df2.get_column(col) if col in df2.columns else pl.Series(name=col, values=[])

    # Normalize in Polars for speed
    tmp = pl.DataFrame({col: pl.concat([s1, s2], rechunk=True)})
    tmp = tmp.with_columns(
        pl.col(col).cast(pl.Utf8).fill_null("").map_elements(
            lambda x: normalize_name_surface(x, replace_yo=True),
            return_dtype=pl.Utf8,
        )
    )
    vc = tmp.select(pl.col(col).value_counts()).to_dict(as_series=False)[col]
    # Polars returns list of structs; easiest to re-create via python
    # But to be robust across versions:
    counts = Counter()
    # value_counts() output shape can vary; handle both common layouts
    try:
        # Newer Polars: a struct list in one series
        # We'll fallback to a safer path:
        df_vc = tmp.select(pl.col(col)).group_by(col).len()
        for row in df_vc.iter_rows():
            val, cnt = row
            if val:
                counts[val] += int(cnt)
    except Exception:
        # Fallback to Python extraction
        for v in tmp.get_column(col).to_list():
            if v:
                counts[v] += 1

    return counts


# -----------------------------
# SymSpell builder
# -----------------------------

def build_symspell(freq: Counter, max_dictionary_edit_distance: int = 2) -> SymSpell:
    sym = SymSpell(
        max_dictionary_edit_distance=max_dictionary_edit_distance,
        prefix_length=7
    )
    for term, count in freq.items():
        if not term:
            continue
        # SymSpell expects positive counts
        c = int(count) if count and count > 0 else 1
        sym.create_dictionary_entry(term, c)
    return sym


# -----------------------------
# Correction logic
# -----------------------------

def choose_max_ed(token: str) -> int:
    L = len(token)
    if L <= 4:
        return 1
    if L <= 8:
        return 2
    return 2


def correct_with_symspell(
    raw: str,
    sym: SymSpell,
    known_set: Set[str],
    min_ratio_accept: int = 90,
    max_ed_cap: int = 2
) -> str:
    """
    Corrects one name-like string that may contain spaces or hyphens.
    - Splits on spaces, then hyphens, corrects each part.
    - Only changes if suggestion is strong.

    We avoid aggressive overcorrection by:
      * skipping tokens already in dictionary set
      * requiring edit distance <= adaptive threshold
      * requiring a high similarity ratio against the suggestion
    """
    if is_blank(raw):
        return ""

    raw_norm = normalize_name_surface(raw, replace_yo=True)
    if raw_norm == "":
        return ""

    # If the whole string is already known, keep it
    if raw_norm in known_set:
        return to_title_cyr(raw_norm)

    def correct_token(tok: str) -> str:
        tok = normalize_name_surface(tok, replace_yo=True)
        if tok == "":
            return ""
        if not is_cyrillicish(tok):
            # Leave non-Cyr tokens untouched
            return tok
        if tok in known_set:
            return tok

        max_ed = min(choose_max_ed(tok), max_ed_cap)
        suggestions = sym.lookup(tok, Verbosity.TOP, max_edit_distance=max_ed)
        if not suggestions:
            return tok

        best = suggestions[0].term
        # quick similarity check without extra dependency
        # simple normalized edit-bound guard:
        if abs(len(best) - len(tok)) > max_ed:
            return tok

        # lightweight ratio approximation using Python's stdlib?
        # We'll do a small safe check using SequenceMatcher:
        try:
            from difflib import SequenceMatcher
            ratio = int(100 * SequenceMatcher(None, tok, best).ratio())
        except Exception:
            ratio = 100

        if ratio < min_ratio_accept:
            return tok

        return best

    corrected_words = []
    for word in raw_norm.split(" "):
        subparts = []
        for part in word.split("-"):
            subparts.append(correct_token(part))
        corrected_words.append("-".join([p for p in subparts if p != ""]))

    out = " ".join([w for w in corrected_words if w != ""])
    out = normalize_whitespace(out)
    return to_title_cyr(out)


def build_mapping_parallel(
    values: List[str],
    correct_fn,
    n_jobs: int
) -> Dict[str, str]:
    """
    Corrects unique values in parallel and returns mapping original->corrected.
    """
    # Keep deterministic order
    def work(v: str) -> Tuple[str, str]:
        return v, correct_fn(v)

    print(len(values))
    results = Parallel(n_jobs=n_jobs, backend="loky", batch_size=500, verbose=10)(
        delayed(work)(v) for v in values
    )
    return {k: v for k, v in results}


def apply_mapping(df: pl.DataFrame, col: str, mapping: Dict[str, str]) -> pl.DataFrame:
    """
    Applies mapping to a column using Polars replace strategy.
    """
    if col not in df.columns:
        return df
    # Polars replace with dict is fast
    return df.with_columns(
        pl.col(col).cast(pl.Utf8).map_elements(
            lambda x: mapping.get(x, x),
            return_dtype=pl.Utf8,
        ).alias(col)
    )


# -----------------------------
# Main procedure
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 2: Correct FIO fields using external dictionaries + SymSpell.")
    parser.add_argument("--df1", required=True, type=str, help="Path to df1 parquet (after step 1).")
    parser.add_argument("--df2", required=True, type=str, help="Path to df2 parquet (after step 1).")
    parser.add_argument("--out1", required=True, type=str, help="Output parquet path for df1.")
    parser.add_argument("--out2", required=True, type=str, help="Output parquet path for df2.")
    parser.add_argument("--cache-dir", default=".cache_names", type=str, help="Cache directory for external dictionaries.")
    parser.add_argument("--threads", default=32, type=int, help="Number of CPU workers for correction.")
    parser.add_argument("--min-ratio", default=90, type=int, help="Minimum similarity ratio to accept a correction.")
    parser.add_argument("--max-ed", default=2, type=int, help="Maximum edit distance cap.")
    parser.add_argument("--re-dedup", action="store_true", help="Re-run deduplication after FIO correction.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df1 = pl.read_parquet(args.df1)
    df2 = pl.read_parquet(args.df2)

    # Ensure FIO columns exist and are strings
    for col in ("name", "surname", "fathername"):
        if col in df1.columns:
            df1 = df1.with_columns(pl.col(col).cast(pl.Utf8))
        if col in df2.columns:
            df2 = df2.with_columns(pl.col(col).cast(pl.Utf8))

    # Normalize surfaces early to improve unique reduction
    def norm_col(df: pl.DataFrame, col: str) -> pl.DataFrame:
        if col not in df.columns:
            return df
        return df.with_columns(
            pl.col(col).fill_null("").map_elements(
                lambda x: normalize_name_surface(x, replace_yo=True),
                return_dtype=pl.Utf8,
            ).alias(col)
        )

    df1 = norm_col(df1, "name")
    df1 = norm_col(df1, "surname")
    df1 = norm_col(df1, "fathername")
    df2 = norm_col(df2, "name")
    df2 = norm_col(df2, "surname")
    df2 = norm_col(df2, "fathername")

    # Load external dicts
    z_names, z_surnames, z_midnames = load_zenodo_russiannames(cache_dir)
    s_names, s_surnames, s_midnames = load_sorokin_lists(cache_dir)

    ext_names = set(z_names) | set(s_names)
    ext_surnames = set(z_surnames) | set(s_surnames)
    ext_midnames = set(z_midnames) | set(s_midnames)

    # If Zenodo fetch failed entirely, still proceed with in-domain data
    # but keep whatever we managed to load.
    print(f"[INFO] External given names loaded: {len(ext_names)}")
    print(f"[INFO] External surnames loaded: {len(ext_surnames)}")
    print(f"[INFO] External midnames loaded: {len(ext_midnames)}")

    # In-domain frequencies (union of both datasets)
    in_name_freq = get_value_counts_union(df1, df2, "name") if "name" in df1.columns or "name" in df2.columns else Counter()
    in_surname_freq = get_value_counts_union(df1, df2, "surname") if "surname" in df1.columns or "surname" in df2.columns else Counter()
    in_mid_freq = get_value_counts_union(df1, df2, "fathername") if "fathername" in df1.columns or "fathername" in df2.columns else Counter()

    # Build frequency dicts for SymSpell:
    # Start with external words at count=1.
    # Then add in-domain counts with weight to prioritize what is actually present in your data.
    def merge_freq(ext: Set[str], in_dom: Counter, in_weight: int = 10) -> Counter:
        freq = Counter()
        for w in ext:
            if w:
                freq[w] += 1
        for w, c in in_dom.items():
            if w:
                freq[w] += in_weight * int(c)
        return freq

    name_freq = merge_freq(ext_names, in_name_freq, in_weight=10)
    surname_freq = merge_freq(ext_surnames, in_surname_freq, in_weight=10)
    mid_freq = merge_freq(ext_midnames, in_mid_freq, in_weight=10)

    # Build SymSpell models
    sym_name = build_symspell(name_freq, max_dictionary_edit_distance=args.max_ed)
    sym_surname = build_symspell(surname_freq, max_dictionary_edit_distance=args.max_ed)
    sym_mid = build_symspell(mid_freq, max_dictionary_edit_distance=args.max_ed)

    # Known sets for fast "already correct" checks
    known_name_set = set(name_freq.keys())
    known_surname_set = set(surname_freq.keys())
    known_mid_set = set(mid_freq.keys())

    # Collect unique values across BOTH dfs per column
    def unique_union(df_a: pl.DataFrame, df_b: pl.DataFrame, col: str) -> List[str]:
        vals = []
        if col in df_a.columns:
            vals.append(df_a.select(pl.col(col).unique()).get_column(col))
        if col in df_b.columns:
            vals.append(df_b.select(pl.col(col).unique()).get_column(col))
        if not vals:
            return []
        allv = pl.concat(vals, rechunk=True).unique()
        # Filter blanks
        out = []
        for v in allv.to_list():
            if not is_blank(v):
                out.append(v)
        return out

    uniq_names = unique_union(df1, df2, "name")
    uniq_surnames = unique_union(df1, df2, "surname")
    uniq_midnames = unique_union(df1, df2, "fathername")

    print(f"[INFO] Unique 'name' candidates: {len(uniq_names)}")
    print(f"[INFO] Unique 'surname' candidates: {len(uniq_surnames)}")
    print(f"[INFO] Unique 'fathername' candidates: {len(uniq_midnames)}")

    # Build correction functions
    def corr_name(x: str) -> str:
        return correct_with_symspell(
            x, sym_name, known_name_set,
            min_ratio_accept=args.min_ratio,
            max_ed_cap=args.max_ed
        )

    def corr_surname(x: str) -> str:
        return correct_with_symspell(
            x, sym_surname, known_surname_set,
            min_ratio_accept=args.min_ratio,
            max_ed_cap=args.max_ed
        )

    def corr_mid(x: str) -> str:
        return correct_with_symspell(
            x, sym_mid, known_mid_set,
            min_ratio_accept=args.min_ratio,
            max_ed_cap=args.max_ed
        )

    # Build mappings in parallel
    name_map = build_mapping_parallel(uniq_names, corr_name, n_jobs=args.threads) if uniq_names else {}
    surname_map = build_mapping_parallel(uniq_surnames, corr_surname, n_jobs=args.threads) if uniq_surnames else {}
    mid_map = build_mapping_parallel(uniq_midnames, corr_mid, n_jobs=args.threads) if uniq_midnames else {}

    # Apply mappings
    if name_map:
        df1 = apply_mapping(df1, "name", name_map)
        df2 = apply_mapping(df2, "name", name_map)
    if surname_map:
        df1 = apply_mapping(df1, "surname", surname_map)
        df2 = apply_mapping(df2, "surname", surname_map)
    if mid_map:
        df1 = apply_mapping(df1, "fathername", mid_map)
        df2 = apply_mapping(df2, "fathername", mid_map)

    # Final title-casing for FIO
    for col in ("name", "surname", "fathername"):
        if col in df1.columns:
            df1 = df1.with_columns(
                pl.col(col).map_elements(lambda x: to_title_cyr(x), return_dtype=pl.Utf8).alias(col)
            )
        if col in df2.columns:
            df2 = df2.with_columns(
                pl.col(col).map_elements(lambda x: to_title_cyr(x), return_dtype=pl.Utf8).alias(col)
            )

    # Optional re-dedup
    if args.re_dedup:
        # Deduplicate full rows
        df1 = df1.unique(maintain_order=True)
        df2 = df2.unique(maintain_order=True)

    # Write outputs
    pl.write_parquet(df1, args.out1, compression="zstd")
    pl.write_parquet(df2, args.out2, compression="zstd")

    print(f"[DONE] Wrote corrected df1 -> {args.out1}")
    print(f"[DONE] Wrote corrected df2 -> {args.out2}")


if __name__ == "__main__":
    main()
