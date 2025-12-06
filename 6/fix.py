#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fio_symspell_pipeline.py

Автоматическое исправление ФИО с ошибками с помощью SymSpell.

Что делает:
1) Загружает датасет (parquet/csv/xlsx).
2) Находит колонку ФИО (или использует --fio-col).
3) Строит три отдельных SymSpell-словаря: фамилии / имена / отчества.
   Источники:
   - (опция) Zenodo-архив с names/surnames/midnames в JSONL
   - (опция) локальные wordlist-файлы пользователя
   - (опция) частоты из входного датасета ФИО (рекомендуется!)
4) Исправляет каждый компонент отдельно.
5) Сохраняет результат.

Требования:
  pip install pandas pyarrow symspellpy requests tqdm

Примеры:
  python fio_symspell_pipeline.py \
      --input employees.parquet \
      --output employees_fixed.parquet \
      --add-user-fio-to-dict \
      --download-zenodo

  python fio_symspell_pipeline.py \
      --input employees.csv \
      --output employees_fixed.csv \
      --fio-col "ФИО" \
      --add-user-fio-to-dict \
      --download-zenodo \
      --max-edit-distance 2
"""

import argparse
import json
import os
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm
from symspellpy import SymSpell, Verbosity


# ============================================================
# Константы источников
# ============================================================

# Архив JSONL с тремя файлами:
# names.jsonl, surnames.jsonl, midnames.jsonl
# (при запуске у пользователя с интернетом скачивается автоматически)
ZENODO_JSONL_URL = "https://zenodo.org/records/2747011/files/russiannames_db_jsonl.zip?download=1"

DEFAULT_CACHE_DIR = "./fio_dict_cache"


# ============================================================
# Нормализация
# ============================================================

# Оставляем буквы/цифры/подчёркивание/дефис.
# Для ФИО обычно ок, т.к. дефисы возможны (Иванов-Петров).
_CLEAN_RE = re.compile(r"[^\w\-]+", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_token(token: str, replace_yo: bool = True) -> str:
    if token is None:
        return ""
    t = str(token).strip().lower()
    t = _CLEAN_RE.sub("", t)
    if replace_yo:
        t = t.replace("ё", "е")
    return t


def titlecase_hyphenated(token: str) -> str:
    if not token:
        return token
    parts = token.split("-")
    out = []
    for p in parts:
        if not p:
            out.append(p)
        else:
            out.append(p[:1].upper() + p[1:].lower())
    return "-".join(out)


def split_fio(fio: str) -> List[str]:
    if fio is None:
        return []
    s = str(fio).strip()
    if not s:
        return []
    s = _SPACE_RE.sub(" ", s)
    return s.split(" ")


# ============================================================
# Загрузка и парсинг источников
# ============================================================

def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with dest.open("wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {dest.name}",
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_string_field(obj: dict) -> Optional[str]:
    preferred_keys = [
        "name", "surname", "midname", "patronymic",
        "first_name", "last_name", "value", "text", "title",
    ]
    for k in preferred_keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    for v in obj.values():
        if isinstance(v, str) and v.strip():
            return v.strip()

    return None


def extract_frequency(obj: dict) -> int:
    preferred_keys = ["count", "freq", "frequency", "n", "popularity"]
    for k in preferred_keys:
        v = obj.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    return 1


def load_wordlist_file(path: Path, replace_yo: bool = True) -> Counter:
    """
    Читает текстовый файл со словами (по одному на строку).
    Можно использовать для локальных словарей имён/фамилий/отчеств.
    """
    counter = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            norm = normalize_token(s, replace_yo=replace_yo)
            if norm:
                counter[norm] += 1
    return counter


# ============================================================
# SymSpell корректор ФИО
# ============================================================

class FIOCorrector:
    def __init__(
        self,
        max_edit_distance: int = 2,
        prefix_length: int = 7,
        replace_yo: bool = True,
        conservative: bool = True,
    ):
        self.max_edit_distance = max_edit_distance
        self.prefix_length = prefix_length
        self.replace_yo = replace_yo
        self.conservative = conservative

        self.sp_surname = SymSpell(max_dictionary_edit_distance=max_edit_distance,
                                   prefix_length=prefix_length)
        self.sp_name = SymSpell(max_dictionary_edit_distance=max_edit_distance,
                                prefix_length=prefix_length)
        self.sp_midname = SymSpell(max_dictionary_edit_distance=max_edit_distance,
                                   prefix_length=prefix_length)

    def add_entries(self, sym: SymSpell, counter: Counter) -> None:
        for term, freq in counter.items():
            if term:
                sym.create_dictionary_entry(term, int(freq))

    def load_zenodo_russiannames(self, cache_dir: Path, download: bool = True) -> bool:
        """
        Загружает базовые списки names/surnames/midnames из Zenodo-архива.
        Возвращает True если удалось загрузить/распарсить.
        """
        cache_dir = Path(cache_dir)
        zip_path = cache_dir / "russiannames_db_jsonl.zip"
        extract_dir = cache_dir / "russiannames_db_jsonl"

        try:
            if download and not zip_path.exists():
                download_file(ZENODO_JSONL_URL, zip_path)

            if zip_path.exists() and not extract_dir.exists():
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_dir)
        except Exception:
            # Интернет/доступ мог отсутствовать
            return False

        name_counter = Counter()
        surname_counter = Counter()
        mid_counter = Counter()

        files_map = {
            "names.jsonl": name_counter,
            "surnames.jsonl": surname_counter,
            "midnames.jsonl": mid_counter,
        }

        any_file = False
        for fname, counter in files_map.items():
            fpath = extract_dir / fname
            if not fpath.exists():
                continue
            any_file = True

            # Итерируем без превращения в list, чтобы не грузить память
            for obj in tqdm(iter_jsonl(fpath), desc=f"Parsing {fname}"):
                s = extract_string_field(obj)
                if not s:
                    continue
                freq = extract_frequency(obj)
                norm = normalize_token(s, replace_yo=self.replace_yo)
                if norm:
                    counter[norm] += max(1, freq)

                # Сохраним и вариант с 'ё', если нормализация её убрала
                if self.replace_yo:
                    raw_norm = normalize_token(s, replace_yo=False)
                    if raw_norm and raw_norm != norm:
                        counter[raw_norm] += 1

        if not any_file:
            return False

        self.add_entries(self.sp_name, name_counter)
        self.add_entries(self.sp_surname, surname_counter)
        self.add_entries(self.sp_midname, mid_counter)
        return True

    def load_local_wordlists(
        self,
        names_files: List[Path],
        surnames_files: List[Path],
        midnames_files: List[Path],
    ) -> None:
        name_counter = Counter()
        surname_counter = Counter()
        mid_counter = Counter()

        for p in names_files:
            if p.exists():
                name_counter.update(load_wordlist_file(p, replace_yo=self.replace_yo))

        for p in surnames_files:
            if p.exists():
                surname_counter.update(load_wordlist_file(p, replace_yo=self.replace_yo))

        for p in midnames_files:
            if p.exists():
                mid_counter.update(load_wordlist_file(p, replace_yo=self.replace_yo))

        self.add_entries(self.sp_name, name_counter)
        self.add_entries(self.sp_surname, surname_counter)
        self.add_entries(self.sp_midname, mid_counter)

    def add_user_fio_corpus(self, fio_list: Iterable[str], weight: int = 3) -> None:
        """
        Добавляет частоты из пользовательского корпуса.
        weight > 1 усиливает защиту от переисправлений.
        """
        surname_counter = Counter()
        name_counter = Counter()
        mid_counter = Counter()

        for fio in fio_list:
            parts = split_fio(fio)
            if not parts:
                continue

            if len(parts) >= 1:
                s = normalize_token(parts[0], replace_yo=self.replace_yo)
                if s:
                    surname_counter[s] += weight
            if len(parts) >= 2:
                n = normalize_token(parts[1], replace_yo=self.replace_yo)
                if n:
                    name_counter[n] += weight
            if len(parts) >= 3:
                m = normalize_token(parts[2], replace_yo=self.replace_yo)
                if m:
                    mid_counter[m] += weight

        self.add_entries(self.sp_surname, surname_counter)
        self.add_entries(self.sp_name, name_counter)
        self.add_entries(self.sp_midname, mid_counter)

    def _lookup_best(self, sym: SymSpell, norm: str):
        return sym.lookup(
            norm,
            Verbosity.TOP,
            max_edit_distance=self.max_edit_distance,
            include_unknown=True
        )

    def _should_accept(self, original_norm: str, suggestion) -> bool:
        """
        Консервативная эвристика принятия исправления.
        """
        if suggestion is None:
            return False

        if suggestion.term == original_norm:
            return True

        # SymSpell suggestion имеет edit_distance и count
        dist = getattr(suggestion, "distance", None)
        cnt = getattr(suggestion, "count", 0)

        if dist is None:
            return True

        if not self.conservative:
            return True

        # Консервативный режим:
        # - всегда принимаем dist==1
        # - dist==2 принимаем только если слово достаточно длинное
        #   и частота кандидата не совсем микроскопическая
        if dist <= 1:
            return True

        if dist == 2 and len(original_norm) >= 6 and cnt >= 2:
            return True

        return False

    def _correct_token(self, sym: SymSpell, token: str) -> str:
        raw = token or ""
        norm = normalize_token(raw, replace_yo=self.replace_yo)
        if not norm:
            return raw

        suggestions = self._lookup_best(sym, norm)
        if not suggestions:
            return titlecase_hyphenated(normalize_token(raw, replace_yo=False) or raw)

        best = suggestions[0]

        if self._should_accept(norm, best):
            return titlecase_hyphenated(best.term)

        # иначе оставляем исходный с аккуратным регистром
        base = normalize_token(raw, replace_yo=False) or raw
        return titlecase_hyphenated(base)

    def correct_fio(self, fio: str, order: str = "FIO") -> str:
        parts = split_fio(fio)
        if not parts:
            return fio

        order = (order or "FIO").upper()
        role_to_sym = {
            "F": self.sp_surname,
            "I": self.sp_name,
            "O": self.sp_midname,
        }

        corrected = []
        for idx, part in enumerate(parts):
            role = order[idx] if idx < len(order) else None
            sym = role_to_sym.get(role)

            if sym is None:
                # fallback: выбираем первый словарь, который даёт осмысленную подсказку
                norm = normalize_token(part, replace_yo=self.replace_yo)
                chosen = None
                for s in (self.sp_surname, self.sp_name, self.sp_midname):
                    sugg = self._lookup_best(s, norm)
                    if sugg and sugg[0].term:
                        chosen = sugg[0].term
                        break
                if chosen:
                    corrected.append(titlecase_hyphenated(chosen))
                else:
                    base = normalize_token(part, replace_yo=False) or part
                    corrected.append(titlecase_hyphenated(base))
                continue

            corrected.append(self._correct_token(sym, part))

        return " ".join(corrected)


# ============================================================
# Работа с датасетом
# ============================================================

def detect_fio_column(columns: List[str]) -> Optional[str]:
    """
    Пытается найти колонку с ФИО по названию.
    """
    lowered = [c.lower() for c in columns]

    # приоритетные точные/почти точные варианты
    patterns = [
        "фио", "ф.и.о", "full_name", "fullname", "name_full",
        "employee_name", "person_name", "client_name",
    ]

    for p in patterns:
        for col, low in zip(columns, lowered):
            if low == p:
                return col

    # более мягкий поиск по вхождению
    for col, low in zip(columns, lowered):
        if "фио" in low:
            return col
    for col, low in zip(columns, lowered):
        if "full" in low and "name" in low:
            return col
    for col, low in zip(columns, lowered):
        if low in ("name", "employee", "person") and len(columns) <= 8:
            # осторожная эвристика
            return col

    return None


def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Неподдерживаемый формат файла: {ext}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    ext = path.suffix.lower()
    if ext == ".parquet":
        df.to_parquet(path, index=False)
        return
    if ext == ".csv":
        df.to_csv(path, index=False)
        return
    if ext in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
        return
    raise ValueError(f"Неподдерживаемый формат файла: {ext}")


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Исправление ФИО с ошибками с помощью SymSpell."
    )

    p.add_argument("--input", required=True, help="Путь к входному файлу (parquet/csv/xlsx).")
    p.add_argument("--output", required=True, help="Путь к выходному файлу.")
    p.add_argument("--fio-col", default="", help="Название колонки с ФИО. Если не задано, попробуем найти автоматически.")

    p.add_argument("--order", default="FIO", help="Порядок компонент в строке (FIO/IFO/...).")

    p.add_argument("--max-edit-distance", type=int, default=2)
    p.add_argument("--prefix-length", type=int, default=7)

    p.add_argument("--no-yo-replace", action="store_true",
                   help="Не заменять 'ё' на 'е' при нормализации.")

    p.add_argument("--conservative", action="store_true",
                   help="Более осторожное принятие исправлений.")
    p.add_argument("--aggressive", action="store_true",
                   help="Менее осторожное принятие исправлений.")

    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                   help="Папка для кеша скачанных словарей.")

    p.add_argument("--download-zenodo", action="store_true",
                   help="Скачать и использовать базовый Zenodo-словарь имён/фамилий/отчеств.")

    p.add_argument("--names-file", action="append", default=[],
                   help="Локальный файл со списком имён (по одной записи на строку). Можно указать несколько раз.")
    p.add_argument("--surnames-file", action="append", default=[],
                   help="Локальный файл со списком фамилий. Можно указать несколько раз.")
    p.add_argument("--midnames-file", action="append", default=[],
                   help="Локальный файл со списком отчеств. Можно указать несколько раз.")

    p.add_argument("--add-user-fio-to-dict", action="store_true",
                   help="Добавить частоты из входного датасета в словари (рекомендуется).")
    p.add_argument("--user-weight", type=int, default=3,
                   help="Вес частот из пользовательского корпуса (по умолчанию 3).")

    p.add_argument("--inplace", action="store_true",
                   help="Заменить исходную колонку ФИО вместо создания новой.")
    p.add_argument("--new-col", default="fio_fixed",
                   help="Название новой колонки (если не --inplace).")

    p.add_argument("--sample-report", type=int, default=0,
                   help="Сколько примеров исправлений вывести в stdout для быстрой проверки.")

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    cache_dir = Path(args.cache_dir)

    if args.aggressive and args.conservative:
        # если указаны оба — пусть aggressive отменяет conservative
        conservative = False
    elif args.aggressive:
        conservative = False
    elif args.conservative:
        conservative = True
    else:
        # дефолт: умеренно осторожный режим
        conservative = True

    replace_yo = not args.no_yo_replace

    df = read_table(input_path)

    fio_col = args.fio_col.strip()
    if not fio_col:
        fio_col = detect_fio_column(list(df.columns)) or ""

    if not fio_col or fio_col not in df.columns:
        cols = ", ".join(map(str, df.columns))
        raise ValueError(
            "Не удалось найти колонку с ФИО автоматически. "
            "Укажи её через --fio-col. "
            f"Доступные колонки: {cols}"
        )

    fio_series = df[fio_col].astype(str).fillna("")

    corrector = FIOCorrector(
        max_edit_distance=args.max_edit_distance,
        prefix_length=args.prefix_length,
        replace_yo=replace_yo,
        conservative=conservative,
    )

    # 1) Zenodo
    zenodo_loaded = False
    if args.download_zenodo:
        zenodo_loaded = corrector.load_zenodo_russiannames(cache_dir=cache_dir, download=True)

    # 2) Локальные списки
    names_files = [Path(p) for p in args.names_file]
    surnames_files = [Path(p) for p in args.surnames_file]
    midnames_files = [Path(p) for p in args.midnames_file]

    if names_files or surnames_files or midnames_files:
        corrector.load_local_wordlists(names_files, surnames_files, midnames_files)

    # 3) Пользовательский корпус
    if args.add_user_fio_to_dict:
        corrector.add_user_fio_corpus(fio_series.tolist(), weight=max(1, args.user_weight))

    # Если пользователь не подключил ничего, всё равно имеет смысл
    # построить словари из собственного корпуса:
    if (not args.download_zenodo) and (not args.add_user_fio_to_dict) and not (names_files or surnames_files or midnames_files):
        corrector.add_user_fio_corpus(fio_series.tolist(), weight=max(1, args.user_weight))

    # Исправление
    corrected = []
    for fio in tqdm(fio_series.tolist(), desc="Correcting FIO"):
        corrected.append(corrector.correct_fio(fio, order=args.order))

    # Отчёт примеров
    if args.sample_report and args.sample_report > 0:
        print("\nПримеры исправлений:")
        shown = 0
        for orig, fixed in zip(fio_series.tolist(), corrected):
            if orig != fixed:
                print(f"  {orig}  ->  {fixed}")
                shown += 1
                if shown >= args.sample_report:
                    break
        if shown == 0:
            print("  Существенных изменений не найдено на первых примерах.")

    # Запись результата
    out_df = df.copy()

    if args.inplace:
        out_df[fio_col] = corrected
    else:
        new_col = args.new_col.strip() or "fio_fixed"
        # чтобы не затереть существующую колонку случайно
        if new_col in out_df.columns:
            # безопасно создадим уникальное имя
            base = new_col
            i = 2
            while new_col in out_df.columns:
                new_col = f"{base}_{i}"
                i += 1
        out_df[new_col] = corrected

    # Метаданные о том, как строили словари (полезно для аудита)
    meta = {
        "source_input": str(input_path),
        "fio_col": fio_col,
        "order": args.order,
        "max_edit_distance": args.max_edit_distance,
        "prefix_length": args.prefix_length,
        "replace_yo": replace_yo,
        "conservative": conservative,
        "zenodo_requested": bool(args.download_zenodo),
        "zenodo_loaded": bool(zenodo_loaded),
        "user_corpus_added": bool(args.add_user_fio_to_dict),
        "user_weight": int(args.user_weight),
        "names_files": [str(p) for p in names_files],
        "surnames_files": [str(p) for p in surnames_files],
        "midnames_files": [str(p) for p in midnames_files],
    }

    # Попробуем сохранить мета-инфо рядом
    try:
        meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    write_table(out_df, output_path)

    print(f"\nГотово. Колонка ФИО: {fio_col}")
    if args.inplace:
        print("ФИО заменены на исправленные (режим --inplace).")
    else:
        print("Добавлена новая колонка с исправленными ФИО.")
    print(f"Результат сохранён: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.", file=sys.stderr)
        sys.exit(130)
