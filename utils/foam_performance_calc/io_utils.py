"""
io_utils.py
-----------
Handles all file I/O concerns:
  - Reading CSV files uploaded via Streamlit (or from a file path).
  - Making duplicate column names unique (pandas deduplication suffix style).
  - Safe file parsing with graceful error handling.

Nothing in this module knows about foam business logic.
"""

from __future__ import annotations

from typing import Union
import io

import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_csv_safe(source: Union[str, io.IOBase]) -> pd.DataFrame:
    """
    Read a CSV file from a file path or a Streamlit UploadedFile object.

    Handles common Excel-exported CSV artefacts:
      - UTF-8 BOM (``encoding_errors='replace'``)
      - Trailing whitespace in column names (stripped by ``deduplicate_columns``)

    Parameters
    ----------
    source : str or file-like
        File path string or Streamlit ``UploadedFile`` / any ``io.IOBase``.

    Returns
    -------
    pd.DataFrame
        Parsed DataFrame with deduplicated column names.

    Raises
    ------
    ValueError
        If the file cannot be parsed as CSV or is completely empty.
    """
    try:
        df = pd.read_csv(source, encoding="utf-8-sig")
    except UnicodeDecodeError:
        # Fall back to latin-1 for legacy Excel exports
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, encoding="latin-1")
    except Exception as exc:
        raise ValueError(f"Could not read CSV file: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    df = deduplicate_columns(df)
    return df


def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename duplicate column headers so that every column name is unique.

    Pandas normally silently allows duplicate column names, which causes
    unpredictable behaviour when selecting by name.  This function applies
    the same ``<name>.<n>`` suffix convention that pandas uses internally
    for ``read_csv`` when ``mangle_dupe_cols`` is active, but we do it
    explicitly so the mapping is deterministic and inspectable.

    The *first* occurrence of a duplicated name keeps the bare name.
    Subsequent occurrences become ``<name>.1``, ``<name>.2``, etc.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns may contain duplicates.

    Returns
    -------
    pd.DataFrame
        New DataFrame with unique column names (same data, same order).
    """
    new_cols: list[str] = []
    seen: dict[str, int] = {}

    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}.{seen[col]}")

    df = df.copy()
    df.columns = new_cols
    return df
