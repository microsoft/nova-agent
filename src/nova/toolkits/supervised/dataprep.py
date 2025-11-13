from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from trident.Converter import OPENSLIDE_EXTENSIONS, PIL_EXTENSIONS

from nova.utils.deterministic import _set_deterministic

# Constants

VALID_SLIDE_EXTS = OPENSLIDE_EXTENSIONS.union(PIL_EXTENSIONS)

# Set deterministic behavior
_set_deterministic()


__all__ = [
    "detect_case_id_column",
    "preview_columns",
    "validate_slide_ids",
    "encode_label_column",
    "drop_rare_categories",
    "get_missing_slide_samples",
    "get_unopenable_slides",
    "reorder_columns",
]


def detect_case_id_column(df: pd.DataFrame) -> str | None:
    """Detect case ID column from common naming patterns."""
    candidates = ["case_id", "caseid", "case"]
    for col in df.columns:
        if col.lower() in candidates:
            return col
    return None


def preview_columns(df: pd.DataFrame, max_rows: int = 5) -> Tuple[List[str], str]:
    """Preview the first few rows of each column in a DataFrame."""
    col_names = df.columns.tolist()
    preview = "\n".join(f"{col}: {df[col].head(max_rows).tolist()}" for col in col_names)
    return col_names, preview


def validate_slide_ids(df: pd.DataFrame, col: str) -> None:
    """Validate that all slide IDs have valid extensions."""
    if not df[col].astype(str).str.lower().str.endswith(tuple(VALID_SLIDE_EXTS)).all():
        raise ValueError(
            "All slide_id values must end with a valid slide extension.\n"
            f"Valid extensions: {sorted(VALID_SLIDE_EXTS)}\n"
            f"First five slide_id values: {df[col].head().tolist()}"
        )


def encode_label_column(df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Encode a label column as categorical integer codes.

    Args:
        df: Input DataFrame containing the label column.
        label_column: Name of the label column to encode.

    Returns:
        Tuple containing:
            - Updated DataFrame with '{label_column}_cats' column of integer codes
            - Mapping from integer code to original label value
    """
    if df[label_column].dtype == "object":
        cats, uniques = pd.factorize(df[label_column])
        df[f"{label_column}_cats"] = cats
        label_map = {i: str(val) for i, val in enumerate(uniques)}
    else:
        df[f"{label_column}_cats"] = df[label_column].astype("int")
        label_map = {int(val): str(val) for val in sorted(df[label_column].unique())}
    return df, label_map


def drop_rare_categories(df: pd.DataFrame, label_column: str, drop_threshold: int) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Remove rows where label category occurs fewer than drop_threshold times.

    Args:
        df: Input DataFrame.
        label_column: Column to use for category frequency.
        drop_threshold: Minimum required count for a category to be kept.

    Returns:
        Tuple of filtered DataFrame and list of dropped rows as dictionaries.
    """
    value_counts = df[label_column].value_counts()
    to_drop = value_counts[value_counts < drop_threshold].index.tolist()
    dropped = df[df[label_column].isin(to_drop)]
    df_filtered = df[~df[label_column].isin(to_drop)].reset_index(drop=True)

    # Convert dropped to a list of dictionaries (one dictionary per row)
    dropped_records = dropped.to_dict(orient='records')

    return df_filtered, dropped_records


def get_missing_slide_samples(df: pd.DataFrame, slide_path_col: str) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Split DataFrame into rows with missing and existing slide files.

    Args:
        df: Input DataFrame with slide path column.
        slide_path_col: Name of column containing slide file paths.

    Returns:
        Tuple of DataFrame with existing files and list of rows with missing files.
    """
    mask = ~df[slide_path_col].apply(lambda x: Path(x).exists())
    dropped = df[mask].copy()
    # change dropped to a list of dictionaries (one dictionary per row)
    dropped = dropped.to_dict(orient='records')
    return df[~mask], dropped


def get_unopenable_slides(df: pd.DataFrame, slide_path_col: str) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Splits DataFrame into slides that cannot and can be opened by OpenSlideWSI.

    Args:
        df (pd.DataFrame): Input DataFrame with slide path column.
        slide_path_col (str): Name of column containing slide file paths.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - DataFrame of slides that cannot be opened.
            - DataFrame of slides that can be opened.

    Notes:
        Relies on trident.OpenSlideWSI. Does not save files.
    """
    from trident import OpenSlideWSI

    cannot_open = []
    for idx, row in df.iterrows():
        try:
            OpenSlideWSI(slide_path=row[slide_path_col], lazy_init=False)
        except Exception:
            cannot_open.append(row["slide_id"])
    mask = df["slide_id"].isin(cannot_open)

    dropped = df[mask].copy()
    dropped = dropped.to_dict(orient='records')

    return df[~mask], dropped


def reorder_columns(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """
    Reorders DataFrame columns to standard order.

    Args:
        df (pd.DataFrame): Input DataFrame.
        label_column (str): Name of the label column.

    Returns:
        pd.DataFrame: DataFrame with columns ordered as ['case_id', 'slide_id', label_column, label_column+'_cats', ...rest].

    Notes:
        Does not modify data; returns new DataFrame view.
    """
    order = ["case_id", "slide_id", label_column, f"{label_column}_cats"]
    rest = [c for c in df.columns if c not in order]
    return df[order + rest]
