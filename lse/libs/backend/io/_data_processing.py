from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pymodules.pandas_utils import extract_calendar_features

from lse.libs.data_models import VariableReference

logger = logging.getLogger()


def _set_index_as_str(df: pd.DataFrame) -> pd.DataFrame:
    df.index = df.index.astype(str)
    return df


def _format_index(df: pd.DataFrame) -> pd.DataFrame:
    # Extract calendar features if available
    try:
        df = extract_calendar_features(df)
    except TypeError:
        # No datetimeindex available
        logger.debug(f"No datetime features in the :obj:`pandas.DataFrame` index: {df.index}")

    # Create a unique index
    df = df.assign(**{VariableReference.UNIQUE_ID: np.arange(len(df))})
    # Reformat the index
    if df.index.name is None and len(df.index.names) == 1:
        # Discard any existing single index
        df = df.set_index(VariableReference.UNIQUE_ID)
    else:
        # else append
        df = df.set_index(VariableReference.UNIQUE_ID, append=True).reorder_levels(
            [VariableReference.UNIQUE_ID, *df.index.names], axis=0
        )
    # Make sure all levels are named
    df.index.names = [
        f"level_{i}" if level is None else level for i, level in enumerate(df.index.names)
    ]
    return df


def _process_data(data: pd.DataFrame) -> pd.DataFrame:
    # Drop null columns
    data = data.dropna(axis=1, how="all")

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data should be provided in a `pandas.DataFrame` format")
    if len(data.columns.names) > 1:
        raise NotImplementedError("Multi-indexed columns DataFrames are not supported")

    # Cast index to string if it is a single index
    if is_numeric_dtype(data.index):
        data = _set_index_as_str(data)
    # Cast column names to string if needed
    data.columns = [f"Feature_{col}" if not isinstance(col, str) else col for col in data.columns]

    # Format the index
    data = _format_index(data)

    return data
