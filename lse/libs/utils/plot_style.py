from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotly.validators.scatter.marker import SymbolValidator
from pydantic import BaseModel

raw_symbols = SymbolValidator().values


def shifted_colormap(
    cmap: str | matplotlib.colors.Colormap,
    start: float = 0,
    midpoint: tuple[float, float] | float = 0.5,
    stop: float = 1.0,
) -> matplotlib.colors.Colormap:
    """
    Offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Args:
      cmap : The matplotlib colormap or colormap reference to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. Given a
          (vmin, vmax) tuple, it will be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          will be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.

    Returns:
        a Matplotlib colormap

    see https://stackoverflow.com/a/20528097

    """

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    if type(midpoint) is tuple:
        vmin, vmax = midpoint
        midpoint = 1 - vmax / (vmax + abs(vmin))

    cdict: dict[Literal["red", "green", "blue", "alpha"], list[tuple[float, ...]]] = {
        "red": [],
        "green": [],
        "blue": [],
        "alpha": [],
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index, strict=True):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    return matplotlib.colors.LinearSegmentedColormap("", cdict)  # type: ignore


def get_marker_symbol_map(series: pd.Series) -> pd.Series:
    symbol_collection = [
        symbol for symbol in np.array(SymbolValidator().values)[2::3] if "-" not in symbol
    ]
    return series.map(dict(zip(series.dropna().unique(), symbol_collection, strict=False)))


class PlotCues(BaseModel):
    slicer: str
    color: str | None
    marker: Literal["marker_symbol"] | None
    opacity: Literal["opacity"] | None
    size: str | None
    text: str
