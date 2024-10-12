import numpy as np
import pandas as pd
from dash import html

from lse.libs.metaparameters import DEFAULT_N_COLOR_BINS
from lse.libs.utils.data_table_formatting import get_data_table_css_style

SAMPLE_DATA = pd.DataFrame(
    {
        "Feature1": np.random.rand(100),
        "Feature2": np.random.rand(100),
        "Feature3": np.random.rand(100),
        "Non_numeric": np.repeat("a", 100),
    }
)


def test_data_table_formatting(caplog):
    # Test with default arguments
    styles, legend = get_data_table_css_style(SAMPLE_DATA)
    assert isinstance(styles, list)
    assert isinstance(legend, html.Div)

    # Test with specified columns
    styles, legend = get_data_table_css_style(SAMPLE_DATA, columns=["Feature1"])
    assert len(styles) == DEFAULT_N_COLOR_BINS
    assert len(legend) == 4

    # Test with non-numeric columns
    styles, legend = get_data_table_css_style(
        SAMPLE_DATA, columns=["Feature1", "Feature2", "Non_numeric"]
    )
    assert len(legend) == 4 * 2
    assert "['Non_numeric'] columns are not numeric and can not be color-coded" in caplog.text
