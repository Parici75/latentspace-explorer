from .computation import add_callbacks as add_computation_callbacks
from .dropdowns import add_callbacks as add_dropdowns_callbacks
from .loading import add_callbacks as add_data_callbacks
from .output import add_callbacks as add_output_callbacks
from .plot import add_callbacks as add_plot_callbacks

__all__ = [
    "add_data_callbacks",
    "add_computation_callbacks",
    "add_dropdowns_callbacks",
    "add_output_callbacks",
    "add_plot_callbacks",
]
