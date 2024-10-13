"""A Dash framework for exploratory data analysis of multidimensional datasets.


This package is organized as follows:

- libs:
    The source code for the layout, callback, and backend logics.

- utils:
    Utility modules for database, cache, and application set up.

- app_builder:
    The entrypoint for building the Dash webapp.

- latent_space_interface:
    The interface for spawning the app from a Flask development server.
"""

from .latent_space_interface import LatentSpaceInterface

__all__ = ["LatentSpaceInterface"]
