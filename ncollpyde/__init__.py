# -*- coding: utf-8 -*-

"""Top-level package for ncollpyde."""

__author__ = """Chris L. Barnes"""
__email__ = "chrislloydbarnes@gmail.com"

from .ncollpyde import _version
from .main import (  # noqa: F401
    Volume,
    N_CPUS,  # noqa: F401
    DEFAULT_THREADS,  # noqa: F401
    DEFAULT_RAYS,  # noqa: F401
    DEFAULT_SEED,  # noqa: F401
    PRECISION,  # noqa: F401
)

__version__ = _version()
__version_info__ = tuple(int(n) for n in __version__.split("-")[0].split("."))

__all__ = ["Volume"]
