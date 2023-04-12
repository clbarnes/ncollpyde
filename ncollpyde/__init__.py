# -*- coding: utf-8 -*-

"""Top-level package for ncollpyde."""

__author__ = """Chris L. Barnes"""
__email__ = "chrislloydbarnes@gmail.com"

from .main import DEFAULT_RAYS  # noqa: F401
from .main import DEFAULT_SEED  # noqa: F401
from .main import DEFAULT_THREADS  # noqa: F401
from .main import INDEX  # noqa: F401
from .main import N_CPUS  # noqa: F401
from .main import PRECISION  # noqa: F401
from .main import Volume  # noqa: F401
from .main import configure_threadpool  # noqa: F401
from .ncollpyde import n_threads  # noqa: F401
from .ncollpyde import _version

__version__ = _version()
__version_info__ = tuple(int(n) for n in __version__.split("-")[0].split("."))

__all__ = ["Volume"]
