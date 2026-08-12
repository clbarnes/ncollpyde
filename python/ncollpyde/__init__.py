"""Top-level package for ncollpyde."""

__author__ = """Chris L. Barnes"""
__email__ = "chrislloydbarnes@gmail.com"

from ._ncollpyde import (
    _version,
    n_threads,  # noqa: F401
)
from .main import (
    DEFAULT_RAYS,  # noqa: F401
    DEFAULT_SEED,  # noqa: F401
    DEFAULT_THREADS,  # noqa: F401
    INDEX,  # noqa: F401
    N_CPUS,  # noqa: F401
    PRECISION,  # noqa: F401
    Volume,
    configure_threadpool,  # noqa: F401
)

__version__ = _version()
__version_info__ = tuple(int(n) for n in __version__.split("-")[0].split("."))

__all__ = ["Volume"]
