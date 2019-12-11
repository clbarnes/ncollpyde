# -*- coding: utf-8 -*-

"""Top-level package for ncollpyde."""

__author__ = """Chris L. Barnes"""
__email__ = "chrislloydbarnes@gmail.com"
__version__ = "0.4.1"
__version_info__ = tuple(int(n) for n in __version__.split("."))

from .main import Volume

__all__ = ["Volume"]
