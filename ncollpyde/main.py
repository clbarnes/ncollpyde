from __future__ import annotations
import itertools
from numbers import Number
from typing import Union, Sequence, Optional

import numpy as np
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count

from .ncollpyde import TriMeshWrapper

N_CPUS = cpu_count()

ArrayLike1D = Union[np.ndarray, Sequence[Number]]
ArrayLike2D = Union[np.ndarray, Sequence[Sequence[Number]]]


class Volume:
    def __init__(self, vertices: ArrayLike2D, triangles: ArrayLike2D, validate=False):
        """
        Create a volume described by a triangular mesh with N vertices and M triangles.

        :param vertices: Nx3 array-like of floats, coordinates of triangle corners
        :param triangles: Mx3 array-like of ints,
            indices of ``vertices`` which describe each triangle
        :param validate: bool, whether to make sure that sizes of ``vertices`` and
            ``triangles`` are compatible
        """
        vertices = np.asarray(vertices, np.float32)
        triangles = np.asarray(triangles, np.uint64)
        if validate:
            self._validate(vertices, triangles)
        self._impl = TriMeshWrapper(vertices.tolist(), triangles.tolist())

    def _validate(self, vertices: np.ndarray, triangles: np.ndarray):
        if vertices.shape[1:] != (3,):
            raise ValueError("Vertices are not in 3D")

        if triangles.shape[1:] != (3,):
            raise ValueError("Triangles do not have 3 points")

        if triangles.max() >= len(vertices):
            raise ValueError("Some triangle vertices do not exist in points")

    def __contains__(self, item: ArrayLike1D) -> bool:
        """Check whether a single point is in the volume.

        :param item:
        :return:
        """
        item = np.asarray(item, np.float32)
        if item.shape != (3,):
            raise ValueError("Item is not a 3-length array-like")
        return self._impl.contains(item.tolist())

    def contains(
        self, coords: ArrayLike2D, threads: Optional[Union[int, bool]] = None
    ) -> np.ndarray:
        """Check whether multiple points (as a Px3 array-like) are in the volume.

        :param coords:
        :param threads: None,
            If ``threads`` is ``None``, the coordinates are tested in serial (but the
            GIL is released).
            If ``threads`` is ``True``, ``threads`` is set to the number of CPUs.
            If ``threads`` is something else, the query array is split into chunks of
            length approximately equal to ``threads``, and they are tested concurrently
            (using a Thread-based Pool: Processes are not necessary because the
            underlying implementation releases the GIL).
        :return: np.ndarray of bools, whether each point is inside the volume
        """
        coords = np.asarray(coords, np.float32)
        if coords.shape[1:] != (3,):
            raise ValueError("Coords is not a Nx3 array-like")

        if threads is None:
            out = self._impl.contains_many(coords.tolist())
        else:
            if threads is True:
                threads = N_CPUS

            threads = min(threads, len(coords))
            with Pool(threads) as p:
                out = list(
                    itertools.chain.from_iterable(
                        p.map(self._impl.contains_many, np.array_split(coords, threads))
                    )
                )
        return np.array(out, dtype=bool)

    @classmethod
    def from_meshio(cls, mesh) -> Volume:
        try:
            return cls(mesh.points, mesh.cells["triangle"])
        except KeyError:
            raise ValueError("Must have triangle cells")
