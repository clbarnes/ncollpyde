from __future__ import annotations

import logging
import warnings
from numbers import Number
from typing import Union, Sequence, Optional
from multiprocessing import cpu_count

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

from .ncollpyde import TriMeshWrapper

logger = logging.getLogger(__name__)

N_CPUS = cpu_count()
DEFAULT_THREADS = None

ArrayLike1D = Union[np.ndarray, Sequence[Number]]
ArrayLike2D = Union[np.ndarray, Sequence[Sequence[Number]]]


class Volume:
    def __init__(self, vertices: ArrayLike2D, triangles: ArrayLike2D, validate=False):
        """
        Create a volume described by a triangular mesh with N vertices and M triangles.

        :param vertices: Nx3 array-like of floats, coordinates of triangle corners
        :param triangles: Mx3 array-like of ints,
            indices of ``vertices`` which describe each triangle
        :param validate: bool, whether to validate mesh.
            If trimesh is installed, the mesh is checked for watertightness and correct
            winding, and repairs made if possible.
            Otherwise, only very basic checks are made.
        """
        vertices = np.asarray(vertices, np.float32)
        triangles = np.asarray(triangles, np.uint64)
        if validate:
            vertices, triangles = self._validate(vertices, triangles)
        self._impl = TriMeshWrapper(vertices.tolist(), triangles.tolist())

    def _validate(self, vertices: np.ndarray, triangles: np.ndarray):
        if trimesh:
            tm = trimesh.Trimesh(vertices, triangles)
            if not tm.is_volume:
                logger.info("Mesh not valid, attempting to fix")
                tm.fill_holes()
                tm.fix_normals()
                if tm.is_volume:
                    return tm.vertices, tm.faces
                else:
                    raise ValueError(
                        "Mesh is not a volume "
                        "(e.g. not watertight, incorrect winding) "
                        "and could not be fixed"
                    )
        else:
            warnings.warn("trimesh not installed; full validation not possible")

            if vertices.shape[1:] != (3,):
                raise ValueError("Vertices are not in 3D")

            if triangles.shape[1:] != (3,):
                raise ValueError("Triangles do not have 3 points")

            if triangles.max() >= len(vertices):
                raise ValueError("Some triangle vertices do not exist in points")

            return vertices, triangles

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
        self, coords: ArrayLike2D, threads: Optional[Union[int, bool]] = DEFAULT_THREADS
    ) -> np.ndarray:
        """Check whether multiple points (as a Px3 array-like) are in the volume.

        :param coords:
        :param threads: None,
            If ``threads`` is ``None``, the coordinates are tested in serial (but the
            GIL is released).
            If ``threads`` is ``True``, ``threads`` is set to the number of CPUs.
            If ``threads`` is something else (a number), the query will be parallelised
            over that number of threads.
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
            out = self._impl.contains_many_threaded(coords.tolist(), threads)
        return np.array(out, dtype=bool)

    @classmethod
    def from_meshio(cls, mesh, validate=False) -> Volume:
        try:
            return cls(mesh.points, mesh.cells["triangle"], validate)
        except KeyError:
            raise ValueError("Must have triangle cells")
