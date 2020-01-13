from __future__ import annotations

import logging
import warnings
from numbers import Number
from typing import Union, Sequence, Optional, TYPE_CHECKING
from multiprocessing import cpu_count

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

from .ncollpyde import TriMeshWrapper

if TYPE_CHECKING:
    import meshio


logger = logging.getLogger(__name__)

N_CPUS = cpu_count()
DEFAULT_THREADS = 0

ArrayLike1D = Union[np.ndarray, Sequence[Number]]
ArrayLike2D = Union[np.ndarray, Sequence[Sequence[Number]]]


class Volume:
    threads = DEFAULT_THREADS

    def __init__(
        self,
        vertices: ArrayLike2D,
        triangles: ArrayLike2D,
        validate=False,
        threads=None,
    ):
        """
        Create a volume described by a triangular mesh with N vertices and M triangles.

        :param vertices: Nx3 array-like of floats, coordinates of triangle corners
        :param triangles: Mx3 array-like of ints,
            indices of ``vertices`` which describe each triangle
        :param validate: bool, whether to validate mesh.
            If trimesh is installed, the mesh is checked for watertightness and correct
            winding, and repairs made if possible.
            Otherwise, only very basic checks are made.
        :param validate: optional number or True, sets default threading for containment
            checks with this instance. Can also be set on the class.
        """
        vertices = np.asarray(vertices, np.float32)
        triangles = np.asarray(triangles, np.uint64)
        if validate:
            vertices, triangles = self._validate(vertices, triangles)
        if threads is not None:
            self.threads = threads
        self._impl = TriMeshWrapper(vertices.tolist(), triangles.tolist())

    def _validate(self, vertices: np.ndarray, triangles: np.ndarray):
        if trimesh:
            tm = trimesh.Trimesh(vertices, triangles)
            if not tm.is_volume:
                logger.info("Mesh not valid, attempting to fix")
                tm.fill_holes()
                tm.fix_normals()
                if not tm.is_volume:
                    raise ValueError(
                        "Mesh is not a volume "
                        "(e.g. not watertight, incorrect winding) "
                        "and could not be fixed"
                    )

            return tm.vertices, tm.faces

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
        self, coords: ArrayLike2D, threads: Optional[Union[int, bool]] = None
    ) -> np.ndarray:
        """Check whether multiple points (as a Px3 array-like) are in the volume.

        :param coords:
        :param threads: None,
            If ``threads`` is ``None``, the instance's ``threads`` attribute (default 0)
            is used.
            If ``threads`` is ``True``, ``threads`` is set to the number of CPUs.
            If ``threads`` is 0, the query will be done in serial (but the GIL will be
            released)
            If ``threads`` is something else (a number), the query will be parallelised
            over that number of threads.
        :return: np.ndarray of bools, whether each point is inside the volume
        """
        coords = np.asarray(coords, np.float32)
        if coords.shape[1:] != (3,):
            raise ValueError("Coords is not a Nx3 array-like")

        if threads is None:
            threads = self.threads

        if not threads:
            out = self._impl.contains_many(coords.tolist())
        else:
            if threads is True:
                threads = N_CPUS
            out = self._impl.contains_many_threaded(coords.tolist(), threads)
        return np.array(out, dtype=bool)

    @classmethod
    def from_meshio(cls, mesh: "meshio.Mesh", validate=False, threads=None) -> Volume:
        """
        Convenience function for instantiating a Volume from a meshio Mesh.

        :param mesh: meshio Mesh whose only cells are triangles.
        :param validate: as passed to __init__, defaults to False
        :param threads: as passed to __init__, defaults to None
        :raises ValueError: if Mesh does not have triangle cells
        :return: Volume instance
        """
        try:
            return cls(mesh.points, mesh.cells["triangle"], validate, threads)
        except KeyError:
            raise ValueError("Must have triangle cells")

    @property
    def points(self) -> np.array:
        """
        Nx3 array of float32 describing vertices
        """
        return np.array(self._impl.points(), np.float32)

    @property
    def faces(self) -> np.array:
        """
        Mx3 array of uint64 describing indexes into points array making up triangles
        """
        return np.array(self._impl.faces(), np.uint64)

    @property
    def extents(self) -> np.array:
        """
        [
            [xmin, ymin, zmin],
            [xmax, ymax, zmax],
        ]
        """
        return np.array(self._impl.aabb(), np.float32)
