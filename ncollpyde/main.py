import logging
import random
import warnings
from multiprocessing import cpu_count
from numbers import Number
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

from .ncollpyde import TriMeshWrapper, _precision

if TYPE_CHECKING:
    import meshio


logger = logging.getLogger(__name__)

N_CPUS = cpu_count()
DEFAULT_THREADS = 0
DEFAULT_RAYS = 3
DEFAULT_SEED = 1991

PRECISION = np.dtype(_precision())

ArrayLike1D = Union[np.ndarray, Sequence[Number]]
ArrayLike2D = Union[np.ndarray, Sequence[Sequence[Number]]]


class Volume:
    dtype = PRECISION
    """Float data type used internally"""

    threads = DEFAULT_THREADS
    """Default number of threads used"""

    def __init__(
        self,
        vertices: ArrayLike2D,
        triangles: ArrayLike2D,
        validate=False,
        threads=None,
        n_rays=DEFAULT_RAYS,
        ray_seed=DEFAULT_SEED,
    ):
        f"""
        Create a volume described by a triangular mesh with N vertices and M triangles.

        :param vertices: Nx3 array-like of floats, coordinates of triangle corners
        :param triangles: Mx3 array-like of ints,
            indices of ``vertices`` which describe each triangle
        :param validate: bool, whether to validate mesh.
            If trimesh is installed, the mesh is checked for watertightness and correct
            winding, and repairs made if possible.
            Otherwise, only very basic checks are made.
        :param threads: optional number or True, sets default threading for containment
            checks with this instance. Can also be set on the class.
        :param n_rays: int (default {DEFAULT_RAYS}), rays used to check containment.
            The underlying library sometimes reports false positives:
            casting multiple rays drastically reduces the chances of this.
            As the bug only affects ray casts and only produces false positives,
            unnecessary ray casts are short-circuited if:
                - the point is not in the bounding box
                - the point is on the hull
                - one ray reports that the point is external.
        :param ray_seed: int >=0 (default {DEFAULT_SEED}), used for generating rays.
            If None, use a random seed.
        """
        vertices = np.asarray(vertices, self.dtype)
        triangles = np.asarray(triangles, np.uint64)
        if validate:
            vertices, triangles = self._validate(vertices, triangles)
        if threads is not None:
            self.threads = threads
        if ray_seed is None:
            ray_seed = random.randrange(0, 2 ** 64)

        self._impl = TriMeshWrapper(
            vertices.tolist(), triangles.tolist(), int(n_rays), ray_seed
        )

    def _validate(self, vertices: np.ndarray, triangles: np.ndarray):
        if trimesh:
            tm = trimesh.Trimesh(vertices, triangles, validate=True)
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
        item = np.asarray(item, self.dtype)
        if item.shape != (3,):
            raise ValueError("Item is not a 3-length array-like")
        return self._impl.contains(item.tolist())

    def _threadcount(self, threads):
        """
        If ``threads`` is ``None``, the instance's ``threads`` attribute (default 0)
        is used.
        If ``threads`` is ``True``, ``threads`` is set to the number of CPUs.
        If ``threads`` is 0, the query will be done in serial (but the GIL will be
        released)
        If ``threads`` is something else (a number), the query will be parallelised
        over that number of threads.

        :return: int, number of threads to use (0 for serial)
        """
        if threads is None:
            threads = self.threads

        if not threads:
            return 0
        elif threads is True:
            return N_CPUS
        else:
            return threads

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
        coords = np.asarray(coords, self.dtype)
        if coords.shape[1:] != (3,):
            raise ValueError("Coords is not a Nx3 array-like")

        coords_list = coords.tolist()

        n_threads = self._threadcount(threads)
        if n_threads == 0:
            out = self._impl.contains_many(coords_list)
        else:
            out = self._impl.contains_many_threaded(coords_list, n_threads)

        return np.array(out, dtype=bool)

    def intersections(
        self,
        src_points: ArrayLike2D,
        tgt_points: ArrayLike2D,
        threads: Optional[Union[int, bool]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get intersections between line segments and volume.

        Line segments are defined by their start (source) and end (target) points.
        Only the first intersection for a given line segment is reported.

        Even if there is only one line segment to check,
        the argument arrays must have 2 dimensions, e.g. ``[[0, 0, 0]], [[1, 1, 1]]``.

        The output arrays are:

        * Which line segment the intersection refers to,
          as an index into the argument arrays
        * The point of intersection
        * Whether the intersection is with the inside face of the mesh

        N.B. the inside face check will report True
        for cases where a line touches ("skims") an external edge; see
        https://github.com/dimforge/ncollide/issues/335
        .

        :param src_points: Nx3 array-like
        :param tgt_points: Nx3 array-like
        :param threads: None,
            If ``threads`` is ``None``, the instance's ``threads`` attribute (default 0)
            is used.
            If ``threads`` is ``True``, ``threads`` is set to the number of CPUs.
            If ``threads`` is 0, the query will be done in serial (but the GIL will be
            released)
            If ``threads`` is something else (a number), the query will be parallelised
            over that number of threads.
        :raises ValueError: Inputs have different shapes or are not Nx3
        :return: tuple of
          uint array of indices (N),
          float array of locations (Nx3),
          bool array of is_backface (N)
        """
        src = np.asarray(src_points, self.dtype)
        tgt = np.asarray(tgt_points, self.dtype)

        if src.shape != tgt.shape:
            raise ValueError("Source and target points arrays must be the same shape")

        if src.shape[1:] != (3,):
            raise ValueError("Points must be Nx3 array-like")

        src_list = src.tolist()
        tgt_list = tgt.tolist()

        n_threads = self._threadcount(threads)
        if n_threads == 0:
            idxs, points, bfs = self._impl.intersections_many(src_list, tgt_list)
        else:
            idxs, points, bfs = self._impl.intersections_many_threaded(
                src_list, tgt_list, n_threads
            )

        return (
            np.array(idxs, np.uint64),
            np.array(points, self.dtype),
            np.array(bfs, bool),
        )

    @classmethod
    def from_meshio(
        cls,
        mesh: "meshio.Mesh",
        validate=False,
        threads=None,
        n_rays=DEFAULT_RAYS,
        ray_seed=None,
    ) -> "Volume":
        """
        Convenience function for instantiating a Volume from a meshio Mesh.

        :param mesh: meshio Mesh whose only cells are triangles.
        :param validate: as passed to ``__init__``, defaults to False
        :param threads: as passed to ``__init__``, defaults to None
        :param n_rays: as passed to ``__init__``, defaults to 3
        :param ray_seed: as passed to ``__init__``, defaults to None (random)
        :raises ValueError: if Mesh does not have triangle cells
        :return: Volume instance
        """
        try:
            return cls(
                mesh.points,
                mesh.cells_dict["triangle"],
                validate,
                threads,
                n_rays,
                ray_seed,
            )
        except KeyError:
            raise ValueError("Must have triangle cells")

    @property
    def points(self) -> np.ndarray:
        """
        Nx3 array of float describing vertices
        """
        return np.array(self._impl.points(), self.dtype)

    @property
    def faces(self) -> np.ndarray:
        """
        Mx3 array of uint64 describing indexes into points array making up triangles
        """
        return np.array(self._impl.faces(), np.uint64)

    @property
    def extents(self) -> np.ndarray:
        """Axis-aligned bounding box of the volume.

        :return: 2x3 numpy array of floats,
          where the first row is mins and the second row is maxes.
        """
        return np.array(self._impl.aabb(), self.dtype)

    @property
    def rays(self) -> np.ndarray:
        """
        Rays used to detect containment as an ndarray of shape (n_rays, 3).

        These rays are in random directions (set with the ray seed on construction),
        and are the length of the diameter of the volume's bounding sphere.
        """
        return np.array(self._impl.rays(), self.dtype)
