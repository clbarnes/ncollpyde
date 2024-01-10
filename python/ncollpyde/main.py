import logging
import random
import warnings
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Optional, Tuple, Union, List

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._ncollpyde import (
    TriMeshWrapper,
    _index,
    _precision,
    _configure_threadpool,
)

if TYPE_CHECKING:
    import meshio


logger = logging.getLogger(__name__)

N_CPUS = cpu_count()
DEFAULT_THREADS = True
DEFAULT_RAYS = 3
DEFAULT_SEED = 1991

PRECISION = np.dtype(_precision())
INDEX = np.dtype(_index())


def configure_threadpool(n_threads: Optional[int], name_prefix: Optional[str]):
    """Configure the thread pool used for parallelisation.

    Must be called a maximum of once,
    and only before the first parallelised ncollpyde query.
    This will be used for all parallelised ncollpyde queries.

    Parameters
    ----------
    n_threads : Optional[int]
        Number of threads to use.
        If None or 0, will use the default, see
        https://docs.rs/rayon/latest/rayon/struct.ThreadPoolBuilder.html#method.num_threads.
    name_prefix : Optional[str]
        How to name threads created by this library.
        Will be suffixed with the thread index.
        If not given, will use the rayon default.

    Raises
    ------
    RuntimeError
        If the pool could not be built for any reason.
    """
    _configure_threadpool(n_threads, name_prefix)


def interpret_threads(threads: Optional[Union[int, bool]], default=DEFAULT_THREADS):
    if isinstance(threads, bool):
        return threads

    if threads is None:
        return interpret_threads(default)

    threads = int(threads)

    warnings.warn(
        "ncollpyde's API has changed; `threads` should now be a boolean. "
        "See https://github.com/clbarnes/ncollpyde/issues/27 for more details",
        DeprecationWarning,
    )
    t = bool(threads)
    logger.warning("Interpreting deprecated `threads=%s` as %s", threads, t)
    return t


class Volume:
    dtype: np.dtype = PRECISION
    """Float data type used internally"""

    threads: bool = DEFAULT_THREADS
    """Whether to use threading"""

    def __init__(
        self,
        vertices: ArrayLike,
        triangles: ArrayLike,
        validate=False,
        threads: Optional[bool] = None,
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
        :param threads: optional bool, whether to parallelise queries.
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
        vert = np.asarray(vertices, self.dtype)
        if len(vert) > np.iinfo(INDEX).max:
            raise ValueError(f"Cannot represent {len(vert)} vertices with {INDEX}")
        tri = np.asarray(triangles, INDEX)
        if validate:
            vert, tri = self._validate(vert, tri)
        self.threads = self._interpret_threads(threads)
        if ray_seed is None:
            logger.warning(
                "Using unseeded random number generator for containment-checking rays; "
                "results may be inconsistent across repeats."
            )
            ray_seed = random.randrange(0, 2**64)

        self.n_rays = int(n_rays)
        inner_rays = 0 if self.n_rays < 0 else self.n_rays

        self._impl = TriMeshWrapper(vert, tri, inner_rays, ray_seed)

    def _validate(
        self, vertices: np.ndarray, triangles: np.ndarray
    ) -> Tuple[NDArray[np.float64], NDArray[np.uint32]]:
        try:
            # todo: may not be necessary now parry can do some topology checks
            import trimesh

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

            return tm.vertices.astype(self.dtype), tm.faces.astype(np.uint32)
        except ImportError:
            warnings.warn("trimesh not installed; full validation not possible")

            if vertices.shape[1:] != (3,):
                raise ValueError("Vertices are not in 3D")

            if triangles.shape[1:] != (3,):
                raise ValueError("Triangles do not have 3 points")

            if triangles.max() >= len(vertices):
                raise ValueError("Some triangle vertices do not exist in points")

            return vertices, triangles

    def __contains__(self, item: ArrayLike) -> bool:
        """Check whether a single point is in the volume.

        Uses the slower, more robust signed distance strategy.
        For more control over the strategy, see the ``self.contains()`` method.
        """
        return self.contains(np.asarray([item]), -1)[0]

    def _interpret_threads(self, threads: Optional[Union[int, bool]]) -> bool:
        return interpret_threads(threads, self.threads)

    def distance(
        self,
        coords: ArrayLike,
        signed: bool = True,
        *,
        threads: Optional[bool] = None,
    ) -> np.ndarray:
        """Check the distance from the volume to multiple points (as a Px3 array-like).

        Distances are reported to the boundary of the volume.
        By default, if the point is inside the volume,
        the distance will be reported as negative.

        :param coords:
        :param signed: bool, default True.
            Whether distances to points inside the volume
            should be reported as negative.
        :param threads: None,
            Whether to parallelise the queries. If ``None`` (default),
            refer to the instance's ``threads`` attribute.
        :return: np.ndarray of float, the distance from the volume to each given point
        """
        coords = np.asarray(coords, self.dtype)
        if coords.shape[1:] != (3,):
            raise ValueError("Coords is not a Nx3 array-like")

        return self._impl.distance(coords, signed, self._interpret_threads(threads))

    def contains(
        self,
        coords: ArrayLike,
        n_rays: Optional[int] = None,
        consensus: Optional[int] = None,
        *,
        threads: Optional[bool] = None,
    ) -> NDArray[np.bool_]:
        """Check whether multiple points (as a Px3 array-like) are in the volume.

        :param coords:
        :param n_rays: Optional[int]
            If None, use the maximum rays defined on construction.
            If < 1, use signed distance strategy
            (more robust, but slower for many meshes).
            Otherwise, use this many meshes, up to the maximum defined on construction.
        :param consensus: Optional[int]
            If using ray casting strategy, how many rays need to hit a backface
            in order to call the point internal?
            Rays from an external point may erroneously report hitting a backface
            if they skim an edge (see https://github.com/clbarnes/ncollpyde/issues/3 )
            If None, will be set to ``n_rays // 2 + 1`` (i.e. >50%).
            Ignored if using signed distance strategy.
        :param threads: None,
            Whether to parallelise the queries. If ``None`` (default),
            refer to the instance's ``threads`` attribute.
        :return: np.ndarray of bools, whether each point is inside the volume
        """
        coords = self._as_points(coords)
        if n_rays is None:
            n_rays = self.n_rays
        elif n_rays < 1:
            n_rays = None
        elif n_rays > self.n_rays:
            logger.warning(
                "Requested %s rays, using the maximum of %s", n_rays, self.n_rays
            )
            n_rays = self.n_rays

        if n_rays is None:
            consensus = 1
            n_rays = 0
        else:
            if consensus is None:
                consensus = n_rays // 2 + 1
            elif consensus > n_rays:
                raise ValueError(
                    "Requested consensus of %s rays but only casting %s",
                    consensus,
                    n_rays,
                )

        return self._impl.contains(
            coords, n_rays, consensus, self._interpret_threads(threads)
        )

    def _as_points(self, points: ArrayLike) -> NDArray:
        p = np.asarray(points, self.dtype)
        if p.shape[1:] != (3,):
            raise ValueError("Points must be Nx3 array-like")
        return p

    def _validate_points(self, *points: ArrayLike) -> List[NDArray]:
        """Ensure that arrays are equal-length sets of points."""
        ndim = None
        out = []

        for p_raw in points:
            p = self._as_points(p_raw)
            nd = p.shape[1:]
            if ndim is None:
                ndim = nd
            elif ndim != nd:
                raise ValueError("Point arrays are not the same shape")
            out.append(p)

        return out

    def _sdf_intersections(
        self, points: ArrayLike, vectors: ArrayLike, *, threads: Optional[bool] = None
    ) -> Tuple[NDArray, NDArray]:
        """Compute values required for signed distance field.

        :param points: Nx3 ndarray of floats
            Points to calculate the distance from.
            Should be within the axis-aligned bounding box of the mesh.
        :param vectors: Nx3 ndarray of floats
            Directions to fire rays from the given points.
            Need not be normalized.
        :param threads: None,
            Whether to parallelise the queries. If ``None`` (default),
            refer to the instance's ``threads`` attribute.
        :return: 2-tuple N-length np.ndarrays of floats.
            The first is the distance,
            which is negative if the collision is with a backface,
            and infinity if there is no collision.
            The second is the absolute dot product of the normalized vector
            with the unit normal of the feature the ray hit,
            NaN if there was no collision.
        """
        p, v = self._validate_points(points, vectors)
        return self._impl.sdf_intersections(p, v, self._interpret_threads(threads))

    def intersections(
        self,
        src_points: ArrayLike,
        tgt_points: ArrayLike,
        *,
        threads: Optional[bool] = None,
    ) -> Tuple[NDArray[np.uint64], NDArray[np.float64], NDArray[np.bool_]]:
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
        https://github.com/dimforge/ncollide/issues/335 .

        N.B. threads=True here uses a slightly different implementation,
        so you may not see the performance increase as with other methods.

        :param src_points: Nx3 array-like
        :param tgt_points: Nx3 array-like
        :param threads: None,
            Whether to parallelise the queries. If ``None`` (default),
            refer to the instance's ``threads`` attribute.
        :raises ValueError: Inputs have different shapes or are not Nx3
        :return: tuple of
          uint array of indices (N),
          float array of locations (Nx3),
          bool array of is_backface (N)
        """
        src, tgt = self._validate_points(src_points, tgt_points)

        if self._interpret_threads(threads):
            return self._impl.intersections_many_threaded2(src, tgt)
        else:
            return self._impl.intersections_many(src, tgt)

    @classmethod
    def from_meshio(
        cls,
        mesh: "meshio.Mesh",
        validate=False,
        threads=None,
        n_rays=DEFAULT_RAYS,
        ray_seed=DEFAULT_SEED,
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
            triangles = mesh.cells_dict["triangle"]
        except KeyError:
            raise ValueError("Must have triangle cells")

        return cls(
            mesh.points,
            triangles,
            validate,
            threads,
            n_rays,
            ray_seed,
        )

    @property
    def points(self) -> NDArray[np.float64]:
        """
        Nx3 array of float describing vertices
        """
        return self._impl.points()

    @property
    def faces(self) -> NDArray[np.uint32]:
        """
        Mx3 array of uint32 describing indexes into points array making up triangles
        """
        return self._impl.faces()

    @property
    def extents(self) -> NDArray[np.float64]:
        """Axis-aligned bounding box of the volume.

        :return: 2x3 numpy array of floats,
          where the first row is mins and the second row is maxes.
        """
        return self._impl.aabb()

    @property
    def rays(self) -> NDArray[np.float64]:
        """
        Rays used to detect containment as an ndarray of shape (n_rays, 3).

        These rays are in random directions (set with the ray seed on construction),
        and are the length of the diameter of the volume's bounding sphere.
        """
        return self._impl.rays()


def points_around_vol(vol: Volume, n: int, pad: float = 0.2, seed=1991):
    ext = vol.extents
    ranges = ext[1] - ext[0]
    to_pad = ranges * pad

    rng = np.random.default_rng(seed)
    return rng.uniform(ext[0] - to_pad, ext[1] + to_pad, (n, 3))
