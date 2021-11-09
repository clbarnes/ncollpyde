from multiprocessing import cpu_count

import meshio
import numpy as np
import pytest

from ncollpyde import Volume

SAMPLES_PER_DIM = 10
PADDING = 0.2
ITERATIONS = 20
CPU_COUNT = cpu_count()

CONTAINS_SERIAL = "containment serial"
INTERSECTION_SERIAL = "intersection serial"
CONTAINS_PARALLEL = "containment parallel"
INTERSECTION_PARALLEL = "intersection parallel"
DISTANCE = "distance"


class PyOctreeWrapper:
    """Wrapper around pyoctree's ray casting for volume checks.

    Adapted from https://github.com/schlegelp/pymaid/blob/master/pymaid/intersect.py
    """

    def __init__(self, vertices, faces, octree):
        self.vertices = np.asarray(vertices, dtype=np.float32, order="C")
        self.faces = np.asarray(faces, dtype=np.int32, order="C")

        # pass in so that the class does not need to re-import pyoctree
        self.octree = octree

    def contains(self, points, multi_ray=False):
        """Use pyoctree's raycsasting to test if points are within volume."""

        # Remove points outside of bounding box
        isin = self._in_bbox(points)
        in_points = points[isin]

        # mx = np.array(volume.vertices).max(axis=0)
        mn = np.array(self.vertices).min(axis=0)

        # Perform ray intersection on points inside bounding box
        rayPointList = np.array(
            [[[p[0], mn[1], mn[2]], p] for p in in_points], dtype=np.float32
        )

        # Get intersections and extract coordinates of intersection
        intersections = [
            np.array([i.p for i in self.octree.rayIntersection(ray)])
            for ray in rayPointList
        ]

        # In a few odd cases we can get the multiple intersections at the exact
        # same coordinate (something funny with the faces).
        unique_int = [
            np.unique(np.round(i), axis=0) if np.any(i) else i for i in intersections
        ]

        # Unfortunately rays are bidirectional -> we have to filter intersections
        # to those that occur "above" the point we are querying
        unilat_int = [
            i[i[:, 2] >= p] if np.any(i) else i
            for i, p in zip(unique_int, in_points[:, 2])
        ]

        # Count intersections
        int_count = [i.shape[0] for i in unilat_int]

        # Get odd (= in volume) numbers of intersections
        is_odd = np.remainder(int_count, 2) != 0

        # If we want to play it safe, run the above again with two additional rays
        # and find a majority decision.
        if multi_ray:
            # Run ray from left back
            rayPointList = np.array(
                [[[mn[0], p[1], mn[2]], p] for p in in_points], dtype=np.float32
            )
            intersections = [
                np.array([i.p for i in self.octree.rayIntersection(ray)])
                for ray in rayPointList
            ]
            unique_int = [
                np.unique(i, axis=0) if np.any(i) else i for i in intersections
            ]
            unilat_int = [
                i[i[:, 0] >= p] if np.any(i) else i
                for i, p in zip(unique_int, in_points[:, 0])
            ]
            int_count = [i.shape[0] for i in unilat_int]
            is_odd2 = np.remainder(int_count, 2) != 0

            # Run ray from lower left
            rayPointList = np.array(
                [[[mn[0], mn[1], p[2]], p] for p in in_points], dtype=np.float32
            )
            intersections = [
                np.array([i.p for i in self.octree.rayIntersection(ray)])
                for ray in rayPointList
            ]
            unique_int = [
                np.unique(i, axis=0) if np.any(i) else i for i in intersections
            ]
            unilat_int = [
                i[i[:, 1] >= p] if np.any(i) else i
                for i, p in zip(unique_int, in_points[:, 1])
            ]
            int_count = [i.shape[0] for i in unilat_int]
            is_odd3 = np.remainder(int_count, 2) != 0

            # Find majority consensus
            is_odd = is_odd.astype(int) + is_odd2.astype(int) + is_odd3.astype(int)
            is_odd = is_odd >= 2

        isin[isin] = is_odd
        return isin

    def _in_bbox(self, points):
        """Test if points are within a bounding box.
        Parameters
        ----------
        points :    numpy array
                    2D  array of xyz coordinates.
        bbox :      numpy array | pymaid.Volume
                    If Volume will use the volumes bounding box. If numpy array
                    must be either::
                        [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
                        [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        Returns
        -------
        numpy array
                    Boolean array with True/False for each point.
        """
        mx = np.array(self.vertices).max(axis=0)
        mn = np.array(self.vertices).min(axis=0)

        # Get points outside of bounding box
        bbox_out = (points >= mx).any(axis=1) | (points <= mn).any(axis=1)
        isin = ~bbox_out

        return isin


@pytest.fixture
def pyoctree_volume(mesh):
    try:
        from pyoctree import pyoctree
    except ImportError:
        pytest.skip("pyoctree not available")
    else:
        vertices = np.asarray(mesh.points, dtype=np.float64, order="C")
        faces = np.asarray(mesh.cells_dict["triangle"], dtype=np.int32, order="C")
        octree = pyoctree.PyOctree(vertices, faces)
        return PyOctreeWrapper(vertices, faces, octree)


@pytest.fixture
def ncollpyde_volume(mesh):
    return Volume.from_meshio(mesh)


@pytest.fixture
def trimesh_volume(mesh):
    try:
        import trimesh
    except ImportError:
        pytest.skip("trimesh not available")
    else:
        return trimesh.Trimesh(vertices=mesh.points, faces=mesh.cells_dict["triangle"])


@pytest.fixture
def expected(trimesh_volume, sample_points):
    return trimesh_volume.contains(sample_points)


def make_sample_points(
    aabb: np.ndarray, padding=PADDING, samples_per_dim=SAMPLES_PER_DIM
):
    """[[minx, miny, minz], [max, maxy, maxz]]"""
    mins, maxes = aabb
    ranges = maxes - mins
    mins -= ranges * padding
    maxes += ranges * padding
    sample1d = np.linspace(mins, maxes, samples_per_dim, dtype=Volume.dtype, axis=1)
    grid = np.meshgrid(*sample1d)
    return np.array([arr.flatten() for arr in grid]).T


@pytest.fixture
def sample_points(mesh: meshio.Mesh):
    return make_sample_points([mesh.points.min(axis=0), mesh.points.max(axis=0)])


def check_internals_equal(expected, actual, tolerance=0):
    """
    tolerance=N allows containment check to differ by N points.

    This is 'nearly equal' because of an intermittent failure.
    Despite seeding every RNG I can find,
    ~1/4 of the time a build will fail because of 82, rather than 81,
    points being internal in test_ncollpyde_contains_threaded.

    This is a problem, but not necessarily one I can fix,
    and it makes it very difficult to make releases.
    """
    assert np.abs(expected.sum() - actual.sum()) <= tolerance
    assert (expected != actual).sum() <= tolerance


@pytest.mark.benchmark(group=CONTAINS_SERIAL)
def test_trimesh_contains(trimesh_volume, sample_points, expected, benchmark):
    actual = benchmark(trimesh_volume.contains, sample_points)
    check_internals_equal(expected, actual)


@pytest.mark.benchmark(group=CONTAINS_SERIAL)
@pytest.mark.parametrize("n_rays", [0, 1, 2, 4, 8, 16])
def test_ncollpyde_contains(mesh, n_rays, sample_points, expected, benchmark):
    ncollpyde_volume = Volume.from_meshio(mesh, n_rays=n_rays)
    ncollpyde_volume.threads = 0
    actual = benchmark(ncollpyde_volume.contains, sample_points)
    if n_rays:
        check_internals_equal(expected, actual, 0)


@pytest.mark.benchmark(group=CONTAINS_SERIAL)
@pytest.mark.parametrize("safe", [True, False], ids=lambda x: ["FAST", "SAFE"][int(x)])
def test_pyoctree_contains(
    pyoctree_volume: PyOctreeWrapper, safe, sample_points, expected, benchmark
):
    actual = benchmark(pyoctree_volume.contains, sample_points, safe)
    if safe:
        check_internals_equal(expected, actual)
    else:
        pytest.xfail("pyoctree results are not consistent unless in SAFE mode")


@pytest.mark.benchmark(group=CONTAINS_PARALLEL)
@pytest.mark.parametrize("threads", [0, 1, 2, 4, 8, 16])
def test_ncollpyde_contains_threaded(mesh, sample_points, expected, benchmark, threads):
    if threads > CPU_COUNT:
        pytest.skip(f"Wanted {threads} threads, only have {CPU_COUNT} CPUs")
    ncollpyde_volume = Volume.from_meshio(mesh, n_rays=3, threads=threads)
    actual = benchmark(ncollpyde_volume.contains, sample_points)
    check_internals_equal(expected, actual, 0)


@pytest.mark.benchmark(group=INTERSECTION_PARALLEL)
@pytest.mark.parametrize("threads", [0, 1, 2, 4, 8, 16])
def test_ncollpyde_intersection(mesh, benchmark, threads):
    if threads > CPU_COUNT:
        pytest.skip(f"Wanted {threads} threads, only have {CPU_COUNT} CPUs")

    n_edges = 1_000
    rng = np.random.default_rng(1991)
    starts = rng.random((n_edges, 3)) * 2 - 0.5
    stops = rng.random((n_edges, 3)) * 2 - 0.5

    vol = Volume.from_meshio(mesh, threads=threads)
    benchmark(vol.intersections, starts, stops)


@pytest.mark.benchmark(group=DISTANCE)
@pytest.mark.parametrize("signed", [True, False])
def test_ncollpyde_distance(mesh, benchmark, signed):
    n_points = 1_000
    rng = np.random.default_rng(1991)
    points = rng.random((n_points, 3)) * 2 - 0.5
    vol = Volume.from_meshio(mesh)
    benchmark(vol.distance, points, signed)
