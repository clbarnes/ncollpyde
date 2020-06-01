import numpy as np
import trimesh

import meshio
import pytest

from ncollpyde import Volume

SAMPLES_PER_DIM = 10
PADDING = 0.2
THREADS = 3
ITERATIONS = 20


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
        faces = np.asarray(mesh.cells["triangle"], dtype=np.int32, order="C")
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
        return trimesh.Trimesh(vertices=mesh.points, faces=mesh.cells["triangle"])


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


def check_internals_equal(expected, actual):
    assert expected.sum() == actual.sum()
    assert np.array_equal(expected, actual)


def test_trimesh(trimesh_volume: trimesh.Trimesh, sample_points, expected, benchmark):
    actual = benchmark(trimesh_volume.contains, sample_points)
    check_internals_equal(expected, actual)


@pytest.mark.parametrize("n_rays", [0, 1, 2, 4, 8, 16])
def test_ncollpyde(mesh, n_rays, sample_points, expected, benchmark):
    ncollpyde_volume = Volume.from_meshio(mesh, n_rays=n_rays)
    actual = benchmark(ncollpyde_volume.contains, sample_points)
    if n_rays:
        check_internals_equal(expected, actual)


# def test_ncollpyde_threaded(
#     ncollpyde_volume: Volume, sample_points, expected, benchmark
# ):
#     actual = benchmark(ncollpyde_volume.contains, sample_points, threads=THREADS)
#     check_internals_equal(expected, actual)


# def test_ncollpyde_complex(sez_right: Volume, benchmark):
#     sample_points = make_sample_points(sez_right.extents, 0.1, 100)
#     benchmark(sez_right.contains, sample_points)


# def test_ncollpyde_complex_threads(sez_right: Volume, benchmark):
#     sample_points = make_sample_points(sez_right.extents, 0.1, 100)
#     benchmark(sez_right.contains, sample_points, threads=THREADS)


@pytest.mark.xfail(reason="Results do not match yet")
def test_pyoctree(pyoctree_volume: PyOctreeWrapper, sample_points, expected, benchmark):
    actual = benchmark(pyoctree_volume.contains, sample_points)
    check_internals_equal(expected, actual)
