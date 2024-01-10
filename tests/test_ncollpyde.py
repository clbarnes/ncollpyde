#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `ncollpyde` package."""
from itertools import product
from math import pi, sqrt
import sys
import subprocess as sp
import logging
from ncollpyde.main import points_around_vol

import numpy as np
import pytest

import trimesh

from ncollpyde import PRECISION, Volume, configure_threadpool

logger = logging.getLogger(__name__)

points_expected = [
    ([-2.3051376, -4.1556454, 1.9047838], True),  # internal
    ([-0.35222054, -0.513299, 7.6191354], False),  # external but in AABB
    ([0.13970017, 0.0928266, 4.7073355], True),  # vertex
    ([10, 10, 10], False),  # out of AABB
]


@pytest.mark.parametrize(["point", "expected"], points_expected)
def test_single(mesh, point, expected):
    """Sample pytest test function with the pytest fixture as an argument."""
    vol = Volume.from_meshio(mesh)
    assert (point in vol) == expected


def test_corner(mesh):
    vol = Volume.from_meshio(mesh)
    assert mesh.points[0] in vol


@pytest.mark.parametrize("threads", [None, False, True])
def test_many(mesh, threads):
    points = []
    expected = []
    for p, e in points_expected:
        points.append(p)
        expected.append(e)

    vol = Volume.from_meshio(mesh)
    assert np.array_equal(vol.contains(points, threads=threads), expected)


def test_contains_results(volume: Volume):
    pts = points_around_vol(volume, 1000, 0.1)
    ray = volume.contains(pts, n_rays=3, consensus=3, threads=False)
    psnorms = volume.contains(pts, n_rays=-1, threads=False)
    assert np.allclose(ray, psnorms)


def test_no_validation(mesh):
    triangles = mesh.cells_dict["triangle"]
    Volume(mesh.points, triangles, True)


@pytest.mark.skipif(not trimesh, reason="Requires trimesh")
def test_can_repair_hole(mesh):
    triangles = mesh.cells_dict["triangle"]
    triangles = triangles[:-1]
    Volume(mesh.points, triangles, True)


@pytest.mark.skipif(not trimesh, reason="Requires trimesh")
def test_can_repair_inversion(mesh):
    triangles = mesh.cells_dict["triangle"]
    triangles[-1] = triangles[-1, ::-1]
    Volume(mesh.points, triangles, True)


@pytest.mark.skipif(not trimesh, reason="Requires trimesh")
def test_can_repair_inversions(mesh):
    triangles = mesh.cells_dict["triangle"]
    triangles = triangles[:, ::-1]
    Volume(mesh.points, triangles, True)


@pytest.mark.skipif(not trimesh, reason="Requires trimesh")
def test_inversions_repaired(simple_mesh):
    center = [0.5, 0.5, 0.5]

    orig_points = simple_mesh.points
    orig_triangles = simple_mesh.cells_dict["triangle"]
    assert center in Volume(orig_points, orig_triangles)

    inv_triangles = orig_triangles[:, ::-1]
    assert center not in Volume(orig_points, inv_triangles)

    assert center in Volume(orig_points, inv_triangles, validate=True)


def test_points(mesh):
    points = mesh.points
    vol = Volume(mesh.points, mesh.cells_dict["triangle"])

    actual = sorted(tuple(p) for p in vol.points)
    expected = sorted(tuple(p) for p in points)
    assert actual == expected


@pytest.mark.skipif(not trimesh, reason="Requires trimesh")
def test_triangles(mesh):
    points = mesh.points
    triangles = mesh.cells_dict["triangle"]
    expected = trimesh.Trimesh(points, triangles)

    vol = Volume(mesh.points, triangles)
    actual = trimesh.Trimesh(vol.points, vol.faces)

    assert expected.volume == actual.volume


def test_extents(mesh):
    points = mesh.points
    vol = Volume(mesh.points, mesh.cells_dict["triangle"])

    expected = np.array([points.min(axis=0), points.max(axis=0)], np.float32)
    actual = vol.extents

    assert np.allclose(expected, actual)


def test_points_roundtrip(mesh):
    points = mesh.points
    vol = Volume(mesh.points, mesh.cells_dict["triangle"])

    expected = np.array(sorted(tuple(p) for p in points), dtype=np.float32)
    actual = np.array(sorted(tuple(p) for p in vol.points), dtype=np.float32)

    assert np.allclose(expected, actual)


def test_extents_validity(mesh):
    points = mesh.points
    vol = Volume(mesh.points, mesh.cells_dict["triangle"])

    expected = np.array([points.min(axis=0), points.max(axis=0)], np.float32)
    actual = vol.extents
    assert np.allclose(expected, actual)


class ParametrizationBuilder:
    def __init__(self, *names):
        self.param_names = names
        self.params = []
        self.test_names = []

    def add(self, test_name, *params):
        if len(params) != len(self.param_names):
            raise ValueError(
                "Wrong number of params given "
                f"(got {len(params)}, expected {len(self.param_names)})"
            )
        self.params.append(params)
        self.test_names.append(test_name)

    def add_many(self, *tuples):
        for tup in tuples:
            self.add(*tup)

    def as_dict(self):
        return {
            "argnames": self.param_names,
            "argvalues": self.params,
            "ids": self.test_names,
        }


cube_params = ParametrizationBuilder("coords", "is_internal")
middle = [0.5, 0.5, 0.5]
cube_params.add("middle", middle, True)
for idx, corner in enumerate(product([0, 1], repeat=3)):
    cube_params.add(f"corner-{idx}", corner, True)

for dim in range(3):
    this = middle.copy()
    this[dim] = 0
    cube_params.add(f"face-low-dim{dim}", this.copy(), True)
    this[dim] = 1
    cube_params.add(f"face-high-dim{dim}", this.copy(), True)

    this[dim] = -0.5
    cube_params.add(f"external-low-dim{dim}", this.copy(), False)
    this[dim] = 1.5
    cube_params.add(f"external-high-dim{dim}", this.copy(), False)


@pytest.mark.parametrize(**cube_params.as_dict())
def test_cube(simple_volume, coords, is_internal):
    assert (coords in simple_volume) == is_internal


intersects_params = ParametrizationBuilder("src", "tgt", "intersects", "is_bf")
intersects_params.add_many(
    ("face-oi", [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], True, False),
    ("face-io", [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], True, True),
    ("edge-oi", [-0.5, -0.5, 0.5], [0.5, 0.5, 0.5], True, False),
    ("edge-io", [0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], True, True),
    ("corner-oi", [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], True, False),
    ("corner-io", [0.5, 0.5, 0.5], [-0.5, -0.5, -0.5], True, True),
    ("miss", [-1, -1, -1], [-1, -1, 0], False, None),
    ("short", [-1, -1, -1], [-0.5, -0.5, -0.5], False, None),
    ("double", [-0.5, 0.5, 0.5], [1.5, 1.5, 1.5], True, False),
)


@pytest.mark.parametrize(**intersects_params.as_dict())
def test_intersections(simple_volume, src, tgt, intersects, is_bf):
    idx, _, backface = simple_volume.intersections([src], [tgt])
    assert intersects == (0 in idx)
    if intersects:
        assert is_bf == backface[0]
    else:
        assert is_bf is None


@pytest.mark.parametrize("threads", [None, True, False])
def test_intersections_threads(simple_volume, threads):
    sources = [
        [0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 1.5],
    ]
    targets = [
        [1.5, 0.5, -0.5],
        [1.5, 0.5, 0.5],
        [1.5, 0.5, 1.5],
    ]

    idxs, _, _ = simple_volume.intersections(sources, targets, threads=threads)
    assert len(idxs) == 1
    assert idxs[0] == 1


@pytest.mark.parametrize("steps", list(range(-5, 6)))
@pytest.mark.parametrize("angle", ["edge", "face"])
def test_near_miss(simple_volume: Volume, steps, angle):
    if angle == "edge":
        start_stop = np.array(
            [
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
            ],
            PRECISION,
        )
    elif angle == "face":
        start_stop = np.array(
            [
                [-1, 0.5, 0],
                [2, 0.5, 0],
            ],
            PRECISION,
        )
    else:
        raise ValueError("Unknown angle '{}', wanted 'edge' or 'face'".format(angle))

    expected_hit = steps >= 0
    fill = np.inf if expected_hit else -np.inf
    toward = np.full(2, fill, PRECISION)

    for _ in range(abs(steps)):
        start_stop[:, 2] = np.nextafter(start_stop[:, 2], toward)

    idxs, _, _ = simple_volume.intersections([start_stop[0]], [start_stop[1]])
    result_hit = len(idxs) > 0
    assert expected_hit == result_hit


@pytest.mark.parametrize(["signed"], [(True,), (False,)])
@pytest.mark.parametrize(
    ["point", "expected"],
    [
        ([1, 1, 1], 0),
        ([2, 2, 2], np.sqrt(3)),
        ([0.5, 0.5, 0.5], -0.5),
    ],
)
def test_distance(simple_volume, point, expected, signed):
    if not signed:
        expected = np.abs(expected)
    assert np.allclose(
        simple_volume.distance([point], signed=signed), np.asarray([expected])
    )


@pytest.mark.xfail(reason="Other tests already configure the pool")
def test_configure_threadpool():
    configure_threadpool(2, "prefix")


def test_configure_threadpool_subprocess():
    # must be run in its own interpreter so that pool is not already configured
    cmd = (
        "from ncollpyde import configure_threadpool; configure_threadpool(2, 'prefix')"
    )
    args = [sys.executable, "-c", cmd]

    result = sp.run(args, text=True, capture_output=True)
    logger.info(result.stdout)
    logger.warning(result.stderr)
    result.check_returncode()


def test_configure_threadpool_twice():
    # configure_threadpool(2, "prefix")
    with pytest.raises(RuntimeError):
        configure_threadpool(3, "prefix")
        configure_threadpool(3, "prefix")


@pytest.mark.parametrize(
    ["point", "vec", "exp_dist", "exp_dot"],
    [
        ([0.5, 0.5, 0.5], [1, 0, 0], 0.5, 1),
        ([-0.5, 0.5, 0.5], [1, 0, 0], -0.5, 1),
        ([0.75, 0.5, 0.5], [1, 1, 0], sqrt(2 * 0.25**2), np.cos(pi / 4)),
    ],
)
def test_sdf_inner(simple_volume: Volume, point, vec, exp_dist, exp_dot):
    dists, dots = simple_volume._sdf_intersections([point], [vec])
    assert np.allclose(dists[0], exp_dist)
    assert np.allclose(dots[0], exp_dot)


def assert_intersection_results(test, ref):
    test_dict = {idx: (tuple(pt), is_bf) for idx, pt, is_bf in zip(*test)}
    ref_dict = {idx: (tuple(pt), is_bf) for idx, pt, is_bf in zip(*ref)}
    assert test_dict == ref_dict


def test_intersections_impls(volume: Volume):
    n_edges = 1000
    starts = points_around_vol(volume, n_edges, seed=1991)
    stops = points_around_vol(volume, n_edges, seed=1992)

    serial = volume.intersections(starts, stops, threads=False)
    par = volume.intersections(starts, stops, threads=True)
    assert_intersection_results(serial, par)
    par2 = volume._impl.intersections_many_threaded(starts, stops)
    assert_intersection_results(serial, par2)
