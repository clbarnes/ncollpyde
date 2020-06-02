#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `ncollpyde` package."""
from itertools import product

import pytest
import numpy as np
from trimesh import Trimesh

from ncollpyde import Volume


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


@pytest.mark.parametrize("threads", [None, 0, 2])
def test_many(mesh, threads):
    points = []
    expected = []
    for p, e in points_expected:
        points.append(p)
        expected.append(e)

    vol = Volume.from_meshio(mesh)
    assert np.array_equal(vol.contains(points, threads=threads), expected)


def test_no_validation(mesh):
    triangles = mesh.cells["triangle"]
    Volume(mesh.points, triangles, True)


def test_can_repair_hole(mesh):
    triangles = mesh.cells["triangle"]
    triangles = triangles[:-1]
    Volume(mesh.points, triangles, True)


def test_can_repair_inversion(mesh):
    triangles = mesh.cells["triangle"]
    triangles[-1] = triangles[-1, ::-1]
    Volume(mesh.points, triangles, True)


def test_can_repair_inversions(mesh):
    triangles = mesh.cells["triangle"]
    triangles = triangles[:, ::-1]
    Volume(mesh.points, triangles, True)


def test_points(mesh):
    points = mesh.points
    vol = Volume(mesh.points, mesh.cells["triangle"])

    actual = sorted(tuple(p) for p in vol.points)
    expected = sorted(tuple(p) for p in points)
    assert actual == expected


def test_triangles(mesh):
    points = mesh.points
    triangles = mesh.cells["triangle"]
    expected = Trimesh(points, triangles)

    vol = Volume(mesh.points, triangles)
    actual = Trimesh(vol.points, vol.faces)

    assert expected.volume == actual.volume


def test_extents(mesh):
    points = mesh.points
    vol = Volume(mesh.points, mesh.cells["triangle"])

    expected = np.array([points.min(axis=0), points.max(axis=0)], np.float32)
    actual = vol.extents

    assert np.allclose(expected, actual)


def test_points_roundtrip(mesh):
    points = mesh.points
    vol = Volume(mesh.points, mesh.cells["triangle"])

    expected = np.array(sorted(tuple(p) for p in points), dtype=np.float32)
    actual = np.array(sorted(tuple(p) for p in vol.points), dtype=np.float32)

    assert np.allclose(expected, actual)


def test_extents_validity(mesh):
    points = mesh.points
    vol = Volume(mesh.points, mesh.cells["triangle"])

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
                f"Wrong number of params given"
                "(got {len(params)}, expected {len(self.param_names)})"
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
