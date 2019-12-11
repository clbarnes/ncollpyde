#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `ncollpyde` package."""

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
