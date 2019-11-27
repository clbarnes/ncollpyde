#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `ncollpyde` package."""

import pytest

from ncollpyde import Volume


points_expected = [
    ([-2.3051376, -4.1556454,  1.9047838], True),  # internal
    ([-0.35222054, -0.513299, 7.6191354], False),  # external but in AABB
    ([0.13970017, 0.0928266, 4.7073355], True),  # vertex
    ([10, 10, 10], False),  # out of AABB
]


@pytest.mark.parametrize(["point", "expected"], points_expected)
def test_single(mesh, point, expected):
    """Sample pytest test function with the pytest fixture as an argument."""
    vol = Volume(*mesh)
    assert vol.contains(point) == expected


def test_corner(mesh):
    vol = Volume(*mesh)
    assert vol.contains(mesh[0][0])


def test_many(mesh):
    points = []
    expected = []
    for p, e in points_expected:
        points.append(p)
        expected.append(e)

    vol = Volume(*mesh)
    assert vol.contains_many(points) == expected
