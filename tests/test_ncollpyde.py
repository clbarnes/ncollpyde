#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `ncollpyde` package."""

import pytest

# from ncollpyde import ncollpyde
import ncollpyde


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    assert ncollpyde.get_42() == 42
