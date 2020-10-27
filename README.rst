=========
ncollpyde
=========


.. image:: https://img.shields.io/pypi/pyversions/ncollpyde.svg
    :target: https://pypi.python.org/pypi/ncollpyde

.. image:: https://img.shields.io/pypi/v/ncollpyde.svg
    :target: https://pypi.python.org/pypi/ncollpyde

.. image:: https://github.com/clbarnes/ncollpyde/workflows/.github/workflows/ci.yml/badge.svg?branch=master
    :target: https://github.com/clbarnes/ncollpyde/workflows/.github/workflows/ci.yml

.. image:: https://readthedocs.org/projects/ncollpyde/badge/?version=latest
    :target: https://ncollpyde.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black



A python wrapper around a subset of the ncollide rust library


* Free software: MIT License
* Documentation: https://ncollpyde.readthedocs.io.

Install
-------

``pip install ncollpyde``

Pre-built wheels are available for Linux, MacOS, and Windows.
If you have a stable rust compiler, you should also be able to install from source.

Features
--------

* Checking whether points are inside a volume defined by a triangular mesh
* Checking the intersection of line segments with the mesh

Usage
-----

.. code-block:: python

    # get an array of vertices and triangles which refer to those points
    import meshio
    mesh = meshio.read("tests/teapot.stl")
    vertices = mesh.points
    triangles = mesh.cells_dict["triangle"]

    # use this library
    from ncollpyde import Volume

    volume = Volume(vertices, triangles)

Containment checks:

.. code-block:: python

    # individual points (as 3-length array-likes) can be checked with `in`
    assert [-2.3051376, -4.1556454,  1.9047838] in volume
    assert [-0.35222054, -0.513299, 7.6191354] not in volume

    # many points (as an Nx3 array-like) can be checked with the `contains` method
    bools = volume.contains(np.array([
        [-2.3051376, -4.1556454,  1.9047838],
        [-0.35222054, -0.513299, 7.6191354],
    ]))
    assert np.array_equal(bools, [True, False])

    # checks can be parallelised
    volume.contains(np.random.random((1000, 3)), threads=4)


Note that v0.11 was the last to support ``meshio < 4.0``.

Known issues
------------

* Benchmarks suggest that multithreaded performance is about the same as serial
* Very rare false positives for containment
   * Due to a `bug in the underlying library <https://github.com/rustsim/ncollide/issues/335>`_
   * Only happens when the point is outside the mesh and fires a ray which touches a single edge or vertex of the mesh.
