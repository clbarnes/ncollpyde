=========
ncollpyde
=========


.. image:: https://img.shields.io/pypi/pyversions/ncollpyde.svg
        :target: https://pypi.python.org/pypi/ncollpyde

.. image:: https://img.shields.io/pypi/v/ncollpyde.svg
        :target: https://pypi.python.org/pypi/ncollpyde

.. image:: https://img.shields.io/travis/clbarnes/ncollpyde.svg
        :target: https://travis-ci.org/clbarnes/ncollpyde

.. image:: https://readthedocs.org/projects/ncollpyde/badge/?version=latest
        :target: https://ncollpyde.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black



A python wrapper around a subset of the ncollide rust library


* Free software: MIT License
* Documentation: https://ncollpyde.readthedocs.io.

Features
--------

* Checking whether points are inside a volume defined by a triangular mesh

Usage
-----

.. code-block:: python

    # get an array of vertices and triangles which refer to those points
    import meshio
    mesh = meshio.read("tests/teapot.stl")
    vertices = mesh.points
    triangles = mesh.cells["triangle"]

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
