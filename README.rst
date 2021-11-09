=========
ncollpyde
=========


.. image:: https://img.shields.io/pypi/v/ncollpyde.svg
    :target: https://pypi.python.org/pypi/ncollpyde

.. image:: https://github.com/clbarnes/ncollpyde/workflows/.github/workflows/ci.yaml/badge.svg
    :target: https://github.com/clbarnes/ncollpyde/actions
    :alt: Actions Status

.. image:: https://readthedocs.org/projects/ncollpyde/badge/?version=latest
    :target: https://ncollpyde.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black


A python library for spatial queries of points and line segments with meshes.
ncollpyde wraps around a subset of the parry rust library (formerly its predecessor ncollide).


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
* Get the (signed) distance from points to the boundary of a mesh

Usage
-----

This library implements most of its functionality through the ``Volume`` class,
instantiated from an array of vertices,
and an array of triangles as indices into the vertex array.

.. code-block:: python

    # get an array of vertices and triangles which refer to those points
    import meshio
    mesh = meshio.read("meshes/teapot.stl")

    # use this library
    from ncollpyde import Volume

    volume = Volume(mesh.points, mesh.cells_dict["triangle"])

    # or, for convenience
    volume = Volume.from_meshio(mesh)

    # containment checks: singular and multiple
    assert [-2.30, -4.15,  1.90] in volume
    assert np.array_equal(
        volume.contains(
            [
                [-2.30, -4.15, 1.90],
                [-0.35, -0.51, 7.61],
            ]
        ),
        [True, False]
    )

    # line segment intersection
    seg_idxs, intersections, is_backface = volume.intersections(
        [[-10, -10, -10], [0, 0, 3], [10, 10, 10]],
        [[0, 0, 3], [10, 10, 10], [20, 20, 20]],
    )
    assert np.array_equal(seg_idxs, [0, 1])  # idx 2 does not intersect
    assert np.array_equal(seg_idxs, [0, 1])
    assert np.allclose(
        intersections,
        [
            [-2.23347309, -2.23347309, 0.09648498],
            [ 3.36591285, 3.36591285, 5.356139],
        ],
    )
    assert np.array_equal(
        is_backface,
        [False, True],
    )

    # distance from boundary (negative means internal)
    assert np.array_equal(
        volume.distance([[10, 10, 10], [0, 0, 3]]),
        [10.08592464, -2.99951118],
    )

See the API docs for more advanced usage.

Known issues
------------

* Performance gains for multi-threaded queries are underwhelming, especially for ray intersections: see `this issue <https://github.com/clbarnes/ncollpyde/issues/12>`_
* Very rare false positives for containment
   * Due to a `bug in the underlying library <https://github.com/rustsim/ncollide/issues/335>`_
   * Only happens when the point is outside the mesh and fires a ray which touches a single edge or vertex of the mesh.
   * Also affects ``is_backface`` result for ray intersection checks
* manylinux-compatible wheels are built on CI but not necessarily in your local environment. Always allow CI to deploy the wheels.
* If you are installing from a source distribution rather than a wheel, you need a compatible `rust toolchain <https://www.rust-lang.org/tools/install>`_
* Meshes with >= ~4.3bn vertices are not supported, as the underlying library uses u32 to address them. This is probably not a problem at time of writing; such a mesh would take up hundreds of GB of RAM to operate on.

ncollpyde v0.11 was the last to support ``meshio < 4.0``.

Acknowledgements
----------------

Thanks to top users
`Philipp Schlegel <https://github.com/schlegelp/>`_ (check out `navis <https://github.com/navis-org/navis>`_!)
and `Nik Drummond <https://github.com/nikdrummond>`_
for their help in debugging and expanding ``ncollpyde`` 's functionality.

Thanks also to ``pyo3``/ ``maturin`` developers
`@konstin <https://github.com/konstin>`_
and `@messense <https://github.com/messense/>`_
for taking an interest in the project and helping along the way.
