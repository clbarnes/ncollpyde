from pathlib import Path

import meshio
import pytest

from ncollpyde import Volume, Validation

test_dir = Path(__file__).resolve().parent
project_dir = test_dir.parent
mesh_dir = project_dir / "meshes"


@pytest.fixture
def mesh():
    return meshio.read(str(mesh_dir / "teapot.stl"))


@pytest.fixture
def volume(mesh):
    return Volume.from_meshio(mesh, validate=Validation.all())


@pytest.fixture
def simple_mesh():
    return meshio.read(str(mesh_dir / "cube.stl"))


@pytest.fixture
def simple_volume(simple_mesh):
    return Volume.from_meshio(simple_mesh, validate=Validation.all())


@pytest.fixture
def sez_right():
    return Volume.from_meshio(
        meshio.read(str(mesh_dir / "SEZ_right.stl")), validate=Validation.all()
    )
