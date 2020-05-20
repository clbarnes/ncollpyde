import meshio
import pytest
from pathlib import Path

from ncollpyde import Volume

test_dir = Path(__file__).resolve().parent
project_dir = test_dir.parent
mesh_dir = project_dir / "meshes"


@pytest.fixture
def mesh():
    return meshio.read(str(mesh_dir / "teapot.stl"))


@pytest.fixture
def volume(mesh):
    return Volume.from_meshio(meshio, validate=True)


@pytest.fixture
def simple_mesh():
    return meshio.read(str(mesh_dir / "cube.stl"))


@pytest.fixture
def simple_volume(simple_mesh):
    return Volume.from_meshio(simple_mesh, validate=True)


@pytest.fixture
def sez_right():
    return Volume.from_meshio(
        meshio.read(str(mesh_dir / "SEZ_right.stl")), validate=True
    )
