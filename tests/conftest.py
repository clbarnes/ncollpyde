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
    mesh = meshio.read(str(mesh_dir / "20mm_cube.stl"))
    points = mesh.points
    points -= points.min(axis=0)
    points /= points.max(axis=0)
    mesh.points = points
    return mesh


@pytest.fixture
def simple_volume(simple_mesh):
    return Volume.from_meshio(simple_mesh, validate=True)
