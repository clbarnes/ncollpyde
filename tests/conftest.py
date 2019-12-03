import meshio
import pytest
from pathlib import Path

test_dir = Path(__file__).resolve().parent
project_dir = test_dir.parent
mesh_dir = project_dir / "meshes"


@pytest.fixture
def mesh():
    return meshio.read(str(mesh_dir / "teapot.stl"))
