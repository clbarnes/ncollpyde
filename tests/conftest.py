import meshio
import pytest
from pathlib import Path

test_dir = Path(__file__).resolve().parent


@pytest.fixture
def mesh():
    stl = meshio.read(str(test_dir / "teapot.stl"))
    return stl.points.tolist(), stl.cells["triangle"].tolist()
