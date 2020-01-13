from pathlib import Path

import numpy as np
import meshio
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from ncollpyde import Volume


MESH_PATH = Path("meshes/teapot.stl")

vol = Volume.from_meshio(meshio.read(str(MESH_PATH)), validate=True, threads=True)

extents = vol.extents
size = np.diff(extents, axis=0).flatten()
extents[0] -= size * 0.1
extents[1] += size * 0.1
new_size = np.diff(extents, axis=0).flatten()

rng = np.random.RandomState(1991)
samples = rng.random_sample((1_000_000, 3)) * new_size + extents[0]

print("containment check")
idxs = vol.contains(samples, True)
print("containment check done")
internals = samples[idxs]
externals = samples[~idxs]

fig: Figure = plt.figure()
ax: Axes3D = fig.add_subplot(111, projection="3d")

ax.set_xlim(extents[:, 0])
ax.set_ylim(extents[:, 1])
ax.set_zlim(extents[:, 2])
ax.scatter(internals[:, 0], internals[:, 1], internals[:, 2])

plt.show()
