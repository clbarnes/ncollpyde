from typing import Optional

import numpy as np
import numpy.typing as npt

def _precision() -> str: ...
def _index() -> str: ...
def _version() -> str: ...
def n_threads() -> int: ...
def _configure_threadpool(n_threads: Optional[int], name_prefix: Optional[str]): ...

Points = npt.NDArray[np.float64]
Indices = npt.NDArray[np.uint32]

class TriMeshWrapper:
    def __init__(
        self,
        points: Points,
        indices: Indices,
        n_rays: int,
        ray_seed: int,
        validate: int,
    ): ...
    def contains(
        self, points: Points, n_rays: int, consensus: int, parallel: bool
    ) -> npt.NDArray[np.bool_]: ...
    def distance(
        self, points: Points, signed: bool, parallel: bool
    ) -> npt.NDArray[np.float64]: ...
    def points(self) -> Points: ...
    def faces(self) -> Indices: ...
    def rays(self) -> Points: ...
    def aabb(self) -> Points: ...
    def intersections_many(
        self, src_points: Points, tgt_points: Points
    ) -> tuple[npt.NDArray[np.uint64], Points, npt.NDArray[np.bool_]]: ...
    def intersections_many_threaded(
        self, src_points: Points, tgt_points: Points
    ) -> tuple[npt.NDArray[np.uint64], Points, npt.NDArray[np.bool_]]: ...
    def sdf_intersections(
        self, points: Points, vectors: Points, threaded: bool
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]: ...
