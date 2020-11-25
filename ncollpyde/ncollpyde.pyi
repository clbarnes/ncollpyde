from typing import List, Tuple

def _precision() -> str: ...
def _version() -> str: ...

Points = List[List[float]]
Indices = List[List[int]]

class TriMeshWrapper:
    def __init__(
        self, points: Points, indices: Indices, n_rays: int, ray_seed: int
    ): ...
    def contains(self, point: List[float]) -> bool: ...
    def contains_many(self, points: Points) -> List[bool]: ...
    def contains_many_threaded(self, points: Points, threads: int) -> List[bool]: ...
    def points(self) -> Points: ...
    def faces(self) -> Indices: ...
    def rays(self) -> Points: ...
    def aabb(self) -> Tuple[List[float], List[float]]: ...
    def intersections_many(
        self, src_points: Points, tgt_points: Points
    ) -> Tuple[List[int], Points, List[bool]]: ...
    def intersections_many_threaded(
        self, src_points: Points, tgt_points: Points, threads: int
    ) -> Tuple[List[int], Points, List[bool]]: ...
