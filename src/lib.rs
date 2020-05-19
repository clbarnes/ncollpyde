use std::fmt::Debug;

use ncollide3d::bounding_volume::{BoundingSphere, HasBoundingVolume};
use ncollide3d::math::{Point, Vector};
use ncollide3d::nalgebra::Isometry3;
use ncollide3d::query::{PointQuery, Ray, RayCast};
use ncollide3d::shape::{TriMesh, TriMeshFace};
use pyo3::prelude::*;
use rayon::prelude::*;

type Precision = f64;
const PRECISION: &'static str = "float64";

const RAY_DIRECTION: [Precision; 3] = [3.1415926535897931, 2.7182818284590451, 1.4142135623730951];

#[pymodule]
fn ncollpyde(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TriMeshWrapper>()?;

    #[pyfn(m, "precision")]
    fn precision_py(_py: Python) -> &'static str {
        PRECISION
    }

    Ok(())
}

fn vec_to_point<T: 'static + Debug + PartialEq + Copy>(v: Vec<T>) -> Point<T> {
    Point::new(v[0], v[1], v[2])
}

fn point_to_vec<T: 'static + Debug + PartialEq + Copy>(p: &Point<T>) -> Vec<T> {
    vec![p.x, p.y, p.z]
}

fn face_to_vec(f: &TriMeshFace<Precision>) -> Vec<usize> {
    f.indices.iter().cloned().collect()
}

fn mesh_contains_point(
    mesh: &TriMesh<Precision>,
    point: &Point<Precision>,
    ray_direction: &Vector<Precision>,
) -> bool {
    if !mesh.aabb().contains_local_point(point) {
        return false;
    }

    let identity = Isometry3::identity();

    // check whether point is on boundary
    if mesh.contains_point(&identity, point) {
        return true;
    }

    match mesh.toi_and_normal_with_ray(
        &identity,
        &Ray::new(*point, *ray_direction),
        1.0,
        false, // unused
    ) {
        Some(intersection) => mesh.is_backface(intersection.feature),
        None => false,
    }
}

#[pyclass]
struct TriMeshWrapper {
    mesh: TriMesh<Precision>,
    ray_direction: Vector<Precision>,
}

#[pymethods]
impl TriMeshWrapper {
    #[new]
    fn __new__(
        obj: &PyRawObject,
        points: Vec<Vec<Precision>>,
        indices: Vec<Vec<usize>>,
    ) -> PyResult<()> {
        let points2 = points.into_iter().map(vec_to_point).collect();
        let indices2 = indices.into_iter().map(vec_to_point).collect();
        let mesh = TriMesh::new(points2, indices2, None);

        let bsphere: BoundingSphere<Precision> = mesh.bounding_volume(&Isometry3::identity());
        let len = bsphere.radius() * 2.0;

        let unscaled_dir: Vector<Precision> = RAY_DIRECTION.into();
        let ray_direction = unscaled_dir.normalize() * len;
        Ok(obj.init(Self {
            mesh,
            ray_direction,
        }))
    }

    fn contains(&self, _py: Python, point: Vec<Precision>) -> bool {
        mesh_contains_point(&self.mesh, &vec_to_point(point), &self.ray_direction)
    }

    fn contains_many(&self, py: Python, points: Vec<Vec<Precision>>) -> Vec<bool> {
        py.allow_threads(|| {
            points
                .into_iter()
                .map(|v| mesh_contains_point(&self.mesh, &vec_to_point(v), &self.ray_direction))
                .collect()
        })
    }

    fn contains_many_threaded(
        &self,
        py: Python,
        points: Vec<Vec<Precision>>,
        threads: usize,
    ) -> Vec<bool> {
        py.allow_threads(|| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                points
                    .into_par_iter()
                    .map(|v| mesh_contains_point(&self.mesh, &vec_to_point(v), &self.ray_direction))
                    .collect()
            })
        })
    }

    fn points(&self, _py: Python) -> Vec<Vec<Precision>> {
        self.mesh.points().iter().map(point_to_vec).collect()
    }

    fn faces(&self, _py: Python) -> Vec<Vec<usize>> {
        self.mesh.faces().iter().map(face_to_vec).collect()
    }

    fn aabb(&self, _py: Python) -> (Vec<Precision>, Vec<Precision>) {
        let aabb = self.mesh.aabb();
        (point_to_vec(aabb.mins()), point_to_vec(aabb.maxs()))
    }
}
