use std::fmt::Debug;

use ncollide3d::bounding_volume::{BoundingSphere, HasBoundingVolume};
use ncollide3d::math::{Point, Vector};
use ncollide3d::nalgebra::Isometry3;
use ncollide3d::shape::{TriMesh, TriMeshFace};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::mesh_contains_point;

pub(crate) type Precision = f64;
const PRECISION: &'static str = "float64";

// pi, e, sqrt(2)
const RAY_DIRECTION: [Precision; 3] = [3.1415926535897931, 2.7182818284590451, 1.4142135623730951];

fn vec_to_point<T: 'static + Debug + PartialEq + Copy>(v: Vec<T>) -> Point<T> {
    Point::new(v[0], v[1], v[2])
}

fn point_to_vec<T: 'static + Debug + PartialEq + Copy>(p: &Point<T>) -> Vec<T> {
    vec![p.x, p.y, p.z]
}

fn face_to_vec(f: &TriMeshFace<Precision>) -> Vec<usize> {
    f.indices.iter().cloned().collect()
}

#[pyclass]
pub struct TriMeshWrapper {
    mesh: TriMesh<Precision>,
    ray_direction: Vector<Precision>,
}

#[pymethods]
impl TriMeshWrapper {
    #[new]
    pub fn __new__(
        points: Vec<Vec<Precision>>,
        indices: Vec<Vec<usize>>,
    ) -> Self {
        let points2 = points.into_iter().map(vec_to_point).collect();
        let indices2 = indices.into_iter().map(vec_to_point).collect();
        let mesh = TriMesh::new(points2, indices2, None);

        let bsphere: BoundingSphere<Precision> = mesh.bounding_volume(&Isometry3::identity());
        let len = bsphere.radius() * 2.0;

        let unscaled_dir: Vector<Precision> = RAY_DIRECTION.into();
        let ray_direction = unscaled_dir.normalize() * len;
        Self { mesh, ray_direction }
    }

    pub fn contains(&self, _py: Python, point: Vec<Precision>) -> bool {
        mesh_contains_point(&self.mesh, &vec_to_point(point), &self.ray_direction)
    }

    pub fn contains_many(&self, py: Python, points: Vec<Vec<Precision>>) -> Vec<bool> {
        py.allow_threads(|| {
            points
                .into_iter()
                .map(|v| mesh_contains_point(&self.mesh, &vec_to_point(v), &self.ray_direction))
                .collect()
        })
    }

    pub fn contains_many_threaded(
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

    pub fn points(&self, _py: Python) -> Vec<Vec<Precision>> {
        self.mesh.points().iter().map(point_to_vec).collect()
    }

    pub fn faces(&self, _py: Python) -> Vec<Vec<usize>> {
        self.mesh.faces().iter().map(face_to_vec).collect()
    }

    pub fn aabb(&self, _py: Python) -> (Vec<Precision>, Vec<Precision>) {
        let aabb = self.mesh.aabb();
        (point_to_vec(aabb.mins()), point_to_vec(aabb.maxs()))
    }
}

#[pymodule]
pub fn ncollpyde(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TriMeshWrapper>()?;

    #[pyfn(m, "precision")]
    pub fn precision_py(_py: Python) -> &'static str {
        PRECISION
    }

    Ok(())
}
