use std::fmt::Debug;
use std::iter::repeat_with;

use ncollide3d::bounding_volume::{BoundingSphere, HasBoundingVolume};
use ncollide3d::math::{Point, Vector};
use ncollide3d::nalgebra::Isometry3;
use ncollide3d::shape::{TriMesh, TriMeshFace};
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;

use crate::utils::{mesh_contains_point, random_dir, Precision, PRECISION};

fn vec_to_point<T: 'static + Debug + PartialEq + Copy>(v: Vec<T>) -> Point<T> {
    Point::new(v[0], v[1], v[2])
}

fn point_to_vec<T: 'static + Debug + PartialEq + Copy>(p: &Point<T>) -> Vec<T> {
    vec![p.x, p.y, p.z]
}

fn vector_to_vec<T: 'static + Debug + PartialEq + Copy>(v: &Vector<T>) -> Vec<T> {
    vec![v[0], v[1], v[2]]
}

fn face_to_vec(f: &TriMeshFace<Precision>) -> Vec<usize> {
    f.indices.iter().cloned().collect()
}

#[cfg(not(test))]
#[pyclass]
pub struct TriMeshWrapper {
    mesh: TriMesh<Precision>,
    ray_directions: Vec<Vector<Precision>>,
}

#[cfg(not(test))]
#[pymethods]
impl TriMeshWrapper {
    #[new]
    pub fn __new__(
        points: Vec<Vec<Precision>>,
        indices: Vec<Vec<usize>>,
        n_rays: usize,
        ray_seed: u64,
    ) -> Self {
        let points2 = points.into_iter().map(vec_to_point).collect();
        let indices2 = indices.into_iter().map(vec_to_point).collect();
        let mesh = TriMesh::new(points2, indices2, None);

        if n_rays > 0 {
            let bsphere: BoundingSphere<Precision> = mesh.bounding_volume(&Isometry3::identity());
            let len = bsphere.radius() * 2.0;

            let mut rng = Pcg64Mcg::seed_from_u64(ray_seed);

            Self {
                mesh,
                ray_directions: repeat_with(|| random_dir(&mut rng, len))
                    .take(n_rays)
                    .collect(),
            }
        } else {
            Self {
                mesh,
                ray_directions: Vec::default(),
            }
        }
    }

    pub fn contains(&self, _py: Python, point: Vec<Precision>) -> bool {
        mesh_contains_point(&self.mesh, &vec_to_point(point), &self.ray_directions)
    }

    pub fn contains_many(&self, py: Python, points: Vec<Vec<Precision>>) -> Vec<bool> {
        py.allow_threads(|| {
            points
                .into_iter()
                .map(|v| mesh_contains_point(&self.mesh, &vec_to_point(v), &self.ray_directions))
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
                    .map(|v| {
                        mesh_contains_point(&self.mesh, &vec_to_point(v), &self.ray_directions)
                    })
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

    pub fn rays(&self, _py: Python) -> Vec<Vec<Precision>> {
        self.ray_directions.iter().map(vector_to_vec).collect()
    }

    pub fn aabb(&self, _py: Python) -> (Vec<Precision>, Vec<Precision>) {
        let aabb = self.mesh.aabb();
        (point_to_vec(aabb.mins()), point_to_vec(aabb.maxs()))
    }
}

#[cfg(not(test))]
#[pymodule]
pub fn ncollpyde(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TriMeshWrapper>()?;

    #[pyfn(m, "_precision")]
    pub fn precision_py(_py: Python) -> &'static str {
        PRECISION
    }

    #[pyfn(m, "_version")]
    pub fn version_py(_py: Python) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    Ok(())
}
