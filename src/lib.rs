use nalgebra::geometry::Point3;
use nalgebra::{Isometry3, RealField, Scalar};
use ncollide3d::math::Vector;
use ncollide3d::query::{PointQuery, Ray, RayCast};
use ncollide3d::shape::{TriMesh, TriMeshFace};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pymodule]
fn ncollpyde(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TriMeshWrapper>()?;

    Ok(())
}

fn vec_to_point<T: Scalar>(v: Vec<T>) -> Point3<T> {
    Point3::from_slice(&v[..3])
}

fn point_to_vec<T: Scalar>(p: &Point3<T>) -> Vec<T> {
    p.iter().cloned().collect()
}

fn face_to_vec<T: RealField>(f: &TriMeshFace<T>) -> Vec<usize> {
    f.indices.iter().cloned().collect()
}

fn mesh_contains_point<T: RealField>(mesh: &TriMesh<T>, point: &Point3<T>) -> bool {
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
        &Ray::new(*point, Vector::new(T::one(), T::zero(), T::zero())),
        false, // unused
    ) {
        Some(intersection) => mesh.is_backface(intersection.feature),
        None => false,
    }
}

#[pyclass]
struct TriMeshWrapper {
    mesh: TriMesh<f32>,
}

#[pymethods]
impl TriMeshWrapper {
    #[new]
    fn __new__(obj: &PyRawObject, points: Vec<Vec<f32>>, indices: Vec<Vec<usize>>) -> PyResult<()> {
        let points2 = points.into_iter().map(vec_to_point).collect();
        let indices2 = indices.into_iter().map(vec_to_point).collect();
        Ok(obj.init(Self {
            mesh: TriMesh::new(points2, indices2, None),
        }))
    }

    fn contains(&self, _py: Python, point: Vec<f32>) -> bool {
        mesh_contains_point(&self.mesh, &vec_to_point(point))
    }

    fn contains_many(&self, py: Python, points: Vec<Vec<f32>>) -> Vec<bool> {
        py.allow_threads(|| {
            points
                .into_iter()
                .map(|v| mesh_contains_point(&self.mesh, &vec_to_point(v)))
                .collect()
        })
    }

    fn contains_many_threaded(
        &self,
        py: Python,
        points: Vec<Vec<f32>>,
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
                    .map(|v| mesh_contains_point(&self.mesh, &vec_to_point(v)))
                    .collect()
            })
        })
    }

    fn points(&self, _py: Python) -> Vec<Vec<f32>> {
        self.mesh.points().iter().map(point_to_vec).collect()
    }

    fn faces(&self, _py: Python) -> Vec<Vec<usize>> {
        self.mesh.faces().iter().map(face_to_vec).collect()
    }

    fn aabb(&self, _py: Python) -> (Vec<f32>, Vec<f32>) {
        let aabb = self.mesh.aabb();
        (point_to_vec(aabb.mins()), point_to_vec(aabb.maxs()))
    }
}
