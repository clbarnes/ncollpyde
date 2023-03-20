use std::fmt::Debug;
use std::iter::repeat_with;

use numpy::ndarray::{Array, Zip};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use parry3d_f64::math::{Point, Vector};
use parry3d_f64::shape::TriMesh;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;

use crate::utils::{dist_from_mesh, mesh_contains_point, points_cross_mesh, random_dir, Precision};

fn vec_to_point<T: 'static + Debug + PartialEq + Copy>(v: Vec<T>) -> Point<T> {
    Point::new(v[0], v[1], v[2])
}

fn point_to_vec<T: 'static + Debug + PartialEq + Copy>(p: &Point<T>) -> Vec<T> {
    vec![p.x, p.y, p.z]
}

fn vector_to_vec<T: 'static + Debug + PartialEq + Copy>(v: &Vector<T>) -> Vec<T> {
    vec![v[0], v[1], v[2]]
}

// fn face_to_vec(f: &TriMeshFace<Precision>) -> Vec<usize> {
//     f.indices.iter().cloned().collect()
// }

#[cfg(not(test))]
#[pyclass]
pub struct TriMeshWrapper {
    mesh: TriMesh,
    ray_directions: Vec<Vector<Precision>>,
    n_rays_inside: usize,
}

#[cfg(not(test))]
#[pymethods]
impl TriMeshWrapper {
    #[new]
    pub fn __new__(
        points: PyReadonlyArray2<Precision>,
        indices: PyReadonlyArray2<u32>,
        n_rays: usize,
        ray_seed: u64,
        n_rays_inside: Option<usize>,
    ) -> Self {
        let points2 = points
            .as_array()
            .rows()
            .into_iter()
            .map(|v| Point::new(v[0], v[1], v[2]))
            .collect();
        let indices2 = indices
            .as_array()
            .rows()
            .into_iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect();
        let mesh = TriMesh::new(points2, indices2);

        let actual_n_rays_inside = n_rays_inside.unwrap_or(n_rays);

        if n_rays > 0 {
            let bsphere = mesh.local_bounding_sphere();
            let len = bsphere.radius() * 2.0;

            let mut rng = Pcg64Mcg::seed_from_u64(ray_seed);

            Self {
                mesh,
                ray_directions: repeat_with(|| random_dir(&mut rng, len))
                    .take(n_rays)
                    .collect(),
                n_rays_inside: actual_n_rays_inside,
            }
        } else {
            Self {
                mesh,
                ray_directions: Vec::default(),
                n_rays_inside: actual_n_rays_inside,
            }
        }
    }

    pub fn distance<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<Precision>,
        signed: bool,
        parallel: bool,
    ) -> &'py PyArray1<Precision> {
        let rays = if signed {
            Some(&self.ray_directions[..])
        } else {
            None
        };
        if parallel {
            Zip::from(points.as_array().rows())
                .par_map_collect(|v| {
                    dist_from_mesh(
                        &self.mesh,
                        &Point::new(v[0], v[1], v[2]),
                        rays,
                        self.n_rays_inside,
                    )
                })
                .into_pyarray(py)
        } else {
            Zip::from(points.as_array().rows())
                .map_collect(|v| {
                    dist_from_mesh(
                        &self.mesh,
                        &Point::new(v[0], v[1], v[2]),
                        rays,
                        self.n_rays_inside,
                    )
                })
                .into_pyarray(py)
        }
    }

    pub fn contains<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<Precision>,
        parallel: bool,
    ) -> &'py PyArray1<bool> {
        if parallel {
            Zip::from(points.as_array().rows())
                .par_map_collect(|r| {
                    mesh_contains_point(
                        &self.mesh,
                        &Point::new(r[0], r[1], r[2]),
                        &self.ray_directions,
                        self.n_rays_inside,
                    )
                })
                .into_pyarray(py)
        } else {
            Zip::from(points.as_array().rows())
                .map_collect(|r| {
                    mesh_contains_point(
                        &self.mesh,
                        &Point::new(r[0], r[1], r[2]),
                        &self.ray_directions,
                        self.n_rays_inside,
                    )
                })
                .into_pyarray(py)
        }
    }

    pub fn points<'py>(&self, py: Python<'py>) -> &'py PyArray2<Precision> {
        let vv: Vec<Vec<Precision>> = self.mesh.vertices().iter().map(point_to_vec).collect();
        PyArray2::from_vec2(py, &vv).unwrap()
    }

    pub fn faces<'py>(&self, py: Python<'py>) -> &'py PyArray2<u32> {
        let vv: Vec<Vec<u32>> = self
            .mesh
            .indices()
            .iter()
            .map(|arr| vec![arr[0], arr[1], arr[2]])
            .collect();
        PyArray2::from_vec2(py, &vv).unwrap()
    }

    pub fn rays<'py>(&self, py: Python<'py>) -> &'py PyArray2<Precision> {
        let vv: Vec<Vec<Precision>> = self.ray_directions.iter().map(vector_to_vec).collect();
        PyArray2::from_vec2(py, &vv).unwrap()
    }

    pub fn aabb<'py>(&self, py: Python<'py>) -> &'py PyArray2<Precision> {
        let aabb = self.mesh.local_aabb();
        PyArray2::from_vec2(py, &[point_to_vec(&aabb.mins), point_to_vec(&aabb.maxs)]).unwrap()
    }

    pub fn intersections_many<'py>(
        &self,
        py: Python<'py>,
        src_points: PyReadonlyArray2<Precision>,
        tgt_points: PyReadonlyArray2<Precision>,
    ) -> (
        &'py PyArray1<u64>,
        &'py PyArray2<Precision>,
        &'py PyArray1<bool>,
    ) {
        let mut idxs = Vec::default();
        let mut intersections = Vec::default();
        let mut is_backface = Vec::default();
        let mut count = 0;
        for (idx, point, is_bf) in src_points
            .as_array()
            .rows()
            .into_iter()
            .zip(tgt_points.as_array().rows().into_iter())
            .zip(0_u64..)
            .filter_map(|((src, tgt), i)| {
                points_cross_mesh(
                    &self.mesh,
                    &Point::new(src[0], src[1], src[2]),
                    &Point::new(tgt[0], tgt[1], tgt[2]),
                )
                .map(|o| (i, o.0, o.1))
            })
        {
            idxs.push(idx);
            intersections.extend(point.iter().cloned());
            is_backface.push(is_bf);
            count += 1;
        }

        (
            PyArray1::from_vec(py, idxs),
            Array::from_shape_vec((count, 3), intersections)
                .unwrap()
                .into_pyarray(py),
            PyArray1::from_vec(py, is_backface),
        )
    }

    pub fn intersections_many_threaded(
        &self,
        src_points: Vec<Vec<Precision>>,
        tgt_points: Vec<Vec<Precision>>,
    ) -> (Vec<u64>, Vec<Vec<Precision>>, Vec<bool>) {
        let (idxs, (intersections, is_backface)) = src_points
            .into_par_iter()
            .zip(tgt_points.into_par_iter())
            .enumerate()
            .filter_map(|(i, (src, tgt))| {
                points_cross_mesh(&self.mesh, &vec_to_point(src), &vec_to_point(tgt))
                    .map(|o| (i as u64, (point_to_vec(&o.0), o.1)))
            })
            .unzip();

        (idxs, intersections, is_backface)
    }
}

#[cfg(not(test))]
#[pymodule]
pub fn ncollpyde(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TriMeshWrapper>()?;

    #[pyfn(m)]
    #[pyo3(name = "_precision")]
    pub fn precision_py(_py: Python) -> &'static str {
        "float64"
    }

    #[pyfn(m)]
    #[pyo3(name = "_index")]
    pub fn index_py(_py: Python) -> &'static str {
        "uint32"
    }

    #[pyfn(m)]
    #[pyo3(name = "_version")]
    pub fn version_py(_py: Python) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    #[pyfn(m)]
    #[pyo3(name = "n_threads")]
    pub fn n_threads(_py: Python) -> usize {
        rayon::current_num_threads()
    }

    Ok(())
}
