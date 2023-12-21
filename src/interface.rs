use std::fmt::Debug;
use std::iter::repeat_with;

use ndarray::{Array2, ArrayView1};
use numpy::ndarray::{Array, Zip};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use parry3d_f64::math::{Point, Vector};
use parry3d_f64::shape::TriMesh;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rayon::{prelude::*, ThreadPoolBuilder};

use crate::utils::{
    aabb_diag, dist_from_mesh, mesh_contains_point, points_cross_mesh, random_dir, sdf_inner,
    Precision,
};

fn vec_to_point<T: 'static + Debug + PartialEq + Copy>(v: Vec<T>) -> Point<T> {
    Point::new(v[0], v[1], v[2])
}

fn point_to_vec<T: 'static + Debug + PartialEq + Copy>(p: &Point<T>) -> Vec<T> {
    vec![p.x, p.y, p.z]
}

#[pyclass]
pub struct TriMeshWrapper {
    mesh: TriMesh,
    ray_directions: Vec<Vector<Precision>>,
}

#[pymethods]
impl TriMeshWrapper {
    #[new]
    pub fn __new__(
        points: PyReadonlyArray2<Precision>,
        indices: PyReadonlyArray2<u32>,
        n_rays: usize,
        ray_seed: u64,
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

        if n_rays > 0 {
            let len = aabb_diag(&mesh);

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
        let p_arr = points.as_array();
        let zipped = Zip::from(p_arr.rows());
        let clos =
            |v: ArrayView1<f64>| dist_from_mesh(&self.mesh, &Point::new(v[0], v[1], v[2]), rays);

        let collected = if parallel {
            zipped.par_map_collect(clos)
        } else {
            zipped.map_collect(clos)
        };
        collected.into_pyarray(py)
    }

    pub fn contains<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<Precision>,
        parallel: bool,
    ) -> &'py PyArray1<bool> {
        let p_arr = points.as_array();
        let zipped = Zip::from(p_arr.rows());
        let clos = |r: ArrayView1<f64>| {
            mesh_contains_point(
                &self.mesh,
                &Point::new(r[0], r[1], r[2]),
                &self.ray_directions,
            )
        };

        let collected = if parallel {
            zipped.par_map_collect(clos)
        } else {
            zipped.map_collect(clos)
        };
        collected.into_pyarray(py)
    }

    pub fn points<'py>(&self, py: Python<'py>) -> &'py PyArray2<Precision> {
        let vs = self.mesh.vertices();
        let v = vs
            .iter()
            .fold(Vec::with_capacity(vs.len() * 3), |mut out, p| {
                out.push(p.x);
                out.push(p.y);
                out.push(p.z);
                out
            });
        Array2::from_shape_vec((vs.len(), 3), v)
            .unwrap()
            .into_pyarray(py)
    }

    pub fn faces<'py>(&self, py: Python<'py>) -> &'py PyArray2<u32> {
        let vs = self.mesh.indices();
        let v: Vec<_> = vs.iter().flatten().cloned().collect();
        Array2::from_shape_vec((vs.len(), 3), v)
            .unwrap()
            .into_pyarray(py)
    }

    pub fn rays<'py>(&self, py: Python<'py>) -> &'py PyArray2<Precision> {
        let vs = &self.ray_directions;
        let v = vs
            .iter()
            .fold(Vec::with_capacity(vs.len() * 3), |mut out, p| {
                out.push(p.x);
                out.push(p.y);
                out.push(p.z);
                out
            });
        Array2::from_shape_vec((vs.len(), 3), v)
            .unwrap()
            .into_pyarray(py)
    }

    pub fn aabb<'py>(&self, py: Python<'py>) -> &'py PyArray2<Precision> {
        let aabb = self.mesh.local_aabb();
        Array2::from_shape_vec(
            (2, 3),
            vec![
                aabb.mins.x,
                aabb.mins.y,
                aabb.mins.z,
                aabb.maxs.x,
                aabb.maxs.y,
                aabb.maxs.z,
            ],
        )
        .unwrap()
        .into_pyarray(py)
    }

    pub fn sdf_intersections<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<Precision>,
        vecs: PyReadonlyArray2<Precision>,
        threaded: bool,
    ) -> (&'py PyArray1<Precision>, &'py PyArray1<Precision>) {
        let diameter = aabb_diag(&self.mesh);

        let n = points.shape()[0];

        let mut dists = Array::from_elem((n,), 0.0);
        let mut dot_norms = Array::from_elem((n,), 0.0);

        let p_arr = points.as_array();
        let v_arr = vecs.as_array();

        let zipped = Zip::from(p_arr.rows())
            .and(v_arr.rows())
            .and(&mut dists)
            .and(&mut dot_norms);

        let clos = |point, vector, dist: &mut f64, dot_norm: &mut f64| {
            let (d, dn) = sdf_inner(point, vector, diameter, &self.mesh);
            *dist = d;
            *dot_norm = dn;
        };

        if threaded {
            zipped.par_for_each(clos);
        } else {
            zipped.for_each(clos);
        }

        (dists.into_pyarray(py), dot_norms.into_pyarray(py))
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

#[pymodule]
#[pyo3(name = "_ncollpyde")]
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

    #[pyfn(m)]
    #[pyo3(name = "_configure_threadpool")]
    pub fn configure_threadpool(
        _py: Python,
        n_threads: Option<usize>,
        name_prefix: Option<String>,
    ) -> PyResult<()> {
        let mut builder = ThreadPoolBuilder::new();
        if let Some(n) = n_threads {
            builder = builder.num_threads(n);
        }
        if let Some(p) = name_prefix {
            builder = builder.thread_name(move |idx| format!("{p}{idx}"));
        }
        builder
            .build_global()
            .map_err(|e| PyRuntimeError::new_err(format!("Error building threadpool: {e}")))
    }

    Ok(())
}
