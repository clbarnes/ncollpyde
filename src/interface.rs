use std::iter::repeat_with;

use ndarray::{Array2, ArrayView1};
use numpy::ndarray::{Array, Zip};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use parry3d_f64::math::{Point, Vector};
use parry3d_f64::shape::TriMesh;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rayon::{prelude::*, ThreadPoolBuilder};

use crate::utils::{
    aabb_diag, dist_from_mesh, mesh_contains_point, mesh_contains_point_oriented,
    points_cross_mesh, random_dir, sdf_inner, Precision, FLAGS,
};

// fn vec_to_point<T: 'static + Debug + PartialEq + Copy>(v: Vec<T>) -> Point<T> {
//     Point::new(v[0], v[1], v[2])
// }

// fn point_to_vec<T: 'static + Debug + PartialEq + Copy>(p: &Point<T>) -> Vec<T> {
//     vec![p.x, p.y, p.z]
// }

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
    ) -> PyResult<Self> {
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
        let mut mesh = TriMesh::new(points2, indices2);

        mesh.set_flags(FLAGS)
            .map_err(|e| PyValueError::new_err(format!("Invalid mesh topology: {e}")))?;

        if n_rays > 0 {
            let len = aabb_diag(&mesh);

            let mut rng = Pcg64Mcg::seed_from_u64(ray_seed);

            Ok(Self {
                mesh,
                ray_directions: repeat_with(|| random_dir(&mut rng, len))
                    .take(n_rays)
                    .collect(),
            })
        } else {
            Ok(Self {
                mesh,
                ray_directions: Vec::default(),
            })
        }
    }

    pub fn distance<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<Precision>,
        signed: bool,
        parallel: bool,
    ) -> &'py PyArray1<Precision> {
        let p_arr = points.as_array();
        let zipped = Zip::from(p_arr.rows());
        let clos =
            |v: ArrayView1<f64>| dist_from_mesh(&self.mesh, &Point::new(v[0], v[1], v[2]), signed);

        let collected = if parallel {
            zipped.par_map_collect(clos)
        } else {
            zipped.map_collect(clos)
        };
        collected.into_pyarray(py)
    }

    // #[pyo3(signature = (points, n_rays=None, consensus=None, parallel=None))]
    pub fn contains<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<Precision>,
        n_rays: usize,
        consensus: usize,
        parallel: bool,
    ) -> &'py PyArray1<bool> {
        let p_arr = points.as_array();
        let zipped = Zip::from(p_arr.rows());

        let collected = if n_rays >= 1 {
            // use ray casting
            let rays = &self.ray_directions[..n_rays];
            let clos = |r: ArrayView1<f64>| {
                mesh_contains_point(&self.mesh, &Point::new(r[0], r[1], r[2]), rays, consensus)
            };

            if parallel {
                zipped.par_map_collect(clos)
            } else {
                zipped.map_collect(clos)
            }
        } else {
            // use pseudonormals
            let clos = |r: ArrayView1<f64>| {
                mesh_contains_point_oriented(&self.mesh, &Point::new(r[0], r[1], r[2]))
            };

            if parallel {
                zipped.par_map_collect(clos)
            } else {
                zipped.map_collect(clos)
            }
        };

        collected.into_pyarray(py)
    }

    pub fn points<'py>(&self, py: Python<'py>) -> &'py PyArray2<Precision> {
        let vs = self.mesh.vertices();
        let v = vs
            .iter()
            .fold(Vec::with_capacity(vs.len() * 3), |mut out, p| {
                out.extend(p.iter().cloned());
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
        src_points
            .as_array()
            .rows()
            .into_iter()
            .zip(tgt_points.as_array().rows())
            .zip(0_u64..)
            .for_each(|((src, tgt), i)| {
                if let Some((pt, is_bf)) = points_cross_mesh(
                    &self.mesh,
                    &Point::new(src[0], src[1], src[2]),
                    &Point::new(tgt[0], tgt[1], tgt[2]),
                ) {
                    idxs.push(i);
                    intersections.extend(pt.iter());
                    is_backface.push(is_bf);
                }
            });

        (
            PyArray1::from_vec(py, idxs),
            Array::from_shape_vec((is_backface.len(), 3), intersections)
                .unwrap()
                .into_pyarray(py),
            PyArray1::from_vec(py, is_backface),
        )
    }

    pub fn intersections_many_threaded<'py>(
        &self,
        py: Python<'py>,
        src_points: PyReadonlyArray2<Precision>,
        tgt_points: PyReadonlyArray2<Precision>,
    ) -> (
        &'py PyArray1<u64>,
        &'py PyArray2<Precision>,
        &'py PyArray1<bool>,
    ) {
        let src_arr = src_points.as_array();
        let tgt_arr = tgt_points.as_array();
        let zipped = Zip::indexed(src_arr.rows()).and(tgt_arr.rows());

        let (out_idxs, out_pts, out_is_bf) = zipped
            .into_par_iter()
            // calculate the output rows, discarding non-intersections
            .filter_map(|(idx, src, tgt)| {
                points_cross_mesh(
                    &self.mesh,
                    &Point::new(src[0], src[1], src[2]),
                    &Point::new(tgt[0], tgt[1], tgt[2]),
                )
                .map(|o| (idx as u64, o.0, o.1))
            })
            // convert chunks of results into flat vecs
            .fold(
                || (vec![], vec![], vec![]),
                |(mut idx, mut pts, mut is_backface), (i, pt, is_bf)| {
                    idx.push(i);
                    pts.extend(pt.iter());
                    is_backface.push(is_bf);
                    (idx, pts, is_backface)
                },
            )
            // concatenate the chunked flat vecs
            .reduce(
                || (vec![], vec![], vec![]),
                |(mut idxs, mut pts, mut is_backfaces), (mut idx, mut pt, mut is_bf)| {
                    idxs.append(&mut idx);
                    pts.append(&mut pt);
                    is_backfaces.append(&mut is_bf);
                    (idxs, pts, is_backfaces)
                },
            );

        (
            PyArray1::from_vec(py, out_idxs),
            Array::from_shape_vec((out_is_bf.len(), 3), out_pts)
                .unwrap()
                .into_pyarray(py),
            PyArray1::from_vec(py, out_is_bf),
        )
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
