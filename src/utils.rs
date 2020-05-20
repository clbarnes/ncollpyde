use ncollide3d::math::{Point, Vector, Isometry};
use ncollide3d::nalgebra::RealField;
use ncollide3d::query::{PointQuery, Ray, RayCast};
use ncollide3d::shape::TriMesh;

pub fn mesh_contains_point<T: RealField>(
    mesh: &TriMesh<T>,
    point: &Point<T>,
    ray_direction: &Vector<T>,
) -> bool {
    if !mesh.aabb().contains_local_point(point) {
        return false;
    }

    let identity = Isometry::identity();

    // check whether point is on boundary
    if mesh.contains_point(&identity, point) {
        return true;
    }

    match mesh.toi_and_normal_with_ray(
        &identity,
        &Ray::new(*point, *ray_direction),
        T::one(),
        false, // unused
    ) {
        Some(intersection) => mesh.is_backface(intersection.feature),
        None => false,
    }
}


// TODO: rust unit tests blocked on this issue https://github.com/PyO3/pyo3/issues/941
//
// #[cfg(test)]
// mod tests {
//     use std::fs::OpenOptions;
//     use std::path::PathBuf;
//     use stl_io::read_stl;
//     use ncollide3d::nalgebra::Point3;

//     use super::*;
//     use crate::interface::Precision;

//     fn read_mesh(name: &'static str) -> TriMesh<Precision> {
//         let mut stl_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
//             .canonicalize()
//             .expect("couldn't resolve");
//         stl_path.push("data");
//         stl_path.push(name);

//         println!("Parsing {:?}", stl_path);
//         let mut f = OpenOptions::new()
//             .read(true)
//             .open(stl_path)
//             .expect("Couldn't open file");
//         let io_obj = read_stl(&mut f).expect("Couldn't parse STL");
//         io_obj.validate().expect("Mesh is invalid");

//         TriMesh::new(
//             io_obj
//                 .vertices
//                 .iter()
//                 .map(|v| Point::new(
//                     v[0] as Precision,
//                     v[1] as Precision,
//                     v[2] as Precision,
//                 ))
//                 .collect(),
//             io_obj
//                 .faces
//                 .iter()
//                 .map(|t| Point3::new(t.vertices[0], t.vertices[1], t.vertices[2]))
//                 .collect(),
//             None,
//         )
//     }

//     fn cube() -> TriMesh<Precision> {
//         read_mesh("cube.stl")
//     }
// }
