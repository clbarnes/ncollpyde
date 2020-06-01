use ncollide3d::math::{Isometry, Point, Vector};
use ncollide3d::nalgebra::RealField;
use ncollide3d::query::{PointQuery, Ray, RayCast};
use ncollide3d::shape::TriMesh;
use rand::Rng;

pub type Precision = f64;
pub const PRECISION: &str = "float64";

pub fn random_dir<R: Rng>(rng: &mut R, length: Precision) -> Vector<Precision> {
    let unscaled: Vector<Precision> = [
        rng.gen::<Precision>() - 0.5,
        rng.gen::<Precision>() - 0.5,
        rng.gen::<Precision>() - 0.5,
    ]
    .into();
    unscaled.normalize() * length
}

pub fn mesh_contains_point_ray<T: RealField>(
    mesh: &TriMesh<T>,
    point: &Point<T>,
    ray_direction: &Vector<T>,
) -> bool {
    let identity = Isometry::identity();
    let intersection_opt = mesh.toi_and_normal_with_ray(
        &identity,
        &Ray::new(*point, *ray_direction),
        T::one(),
        false, // unused
    );

    if let Some(intersection) = intersection_opt {
        mesh.is_backface(intersection.feature)
    } else {
        false
    }
}

pub fn mesh_contains_point<T: RealField>(
    mesh: &TriMesh<T>,
    point: &Point<T>,
    ray_directions: &[Vector<T>],
) -> bool {
    if !mesh.aabb().contains_local_point(point) {
        return false;
    }

    // check whether point is on boundary
    if mesh.contains_point(&Isometry::identity(), point) {
        return true;
    }

    if ray_directions.is_empty() {
        false
    } else {
        ray_directions
            .iter()
            .all(|v| mesh_contains_point_ray(mesh, point, v))
    }
}

#[cfg(test)]
mod tests {
    use ncollide3d::nalgebra::Point3;
    use std::fs::OpenOptions;
    use std::path::PathBuf;
    use stl_io::read_stl;

    use super::*;

    fn read_mesh(name: &'static str) -> TriMesh<Precision> {
        let mut stl_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .canonicalize()
            .expect("couldn't resolve");
        stl_path.push("meshes");
        stl_path.push(name);

        let mut f = OpenOptions::new()
            .read(true)
            .open(stl_path)
            .expect("Couldn't open file");
        let io_obj = read_stl(&mut f).expect("Couldn't parse STL");
        io_obj.validate().expect("Mesh is invalid");

        TriMesh::new(
            io_obj
                .vertices
                .iter()
                .map(|v| Point::new(v[0] as Precision, v[1] as Precision, v[2] as Precision))
                .collect(),
            io_obj
                .faces
                .iter()
                .map(|t| Point3::new(t.vertices[0], t.vertices[1], t.vertices[2]))
                .collect(),
            None,
        )
    }

    fn cube() -> TriMesh<Precision> {
        read_mesh("cube.stl")
    }

    #[test]
    fn corner_contains() {
        assert!(cube().contains_point(&Isometry::identity(), &Point::new(0.0, 0.0, 0.0)))
    }

    #[test]
    fn edge_contains() {
        assert!(cube().contains_point(&Isometry::identity(), &Point::new(0.5, 0.0, 0.0)))
    }

    #[test]
    fn face_contains() {
        assert!(cube().contains_point(&Isometry::identity(), &Point::new(0.5, 0.5, 0.0)))
    }

    fn assert_ray(p: [Precision; 3], v: [Precision; 3], is_inside: bool) {
        let mesh = cube();
        let actual = mesh_contains_point_ray(
            &mesh,
            &Point::new(p[0], p[1], p[2]),
            &Vector::new(v[0], v[1], v[2]),
        );
        if actual != is_inside {
            panic!(
                "Test failure.\nExpected `is_inside = {:?}` but got `{:?}` for\n\tPoint    {:?}\n\tRay dir. {:?}\n",
                is_inside, actual, p, v
            )
        }
    }

    #[test]
    fn simple() {
        assert_ray([0.5, 0.5, 0.5], [1.0, 0.0, 0.0], true);
    }

    #[test]
    fn outside_away() {
        assert_ray([-0.5, 0.5, 0.5], [-1.0, 0.0, 0.0], false);
    }

    #[test]
    fn outside_face() {
        assert_ray([-0.5, 0.5, 0.5], [1.0, 0.0, 0.0], false);
    }

    #[test]
    fn inside_far() {
        assert_ray([0.5, 0.5, 0.5], [0.1, 0.0, 0.0], false);
    }

    #[test]
    fn inside_face() {
        assert_ray([0.5, 0.5, 0.5], [1.0, 0.0, 0.0], true);
    }

    #[test]
    fn inside_edge() {
        assert_ray([0.5, 0.5, 0.5], [1.0, 1.0, 0.0], true);
    }

    #[test]
    fn inside_corner() {
        assert_ray([0.5, 0.5, 0.5], [1.0, 1.0, 1.0], true);
    }

    #[test]
    fn outside_edge() {
        assert_ray([-0.5, -0.5, 0.5], [1.0, 1.0, 0.0], false);
        assert_ray([-0.5, -0.5, 0.5], [10.0, 10.0, 0.0], false);
    }

    #[test]
    fn outside_corner() {
        assert_ray([-0.5, -0.5, -0.5], [1.0, 1.0, 1.0], false);
        assert_ray([-0.5, -0.5, -0.5], [10.0, 10.0, 10.0], false);
    }

    #[ignore]
    #[test]
    fn outside_touch_edge() {
        assert_ray([-0.5, 0.5, 0.5], [1.0, 1.0, 0.0], false);
    }

    #[ignore]
    #[test]
    fn outside_touch_corner() {
        assert_ray([-0.5, 0.5, 0.5], [1.0, 1.0, 1.0], false);
    }

    #[test]
    fn outside_touch_face() {
        assert_ray([-0.5, 0.0, 0.0], [1.0, 0.0, 0.0], false);
        assert_ray([-0.5, 0.0, 0.0], [2.0, 0.0, 0.0], false);
    }
}
