use parry3d_f64::math::{Isometry, Point, Vector};
use parry3d_f64::query::{PointQuery, Ray, RayCast};
use parry3d_f64::shape::TriMesh;
use rand::Rng;

pub type Precision = f64;

pub fn random_dir<R: Rng>(rng: &mut R, length: Precision) -> Vector<Precision> {
    let unscaled: Vector<Precision> = [
        rng.gen::<Precision>() - 0.5,
        rng.gen::<Precision>() - 0.5,
        rng.gen::<Precision>() - 0.5,
    ]
    .into();
    unscaled.normalize() * length
}

pub fn mesh_contains_point_ray(
    mesh: &TriMesh,
    point: &Point<f64>,
    ray_direction: &Vector<f64>,
) -> bool {
    let intersection_opt = mesh.cast_local_ray_and_get_normal(
        &Ray::new(*point, *ray_direction),
        1.0,
        false, // unused
    );

    if let Some(intersection) = intersection_opt {
        mesh.is_backface(intersection.feature)
    } else {
        false
    }
}

pub fn mesh_contains_point(
    mesh: &TriMesh,
    point: &Point<f64>,
    ray_directions: &[Vector<f64>],
) -> bool {
    if !mesh.local_aabb().contains_local_point(point) {
        return false;
    }

    // check whether point is on boundary
    if mesh.contains_point(&Isometry::identity(), point) {
        return true;
    }

    if ray_directions.is_empty() {
        false
    } else {
        let n_containments = ray_directions
            .iter()
            .filter(|v| mesh_contains_point_ray(mesh, point, v))
            .count();
        n_containments as f64 / ray_directions.len() as f64 > 0.5
    }
}

pub fn points_cross_mesh(
    mesh: &TriMesh,
    src_point: &Point<f64>,
    tgt_point: &Point<f64>,
) -> Option<(Point<f64>, bool)> {
    let ray = Ray::new(*src_point, tgt_point - src_point);
    mesh.cast_local_ray_and_get_normal(
        &ray, 1.0, false, // unused
    )
    .map(|i| (ray.point_at(i.toi), mesh.is_backface(i.feature)))
}

pub fn dist_from_mesh(mesh: &TriMesh, point: &Point<f64>, rays: Option<&[Vector<f64>]>) -> f64 {
    let mut dist = mesh.distance_to_point(&Isometry::identity(), point, true);
    if let Some(r) = rays {
        if mesh_contains_point(mesh, point, r) {
            dist = -dist;
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;
    use std::path::PathBuf;
    use stl_io::read_stl;

    use super::*;

    const EPSILON: Precision = 0.001;

    fn read_mesh(name: &'static str) -> TriMesh {
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
                .map(|t| {
                    [
                        t.vertices[0] as u32,
                        t.vertices[1] as u32,
                        t.vertices[2] as u32,
                    ]
                })
                .collect(),
        )
    }

    fn cube() -> TriMesh {
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

    fn get_cross(src: [Precision; 3], tgt: [Precision; 3]) -> Option<([Precision; 3], bool)> {
        let mesh = cube();
        points_cross_mesh(
            &mesh,
            &Point::new(src[0], src[1], src[2]),
            &Point::new(tgt[0], tgt[1], tgt[2]),
        )
        .map(|(p, bf)| ([p.x, p.y, p.z], bf))
    }

    fn assert_array_eq(test: &[Precision; 3], refer: &[Precision; 3]) {
        if !test
            .iter()
            .zip(refer.iter())
            .all(|(a, b)| (a - b).abs() < EPSILON)
        {
            panic!(
                "Test failure: arrays unequal.\n\ttest {:?}\n\tref  {:?}\n",
                test, refer
            )
        }
    }

    #[test]
    fn cross_oi() {
        let (loc, is_bf) =
            get_cross([-0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).expect("Fail: no collision");
        assert!(!is_bf);
        assert_array_eq(&loc, &[0.0, 0.5, 0.5]);
    }

    #[test]
    fn cross_io() {
        let (loc, is_bf) =
            get_cross([0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]).expect("Fail: no collision");
        assert!(is_bf);
        assert_array_eq(&loc, &[0.0, 0.5, 0.5]);
    }

    #[test]
    fn nocross() {
        assert!(get_cross([1.5, 1.5, 1.5], [2.0, 2.0, 2.0]).is_none());
    }

    fn axis_rays() -> Vec<Vector<Precision>> {
        vec![
            Vector::new(1.0, 0.0, 0.0),
            Vector::new(0.0, 1.0, 0.0),
            Vector::new(0.0, 0.0, 1.0),
        ]
    }

    fn assert_dist(
        mesh: &TriMesh,
        point: &Point<Precision>,
        rays: Option<&[Vector<Precision>]>,
        expected: Precision,
    ) {
        assert_eq!(dist_from_mesh(mesh, point, rays), expected)
    }

    #[test]
    fn distance_signed() {
        let rays = axis_rays();
        let cube = cube();
        assert_dist(&cube, &Point::new(1.0, 1.0, 1.0), Some(&rays), 0.0);
        assert_dist(&cube, &Point::new(0.5, 0.5, 0.5), Some(&rays), -0.5);
        assert_dist(&cube, &Point::new(2.0, 1.0, 1.0), Some(&rays), 1.0);
        let three: Precision = 3.0;
        assert_dist(&cube, &Point::new(2.0, 2.0, 2.0), Some(&rays), three.sqrt());
    }

    #[test]
    fn distance_unsigned() {
        let cube = cube();
        assert_dist(&cube, &Point::new(1.0, 1.0, 1.0), None, 0.0);
        assert_dist(&cube, &Point::new(0.5, 0.5, 0.5), None, 0.5);
        assert_dist(&cube, &Point::new(2.0, 1.0, 1.0), None, 1.0);
        let three: Precision = 3.0;
        assert_dist(&cube, &Point::new(2.0, 2.0, 2.0), None, three.sqrt());
    }
}
