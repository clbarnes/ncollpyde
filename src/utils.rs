use ndarray::ArrayView1;
use parry3d_f64::math::{Point, Vector};
use parry3d_f64::na::{distance, Unit};
use parry3d_f64::query::{PointQuery, Ray, RayCast};
use parry3d_f64::shape::{FeatureId, TriMesh, TriMeshFlags};
use rand::Rng;

pub type Precision = f64;

pub const FLAGS: TriMeshFlags = TriMeshFlags::empty()
    .union(TriMeshFlags::ORIENTED)
    .union(TriMeshFlags::DELETE_BAD_TOPOLOGY_TRIANGLES)
    .union(TriMeshFlags::HALF_EDGE_TOPOLOGY)
    .union(TriMeshFlags::MERGE_DUPLICATE_VERTICES)
    .union(TriMeshFlags::DELETE_DEGENERATE_TRIANGLES)
    .union(TriMeshFlags::DELETE_DUPLICATE_TRIANGLES);

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
    consensus: usize,
) -> bool {
    if !mesh.local_aabb().contains_local_point(point) {
        return false;
    }

    // this previously checked if point was on boundary,
    // now it does a full (slow) containment check
    // if mesh.contains_local_point(point) {
    //     return true;
    // }

    // ray_directions
    //     .iter()
    //     .filter(|v| mesh_contains_point_ray(mesh, point, v))
    //     .count()
    //     >= consensus

    let mut inside_remaining = consensus;
    let mut remaining = ray_directions.len();
    for contains in ray_directions
        .iter()
        .map(|v| mesh_contains_point_ray(mesh, point, v))
    {
        remaining -= 1;
        if contains {
            if inside_remaining == 1 {
                // early return if we've met consensus
                return true;
            }
            inside_remaining -= 1;
        } else if remaining < inside_remaining {
            // early return if there aren't enough rays left to reach consensus
            return false;
        }
    }
    false
}

pub fn mesh_contains_point_oriented(mesh: &TriMesh, point: &Point<f64>) -> bool {
    mesh.local_aabb().contains_local_point(point) && mesh.contains_local_point(point)
}

pub fn points_cross_mesh(
    mesh: &TriMesh,
    src_point: &Point<f64>,
    tgt_point: &Point<f64>,
) -> Option<(Point<f64>, bool)> {
    points_cross_mesh_info(mesh, src_point, tgt_point)
        .map(|(inter, _, ft)| (inter, mesh.is_backface(ft)))
}

pub fn points_cross_mesh_info(
    mesh: &TriMesh,
    src_point: &Point<f64>,
    tgt_point: &Point<f64>,
) -> Option<(Point<f64>, Vector<f64>, FeatureId)> {
    let ray = Ray::new(*src_point, tgt_point - src_point);
    mesh.cast_local_ray_and_get_normal(
        &ray, 1.0, false, // unused
    )
    .map(|i| (ray.point_at(i.toi), i.normal, i.feature))
}

pub fn dist_from_mesh(mesh: &TriMesh, point: &Point<f64>, signed: bool) -> f64 {
    let pp = mesh.project_local_point(point, true);
    let dist = distance(&pp.point, point);
    if signed && pp.is_inside {
        -dist
    } else {
        dist
    }
}

/// The diagonal length of the mesh's axis-aligned bounding box.
///
/// Useful as an upper bound for ray length.
pub fn aabb_diag(mesh: &TriMesh) -> f64 {
    mesh.local_aabb().extents().norm()
}

pub fn ray_toi_dot(
    p: Point<Precision>,
    v: Unit<Vector<Precision>>,
    length: Precision,
    mesh: &TriMesh,
) -> (Precision, Precision) {
    let ray = Ray::new(p, *v);
    if let Some(inter) = mesh.cast_local_ray_and_get_normal(
        &ray, length, true, // unused
    ) {
        let dot = v.dot(&inter.normal).abs();
        if mesh.is_backface(inter.feature) {
            (inter.toi, dot)
        } else {
            (-inter.toi, dot)
        }
    } else {
        (Precision::INFINITY, Precision::NAN)
    }
}

/// Find the distance to a mesh boundary from a point along a particular direction.
///
/// Returns a tuple of the distance and the absolute dot product between the vector and the face normal.
/// The distance is positive if the intersection was with a backface,
/// negative if the intersection was with an external face.
pub fn sdf_inner(
    point: ArrayView1<Precision>,
    vector: ArrayView1<Precision>,
    diameter: Precision,
    mesh: &TriMesh,
) -> (Precision, Precision) {
    let p = Point::new(point[0], point[1], point[2]);
    let v = Unit::new_normalize(Vector::new(vector[0], vector[1], vector[2]));
    ray_toi_dot(p, v, diameter, mesh)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use std::f64::consts::TAU;
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

        let mut tm = TriMesh::new(
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
        );
        tm.set_flags(FLAGS).unwrap();
        tm
    }

    fn cube() -> TriMesh {
        read_mesh("cube.stl")
    }

    #[test]
    fn corner_contains() {
        assert!(cube().contains_local_point(&Point::new(0.0, 0.0, 0.0)))
    }

    #[test]
    fn edge_contains() {
        assert!(cube().contains_local_point(&Point::new(0.5, 0.0, 0.0)))
    }

    #[test]
    fn face_contains() {
        assert!(cube().contains_local_point(&Point::new(0.5, 0.5, 0.0)))
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

    fn assert_dist(mesh: &TriMesh, point: &Point<Precision>, signed: bool, expected: Precision) {
        assert_eq!(dist_from_mesh(mesh, point, signed), expected)
    }

    #[test]
    fn distance_signed() {
        let cube = cube();
        assert_dist(&cube, &Point::new(1.0, 1.0, 1.0), true, 0.0);
        assert_dist(&cube, &Point::new(0.5, 0.5, 0.5), true, -0.5);
        assert_dist(&cube, &Point::new(2.0, 1.0, 1.0), true, 1.0);
        let three: Precision = 3.0;
        assert_dist(&cube, &Point::new(2.0, 2.0, 2.0), true, three.sqrt());
    }

    #[test]
    fn distance_unsigned() {
        let cube = cube();
        assert_dist(&cube, &Point::new(0.5, 0.5, 0.5), false, 0.5);
    }

    #[test]
    fn containment_with_psnorms() {
        let cube = cube();
        assert!(
            !cube.contains_local_point(&[1.5, 0.5, 0.5].into()),
            "containment check failed for outside"
        );
        assert!(
            cube.contains_local_point(&[0.5, 0.5, 0.5].into()),
            "containment check failed for center"
        );
        assert!(
            cube.contains_local_point(&[0.0, 0.0, 0.0].into()),
            "containment check failed for vertex"
        );
        assert!(
            cube.contains_local_point(&[0.5, 0.0, 0.0].into()),
            "containment check failed for edge"
        );
        assert!(
            cube.contains_local_point(&[0.5, 0.5, 0.0].into()),
            "containment check failed for face"
        );
    }

    fn get_sdf_results(
        point: &[Precision; 3],
        direction: &[Precision; 3],
        mesh: &TriMesh,
    ) -> (Precision, Precision) {
        let length = aabb_diag(mesh);
        let v = Unit::new_normalize(Vector::new(direction[0], direction[1], direction[2]));
        ray_toi_dot(point.clone().into(), v, length, mesh)
    }

    #[test]
    fn sdf_miss() {
        let vol = cube();
        let (dist, dot) = get_sdf_results(&[2.0, 2.0, 2.0], &[1.0, 1.0, 1.0], &vol);
        assert_eq!(dist, Precision::INFINITY);
        assert!(dot.is_nan());
    }

    #[test]
    fn sdf_direct() {
        let vol = cube();
        let res = get_sdf_results(&[-0.5, 0.5, 0.5], &[1.0, 0.0, 0.0], &vol);
        assert_eq!(res, (-0.5, 1.0));
    }

    #[test]
    fn sdf_diag() {
        let vol = cube();
        let offset = -0.1;
        let angle = TAU / 8.; // 45deg
        let exp_dist = offset / angle.cos();
        let exp_dot = (TAU / 8.).cos();
        let (dist, dot) = get_sdf_results(&[offset, 0.5, 0.5], &[1.0, 1.0, 0.0], &vol);
        assert_relative_eq!(dist, exp_dist);
        assert_relative_eq!(dot, exp_dot);
    }
}
