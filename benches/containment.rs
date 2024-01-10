use std::{fs::OpenOptions, iter::repeat_with, path::PathBuf};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use log::trace;
use parry3d_f64::{
    math::{Point, Vector},
    na::{Point3, Vector3},
    query::{PointQuery, Ray, RayCast},
    shape::{TriMesh, TriMeshFlags},
};
use rand::Rng;
use rand_pcg::Pcg64;
use stl_io::read_stl;

type Precision = f64;

const SEED: u128 = 1991;

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

pub fn mesh_contains_point_rays_simple(
    mesh: &TriMesh,
    point: &Point<f64>,
    ray_directions: &[Vector<f64>],
) -> bool {
    if ray_directions.is_empty() {
        false
    } else {
        ray_directions
            .iter()
            .all(|v| mesh_contains_point_ray(mesh, point, v))
    }
}

pub fn mesh_contains_point_rays(
    mesh: &TriMesh,
    point: &Point<f64>,
    ray_directions: &[Vector<f64>],
) -> bool {
    if !mesh.local_aabb().contains_local_point(point) {
        trace!("Returning false, not in AABB");
        return false;
    }

    // check whether point is on boundary
    if mesh.contains_local_point(point) {
        trace!("Returning true, point is on boundary");
        return true;
    }

    mesh_contains_point_rays_simple(mesh, point, ray_directions)
}

pub fn mesh_contains_point_oriented(mesh: &TriMesh, point: &Point<f64>) -> bool {
    mesh.pseudo_normals().expect("Mesh orientation not checked");
    mesh.contains_local_point(point)
}

pub fn mesh_contains_point_oriented_aabb(mesh: &TriMesh, point: &Point<f64>) -> bool {
    mesh.pseudo_normals().expect("Mesh orientation not checked");
    mesh.local_aabb().contains_local_point(point) && mesh.contains_local_point(point)
}

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

fn orient(mesh: &TriMesh) -> TriMesh {
    let mut m2 = mesh.clone();
    m2.set_flags(TriMeshFlags::ORIENTED).unwrap();
    m2
}

fn points_for_mesh(m: &TriMesh, n: usize) -> Vec<Point3<Precision>> {
    let aabb = m.local_aabb();
    let range = aabb.maxs - aabb.mins;
    let frac = range * 0.1;
    let mins = aabb.mins - frac;
    let new_range = range + 2.0 * frac;

    let mut rng = Pcg64::new(0, SEED);

    repeat_with(|| {
        Point3::from([
            rng.gen::<f64>() * new_range.x + mins.x,
            rng.gen::<f64>() * new_range.y + mins.y,
            rng.gen::<f64>() * new_range.z + mins.z,
        ])
    })
    .take(n)
    .collect()
}

fn rays_for_mesh(mesh: &TriMesh, n: usize) -> Vec<Vector3<Precision>> {
    let diameter = mesh.local_bounding_sphere().radius() * 2.0;
    let mut rng = Pcg64::new(0, SEED);

    repeat_with(|| {
        Vector3::from([rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>()]).normalize() * diameter
    })
    .take(n)
    .collect()
}

pub fn containment(c: &mut Criterion) {
    let mut group = c.benchmark_group("containment");
    let max_rays = 100;
    for name in &[
        "cube.stl",
        "teapot.stl",
        // "SEZ_right.stl",
    ] {
        let mesh_raw = read_mesh(name);
        let points = points_for_mesh(&mesh_raw, 100);
        let rays = rays_for_mesh(&mesh_raw, max_rays);
        group.throughput(criterion::Throughput::Elements(points.len() as u64));

        for n_rays in vec![0, 1, 2, 3, 4, 5] {
            group.bench_with_input(
                BenchmarkId::new(*name, format!("ray{n_rays}")),
                &points,
                |b, i| {
                    b.iter(|| {
                        i.iter().for_each(|p| {
                            mesh_contains_point_rays_simple(
                                &mesh_raw,
                                black_box(p),
                                &rays[..n_rays],
                            );
                        })
                    })
                },
            );
        }

        let mesh_oriented = orient(&mesh_raw);
        group.bench_with_input(BenchmarkId::new(*name, "pseudonormal"), &points, |b, i| {
            b.iter(|| {
                i.iter().for_each(|p| {
                    mesh_contains_point_oriented(&mesh_oriented, black_box(p));
                })
            })
        });

        group.bench_with_input(
            BenchmarkId::new(*name, "pseudonormal_plus_aabb"),
            &points,
            |b, i| {
                b.iter(|| {
                    i.iter().for_each(|p| {
                        mesh_contains_point_oriented_aabb(&mesh_oriented, black_box(p));
                    })
                })
            },
        );
    }
}

criterion_group!(benches, containment);
criterion_main!(benches);
