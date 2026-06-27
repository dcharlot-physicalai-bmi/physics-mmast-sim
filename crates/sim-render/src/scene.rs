//! Scene builder — constructs the meshes for a given vehicle + environment.

use crate::mesh::CpuMesh;

/// Build the terrain mesh.
pub fn build_terrain() -> CpuMesh {
    CpuMesh::terrain(
        400.0,
        80,
        [0.08, 0.13, 0.2, 1.0], // dark blue-grey
    )
}

/// Build a simplified HALE-style fixed-wing UAV.
pub fn build_hale_uav() -> Vec<CpuMesh> {
    let body_color = [0.04, 0.05, 0.09, 1.0];
    let wing_color = [0.03, 0.04, 0.08, 1.0];
    let pv_color = [0.04, 0.08, 0.22, 1.0]; // dark blue PV surface

    vec![
        // Fuselage
        CpuMesh::box_mesh(2.6, 0.7, 0.7, body_color),
        // Wing
        CpuMesh::wing(12.0, 1.1, 0.12, wing_color),
        // PV surface (slightly above wing)
        CpuMesh::wing(11.8, 1.05, 0.02, pv_color),
        // Tail vertical
        CpuMesh::box_mesh(0.06, 1.0, 0.06, wing_color),
        // Tail horizontal
        CpuMesh::box_mesh(1.4, 0.05, 0.06, wing_color),
        // Prop hub
        CpuMesh::cylinder(0.08, 0.1, 12, [0.13, 0.16, 0.25, 1.0]),
    ]
}

/// Build a simplified quadcopter.
pub fn build_quadcopter() -> Vec<CpuMesh> {
    let arm_color = [0.06, 0.07, 0.12, 1.0];
    let motor_color = [0.15, 0.15, 0.2, 1.0];

    let mut meshes = vec![
        // Central body
        CpuMesh::box_mesh(0.15, 0.04, 0.15, arm_color),
    ];

    // Four arms + motor pods
    for (_dx, _dz) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        meshes.push(CpuMesh::box_mesh(0.2, 0.02, 0.02, arm_color));
        meshes.push(CpuMesh::cylinder(0.03, 0.04, 8, motor_color));
    }

    meshes
}

/// Build a simplified ground rover.
pub fn build_rover() -> Vec<CpuMesh> {
    let body_color = [0.12, 0.10, 0.06, 1.0];
    let wheel_color = [0.05, 0.05, 0.05, 1.0];
    let panel_color = [0.04, 0.08, 0.22, 1.0];

    vec![
        // Chassis
        CpuMesh::box_mesh(1.0, 0.25, 0.6, body_color),
        // Solar panel on top
        CpuMesh::box_mesh(0.9, 0.02, 0.55, panel_color),
        // Wheels (4 cylinders)
        CpuMesh::cylinder(0.12, 0.08, 12, wheel_color),
        CpuMesh::cylinder(0.12, 0.08, 12, wheel_color),
        CpuMesh::cylinder(0.12, 0.08, 12, wheel_color),
        CpuMesh::cylinder(0.12, 0.08, 12, wheel_color),
    ]
}
