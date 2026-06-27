//! Adapter for converting CadFuture's `TessMesh` into a flat (positions,
//! normals, indices) layout that sim-render can upload as vertex buffers.
//!
//! The kernel stores mesh data per-vertex (position + normal + uv) in
//! millimeters. sim-render expects positions and normals to be in meters
//! (the world-frame units the simulator uses).
//!
//! This module is the one place that knows about both unit systems.

use physical_tessellation::TessMesh;

/// SI-unit mesh ready for sim-render upload.
#[derive(Debug, Clone)]
pub struct RenderableMesh {
    /// Per-vertex [x, y, z] in meters.
    pub positions: Vec<[f32; 3]>,
    /// Per-vertex [nx, ny, nz] unit normals.
    pub normals: Vec<[f32; 3]>,
    /// Triangle indices.
    pub indices: Vec<u32>,
}

impl RenderableMesh {
    /// Convert a kernel `TessMesh` (mm units) to a sim-render-ready mesh (m units).
    pub fn from_kernel(mesh: &TessMesh) -> Self {
        let vc = mesh.vertices.len();
        let mut positions = Vec::with_capacity(vc);
        let mut normals = Vec::with_capacity(vc);
        for v in &mesh.vertices {
            positions.push([
                v.position[0] * 1e-3, // mm → m
                v.position[1] * 1e-3,
                v.position[2] * 1e-3,
            ]);
            normals.push([v.normal[0], v.normal[1], v.normal[2]]);
        }
        let indices = mesh.indices.clone();
        Self { positions, normals, indices }
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }
}
