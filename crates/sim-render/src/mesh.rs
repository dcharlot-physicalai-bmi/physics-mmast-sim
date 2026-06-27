//! CPU and GPU mesh types, with procedural generators for vehicle archetypes.

use crate::vertex::Vertex;
use wgpu::util::DeviceExt;

/// CPU-side mesh before upload.
#[derive(Debug, Clone)]
pub struct CpuMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

/// GPU-side mesh ready to render.
pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

impl GpuMesh {
    pub fn from_cpu(device: &wgpu::Device, mesh: &CpuMesh) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex_buffer"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index_buffer"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices.len() as u32,
        }
    }
}

// ---- Procedural generators ----

impl CpuMesh {
    /// Flat grid terrain.
    pub fn terrain(size: f32, subdivisions: u32, color: [f32; 4]) -> Self {
        let n = subdivisions + 1;
        let step = size / subdivisions as f32;
        let half = size / 2.0;
        let mut vertices = Vec::with_capacity((n * n) as usize);
        let mut indices = Vec::new();

        for z in 0..n {
            for x in 0..n {
                let px = x as f32 * step - half;
                let pz = z as f32 * step - half;
                // Simple perlin-ish noise for terrain.
                let py = (px * 0.05).sin() * 0.8 + (pz * 0.07).cos() * 0.6;
                vertices.push(Vertex {
                    position: [px, py, pz],
                    normal: [0.0, 1.0, 0.0],
                    color,
                });
            }
        }

        for z in 0..subdivisions {
            for x in 0..subdivisions {
                let i = z * n + x;
                indices.extend_from_slice(&[i, i + n, i + 1, i + 1, i + n, i + n + 1]);
            }
        }

        Self { vertices, indices }
    }

    /// Simple box (fuselage / body).
    pub fn box_mesh(sx: f32, sy: f32, sz: f32, color: [f32; 4]) -> Self {
        let hx = sx / 2.0;
        let hy = sy / 2.0;
        let hz = sz / 2.0;

        let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
            ([0.0, 0.0, 1.0],  [[-hx,-hy, hz],[hx,-hy, hz],[hx, hy, hz],[-hx, hy, hz]]),
            ([0.0, 0.0,-1.0],  [[ hx,-hy,-hz],[-hx,-hy,-hz],[-hx, hy,-hz],[ hx, hy,-hz]]),
            ([0.0, 1.0, 0.0],  [[-hx, hy, hz],[hx, hy, hz],[hx, hy,-hz],[-hx, hy,-hz]]),
            ([0.0,-1.0, 0.0],  [[-hx,-hy,-hz],[hx,-hy,-hz],[hx,-hy, hz],[-hx,-hy, hz]]),
            ([1.0, 0.0, 0.0],  [[hx,-hy, hz],[hx,-hy,-hz],[hx, hy,-hz],[hx, hy, hz]]),
            ([-1.0,0.0, 0.0],  [[-hx,-hy,-hz],[-hx,-hy, hz],[-hx, hy, hz],[-hx, hy,-hz]]),
        ];

        let mut vertices = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(36);

        for (normal, corners) in &faces {
            let base = vertices.len() as u32;
            for c in corners {
                vertices.push(Vertex {
                    position: *c,
                    normal: *normal,
                    color,
                });
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        Self { vertices, indices }
    }

    /// Flat wing (thin box extruded along Z).
    pub fn wing(span: f32, chord: f32, thickness: f32, color: [f32; 4]) -> Self {
        Self::box_mesh(chord, thickness, span, color)
    }

    /// Cylinder (for propeller hub, tower shaft, etc).
    pub fn cylinder(radius: f32, height: f32, segments: u32, color: [f32; 4]) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let half_h = height / 2.0;

        for i in 0..=segments {
            let theta = (i as f32 / segments as f32) * std::f32::consts::TAU;
            let (s, c) = theta.sin_cos();
            let nx = c;
            let nz = s;
            // Bottom
            vertices.push(Vertex {
                position: [radius * c, -half_h, radius * s],
                normal: [nx, 0.0, nz],
                color,
            });
            // Top
            vertices.push(Vertex {
                position: [radius * c, half_h, radius * s],
                normal: [nx, 0.0, nz],
                color,
            });
        }

        for i in 0..segments {
            let b = i * 2;
            indices.extend_from_slice(&[b, b + 1, b + 3, b, b + 3, b + 2]);
        }

        Self { vertices, indices }
    }
}
