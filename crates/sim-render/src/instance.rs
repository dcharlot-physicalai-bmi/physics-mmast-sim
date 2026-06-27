//! Instanced rendering — Primitive #40 Scatter/Gather (L₀, thermodynamically free).
//!
//! Instead of N draw calls for N objects of the same type, we upload one
//! geometry + one instance buffer containing per-instance transforms and
//! colors. The GPU scatters the geometry across all positions in a single
//! draw call. This moves the operation from B₄ (CPU loop, L₂ cost) to
//! B₂ (GPU hardware instancing, L₀ cost).

use wgpu::util::DeviceExt;

/// Per-instance data sent to the GPU.
/// Each instance carries a 4x4 model matrix (position/rotation/scale)
/// and an RGBA color override.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    /// Model matrix (column-major, 4×4).
    pub model: [[f32; 4]; 4],
    /// Per-instance color (RGBA).
    pub color: [f32; 4],
}

impl InstanceRaw {
    /// Vertex buffer layout for the instance data (slots 3-6 for matrix, 7 for color).
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            // mat4 col 0
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32x4,
            },
            // mat4 col 1
            wgpu::VertexAttribute {
                offset: 16,
                shader_location: 4,
                format: wgpu::VertexFormat::Float32x4,
            },
            // mat4 col 2
            wgpu::VertexAttribute {
                offset: 32,
                shader_location: 5,
                format: wgpu::VertexFormat::Float32x4,
            },
            // mat4 col 3
            wgpu::VertexAttribute {
                offset: 48,
                shader_location: 6,
                format: wgpu::VertexFormat::Float32x4,
            },
            // color
            wgpu::VertexAttribute {
                offset: 64,
                shader_location: 7,
                format: wgpu::VertexFormat::Float32x4,
            },
        ],
    };

    /// Create an instance at a position with uniform scale and a color.
    pub fn at(x: f32, y: f32, z: f32, scale: f32, color: [f32; 4]) -> Self {
        Self {
            model: [
                [scale, 0.0, 0.0, 0.0],
                [0.0, scale, 0.0, 0.0],
                [0.0, 0.0, scale, 0.0],
                [x, y, z, 1.0],
            ],
            color,
        }
    }

    /// Create from a full glam Mat4 + color.
    pub fn from_mat4(m: glam::Mat4, color: [f32; 4]) -> Self {
        Self {
            model: m.to_cols_array_2d(),
            color,
        }
    }
}

/// An instanced batch: one shared geometry + N instances = 1 draw call.
pub struct InstancedBatch {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub instance_buffer: wgpu::Buffer,
    pub instance_count: u32,
}

impl InstancedBatch {
    /// Create from a CPU mesh (the prototype shape) and a list of instances.
    pub fn new(
        device: &wgpu::Device,
        mesh: &crate::mesh::CpuMesh,
        instances: &[InstanceRaw],
    ) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instanced_vb"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instanced_ib"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instance_data"),
            contents: bytemuck::cast_slice(instances),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices.len() as u32,
            instance_buffer,
            instance_count: instances.len() as u32,
        }
    }

    /// Update instance data (e.g., when objects move or sim state changes).
    pub fn update_instances(&self, queue: &wgpu::Queue, instances: &[InstanceRaw]) {
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));
    }

    /// Issue the instanced draw call.
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..self.instance_count);
    }
}
