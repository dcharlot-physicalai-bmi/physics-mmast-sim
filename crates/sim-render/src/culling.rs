//! GPU frustum culling with indirect draw compaction.
//!
//! Replaces CPU-side "draw everything and let the depth test sort it out"
//! with a single compute dispatch that:
//!   1. Tests each instance's bounding sphere against the camera frustum
//!   2. Atomically compacts visible instances into a contiguous output buffer
//!   3. Writes the visible instance count into an indirect draw command
//!
//! The render pass then issues `draw_indexed_indirect` — the GPU reads
//! the draw command from the buffer the compute pass just wrote. The
//! visible count never touches the CPU.
//!
//! Compute stack primitives:
//!   #5  Comparison/Predicate (L₂max) — plane-sphere tests
//!   #40 Scatter/Gather (L₀)          — compaction write
//!   D₄  Embarrassingly parallel      — 64-wide workgroup

use crate::instance::InstanceRaw;
use crate::mesh::CpuMesh;
use glam::{Mat4, Vec4};
use wgpu::util::DeviceExt;

/// Per-instance bounding sphere.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceBounds {
    pub center: [f32; 3],
    pub radius: f32,
}

/// Frustum uniform sent to the cull shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CullUniform {
    /// Left, right, bottom, top, near, far.
    pub frustum_planes: [[f32; 4]; 6],
    pub source_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Shared compute pipeline used by all CullableBatch instances.
pub struct CullPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl CullPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cull_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("cull.wgsl").into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cull_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cull_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cull_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("cull"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, bind_group_layout }
    }
}

/// Instanced batch with GPU frustum culling.
pub struct CullableBatch {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub source_buffer: wgpu::Buffer,
    pub source_count: u32,
    pub bounds_buffer: wgpu::Buffer,
    /// Compacted visible instances — written by cull shader, read by render pass.
    pub visible_buffer: wgpu::Buffer,
    /// [index_count, instance_count, first_index, base_vertex, first_instance]
    /// Cull shader atomically increments instance_count; render uses as indirect arg.
    pub indirect_buffer: wgpu::Buffer,
    pub cull_uniform_buffer: wgpu::Buffer,
    pub cull_bind_group: wgpu::BindGroup,
    pub workgroup_count: u32,
}

impl CullableBatch {
    pub fn new(
        device: &wgpu::Device,
        cull_pipeline: &CullPipeline,
        mesh: &CpuMesh,
        instances: &[InstanceRaw],
    ) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cullable_vb"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cullable_ib"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let source_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cullable_source"),
            contents: bytemuck::cast_slice(instances),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Compute per-instance bounding spheres from model matrix translation + prototype radius.
        let radius = prototype_radius(mesh);
        let bounds: Vec<InstanceBounds> = instances
            .iter()
            .map(|i| compute_bounds(i, radius))
            .collect();
        let bounds_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cullable_bounds"),
            contents: bytemuck::cast_slice(&bounds),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Visible buffer: sized for worst case (all instances visible).
        let visible_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cullable_visible"),
            size: (instances.len() * std::mem::size_of::<InstanceRaw>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        // Indirect draw command. index_count is fixed; instance_count gets
        // reset to 0 each frame via write_buffer, then atomicAdd'd by cull.
        let initial_indirect: [u32; 5] = [mesh.indices.len() as u32, 0, 0, 0, 0];
        let indirect_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cullable_indirect"),
            contents: bytemuck::cast_slice(&initial_indirect),
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });

        let cull_uniform = CullUniform {
            frustum_planes: [[0.0; 4]; 6],
            source_count: instances.len() as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let cull_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cull_uniform"),
            contents: bytemuck::cast_slice(&[cull_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let cull_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cull_bg"),
            layout: &cull_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: cull_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: source_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bounds_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: visible_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: indirect_buffer.as_entire_binding() },
            ],
        });

        let workgroup_count = (instances.len() as u32 + 63) / 64;

        Self {
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices.len() as u32,
            source_buffer,
            source_count: instances.len() as u32,
            bounds_buffer,
            visible_buffer,
            indirect_buffer,
            cull_uniform_buffer,
            cull_bind_group,
            workgroup_count,
        }
    }

    /// Reset instance_count (bytes 4..8) to 0. Call before dispatching cull.
    pub fn reset_count(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.indirect_buffer, 4, bytemuck::cast_slice(&[0u32]));
    }

    /// Update the frustum planes uniform from a view-projection matrix.
    pub fn update_frustum(&self, queue: &wgpu::Queue, view_proj: &Mat4) {
        let planes = extract_frustum_planes(view_proj);
        let mut planes_arr = [[0.0f32; 4]; 6];
        for (i, p) in planes.iter().enumerate() {
            planes_arr[i] = [p.x, p.y, p.z, p.w];
        }
        let uniform = CullUniform {
            frustum_planes: planes_arr,
            source_count: self.source_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(&self.cull_uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));
    }

    /// Dispatch the cull compute pass for this batch.
    pub fn dispatch_cull<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>) {
        pass.set_bind_group(0, &self.cull_bind_group, &[]);
        pass.dispatch_workgroups(self.workgroup_count, 1, 1);
    }

    /// Draw using the indirect buffer (instance_count populated by cull).
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, self.visible_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed_indirect(&self.indirect_buffer, 0);
    }
}

/// Prototype radius: max distance from local origin across all vertices.
pub fn prototype_radius(mesh: &CpuMesh) -> f32 {
    mesh.vertices
        .iter()
        .map(|v| {
            (v.position[0].powi(2) + v.position[1].powi(2) + v.position[2].powi(2)).sqrt()
        })
        .fold(0.0_f32, f32::max)
}

/// Compute bounding sphere for an instance.
/// Center is the model matrix translation; radius is prototype_radius × uniform scale.
fn compute_bounds(instance: &InstanceRaw, prototype_radius: f32) -> InstanceBounds {
    let center = [
        instance.model[3][0],
        instance.model[3][1],
        instance.model[3][2],
    ];
    let c0 = instance.model[0];
    let scale = (c0[0].powi(2) + c0[1].powi(2) + c0[2].powi(2)).sqrt();
    InstanceBounds {
        center,
        radius: prototype_radius * scale,
    }
}

/// Extract 6 frustum planes from a view-projection matrix.
/// Each plane is (nx, ny, nz, d) with inward-pointing normal.
/// A point p is inside the frustum iff (n · p + d) >= 0 for all planes.
pub fn extract_frustum_planes(view_proj: &Mat4) -> [Vec4; 6] {
    // Row i of the matrix (column-major storage).
    let cols = view_proj.to_cols_array_2d();
    let row = |i: usize| Vec4::new(cols[0][i], cols[1][i], cols[2][i], cols[3][i]);
    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);

    let normalize = |p: Vec4| {
        let n = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt().max(1e-12);
        Vec4::new(p.x / n, p.y / n, p.z / n, p.w / n)
    };

    [
        normalize(r3 + r0), // left
        normalize(r3 - r0), // right
        normalize(r3 + r1), // bottom
        normalize(r3 - r1), // top
        normalize(r3 + r2), // near
        normalize(r3 - r2), // far
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frustum_extraction_is_normalized() {
        let view = Mat4::look_at_rh(glam::Vec3::new(0.0, 5.0, 10.0), glam::Vec3::ZERO, glam::Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);
        let vp = proj * view;
        let planes = extract_frustum_planes(&vp);
        for p in planes {
            let n = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
            assert!((n - 1.0).abs() < 1e-4, "plane normal not unit: {n}");
        }
    }

    #[test]
    fn origin_inside_frustum() {
        let view = Mat4::look_at_rh(glam::Vec3::new(0.0, 5.0, 10.0), glam::Vec3::ZERO, glam::Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);
        let planes = extract_frustum_planes(&(proj * view));
        // Origin is the target — should be inside all 6 planes.
        for (i, p) in planes.iter().enumerate() {
            let dist = p.w; // distance from origin
            assert!(dist >= -1e-3, "origin outside plane {i}: dist={dist}");
        }
    }
}
