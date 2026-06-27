//! GPU compute-shader terrain generation.
//!
//! Dispatches `terrain_compute.wgsl` once at startup to fill a storage
//! buffer with (position, normal, color) per vertex, then builds the
//! triangle index list on the CPU (a tiny fraction of the work compared
//! to the noise sampling the GPU just did in parallel).
//!
//! Compute stack primitives used:
//!   #23/#24 FFT/Convolution (B₂) — FBM octave accumulation on GPU
//!   #2  Bitwise Logic (L₀)       — hash-based noise seed
//!   D₄  Embarrassingly parallel  — each vertex independent, 8×8 workgroups
//!
//! Replaces CPU sin/cos terrain with one `dispatch_workgroups()` call.

use crate::mesh::CpuMesh;
use crate::vertex::Vertex;
use wgpu::util::DeviceExt;

/// Uniform parameters for the terrain compute shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainParams {
    pub grid_size: u32,
    pub world_size: f32,
    pub seed: u32,
    pub octaves: u32,
    pub lacunarity: f32,
    pub persistence: f32,
    pub amplitude: f32,
    pub frequency: f32,
}

impl Default for TerrainParams {
    fn default() -> Self {
        Self {
            grid_size: 128,
            world_size: 400.0,
            seed: 42,
            octaves: 5,
            lacunarity: 2.0,
            persistence: 0.5,
            amplitude: 2.5,
            frequency: 0.012,
        }
    }
}

/// Dispatches the terrain compute shader and reads the result back
/// into a CPU-side mesh. One call.
pub fn generate_terrain_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    params: TerrainParams,
) -> CpuMesh {
    let n = params.grid_size as usize;
    let vertex_count = n * n;
    let float_count = vertex_count * 10; // 10 floats per vertex

    // Uniform buffer for params.
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("terrain_params"),
        contents: bytemuck::cast_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Storage buffer: GPU-writable, CPU-readable via staging.
    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain_storage"),
        size: (float_count * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Staging buffer for readback.
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain_staging"),
        size: storage_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Shader module.
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("terrain_compute"),
        source: wgpu::ShaderSource::Wgsl(include_str!("terrain_compute.wgsl").into()),
    });

    // Bind group layout: uniform params + storage buffer.
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("terrain_bgl"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("terrain_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: storage_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("terrain_pipeline_layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("terrain_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("terrain_gen"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Dispatch: ceil(n / 8) workgroups per axis.
    let workgroups = (params.grid_size + 7) / 8;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("terrain_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("terrain_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging, 0, storage_buffer.size());
    queue.submit(std::iter::once(encoder.finish()));

    // Map staging for readback. Block synchronously — this runs once at startup.
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::Wait).ok();
    rx.recv().unwrap().expect("terrain staging map failed");

    let data = slice.get_mapped_range();
    let floats: &[f32] = bytemuck::cast_slice(&data);

    // Build CPU mesh from the GPU-generated vertex data.
    let mut vertices = Vec::with_capacity(vertex_count);
    for i in 0..vertex_count {
        let b = i * 10;
        vertices.push(Vertex {
            position: [floats[b], floats[b + 1], floats[b + 2]],
            normal: [floats[b + 3], floats[b + 4], floats[b + 5]],
            color: [floats[b + 6], floats[b + 7], floats[b + 8], floats[b + 9]],
        });
    }

    drop(data);
    staging.unmap();

    // Build triangle indices — O(n²), negligible compared to noise sampling.
    let mut indices = Vec::with_capacity((n - 1) * (n - 1) * 6);
    for z in 0..(n - 1) {
        for x in 0..(n - 1) {
            let i = (z * n + x) as u32;
            let n_u = n as u32;
            indices.extend_from_slice(&[
                i, i + n_u, i + 1,
                i + 1, i + n_u, i + n_u + 1,
            ]);
        }
    }

    CpuMesh { vertices, indices }
}
