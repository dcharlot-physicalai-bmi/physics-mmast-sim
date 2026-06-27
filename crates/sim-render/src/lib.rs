//! `sim-render` — WebGPU renderer for the MMAST simulator.
//!
//! Optimized using the periodic stack of compute primitives:
//!   #40 Scatter/Gather (L₀) — instanced rendering, 1 draw call per type
//!   #41 Embedding Lookup (L₀) — per-instance color, atmosphere LUT
//!   #2  Bitwise Logic (L₀) — seeded terrain hash
//!   #1  FMA (L₁) — matrix × vertex in hardware
//!   #23/#24 FFT/Convolution (B₂) — GPU compute-shader terrain gen

pub mod vertex;
pub mod mesh;
pub mod camera;
pub mod pipeline;
pub mod scene;
pub mod instance;
pub mod terrain_compute;
pub mod culling;

use culling::{CullableBatch, CullPipeline};
use instance::InstancedBatch;
use mesh::GpuMesh;

/// Renderer configuration.
#[derive(Debug, Clone)]
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
    pub pixel_ratio: f32,
    pub clear_color: wgpu::Color,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            pixel_ratio: 1.0,
            clear_color: wgpu::Color {
                r: 0.02,
                g: 0.023,
                b: 0.04,
                a: 1.0,
            },
        }
    }
}

/// Per-frame stats for the HUD.
#[derive(Debug, Clone, Default)]
pub struct RenderStats {
    pub draw_calls: u32,
    pub total_instances: u32,
    pub total_triangles: u32,
    /// Source (pre-cull) instance count in cullable batches.
    pub cullable_source: u32,
}

/// The top-level renderer.
pub struct Renderer {
    pub config: RenderConfig,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: wgpu::RenderPipeline,
    pub cull_pipeline: CullPipeline,
    pub camera_uniform: camera::CameraUniform,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub depth_texture: wgpu::TextureView,

    /// Non-instanced meshes (terrain, unique geometry).
    pub meshes: Vec<GpuMesh>,
    /// Instanced batches — 1 draw call per batch, fixed instance count, no culling.
    pub batches: Vec<InstancedBatch>,
    /// GPU-culled instanced batches — visibility tested each frame via compute.
    pub cullable: Vec<CullableBatch>,

    /// Last frame stats.
    pub stats: RenderStats,
}

impl Renderer {
    /// Create a new renderer.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        config: RenderConfig,
    ) -> Self {
        use wgpu::util::DeviceExt;

        let camera_uniform = camera::CameraUniform::default();
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera_uniform"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bg"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let pipeline = pipeline::create_instanced_pipeline(
            &device,
            surface_format,
            &camera_bind_group_layout,
        );

        let cull_pipeline = CullPipeline::new(&device);

        let depth_texture = pipeline::create_depth_texture(&device, config.width, config.height);

        Self {
            config,
            device,
            queue,
            pipeline,
            cull_pipeline,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            depth_texture,
            meshes: Vec::new(),
            batches: Vec::new(),
            cullable: Vec::new(),
            stats: RenderStats::default(),
        }
    }

    /// Add a GPU-culled instanced batch. Per-frame compute pass tests each
    /// instance's bounding sphere against the frustum and atomically compacts
    /// visible instances into the draw buffer.
    pub fn add_cullable(
        &mut self,
        prototype: &mesh::CpuMesh,
        instances: &[instance::InstanceRaw],
    ) -> usize {
        let batch = CullableBatch::new(&self.device, &self.cull_pipeline, prototype, instances);
        let idx = self.cullable.len();
        self.cullable.push(batch);
        idx
    }

    /// Upload a non-instanced mesh (rendered as 1 instance with identity transform).
    pub fn add_mesh(&mut self, cpu_mesh: &mesh::CpuMesh) -> usize {
        // Wrap in a single-instance batch so everything goes through the instanced pipeline.
        let identity = instance::InstanceRaw {
            model: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            color: [0.0, 0.0, 0.0, 0.0], // zero alpha = use vertex color
        };
        let batch = InstancedBatch::new(&self.device, cpu_mesh, &[identity]);
        let idx = self.batches.len();
        self.batches.push(batch);
        idx
    }

    /// Add an instanced batch: one geometry drawn at N positions. 1 draw call total.
    pub fn add_instanced(
        &mut self,
        prototype: &mesh::CpuMesh,
        instances: &[instance::InstanceRaw],
    ) -> usize {
        let batch = InstancedBatch::new(&self.device, prototype, instances);
        let idx = self.batches.len();
        self.batches.push(batch);
        idx
    }

    /// Render one frame.
    pub fn render(
        &mut self,
        view: &wgpu::TextureView,
        camera: &camera::Camera,
        sim_state: Option<&sim_core::state::SimState>,
    ) {
        // Build camera uniform with solar elevation from sim state.
        let aspect = self.config.width as f32 / self.config.height as f32;
        let solar_elev = sim_state
            .map(|s| s.clock.solar_elevation_deg as f32)
            .unwrap_or(55.0);
        self.camera_uniform = camera.build_uniform_with_sun(aspect, solar_elev);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Reset cullable batch instance counts and update frustum uniforms.
        // These writes are ordered before encoder submission by wgpu.
        if !self.cullable.is_empty() {
            let view_proj = glam::Mat4::from_cols_array_2d(&self.camera_uniform.view_proj);
            for batch in &self.cullable {
                batch.reset_count(&self.queue);
                batch.update_frustum(&self.queue, &view_proj);
            }
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder"),
            });

        let mut draw_calls = 0u32;
        let mut total_instances = 0u32;
        let mut total_triangles = 0u32;
        let mut cullable_source = 0u32;

        // ---- Compute pass: GPU frustum culling ----
        if !self.cullable.is_empty() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cull_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cull_pipeline.pipeline);
            for batch in &self.cullable {
                batch.dispatch_cull(&mut pass);
                cullable_source += batch.source_count;
            }
        }

        // ---- Main render pass ----
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.config.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);

            // Non-cullable instanced batches: fixed instance count.
            for batch in &self.batches {
                batch.draw(&mut pass);
                draw_calls += 1;
                total_instances += batch.instance_count;
                total_triangles += (batch.index_count / 3) * batch.instance_count;
            }

            // Cullable batches: draw_indexed_indirect reads count from GPU buffer.
            for batch in &self.cullable {
                batch.draw(&mut pass);
                draw_calls += 1;
                // Note: actual visible count is on GPU; we report source count here.
                total_instances += batch.source_count;
                total_triangles += (batch.index_count / 3) * batch.source_count;
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        self.stats = RenderStats {
            draw_calls,
            total_instances,
            total_triangles,
            cullable_source,
        };
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        self.depth_texture = pipeline::create_depth_texture(&self.device, width, height);
    }
}
