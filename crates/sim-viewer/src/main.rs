//! `mmast-viewer` — Native window viewer for live MMAST simulation.
//!
//! Controls:
//!   Left-drag:   orbit camera
//!   Scroll:      zoom
//!   Space:       pause / resume
//!   R:           restart
//!   1-6:         switch vehicle preset
//!   +/-:         sim speed

use anyhow::Result;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

use sim_core::SimConfig;
use sim_dynamics::Solver;
use sim_environment::atmosphere::StandardAtmosphere;
use sim_geometry::mesh_adapter::RenderableMesh;
use sim_geometry::VehicleGeometry;
use sim_render::{
    camera::Camera,
    instance::InstanceRaw,
    mesh::CpuMesh,
    terrain_compute::{generate_terrain_gpu, TerrainParams},
    vertex::Vertex,
    Renderer, RenderConfig,
};

/// Convert a cad-kernel-tessellated mesh (SI units, from sim-geometry)
/// into a sim-render CpuMesh with a uniform color.
fn renderable_to_cpu_mesh(r: &RenderableMesh, color: [f32; 4]) -> CpuMesh {
    let vertices: Vec<Vertex> = r
        .positions
        .iter()
        .zip(r.normals.iter())
        .map(|(&p, &n)| Vertex {
            position: p,
            normal: n,
            color,
        })
        .collect();
    CpuMesh {
        vertices,
        indices: r.indices.clone(),
    }
}
use sim_vehicle::VehicleParams;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let event_loop = EventLoop::new()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    camera: Camera,
}

struct App {
    gpu: Option<GpuState>,
    window: Option<Arc<Window>>,
    solver: Solver,
    env: StandardAtmosphere,
    last_state: Option<sim_core::state::SimState>,
    paused: bool,
    sim_speed: f64,
    mouse_pressed: bool,
    last_mouse: (f64, f64),
}

impl App {
    fn new() -> Self {
        let vehicle = VehicleParams::hale_solar_uav();
        let (solver, env) = create_solver(vehicle);
        Self {
            gpu: None,
            window: None,
            solver,
            env,
            last_state: None,
            paused: false,
            sim_speed: 60.0,
            mouse_pressed: false,
            last_mouse: (0.0, 0.0),
        }
    }

    fn reset_vehicle(&mut self, vehicle: VehicleParams) {
        let name = vehicle.name.clone();
        let (solver, env) = create_solver(vehicle);
        self.solver = solver;
        self.env = env;
        self.last_state = None;
        tracing::info!("Vehicle: {name}");
    }

    fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("No suitable GPU adapter");

        tracing::info!("GPU: {:?}", adapter.get_info().name);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("mmast_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
        ))
        .expect("Failed to create device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let config = RenderConfig {
            width: size.width.max(1),
            height: size.height.max(1),
            ..Default::default()
        };

        let mut renderer = Renderer::new(device, queue, format, config);

        // ---- Terrain: GPU compute-shader FBM noise (Primitives #23/#24, B₂) ----
        // 128×128 grid = 16,384 vertices generated in one dispatch.
        // Replaces the CPU sin/cos terrain generator.
        let t_start = std::time::Instant::now();
        let terrain = generate_terrain_gpu(
            &renderer.device,
            &renderer.queue,
            TerrainParams {
                grid_size: 128,
                world_size: 400.0,
                seed: 42,
                octaves: 5,
                amplitude: 2.5,
                frequency: 0.012,
                ..Default::default()
            },
        );
        tracing::info!(
            "Terrain: {} verts, {} tris generated on GPU in {:?}",
            terrain.vertices.len(),
            terrain.indices.len() / 3,
            t_start.elapsed()
        );
        renderer.add_mesh(&terrain);

        // ---- UAV (real B-Rep geometry from openie-cad's cad-kernel) ----
        // The HALE airframe is built by the cad-kernel as NURBS primitives
        // (fuselage + wing + tail + vstab), tessellated by Truck, and the
        // resulting Mesh3D is converted to renderer vertices here.
        let t_geom = std::time::Instant::now();
        match VehicleGeometry::hale_solar_uav() {
            Ok(geom) => {
                let rm = RenderableMesh::from_kernel(&geom.mesh);
                tracing::info!(
                    "HALE from cad-kernel: {} vertices, {} triangles, mass {:.2} kg, upward area {:.2} m² ({:.2?})",
                    rm.vertex_count(),
                    rm.triangle_count(),
                    geom.derived.mass_kg,
                    geom.derived.upward_projected_area_m2,
                    t_geom.elapsed(),
                );
                let cpu = renderable_to_cpu_mesh(&rm, [0.04, 0.06, 0.12, 1.0]);
                renderer.add_mesh(&cpu);
            }
            Err(e) => {
                tracing::error!("Failed to build HALE from cad-kernel: {e}. Falling back to procedural meshes.");
                for m in &sim_render::scene::build_hale_uav() {
                    renderer.add_mesh(m);
                }
            }
        }

        // ---- Runway (1 draw call) ----
        renderer.add_mesh(&CpuMesh::box_mesh(90.0, 0.05, 8.0, [0.05, 0.06, 0.08, 1.0]));

        // ---- Runway edge lights: INSTANCED ----
        // Old: 46 individual meshes = 46 draw calls.
        // New: 1 prototype + 46 instances = 1 draw call. Primitive #40 L₀.
        {
            let light_proto = CpuMesh::box_mesh(0.2, 0.15, 0.15, [0.9, 0.9, 0.7, 1.0]);
            let mut instances = Vec::new();
            for x in (-45..=45).step_by(4) {
                for &z in &[-4.4_f32, 4.4] {
                    instances.push(InstanceRaw::at(
                        x as f32, 0.07, z, 1.0,
                        [0.9, 0.9, 0.7, 1.0],
                    ));
                }
            }
            // Red threshold lights.
            for &z in &[-4.4, -3.4, 3.4, 4.4] {
                instances.push(InstanceRaw::at(-45.5, 0.08, z, 1.2, [1.0, 0.2, 0.25, 1.0]));
            }
            // Green departure lights.
            for &z in &[-4.4, -3.4, 3.4, 4.4] {
                instances.push(InstanceRaw::at(45.5, 0.08, z, 1.2, [0.2, 1.0, 0.4, 1.0]));
            }
            renderer.add_cullable(&light_proto, &instances);
        }

        // ---- Procedural trees: INSTANCED ----
        // 200 trees = 1 draw call (trunk) + 1 draw call (canopy) = 2 draw calls.
        // Old approach: 200 × 2 = 400 draw calls.
        {
            let trunk = CpuMesh::cylinder(0.08, 1.2, 6, [0.25, 0.15, 0.08, 1.0]);
            let canopy = CpuMesh::box_mesh(0.8, 0.6, 0.8, [0.1, 0.28, 0.08, 1.0]);

            let mut trunk_instances = Vec::new();
            let mut canopy_instances = Vec::new();
            let mut rng = 42u32;

            for _ in 0..200 {
                // Seeded hash — Primitive #2 Bitwise Logic (L₀).
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                let x = (rng as f32 / u32::MAX as f32) * 300.0 - 150.0;
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                let z = (rng as f32 / u32::MAX as f32) * 300.0 - 150.0;

                // Skip runway corridor.
                if x.abs() < 50.0 && z.abs() < 10.0 {
                    continue;
                }

                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                let scale = 0.7 + (rng as f32 / u32::MAX as f32) * 0.8;

                let height = (x * 0.05).sin() * 0.8 + (z * 0.07).cos() * 0.6;

                trunk_instances.push(InstanceRaw::at(x, height + 0.6 * scale, z, scale, [0.0; 4]));
                canopy_instances.push(InstanceRaw::at(
                    x,
                    height + 1.3 * scale,
                    z,
                    scale,
                    [0.08 + (rng as f32 / u32::MAX as f32) * 0.06,
                     0.22 + (rng as f32 / u32::MAX as f32) * 0.12,
                     0.06,
                     1.0],
                ));
            }
            renderer.add_cullable(&trunk, &trunk_instances);
            renderer.add_cullable(&canopy, &canopy_instances);
        }

        // ---- Rocks: INSTANCED ----
        {
            let rock = CpuMesh::box_mesh(0.4, 0.25, 0.35, [0.18, 0.16, 0.14, 1.0]);
            let mut instances = Vec::new();
            let mut rng = 137u32;
            for _ in 0..120 {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                let x = (rng as f32 / u32::MAX as f32) * 300.0 - 150.0;
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                let z = (rng as f32 / u32::MAX as f32) * 300.0 - 150.0;
                if x.abs() < 50.0 && z.abs() < 10.0 { continue; }
                let height = (x * 0.05).sin() * 0.8 + (z * 0.07).cos() * 0.6;
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                let scale = 0.5 + (rng as f32 / u32::MAX as f32) * 1.0;
                instances.push(InstanceRaw::at(x, height + 0.1 * scale, z, scale, [0.0; 4]));
            }
            renderer.add_cullable(&rock, &instances);
        }

        tracing::info!(
            "Scene built: {} non-cull batches + {} cullable batches = {} draw calls",
            renderer.batches.len(),
            renderer.cullable.len(),
            renderer.batches.len() + renderer.cullable.len()
        );

        self.gpu = Some(GpuState {
            surface,
            surface_config,
            renderer,
            camera: Camera {
                eye: glam::Vec3::new(30.0, 15.0, 35.0),
                target: glam::Vec3::new(0.0, 3.0, 0.0),
                ..Default::default()
            },
        });
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("MMAST Physics Simulator")
                            .with_inner_size(winit::dpi::LogicalSize::new(1600, 900)),
                    )
                    .unwrap(),
            );
            self.init_gpu(window.clone());
            self.window = Some(window);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    let w = size.width.max(1);
                    let h = size.height.max(1);
                    gpu.surface_config.width = w;
                    gpu.surface_config.height = h;
                    gpu.surface.configure(&gpu.renderer.device, &gpu.surface_config);
                    gpu.renderer.resize(w, h);
                }
            }

            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match &event.logical_key {
                    Key::Named(NamedKey::Space) => {
                        self.paused = !self.paused;
                    }
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Character(c) => match c.as_str() {
                        "r" | "1" => self.reset_vehicle(VehicleParams::hale_solar_uav()),
                        "2" => self.reset_vehicle(VehicleParams::recon_quadcopter()),
                        "3" => self.reset_vehicle(VehicleParams::stratospheric_glider()),
                        "4" => self.reset_vehicle(VehicleParams::auv()),
                        "5" => self.reset_vehicle(VehicleParams::airship()),
                        "6" => self.reset_vehicle(VehicleParams::planetary_rover()),
                        "+" | "=" => {
                            self.sim_speed = (self.sim_speed * 2.0).min(86_400.0);
                            tracing::info!("Speed: {}x", self.sim_speed);
                        }
                        "-" => {
                            self.sim_speed = (self.sim_speed / 2.0).max(1.0);
                            tracing::info!("Speed: {}x", self.sim_speed);
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }

            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.mouse_pressed = state == ElementState::Pressed;
            }

            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    if let Some(gpu) = &mut self.gpu {
                        let dx = (position.x - self.last_mouse.0) as f32;
                        let dy = (position.y - self.last_mouse.1) as f32;
                        gpu.camera.orbit(-dx * 0.005, -dy * 0.005);
                    }
                }
                self.last_mouse = (position.x, position.y);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(gpu) = &mut self.gpu {
                    let scroll = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                    };
                    gpu.camera.dolly(1.0 - scroll * 0.1);
                }
            }

            WindowEvent::RedrawRequested => {
                // Advance sim. At 86,400× speed = 1 sim-day per real-second,
                // so 1/60 s real = 1440 s sim = 1440 tick() calls per frame.
                // Cap step count per frame to 10,000 to avoid stalling on
                // insanely high speeds.
                if !self.paused {
                    let sim_dt = (1.0 / 60.0) * self.sim_speed;
                    let steps = ((sim_dt / self.solver.config.dt).ceil() as usize).min(10_000);

                    // Only build a snapshot at the end — tick() is the fast path.
                    for _ in 0..steps {
                        if self.solver.clock.elapsed >= self.solver.config.duration {
                            // Loop back to start so viewer runs indefinitely.
                            self.solver.clock.elapsed = 0.0;
                            self.solver.clock.solar_hour =
                                self.solver.config.start_hour;
                        }
                        self.solver.tick(&self.env);
                    }
                    if steps > 0 {
                        self.last_state = Some(self.solver.snapshot(&self.env));
                    }
                }

                // Update title with stats
                if let (Some(st), Some(win)) = (&self.last_state, &self.window) {
                    let h = st.clock.elapsed / 3600.0;
                    let net = st.net_power_w;
                    let s = if net >= 0.0 { "+" } else { "" };
                    let p = if self.paused { " [PAUSED]" } else { "" };
                    let stats = self.gpu.as_ref().map(|g| &g.renderer.stats);
                    let dc = stats.map(|s| s.draw_calls).unwrap_or(0);
                    let inst = stats.map(|s| s.total_instances).unwrap_or(0);
                    let culled = stats.map(|s| s.cullable_source).unwrap_or(0);
                    win.set_title(&format!(
                        "MMAST | {:.1}h | {s}{:.0}W | SOC {:.0} Wh | {dc} draws / {inst} inst ({culled} cullable) | {:.0}x{p}",
                        h, net, st.battery_soc_wh, self.sim_speed
                    ));
                }

                // Render
                if let Some(gpu) = &mut self.gpu {
                    let output = match gpu.surface.get_current_texture() {
                        Ok(t) => t,
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            gpu.surface.configure(&gpu.renderer.device, &gpu.surface_config);
                            return;
                        }
                        Err(e) => {
                            tracing::error!("Surface: {e:?}");
                            return;
                        }
                    };
                    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                    gpu.renderer.render(&view, &gpu.camera, self.last_state.as_ref());
                    output.present();
                }

                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }

            _ => {}
        }
    }
}

fn create_solver(vehicle: VehicleParams) -> (Solver, StandardAtmosphere) {
    let config = SimConfig {
        dt: 1.0,
        duration: 86_400.0 * 7.0,
        ..Default::default()
    };
    let modules = sim_mmast::full_stack();
    let mut solver = Solver::new(config, vehicle, modules);
    solver.set_position([0.0, 500.0, 0.0]);
    let env = StandardAtmosphere {
        wind: [3.0, 0.0, 1.0],
        ..Default::default()
    }
    .with_lut(172);
    // Bake the MMAST LUT so tick() uses the fast path. ~4 ms one-time cost.
    solver.bake_mmast_lut(&env);
    (solver, env)
}
