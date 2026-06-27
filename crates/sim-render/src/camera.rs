//! Camera + scene uniforms.

use glam::{Mat4, Vec3};

/// Camera + lighting uniform sent to GPU.
/// Matches the CameraUniform struct in shader.wgsl.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    /// Sun direction (xyz) + intensity (w).
    pub sun_dir: [f32; 4],
    /// Ambient color (rgb) + intensity (a).
    pub ambient: [f32; 4],
    /// Fog color (rgb) + density (a).
    pub fog_color: [f32; 4],
    /// Camera world position (xyz) + unused (w).
    pub camera_pos: [f32; 4],
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            sun_dir: [0.4, 0.8, 0.3, 1.5],
            ambient: [0.45, 0.5, 0.65, 0.2],
            fog_color: [0.05, 0.06, 0.10, 0.008],
            camera_pos: [0.0, 10.0, 30.0, 0.0],
        }
    }
}

/// Orbit camera state.
#[derive(Debug, Clone)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_deg: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            eye: Vec3::new(30.0, 20.0, 40.0),
            target: Vec3::new(0.0, 5.0, 0.0),
            up: Vec3::Y,
            fov_y_deg: 45.0,
            near: 0.1,
            far: 5000.0,
        }
    }
}

impl Camera {
    /// Build the full uniform including lighting driven by solar elevation.
    pub fn build_uniform_with_sun(&self, aspect: f32, solar_elev_deg: f32) -> CameraUniform {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        let proj = Mat4::perspective_rh(
            self.fov_y_deg.to_radians(),
            aspect,
            self.near,
            self.far,
        );

        // Sun direction from elevation.
        let elev_rad = solar_elev_deg.to_radians();
        let sun_y = elev_rad.sin().max(-0.2);
        let sun_xz = elev_rad.cos();
        let sun_dir = Vec3::new(sun_xz * 0.7, sun_y, sun_xz * 0.3).normalize();
        let sun_intensity = (sun_y * 1.8 + 0.2).clamp(0.05, 2.0);

        // Warmth at low sun.
        let warmth = (1.0 - sun_y.max(0.0)).max(0.0);

        // Ambient — bluer at night, warmer at dusk.
        let ambient_intensity = (sun_y * 0.3 + 0.15).clamp(0.05, 0.35);

        // Fog — warmer and denser at dusk, thin during day.
        let fog_density = if solar_elev_deg < 10.0 { 0.012 } else { 0.006 };
        let fog_r = 0.05 + warmth * 0.15;
        let fog_g = 0.06 + warmth * 0.05;
        let fog_b = 0.10 - warmth * 0.04;

        CameraUniform {
            view_proj: (proj * view).to_cols_array_2d(),
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, sun_intensity],
            ambient: [0.4 + warmth * 0.2, 0.45, 0.6 - warmth * 0.2, ambient_intensity],
            fog_color: [fog_r, fog_g, fog_b, fog_density],
            camera_pos: [self.eye.x, self.eye.y, self.eye.z, 0.0],
        }
    }

    pub fn build_uniform(&self, aspect: f32) -> CameraUniform {
        self.build_uniform_with_sun(aspect, 55.0)
    }

    pub fn orbit(&mut self, d_yaw: f32, d_pitch: f32) {
        let offset = self.eye - self.target;
        let r = offset.length();
        let yaw = offset.z.atan2(offset.x) + d_yaw;
        let pitch = (offset.y / r).asin().clamp(-1.4, 1.4) + d_pitch;
        let pitch = pitch.clamp(-1.4, 1.4);

        self.eye = self.target
            + Vec3::new(
                r * pitch.cos() * yaw.cos(),
                r * pitch.sin(),
                r * pitch.cos() * yaw.sin(),
            );
    }

    pub fn dolly(&mut self, factor: f32) {
        let dir = (self.eye - self.target).normalize();
        let dist = (self.eye - self.target).length();
        let new_dist = (dist * factor).clamp(2.0, 500.0);
        self.eye = self.target + dir * new_dist;
    }
}
