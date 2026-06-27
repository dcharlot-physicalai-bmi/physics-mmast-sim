//! `sim-geometry` — Parametric vehicle geometry built on CadFuture's
//! B-Rep kernel.
//!
//! This is the bridge between `cad-future` (the lab's computable-world-model
//! engine) and `physics-mmast-sim` (the application). Every `VehicleGeometry`
//! is a real B-Rep solid built via `physical_brep::make_box` and tessellated
//! by `physical_tessellation`. The derived quantities (mass, solar surface
//! area, bounding box, tessellated mesh for GPU rendering) all flow from the
//! kernel output.
//!
//! Units:
//!   - the kernel works in **millimeters** (coordinates) and **g/mm³** (density).
//!   - sim-geometry returns **SI units** at its external API (m, kg, m², m³).
//!   - All mm → m conversion happens here so the rest of the simulator can
//!     stay in SI without knowing anything about the kernel's internal units.

pub mod hale_uav;
pub mod vehicles;
pub mod mesh_adapter;

use physical_brep::make_box;
use physical_tessellation::{tessellate, TessMesh};
use serde::{Deserialize, Serialize};

/// Error returned when vehicle geometry cannot be built.
///
/// CadFuture's primitive + tessellation operations are infallible, so this is
/// retained only to preserve the `Result`-returning constructor API that
/// downstream callers expect.
#[derive(Debug)]
pub struct GeometryError(pub String);

impl std::fmt::Display for GeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "geometry build failed: {}", self.0)
    }
}
impl std::error::Error for GeometryError {}

/// Mass properties of a tessellated solid, in the kernel's internal units
/// (mm³, mm², g, g·mm²). Computed from the triangle mesh via the divergence
/// theorem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MassProperties {
    /// Volume (mm³).
    pub volume: f64,
    /// Surface area (mm²).
    pub surface_area: f64,
    /// Center of mass (mm).
    pub center_of_mass: [f64; 3],
    /// Mass (grams).
    pub mass: f64,
    /// Moments of inertia about center of mass (g·mm²) [Ixx, Iyy, Izz].
    pub moments_of_inertia: [f64; 3],
    /// Products of inertia (g·mm²) [Ixy, Ixz, Iyz].
    pub products_of_inertia: [f64; 3],
}

/// Result of computing geometric properties from a tessellated B-Rep solid,
/// converted to SI units.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedProperties {
    /// Mass (kg), from the kernel's divergence-theorem volume × material density.
    pub mass_kg: f64,
    /// Total surface area (m²), integrated over every triangle.
    pub surface_area_m2: f64,
    /// Volume (m³).
    pub volume_m3: f64,
    /// Center of mass (m, world frame).
    pub center_of_mass_m: [f64; 3],
    /// Principal moments of inertia about the COM (kg·m²) [Ixx, Iyy, Izz].
    pub moments_of_inertia: [f64; 3],
    /// Upward-projected surface area (m²) — sum of every triangle's area
    /// weighted by `max(0, normal · Ŷ)`. This is the area exposed to the
    /// zenith sun at noon — directly usable as the PV footprint.
    pub upward_projected_area_m2: f64,
    /// Axis-aligned bounding box in world frame (m). [min, max].
    pub bbox_min_m: [f64; 3],
    pub bbox_max_m: [f64; 3],
}

/// A parametric vehicle built on CadFuture's B-Rep kernel.
///
/// Owns a single tessellated mesh (the union of all component solids) and the
/// derived SI-unit properties.
pub struct VehicleGeometry {
    pub name: String,
    pub mesh: TessMesh,
    pub raw_properties: MassProperties,
    pub derived: DerivedProperties,
    /// Density used for the mass computation (g/mm³), preserved so the
    /// caller knows what material assumption was baked in.
    pub density_g_per_mm3: f64,
}

impl VehicleGeometry {
    /// Tessellation tolerance in mm. 0.5 mm is a good default for vehicles
    /// at meter scale.
    pub const TESSELLATION_TOLERANCE_MM: f64 = 0.5;

    /// Build the HALE Solar UAV reference geometry (7 kg, 6 m span, ~4 m² wing).
    pub fn hale_solar_uav() -> Result<Self, GeometryError> {
        Ok(hale_uav::build())
    }

    pub fn recon_quadcopter() -> Result<Self, GeometryError> {
        Ok(vehicles::build_quad())
    }

    pub fn stratospheric_glider() -> Result<Self, GeometryError> {
        Ok(vehicles::build_strato())
    }

    pub fn auv() -> Result<Self, GeometryError> {
        Ok(vehicles::build_auv())
    }

    pub fn airship() -> Result<Self, GeometryError> {
        Ok(vehicles::build_airship())
    }

    /// Cloud Carrier — autonomous stratospheric platform for the orbital economy.
    pub fn cloud_carrier() -> Result<Self, GeometryError> {
        Ok(vehicles::build_cloud_carrier())
    }

    pub fn planetary_rover() -> Result<Self, GeometryError> {
        Ok(vehicles::build_rover())
    }

    /// Calibrate the effective density so that the computed mass matches
    /// an engineering-target total mass (kg). The geometry shape is
    /// unchanged; only the density and derived mass/inertia fields move.
    pub fn with_target_mass(mut self, target_mass_kg: f64) -> Self {
        let current_mass = self.derived.mass_kg;
        if current_mass > 1e-9 {
            let scale = target_mass_kg / current_mass;
            self.density_g_per_mm3 *= scale;
            self.raw_properties.mass *= scale;
            for v in self.raw_properties.moments_of_inertia.iter_mut() {
                *v *= scale;
            }
            for v in self.raw_properties.products_of_inertia.iter_mut() {
                *v *= scale;
            }
            self.derived.mass_kg = target_mass_kg;
            for v in self.derived.moments_of_inertia.iter_mut() {
                *v *= scale;
            }
        }
        self
    }
}

/// Build a translated, tessellated box mesh.
///
/// CadFuture's `make_box` is centered at the origin on every axis. The
/// original kernel's box convention was "center of bottom" — centered in X/Z
/// but spanning `y ∈ [0, h]`. To keep the vehicle builders' translation
/// offsets identical, we lift the centered box by `+h/2` in Y, then apply the
/// caller's translation. The result spans `y ∈ [ty, ty + h]`, exactly as the
/// old `create_box(w, h, d); translate(tx, ty, tz)` pair did.
pub fn box_mesh(w: f64, h: f64, d: f64, tx: f64, ty: f64, tz: f64, tol: f64) -> TessMesh {
    let solid = make_box(w, h, d);
    let mut mesh = tessellate(&solid, tol);
    let ox = tx as f32;
    let oy = (ty + h * 0.5) as f32;
    let oz = tz as f32;
    for v in &mut mesh.vertices {
        v.position[0] += ox;
        v.position[1] += oy;
        v.position[2] += oz;
    }
    mesh
}

/// Compute mass properties from a tessellated triangle mesh (mm units) at the
/// given density (g/mm³), via the divergence theorem.
pub fn compute_mass_properties(mesh: &TessMesh, density: f64) -> MassProperties {
    let mut volume = 0.0;
    let mut surface_area = 0.0;
    let (mut cx, mut cy, mut cz) = (0.0, 0.0, 0.0);

    let tc = mesh.triangle_count();
    for i in 0..tc {
        let i0 = mesh.indices[i * 3] as usize;
        let i1 = mesh.indices[i * 3 + 1] as usize;
        let i2 = mesh.indices[i * 3 + 2] as usize;
        let v0 = vertex_pos(mesh, i0);
        let v1 = vertex_pos(mesh, i1);
        let v2 = vertex_pos(mesh, i2);

        let e1 = sub3(v1, v0);
        let e2 = sub3(v2, v0);
        let cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let area = 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        surface_area += area;

        let sv = signed_volume(v0, v1, v2);
        volume += sv;

        let tri_cx = (v0[0] + v1[0] + v2[0]) / 4.0;
        let tri_cy = (v0[1] + v1[1] + v2[1]) / 4.0;
        let tri_cz = (v0[2] + v1[2] + v2[2]) / 4.0;
        cx += sv * tri_cx;
        cy += sv * tri_cy;
        cz += sv * tri_cz;
    }

    volume = volume.abs();
    let mass = volume * density;
    let com = if volume > 1e-15 {
        [cx / volume, cy / volume, cz / volume]
    } else {
        [0.0, 0.0, 0.0]
    };

    let (mut ixx, mut iyy, mut izz) = (0.0, 0.0, 0.0);
    let (mut ixy, mut ixz, mut iyz) = (0.0, 0.0, 0.0);
    if volume > 1e-15 {
        for i in 0..tc {
            let i0 = mesh.indices[i * 3] as usize;
            let i1 = mesh.indices[i * 3 + 1] as usize;
            let i2 = mesh.indices[i * 3 + 2] as usize;
            let v0 = vertex_pos(mesh, i0);
            let v1 = vertex_pos(mesh, i1);
            let v2 = vertex_pos(mesh, i2);
            let e1 = sub3(v1, v0);
            let e2 = sub3(v2, v0);
            let cross_mag = ((e1[1] * e2[2] - e1[2] * e2[1]).powi(2)
                + (e1[2] * e2[0] - e1[0] * e2[2]).powi(2)
                + (e1[0] * e2[1] - e1[1] * e2[0]).powi(2))
            .sqrt();
            let area = 0.5 * cross_mag;
            for v in [v0, v1, v2] {
                let dx = v[0] - com[0];
                let dy = v[1] - com[1];
                let dz = v[2] - com[2];
                let w = area / 3.0 * density;
                ixx += w * (dy * dy + dz * dz);
                iyy += w * (dx * dx + dz * dz);
                izz += w * (dx * dx + dy * dy);
                ixy -= w * dx * dy;
                ixz -= w * dx * dz;
                iyz -= w * dy * dz;
            }
        }
    }

    MassProperties {
        volume,
        surface_area,
        center_of_mass: com,
        mass,
        moments_of_inertia: [ixx, iyy, izz],
        products_of_inertia: [ixy, ixz, iyz],
    }
}

/// Compute SI-unit derived properties from a tessellated mesh.
pub fn derive_properties(mesh: &TessMesh, density_g_per_mm3: f64) -> DerivedProperties {
    let raw = compute_mass_properties(mesh, density_g_per_mm3);

    let surface_area_m2 = raw.surface_area * 1e-6;
    let volume_m3 = raw.volume * 1e-9;
    let mass_kg = raw.mass * 1e-3;

    let center_of_mass_m = [
        raw.center_of_mass[0] * 1e-3,
        raw.center_of_mass[1] * 1e-3,
        raw.center_of_mass[2] * 1e-3,
    ];

    // Moments of inertia: g·mm² → kg·m² is a factor of 1e-9.
    let moments_of_inertia = [
        raw.moments_of_inertia[0] * 1e-9,
        raw.moments_of_inertia[1] * 1e-9,
        raw.moments_of_inertia[2] * 1e-9,
    ];

    let (bbox_min_mm, bbox_max_mm) = bbox_mm(mesh);
    let upward_area_mm2 = upward_projected_area_mm2(mesh);
    let upward_projected_area_m2 = upward_area_mm2 * 1e-6;

    DerivedProperties {
        mass_kg,
        surface_area_m2,
        volume_m3,
        center_of_mass_m,
        moments_of_inertia,
        upward_projected_area_m2,
        bbox_min_m: [bbox_min_mm[0] * 1e-3, bbox_min_mm[1] * 1e-3, bbox_min_mm[2] * 1e-3],
        bbox_max_m: [bbox_max_mm[0] * 1e-3, bbox_max_mm[1] * 1e-3, bbox_max_mm[2] * 1e-3],
    }
}

/// Sum triangle areas weighted by `max(0, n · ŷ)` — the effective solar
/// footprint at solar noon.
fn upward_projected_area_mm2(mesh: &TessMesh) -> f64 {
    let mut total = 0.0;
    let tc = mesh.triangle_count();
    for i in 0..tc {
        let i0 = mesh.indices[i * 3] as usize;
        let i1 = mesh.indices[i * 3 + 1] as usize;
        let i2 = mesh.indices[i * 3 + 2] as usize;

        let v0 = vertex_pos(mesh, i0);
        let v1 = vertex_pos(mesh, i1);
        let v2 = vertex_pos(mesh, i2);

        let e1 = sub3(v1, v0);
        let e2 = sub3(v2, v0);
        let cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let mag = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        let area = 0.5 * mag;
        if mag > 1e-12 {
            let up = cross[1] / mag;
            if up > 0.0 {
                total += area * up;
            }
        }
    }
    total
}

fn bbox_mm(mesh: &TessMesh) -> ([f64; 3], [f64; 3]) {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for v in &mesh.vertices {
        let p = [v.position[0] as f64, v.position[1] as f64, v.position[2] as f64];
        for k in 0..3 {
            if p[k] < min[k] {
                min[k] = p[k];
            }
            if p[k] > max[k] {
                max[k] = p[k];
            }
        }
    }
    (min, max)
}

fn vertex_pos(mesh: &TessMesh, idx: usize) -> [f64; 3] {
    let p = mesh.vertices[idx].position;
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Signed volume of the tetrahedron formed by a triangle and the origin.
fn signed_volume(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3]) -> f64 {
    (v0[0] * (v1[1] * v2[2] - v1[2] * v2[1]) - v0[1] * (v1[0] * v2[2] - v1[2] * v2[0])
        + v0[2] * (v1[0] * v2[1] - v1[1] * v2[0]))
        / 6.0
}

/// Common vehicle material densities (g/mm³).
pub mod density {
    /// Carbon fiber composite — representative HALE UAV primary structure.
    pub const CARBON_COMPOSITE: f64 = 0.0016;
    /// Aluminum 6061 — airframes, airship gondolas, rover chassis.
    pub const ALUMINUM_6061: f64 = 0.0027;
    /// Li-ion battery pack average.
    pub const LI_ION_PACK: f64 = 0.0025;
    /// Titanium 6Al-4V — space structures.
    pub const TITANIUM: f64 = 0.0044;
    /// Thermoplastic (PLA/ABS) — quadcopter frames.
    pub const PLASTIC: f64 = 0.0012;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hale_geometry_has_sane_derived_properties() {
        let geom = VehicleGeometry::hale_solar_uav().expect("build HALE geometry");
        let span = (geom.derived.bbox_max_m[2] - geom.derived.bbox_min_m[2]).abs();
        eprintln!(
            "\nHALE derived: mass={:.2} kg, upward_area={:.3} m², volume={:.4} m³, span={:.2} m, tris={}",
            geom.derived.mass_kg,
            geom.derived.upward_projected_area_m2,
            geom.derived.volume_m3,
            span,
            geom.mesh.triangle_count(),
        );

        assert!(
            geom.derived.mass_kg > 3.0 && geom.derived.mass_kg < 25.0,
            "HALE mass was {} kg (expected 3-25 kg)",
            geom.derived.mass_kg
        );

        assert!(
            geom.derived.upward_projected_area_m2 > 1.0
                && geom.derived.upward_projected_area_m2 < 6.0,
            "HALE upward area was {} m² (expected 1-6 m²)",
            geom.derived.upward_projected_area_m2
        );

        let span = (geom.derived.bbox_max_m[2] - geom.derived.bbox_min_m[2]).abs();
        assert!(
            span > 4.0 && span < 8.0,
            "HALE wingspan was {} m (expected 4-8 m)",
            span
        );

        assert!(geom.mesh.triangle_count() > 10);
    }
}
