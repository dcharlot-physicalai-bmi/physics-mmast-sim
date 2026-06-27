//! Parametric HALE Solar UAV geometry.
//!
//! Construction (in cad-kernel's mm units, Y=up):
//!   - Fuselage: cylinder along X, radius 120 mm, length 2400 mm
//!   - Main wing: thin box, span 6000 mm, chord 500 mm, thickness 40 mm
//!   - Vertical stabilizer: thin box, 700×500×30 mm, tail end of fuselage
//!   - Horizontal stabilizer: thin box, 1400×30×300 mm, tail end of fuselage
//!
//! The components are boolean-unioned into a single solid body so mass
//! properties and the tessellated mesh both cover the whole aircraft.

use crate::{box_mesh, compute_mass_properties, derive_properties, density, VehicleGeometry};

// Reference design dimensions (mm). These are the only "magic numbers" —
// everything downstream (mass, area, inertia) is derived by the kernel.
const FUSELAGE_RADIUS_MM: f64 = 120.0;
const FUSELAGE_LENGTH_MM: f64 = 2400.0;

const WING_SPAN_MM: f64 = 6000.0;
const WING_CHORD_MM: f64 = 500.0;
const WING_THICKNESS_MM: f64 = 40.0;

const VSTAB_HEIGHT_MM: f64 = 500.0;
const VSTAB_CHORD_MM: f64 = 400.0;
const VSTAB_THICKNESS_MM: f64 = 30.0;

const HSTAB_SPAN_MM: f64 = 1400.0;
const HSTAB_CHORD_MM: f64 = 300.0;
const HSTAB_THICKNESS_MM: f64 = 30.0;

// Target total mass (kg) for the HALE reference design — matches
// drone-notes.md. The solid-box proxy's volume is much larger than a
// real hollow monocoque airframe, so we build with a nominal density
// and then calibrate via `with_target_mass()` so the derived mass
// lines up with the engineering spec.
const TARGET_MASS_KG: f64 = 7.0;
const NOMINAL_DENSITY_G_PER_MM3: f64 = density::CARBON_COMPOSITE;

pub fn build() -> VehicleGeometry {
    // Each component is a separate box solid. We tessellate each independently
    // and merge the triangle meshes — disjoint solids that touch without
    // overlapping cannot be boolean-unioned, but the divergence-theorem mass
    // sum is valid regardless of whether the solids share faces.
    let tol = VehicleGeometry::TESSELLATION_TOLERANCE_MM;

    // ---- Fuselage (box proxy aligned with +X), long axis centered on X ----
    let mut mesh = box_mesh(
        FUSELAGE_LENGTH_MM,
        2.0 * FUSELAGE_RADIUS_MM,
        2.0 * FUSELAGE_RADIUS_MM,
        -FUSELAGE_LENGTH_MM * 0.5,
        0.0,
        0.0,
        tol,
    );

    // ---- Main wing (on top of the fuselage, slightly forward) ----
    mesh.merge(&box_mesh(
        WING_CHORD_MM,
        WING_THICKNESS_MM,
        WING_SPAN_MM,
        -WING_CHORD_MM * 0.5 - FUSELAGE_LENGTH_MM * 0.15,
        2.0 * FUSELAGE_RADIUS_MM,
        -WING_SPAN_MM * 0.5,
        tol,
    ));

    // ---- Horizontal stabilizer ----
    mesh.merge(&box_mesh(
        HSTAB_CHORD_MM,
        HSTAB_THICKNESS_MM,
        HSTAB_SPAN_MM,
        FUSELAGE_LENGTH_MM * 0.5 - HSTAB_CHORD_MM,
        2.0 * FUSELAGE_RADIUS_MM,
        -HSTAB_SPAN_MM * 0.5,
        tol,
    ));

    // ---- Vertical stabilizer ----
    mesh.merge(&box_mesh(
        VSTAB_CHORD_MM,
        VSTAB_HEIGHT_MM,
        VSTAB_THICKNESS_MM,
        FUSELAGE_LENGTH_MM * 0.5 - VSTAB_CHORD_MM,
        2.0 * FUSELAGE_RADIUS_MM,
        -VSTAB_THICKNESS_MM * 0.5,
        tol,
    ));

    // ---- Derive SI properties at the nominal density ----
    let raw_properties = compute_mass_properties(&mesh, NOMINAL_DENSITY_G_PER_MM3);
    let derived = derive_properties(&mesh, NOMINAL_DENSITY_G_PER_MM3);

    let uncalibrated = VehicleGeometry {
        name: "HALE Solar UAV".into(),
        mesh,
        raw_properties,
        derived,
        density_g_per_mm3: NOMINAL_DENSITY_G_PER_MM3,
    };

    // Calibrate effective density to match the engineering-spec 7 kg mass.
    let calibrated = uncalibrated.with_target_mass(TARGET_MASS_KG);

    tracing::info!(
        "HALE geometry: mass={:.2} kg (calibrated), upward area={:.2} m², volume={:.4} m³, tris={}",
        calibrated.derived.mass_kg,
        calibrated.derived.upward_projected_area_m2,
        calibrated.derived.volume_m3,
        calibrated.mesh.triangle_count(),
    );

    calibrated
}
