//! Geometry builders for all vehicle archetypes.
//!
//! Each builder follows the same pattern as hale_uav.rs:
//!   box_mesh (create + translate, in one call) → merge → derive → calibrate mass.

use crate::{box_mesh, compute_mass_properties, derive_properties, density, VehicleGeometry};

// ---- Recon Quadcopter ----

pub fn build_quad() -> VehicleGeometry {
    let tol = VehicleGeometry::TESSELLATION_TOLERANCE_MM;

    // Central body: 150×50×150 mm
    let mut mesh = box_mesh(150.0, 50.0, 150.0, 0.0, 0.0, 0.0, tol);

    // Four arms: 200×20×20 mm each
    mesh.merge(&box_mesh(200.0, 20.0, 20.0, 100.0, 25.0, 50.0, tol));
    mesh.merge(&box_mesh(200.0, 20.0, 20.0, 100.0, 25.0, -50.0, tol));
    mesh.merge(&box_mesh(200.0, 20.0, 20.0, -100.0, 25.0, 50.0, tol));
    mesh.merge(&box_mesh(200.0, 20.0, 20.0, -100.0, 25.0, -50.0, tol));

    let raw_properties = compute_mass_properties(&mesh, density::PLASTIC);
    let derived = derive_properties(&mesh, density::PLASTIC);

    VehicleGeometry {
        name: "Recon Quadcopter".into(),
        mesh,
        raw_properties,
        derived,
        density_g_per_mm3: density::PLASTIC,
    }
    .with_target_mass(1.56)
}

// ---- Stratospheric Glider ----

pub fn build_strato() -> VehicleGeometry {
    let tol = VehicleGeometry::TESSELLATION_TOLERANCE_MM;

    // Fuselage: slender, 5000 mm long, 200 mm diameter
    let mut mesh = box_mesh(5000.0, 400.0, 200.0, -2500.0, 0.0, -100.0, tol);

    // Wing: 25 m span, 720 mm chord, 50 mm thick
    mesh.merge(&box_mesh(720.0, 50.0, 25_000.0, -360.0, 400.0, -12_500.0, tol));

    // H-stab: 4 m span
    mesh.merge(&box_mesh(500.0, 30.0, 4000.0, 2200.0, 400.0, -2000.0, tol));

    // V-stab
    mesh.merge(&box_mesh(500.0, 800.0, 30.0, 2200.0, 400.0, -15.0, tol));

    let raw_properties = compute_mass_properties(&mesh, density::CARBON_COMPOSITE);
    let derived = derive_properties(&mesh, density::CARBON_COMPOSITE);

    VehicleGeometry {
        name: "Stratospheric Glider".into(),
        mesh,
        raw_properties,
        derived,
        density_g_per_mm3: density::CARBON_COMPOSITE,
    }
    .with_target_mass(57.0)
}

// ---- AUV (torpedo body) ----

pub fn build_auv() -> VehicleGeometry {
    let tol = VehicleGeometry::TESSELLATION_TOLERANCE_MM;

    // Torpedo body: 2000 mm long, 200 mm diameter
    let mut mesh = box_mesh(2000.0, 200.0, 200.0, -1000.0, 0.0, -100.0, tol);

    // Four fins: 100×300×10 mm
    mesh.merge(&box_mesh(100.0, 300.0, 10.0, 800.0, 100.0, -5.0, tol));
    mesh.merge(&box_mesh(100.0, 300.0, 10.0, 800.0, -200.0, -5.0, tol));
    mesh.merge(&box_mesh(100.0, 10.0, 300.0, 800.0, -5.0, -150.0, tol));
    mesh.merge(&box_mesh(100.0, 10.0, 300.0, 800.0, -5.0, -50.0, tol));

    let raw_properties = compute_mass_properties(&mesh, density::ALUMINUM_6061);
    let derived = derive_properties(&mesh, density::ALUMINUM_6061);

    VehicleGeometry {
        name: "AUV Glider".into(),
        mesh,
        raw_properties,
        derived,
        density_g_per_mm3: density::ALUMINUM_6061,
    }
    .with_target_mass(33.2)
}

// ---- Station-Keeping Airship ----

pub fn build_airship() -> VehicleGeometry {
    let tol = 2.0; // coarser tolerance for this big vehicle

    // Envelope: 12 m long, 4 m diameter (box proxy for mass/area)
    let mut mesh = box_mesh(12_000.0, 4000.0, 4000.0, -6000.0, 0.0, -2000.0, tol);

    // Gondola: 2 m × 0.8 m × 1 m
    mesh.merge(&box_mesh(2000.0, 800.0, 1000.0, -1000.0, -1200.0, -500.0, tol));

    // Solar panel on top: 10 m × 3.5 m × 20 mm
    mesh.merge(&box_mesh(10_000.0, 20.0, 3500.0, -5000.0, 4000.0, -1750.0, tol));

    let raw_properties = compute_mass_properties(&mesh, density::PLASTIC);
    let derived = derive_properties(&mesh, density::PLASTIC);

    VehicleGeometry {
        name: "Station-Keeping Airship".into(),
        mesh,
        raw_properties,
        derived,
        density_g_per_mm3: density::PLASTIC,
    }
    .with_target_mass(144.0)
}

// ---- Cloud Carrier ----
//
// Autonomous stratospheric platform for the orbital economy.
// Operates at 30 km altitude, persistent, robotic (no human life support).
//
// REVISED SCALE (v2): 170 m diameter, 30 m tall lenticular.
// For simulation: model as 10-ton platform with 170m solar planform.
// This gives ~22,700 m² solar collection (massive) with buildable mass.

pub fn build_cloud_carrier() -> VehicleGeometry {
    let tol = 20.0; // coarse for very large vehicle

    // Main envelope: 170 m × 170 m × 30 m lenticular (box proxy)
    let mut mesh = box_mesh(170_000.0, 30_000.0, 170_000.0, -85_000.0, 0.0, -85_000.0, tol);

    // Upper solar skin: 160 m × 160 m × 50 mm thin-film PV
    mesh.merge(&box_mesh(160_000.0, 50.0, 160_000.0, -80_000.0, 30_000.0, -80_000.0, tol));

    // Keel structure: 120 m × 3 m × 10 m — backbone for payload bays
    mesh.merge(&box_mesh(120_000.0, 3000.0, 10_000.0, -60_000.0, -10_000.0, -5000.0, tol));

    // Cubesat fab bay: 15 m × 5 m × 5 m
    mesh.merge(&box_mesh(15_000.0, 5000.0, 5000.0, -40_000.0, -18_000.0, -2500.0, tol));

    // Repair bay: 12 m × 6 m × 8 m (robotic arms, satellite docking)
    mesh.merge(&box_mesh(12_000.0, 6000.0, 8000.0, -10_000.0, -18_000.0, -4000.0, tol));

    // Launch rail: 40 m × 1.5 m × 1.5 m
    mesh.merge(&box_mesh(40_000.0, 1500.0, 1500.0, 15_000.0, -14_000.0, -750.0, tol));

    // Station-keeping EDF nacelles: four at corners
    mesh.merge(&box_mesh(3000.0, 2000.0, 2000.0, 82_000.0, -5000.0, 82_000.0, tol));
    mesh.merge(&box_mesh(3000.0, 2000.0, 2000.0, -85_000.0, -5000.0, 82_000.0, tol));
    mesh.merge(&box_mesh(3000.0, 2000.0, 2000.0, 82_000.0, -5000.0, -85_000.0, tol));
    mesh.merge(&box_mesh(3000.0, 2000.0, 2000.0, -85_000.0, -5000.0, -85_000.0, tol));

    let raw_properties = compute_mass_properties(&mesh, density::PLASTIC);
    let derived = derive_properties(&mesh, density::PLASTIC);

    VehicleGeometry {
        name: "Cloud Carrier".into(),
        mesh,
        raw_properties,
        derived,
        density_g_per_mm3: density::PLASTIC,
    }
    .with_target_mass(10_000.0) // 10 tons: buildable at 170m with current gas barriers
}

// ---- Planetary Rover ----

pub fn build_rover() -> VehicleGeometry {
    let tol = VehicleGeometry::TESSELLATION_TOLERANCE_MM;

    // Chassis: 1000 × 350 × 600 mm
    let mut mesh = box_mesh(1000.0, 350.0, 600.0, -500.0, 150.0, -300.0, tol);

    // Solar panel on top: 900 × 20 × 550 mm
    mesh.merge(&box_mesh(900.0, 20.0, 550.0, -450.0, 500.0, -275.0, tol));

    // Four wheel axles (box proxy): 80 mm cube
    mesh.merge(&box_mesh(80.0, 80.0, 80.0, -350.0, 40.0, -370.0, tol));
    mesh.merge(&box_mesh(80.0, 80.0, 80.0, -350.0, 40.0, 290.0, tol));
    mesh.merge(&box_mesh(80.0, 80.0, 80.0, 270.0, 40.0, -370.0, tol));
    mesh.merge(&box_mesh(80.0, 80.0, 80.0, 270.0, 40.0, 290.0, tol));

    let raw_properties = compute_mass_properties(&mesh, density::ALUMINUM_6061);
    let derived = derive_properties(&mesh, density::ALUMINUM_6061);

    VehicleGeometry {
        name: "Planetary Rover".into(),
        mesh,
        raw_properties,
        derived,
        density_g_per_mm3: density::ALUMINUM_6061,
    }
    .with_target_mass(84.8)
}
