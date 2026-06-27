//! `sim-wasm` — WASM boundary for the MMAST simulator.
//!
//! Exposes the solver and reporter to JavaScript. The renderer will
//! either run in WASM (via wgpu's WebGPU backend) or delegate to a
//! JS three.js/WebGPU renderer that consumes the state stream as JSON.

use wasm_bindgen::prelude::*;

use sim_core::SimConfig;
use sim_dynamics::Solver;
use sim_environment::atmosphere::StandardAtmosphere;
use sim_vehicle::VehicleParams;

/// Run a HALE solar UAV simulation and return the energy summary as JSON.
#[wasm_bindgen]
pub fn run_hale_simulation(duration_hours: f64, cloud_cover: f64) -> String {
    let config = SimConfig {
        dt: 1.0,
        duration: duration_hours * 3600.0,
        ..Default::default()
    };
    let vehicle = VehicleParams::hale_solar_uav();
    let modules = sim_mmast::full_stack();
    let mut solver = Solver::new(config, vehicle, modules);
    solver.set_position([0.0, 500.0, 0.0]);

    let atm = StandardAtmosphere {
        cloud_cover,
        ..Default::default()
    }
    .with_lut(172);
    let series = solver.run(&atm, 60.0); // sample every 60s
    let summary = sim_report::summarize(&series);

    serde_json::to_string_pretty(&summary).unwrap_or_default()
}

/// Run a simulation and return the full time-series as JSON.
#[wasm_bindgen]
pub fn run_simulation_timeseries(
    vehicle_preset: &str,
    duration_hours: f64,
    cloud_cover: f64,
    latitude: f64,
    altitude_m: f64,
) -> String {
    let vehicle = match vehicle_preset {
        "hale" => VehicleParams::hale_solar_uav(),
        "quad" => VehicleParams::recon_quadcopter(),
        "strato" => VehicleParams::stratospheric_glider(),
        "auv" => VehicleParams::auv(),
        "airship" => VehicleParams::airship(),
        "rover" => VehicleParams::planetary_rover(),
        _ => VehicleParams::hale_solar_uav(),
    };

    let config = SimConfig {
        dt: 1.0,
        duration: duration_hours * 3600.0,
        latitude,
        ..Default::default()
    };

    let modules = sim_mmast::full_stack();
    let mut solver = Solver::new(config, vehicle, modules);
    solver.set_position([0.0, altitude_m, 0.0]);

    let atm = StandardAtmosphere {
        cloud_cover,
        ..Default::default()
    }
    .with_lut(172);
    let series = solver.run(&atm, 60.0);
    sim_report::to_json(&series)
}

/// List available vehicle presets as JSON.
#[wasm_bindgen]
pub fn list_vehicle_presets() -> String {
    serde_json::to_string(&[
        "hale", "quad", "strato", "auv", "airship", "rover",
    ])
    .unwrap_or_default()
}
