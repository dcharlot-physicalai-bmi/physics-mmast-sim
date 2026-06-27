//! Benchmark: trait-dispatch MMAST vs LUT-baked MMAST.
//!
//! Runs a full 24h simulation (86,400 steps at dt=1s) both ways and
//! reports wall-clock time and per-step cost.
//!
//! Run with: `cargo bench -p sim-dynamics`

use sim_core::SimConfig;
use sim_dynamics::Solver;
use sim_environment::atmosphere::StandardAtmosphere;
use sim_vehicle::VehicleParams;
use std::time::Instant;

fn main() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}

fn run() {
    let config = SimConfig {
        dt: 1.0,
        duration: 86_400.0,
        ..Default::default()
    };
    let vehicle = VehicleParams::hale_solar_uav();
    let atm = StandardAtmosphere::default().with_lut(172);

    // ---- Trait dispatch path ----
    {
        let modules = sim_mmast::full_stack();
        let mut solver = Solver::new(config.clone(), vehicle.clone(), modules);
        solver.set_position([0.0, 500.0, 0.0]);

        let t0 = Instant::now();
        let series = solver.run(&atm, 86_400.0); // sample once at end
        let elapsed = t0.elapsed();
        let steps = (config.duration / config.dt) as u64;
        let ns_per_step = elapsed.as_nanos() as f64 / steps as f64;

        let last = series.snapshots.last().unwrap();
        // Margin from the per-module accumulators in the final snapshot
        // (track_energy_balance is off by default for hot-path perf).
        let harvested: f64 = last.modules.iter()
            .filter(|m| m.accumulated_wh > 0.0)
            .map(|m| m.accumulated_wh)
            .sum();
        let consumed: f64 = last.modules.iter()
            .filter(|m| m.accumulated_wh < 0.0)
            .map(|m| m.accumulated_wh.abs())
            .sum();
        let margin = harvested - consumed;
        println!("=== MMAST solver benchmark (86,400 steps) ===");
        println!();
        println!("Trait dispatch (B₄):");
        println!("  Total:        {:>8.2?}", elapsed);
        println!("  Per step:     {:>8.0} ns", ns_per_step);
        println!("  Net margin:   {:>+8.0} Wh (accumulated)", margin);
        println!("  Final SOC:    {:>8.0} Wh", last.battery_soc_wh);
    }

    // ---- LUT path ----
    {
        let modules = sim_mmast::full_stack();
        let mut solver = Solver::new(config.clone(), vehicle.clone(), modules);
        solver.set_position([0.0, 500.0, 0.0]);
        solver.bake_mmast_lut(&atm);

        let t0 = Instant::now();
        let series = solver.run(&atm, 86_400.0);
        let elapsed = t0.elapsed();
        let steps = (config.duration / config.dt) as u64;
        let ns_per_step = elapsed.as_nanos() as f64 / steps as f64;

        let last = series.snapshots.last().unwrap();
        // Margin from the per-module accumulators in the final snapshot
        // (track_energy_balance is off by default for hot-path perf).
        let harvested: f64 = last.modules.iter()
            .filter(|m| m.accumulated_wh > 0.0)
            .map(|m| m.accumulated_wh)
            .sum();
        let consumed: f64 = last.modules.iter()
            .filter(|m| m.accumulated_wh < 0.0)
            .map(|m| m.accumulated_wh.abs())
            .sum();
        let margin = harvested - consumed;
        println!();
        println!("LUT path (L₀):");
        println!("  Total:        {:>8.2?}", elapsed);
        println!("  Per step:     {:>8.0} ns", ns_per_step);
        println!("  Net margin:   {:>+8.0} Wh (accumulated)", margin);
        println!("  Final SOC:    {:>8.0} Wh", last.battery_soc_wh);
    }
}
