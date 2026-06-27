//! `mmast-sim` CLI — run simulations headlessly.
//!
//! ```bash
//! mmast-sim run --vehicle hale --duration 24 --latitude 35
//! mmast-sim sweep --vehicle hale --lat-min 0 --lat-max 60 --lat-bins 30 \
//!                 --cloud-min 0 --cloud-max 0.9 --cloud-bins 10
//! ```

use anyhow::Result;
use clap::{Parser, Subcommand};
use rayon::prelude::*;

use sim_core::SimConfig;
use sim_dynamics::Solver;
use sim_environment::atmosphere::StandardAtmosphere;
use sim_environment::atmosphere_lut::AtmosphereLut;
use sim_vehicle::VehicleParams;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "mmast-sim", about = "MMAST multi-vehicle physics simulator", allow_negative_numbers = true)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a single simulation and emit a report.
    Run(RunArgs),
    /// Monte Carlo sweep across a parameter space.
    Sweep(SweepArgs),
    /// Bake a mission envelope LUT (latitude × cloud × altitude × doy).
    Envelope(EnvelopeArgs),
    /// Bake a fleet atlas — envelopes for multiple vehicles in one HTML.
    Fleet(FleetArgs),
}

#[derive(Parser)]
struct FleetArgs {
    /// Comma-separated vehicle presets.
    #[arg(long, default_value = "hale,quad,strato,auv,airship,carrier,rover")]
    vehicles: String,

    #[arg(long, default_value_t = 0.0)]   lat_min: f64,
    #[arg(long, default_value_t = 60.0)]  lat_max: f64,
    #[arg(long, default_value_t = 25)]    lat_bins: usize,
    #[arg(long, default_value_t = 0.0)]   cloud_min: f64,
    #[arg(long, default_value_t = 0.9)]   cloud_max: f64,
    #[arg(long, default_value_t = 16)]    cloud_bins: usize,
    #[arg(long, default_value_t = 500.0)] alt_min: f64,
    #[arg(long, default_value_t = 15_000.0)] alt_max: f64,
    #[arg(long, default_value_t = 5)]     alt_bins: usize,
    #[arg(long, default_value_t = 1)]     doy_min: u16,
    #[arg(long, default_value_t = 365)]   doy_max: u16,
    #[arg(long, default_value_t = 12)]    doy_bins: usize,

    /// Output format: html, json.
    #[arg(long, default_value = "html")]
    output: String,
}

#[derive(Parser, Clone)]
struct RunArgs {
    /// Vehicle preset: hale, quad, strato, auv, airship, carrier, rover.
    #[arg(long, default_value = "hale")]
    vehicle: String,

    /// Simulation duration (hours).
    #[arg(long, default_value_t = 24.0)]
    duration: f64,

    /// Geographic latitude (degrees).
    #[arg(long, default_value_t = 35.0)]
    latitude: f64,

    /// Initial altitude (meters).
    #[arg(long, default_value_t = 500.0)]
    altitude: f64,

    /// Cloud cover fraction (0.0–1.0).
    #[arg(long, default_value_t = 0.0)]
    cloud_cover: f64,

    /// Output format: summary, timeseries, json.
    #[arg(long, default_value = "summary")]
    output: String,

    /// Sample interval for time-series output (seconds).
    #[arg(long, default_value_t = 60.0)]
    sample_interval: f64,
}

#[derive(Parser)]
struct EnvelopeArgs {
    #[arg(long, default_value = "hale")]
    vehicle: String,

    // ---- Latitude axis ----
    #[arg(long, default_value_t = 0.0)]
    lat_min: f64,
    #[arg(long, default_value_t = 60.0)]
    lat_max: f64,
    #[arg(long, default_value_t = 31)]
    lat_bins: usize,

    // ---- Cloud cover axis ----
    #[arg(long, default_value_t = 0.0)]
    cloud_min: f64,
    #[arg(long, default_value_t = 0.9)]
    cloud_max: f64,
    #[arg(long, default_value_t = 19)]
    cloud_bins: usize,

    // ---- Altitude axis ----
    #[arg(long, default_value_t = 500.0)]
    alt_min: f64,
    #[arg(long, default_value_t = 15_000.0)]
    alt_max: f64,
    #[arg(long, default_value_t = 6)]
    alt_bins: usize,

    // ---- Day-of-year axis ----
    #[arg(long, default_value_t = 1)]
    doy_min: u16,
    #[arg(long, default_value_t = 365)]
    doy_max: u16,
    #[arg(long, default_value_t = 12)]
    doy_bins: usize,

    /// Output format: csv, json, html, summary.
    #[arg(long, default_value = "summary")]
    output: String,
}

#[derive(Parser)]
struct SweepArgs {
    /// Vehicle preset.
    #[arg(long, default_value = "hale")]
    vehicle: String,

    /// Simulation duration per run (hours).
    #[arg(long, default_value_t = 24.0)]
    duration: f64,

    /// Day of year (1-366) for solar calculations.
    #[arg(long, default_value_t = 172)]
    day_of_year: u16,

    /// Initial altitude (meters).
    #[arg(long, default_value_t = 500.0)]
    altitude: f64,

    // ---- Latitude axis ----
    #[arg(long, default_value_t = 0.0)]
    lat_min: f64,
    #[arg(long, default_value_t = 60.0)]
    lat_max: f64,
    #[arg(long, default_value_t = 13)]
    lat_bins: usize,

    // ---- Cloud cover axis ----
    #[arg(long, default_value_t = 0.0)]
    cloud_min: f64,
    #[arg(long, default_value_t = 0.9)]
    cloud_max: f64,
    #[arg(long, default_value_t = 10)]
    cloud_bins: usize,

    /// Output format: csv, json.
    #[arg(long, default_value = "csv")]
    output: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    match cli.cmd {
        Command::Run(args) => run_single(args),
        Command::Sweep(args) => run_sweep(args),
        Command::Envelope(args) => run_envelope(args),
        Command::Fleet(args) => run_fleet(args),
    }
}

fn run_fleet(args: FleetArgs) -> Result<()> {
    let vehicle_names: Vec<&str> = args.vehicles.split(',').map(|s| s.trim()).collect();
    let vehicles: Vec<_> = vehicle_names.iter().map(|n| pick_vehicle(n)).collect();
    let total_cells = args.lat_bins * args.cloud_bins * args.alt_bins * args.doy_bins * vehicles.len();
    eprintln!(
        "Baking fleet atlas: {} vehicles × ({} × {} × {} × {}) = {} cells",
        vehicles.len(),
        args.lat_bins, args.cloud_bins, args.alt_bins, args.doy_bins,
        total_cells,
    );

    let t_start = std::time::Instant::now();
    let atlas = sim_report::fleet::FleetAtlas::bake(
        vehicles,
        args.lat_min, args.lat_max, args.lat_bins,
        args.cloud_min, args.cloud_max, args.cloud_bins,
        args.alt_min, args.alt_max, args.alt_bins,
        args.doy_min, args.doy_max, args.doy_bins,
        |name, i, n| {
            eprintln!("  [{:>2}/{:>2}] {}…", i + 1, n, name);
        },
    );
    let elapsed = t_start.elapsed();
    eprintln!(
        "Atlas baked in {:.2?} ({:.0} cells/sec)",
        elapsed,
        total_cells as f64 / elapsed.as_secs_f64(),
    );

    match args.output.as_str() {
        "html" => print!("{}", atlas.to_html()),
        "json" => println!("{}", atlas.to_json()),
        other => {
            eprintln!("Unknown output format: {other}. Using html.");
            print!("{}", atlas.to_html());
        }
    }
    Ok(())
}

fn run_envelope(args: EnvelopeArgs) -> Result<()> {
    let vehicle = pick_vehicle(&args.vehicle);
    let total = args.lat_bins * args.cloud_bins * args.alt_bins * args.doy_bins;
    eprintln!(
        "Baking 4D mission envelope for {}:\n  latitude  {:.0}–{:.0}° × {} bins\n  cloud     {:.0}–{:.0}% × {} bins\n  altitude  {:.0}–{:.0} m × {} bins\n  day range {}–{} × {} bins\n  = {} cells total",
        args.vehicle,
        args.lat_min, args.lat_max, args.lat_bins,
        args.cloud_min * 100.0, args.cloud_max * 100.0, args.cloud_bins,
        args.alt_min, args.alt_max, args.alt_bins,
        args.doy_min, args.doy_max, args.doy_bins,
        total,
    );

    let t_start = std::time::Instant::now();
    let env = sim_report::envelope::MissionEnvelope::bake(
        vehicle,
        args.lat_min, args.lat_max, args.lat_bins,
        args.cloud_min, args.cloud_max, args.cloud_bins,
        args.alt_min, args.alt_max, args.alt_bins,
        args.doy_min, args.doy_max, args.doy_bins,
    );
    let elapsed = t_start.elapsed();
    eprintln!(
        "Baked {} cells in {:.2?} ({:.0} cells/sec). Feasibility: {:.1}%",
        env.cells.len(),
        elapsed,
        env.cells.len() as f64 / elapsed.as_secs_f64(),
        env.feasibility_fraction() * 100.0,
    );

    match args.output.as_str() {
        "csv" => print!("{}", env.to_csv()),
        "json" => println!("{}", env.to_json()),
        "html" => print!("{}", env.to_html()),
        "summary" | _ => {
            println!("Mission envelope for {}", args.vehicle);
            println!("  Latitude:     {:.0}° to {:.0}° ({} bins)", args.lat_min, args.lat_max, args.lat_bins);
            println!("  Cloud cover:  {:.0}% to {:.0}% ({} bins)", args.cloud_min * 100.0, args.cloud_max * 100.0, args.cloud_bins);
            println!("  Altitude:     {:.0}m to {:.0}m ({} bins)", args.alt_min, args.alt_max, args.alt_bins);
            println!("  Day of year:  {} to {} ({} bins)", args.doy_min, args.doy_max, args.doy_bins);
            println!("  Total cells:  {}", env.cells.len());
            println!("  Feasible:     {:.1}%", env.feasibility_fraction() * 100.0);
        }
    }
    Ok(())
}

// ---- Single run ----

fn pick_vehicle(name: &str) -> VehicleParams {
    match name {
        "hale" => VehicleParams::hale_solar_uav(),
        "quad" => VehicleParams::recon_quadcopter(),
        "strato" => VehicleParams::stratospheric_glider(),
        "auv" => VehicleParams::auv(),
        "airship" => VehicleParams::airship(),
        "carrier" | "cloud_carrier" => VehicleParams::cloud_carrier(),
        "rover" => VehicleParams::planetary_rover(),
        other => {
            eprintln!("Unknown vehicle preset: {other}. Using 'hale'.");
            VehicleParams::hale_solar_uav()
        }
    }
}

fn run_single(args: RunArgs) -> Result<()> {
    let vehicle = pick_vehicle(&args.vehicle);
    let config = SimConfig {
        dt: 1.0,
        duration: args.duration * 3600.0,
        latitude: args.latitude,
        ..Default::default()
    };

    let modules = sim_mmast::full_stack();
    let mut solver = Solver::new(config, vehicle, modules);
    solver.set_position([0.0, args.altitude, 0.0]);

    let atm = StandardAtmosphere {
        cloud_cover: args.cloud_cover,
        latitude_deg: args.latitude,
        ..Default::default()
    }
    .with_lut(172);

    eprintln!(
        "Running {} for {:.0}h at lat {:.1}°, alt {:.0}m, cloud {:.0}%…",
        args.vehicle, args.duration, args.latitude, args.altitude, args.cloud_cover * 100.0,
    );

    let series = solver.run(&atm, args.sample_interval);

    match args.output.as_str() {
        "summary" => {
            let summary = sim_report::summarize(&series);
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        "timeseries" | "json" => {
            println!("{}", sim_report::to_json(&series));
        }
        other => {
            eprintln!("Unknown output format: {other}. Using 'summary'.");
            let summary = sim_report::summarize(&series);
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
    }

    Ok(())
}

// ---- Monte Carlo sweep ----

#[derive(Debug, Clone, serde::Serialize)]
struct SweepRow {
    latitude: f64,
    cloud_cover: f64,
    harvested_wh: f64,
    consumed_wh: f64,
    margin_wh: f64,
    feasible: bool,
}

fn run_sweep(args: SweepArgs) -> Result<()> {
    let total = args.lat_bins * args.cloud_bins;
    eprintln!(
        "Sweeping {} × {} = {} runs ({} cores available)…",
        args.lat_bins,
        args.cloud_bins,
        total,
        rayon::current_num_threads(),
    );

    let t_start = std::time::Instant::now();

    let lat_step = if args.lat_bins > 1 {
        (args.lat_max - args.lat_min) / (args.lat_bins - 1) as f64
    } else {
        0.0
    };
    let cloud_step = if args.cloud_bins > 1 {
        (args.cloud_max - args.cloud_min) / (args.cloud_bins - 1) as f64
    } else {
        0.0
    };

    // Build the cartesian product.
    let tasks: Vec<(f64, f64)> = (0..args.lat_bins)
        .flat_map(|li| {
            (0..args.cloud_bins).map(move |ci| {
                let latitude = args.lat_min + li as f64 * lat_step;
                let cloud = args.cloud_min + ci as f64 * cloud_step;
                (latitude, cloud)
            })
        })
        .collect();

    let vehicle_name = args.vehicle.clone();
    let altitude = args.altitude;
    let duration_s = args.duration * 3600.0;
    let day_of_year = args.day_of_year;

    // Bake the atmosphere LUT once and share it across all workers via Arc.
    // The LUT is latitude-independent at bake time (all latitudes in grid),
    // and cloud cover is applied at read time.
    let t_bake = std::time::Instant::now();
    let atmo_lut = Arc::new(AtmosphereLut::bake(day_of_year));
    eprintln!("Atmosphere LUT baked in {:.2?}", t_bake.elapsed());

    // Parallel execution via rayon. Each worker clones the Arc — cheap.
    let rows: Vec<SweepRow> = tasks
        .par_iter()
        .map(|&(latitude, cloud)| {
            let vehicle = pick_vehicle(&vehicle_name);
            let config = SimConfig {
                dt: 1.0,
                duration: duration_s,
                latitude,
                day_of_year,
                ..Default::default()
            };
            let modules = sim_mmast::full_stack();
            let mut solver = Solver::new(config, vehicle, modules);
            solver.set_position([0.0, altitude, 0.0]);

            let atm = StandardAtmosphere {
                cloud_cover: cloud,
                latitude_deg: latitude,
                ..Default::default()
            }
            .with_shared_lut(atmo_lut.clone());

            // Bake MMAST LUT for this run's vehicle×env combo.
            solver.bake_mmast_lut(&atm);

            // One sample at end — no intermediate snapshots.
            let series = solver.run(&atm, duration_s);
            let last = series.snapshots.last().expect("at least one snapshot");

            let harvested: f64 = last
                .modules
                .iter()
                .filter(|m| m.accumulated_wh > 0.0)
                .map(|m| m.accumulated_wh)
                .sum();
            let consumed: f64 = last
                .modules
                .iter()
                .filter(|m| m.accumulated_wh < 0.0)
                .map(|m| m.accumulated_wh.abs())
                .sum();
            let margin = harvested - consumed;

            SweepRow {
                latitude,
                cloud_cover: cloud,
                harvested_wh: harvested,
                consumed_wh: consumed,
                margin_wh: margin,
                feasible: margin > 0.0,
            }
        })
        .collect();

    let elapsed = t_start.elapsed();
    eprintln!(
        "Done: {} runs in {:.2?} ({:.1} runs/sec)",
        rows.len(),
        elapsed,
        rows.len() as f64 / elapsed.as_secs_f64(),
    );

    match args.output.as_str() {
        "csv" => {
            println!("latitude,cloud_cover,harvested_wh,consumed_wh,margin_wh,feasible");
            for r in &rows {
                println!(
                    "{:.2},{:.3},{:.1},{:.1},{:.1},{}",
                    r.latitude, r.cloud_cover, r.harvested_wh, r.consumed_wh, r.margin_wh, r.feasible
                );
            }
        }
        "json" => {
            println!("{}", serde_json::to_string_pretty(&rows)?);
        }
        other => {
            eprintln!("Unknown output format: {other}. Using csv.");
            println!("latitude,cloud_cover,harvested_wh,consumed_wh,margin_wh,feasible");
            for r in &rows {
                println!(
                    "{:.2},{:.3},{:.1},{:.1},{:.1},{}",
                    r.latitude, r.cloud_cover, r.harvested_wh, r.consumed_wh, r.margin_wh, r.feasible
                );
            }
        }
    }

    Ok(())
}
