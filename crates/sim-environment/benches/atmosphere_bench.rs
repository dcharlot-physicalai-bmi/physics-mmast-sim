//! Benchmark: ISA math path vs atmosphere LUT path.
//!
//! Runs N environment queries at varying (altitude, hour) and measures
//! wall-clock time per query. Prints a clean speedup ratio.

use sim_core::time::SimClock;
use sim_environment::atmosphere::StandardAtmosphere;
use sim_environment::Environment;
use std::time::Instant;

const N: usize = 1_000_000;

fn main() {
    // Run on a thread with a larger stack to avoid macOS's default 8 MB limit
    // interacting badly with the LUT + 16 MB query buffer.
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}

fn run() {
    let atm_math = StandardAtmosphere::default();
    let atm_lut = StandardAtmosphere::default().with_lut(172);

    // Heap-allocated queries to avoid stack overflow on large N.
    let queries: Box<[(f64, f64)]> = (0..N)
        .map(|i| {
            let alt = (i as f64 * 37.0) % 20_000.0;
            let hour = (i as f64 * 0.013) % 24.0;
            (alt, hour)
        })
        .collect();

    let latitude = 35.0_f64;

    // Pre-compute clocks so the benchmark measures only the atmosphere model,
    // not SimClock::new's solar_position trig cost.
    let clocks: Box<[SimClock]> = queries
        .iter()
        .map(|&(_alt, hour)| SimClock::new(hour, latitude, 172))
        .collect();

    // Warm-up.
    let mut warm = 0.0f64;
    for (i, &(alt, _)) in queries.iter().take(10_000).enumerate() {
        let c = atm_math.conditions([0.0, alt, 0.0], &clocks[i]);
        warm += c.temperature_k;
    }
    std::hint::black_box(warm);

    // ---- Scenario 1: pre-computed clocks (isolates atmosphere model) ----
    let t0 = Instant::now();
    let mut sum_math = 0.0_f64;
    for (i, &(alt, _)) in queries.iter().enumerate() {
        let c = atm_math.conditions([0.0, alt, 0.0], &clocks[i]);
        sum_math += c.temperature_k + c.pressure_pa + c.density + c.solar_irradiance;
    }
    let t_math = t0.elapsed();
    std::hint::black_box(sum_math);

    let t0 = Instant::now();
    let mut sum_lut = 0.0_f64;
    for (i, &(alt, _)) in queries.iter().enumerate() {
        let c = atm_lut.conditions([0.0, alt, 0.0], &clocks[i]);
        sum_lut += c.temperature_k + c.pressure_pa + c.density + c.solar_irradiance;
    }
    let t_lut = t0.elapsed();
    std::hint::black_box(sum_lut);

    let ns_math_iso = t_math.as_nanos() as f64 / N as f64;
    let ns_lut_iso = t_lut.as_nanos() as f64 / N as f64;

    // ---- Scenario 2: full step (clock + atmosphere) ----
    let t0 = Instant::now();
    let mut sum_math2 = 0.0_f64;
    for &(alt, hour) in queries.iter() {
        let clock = SimClock::new(hour, latitude, 172);
        let c = atm_math.conditions([0.0, alt, 0.0], &clock);
        sum_math2 += c.temperature_k + c.pressure_pa + c.density + c.solar_irradiance;
    }
    let t_math2 = t0.elapsed();
    std::hint::black_box(sum_math2);

    let t0 = Instant::now();
    let mut sum_lut2 = 0.0_f64;
    for &(alt, hour) in queries.iter() {
        let clock = SimClock::new(hour, latitude, 172);
        let c = atm_lut.conditions([0.0, alt, 0.0], &clock);
        sum_lut2 += c.temperature_k + c.pressure_pa + c.density + c.solar_irradiance;
    }
    let t_lut2 = t0.elapsed();
    std::hint::black_box(sum_lut2);

    let ns_math_full = t_math2.as_nanos() as f64 / N as f64;
    let ns_lut_full = t_lut2.as_nanos() as f64 / N as f64;

    println!("=== Atmosphere query benchmark ({N} queries) ===");
    println!();
    println!("Scenario 1: atmosphere model only (clocks pre-computed)");
    println!("  ISA math:     {:>9.1} ns/query", ns_math_iso);
    println!("  LUT:          {:>9.1} ns/query", ns_lut_iso);
    println!("  Speedup:      {:>9.2}x", ns_math_iso / ns_lut_iso);
    println!();
    println!("Scenario 2: full step (clock + atmosphere)");
    println!("  ISA math:     {:>9.1} ns/query", ns_math_full);
    println!("  LUT:          {:>9.1} ns/query", ns_lut_full);
    println!("  Speedup:      {:>9.2}x", ns_math_full / ns_lut_full);
}
