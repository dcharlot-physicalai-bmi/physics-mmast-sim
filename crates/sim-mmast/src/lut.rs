//! MMAST Module LUT — Primitive #41 Embedding Lookup (L₀).
//!
//! Pre-computes the full MMAST module stack into a 4D table indexed by:
//!   (solar_hour_bin, altitude_bin, density_bin, battery_temp_bin)
//!   → [power_w; N_MODULES]
//!
//! At runtime, the solver reads one array entry instead of dispatching
//! 9 trait objects through vtable calls. This is the 1A→1B transition:
//! the modules were the "stochastic discovery" (designed, validated
//! against literature). The LUT is the "deterministic deployment."
//!
//! Compute stack mapping:
//!   Before: 9× Box<dyn MmastModule>::power_w() per step = B₄ trait dispatch, L₂
//!   After:  1 array read per step = L₀ embedding lookup
//!
//! The LUT bakes a specific (vehicle, environment config). If the vehicle
//! or environment changes, rebake.

use sim_core::time::SimClock;
use sim_environment::Environment;
use sim_vehicle::VehicleParams;

use crate::full_stack;

/// Number of bins per axis.
pub const HOUR_BINS: usize = 96;       // 24h in 15-min steps
pub const ALT_BINS: usize = 32;        // 0–20 km in 625 m steps
pub const BATT_TEMP_BINS: usize = 16;  // 220–320 K in 6.25 K steps

pub const MODULE_COUNT: usize = 9;

/// One LUT entry: per-module power (W) and applicability flags.
#[derive(Clone, Debug)]
pub struct MmastLutEntry {
    /// Power per module (W). Positive = harvest, negative = cost.
    pub power: [f64; MODULE_COUNT],
    /// Whether each module is active at this point.
    pub active: [bool; MODULE_COUNT],
}

impl Default for MmastLutEntry {
    fn default() -> Self {
        Self {
            power: [0.0; MODULE_COUNT],
            active: [false; MODULE_COUNT],
        }
    }
}

/// The pre-computed MMAST module LUT.
pub struct MmastLut {
    pub data: Vec<MmastLutEntry>,
    /// Module names — `&'static str` because every MMAST module's name
    /// is a compile-time string literal. No allocation on lookup.
    pub module_names: Vec<&'static str>,
}

impl MmastLut {
    /// Bake the LUT from the *actual* Environment the solver will use.
    /// This eliminates any divergence between bake-time and runtime.
    pub fn bake(
        vehicle: &VehicleParams,
        env: &dyn Environment,
        latitude: f64,
        day_of_year: u16,
    ) -> Self {
        let total = HOUR_BINS * ALT_BINS * BATT_TEMP_BINS;
        let mut data = vec![MmastLutEntry::default(); total];

        let modules = full_stack();
        let module_names: Vec<&'static str> = modules.iter().map(|m| m.name()).collect();

        for hi in 0..HOUR_BINS {
            let hour = hi as f64 * 0.25;
            let clock = SimClock::new(hour, latitude, day_of_year);

            for ai in 0..ALT_BINS {
                let alt = ai as f64 * 625.0;
                // Query the real environment — same path the solver uses.
                let conditions = env.conditions([0.0, alt, 0.0], &clock);

                for bi in 0..BATT_TEMP_BINS {
                    let batt_temp = 220.0 + bi as f64 * 6.25;
                    let idx = (hi * ALT_BINS + ai) * BATT_TEMP_BINS + bi;

                    let mut entry = MmastLutEntry::default();
                    for (j, m) in modules.iter().enumerate() {
                        let active = m.applicable(vehicle, &conditions);
                        let power = if active {
                            m.power_w(vehicle, &conditions, &clock, batt_temp)
                        } else {
                            0.0
                        };
                        entry.power[j] = power;
                        entry.active[j] = active;
                    }
                    data[idx] = entry;
                }
            }
        }

        Self { data, module_names }
    }

    /// Sample the LUT by (hour, altitude, battery_temp). Nearest-neighbor.
    pub fn sample(
        &self,
        solar_hour: f64,
        altitude_m: f64,
        battery_temp_k: f64,
    ) -> &MmastLutEntry {
        let hi = ((solar_hour.rem_euclid(24.0) * 4.0) as usize).min(HOUR_BINS - 1);
        let ai = ((altitude_m / 625.0) as usize).min(ALT_BINS - 1);
        let bi = (((battery_temp_k - 220.0) / 6.25).max(0.0) as usize).min(BATT_TEMP_BINS - 1);

        let idx = (hi * ALT_BINS + ai) * BATT_TEMP_BINS + bi;
        &self.data[idx]
    }

    /// Total entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Size in bytes (approximate).
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<MmastLutEntry>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_environment::atmosphere::StandardAtmosphere;
    use sim_vehicle::VehicleParams;

    #[test]
    fn bake_and_sample() {
        let vehicle = VehicleParams::hale_solar_uav();
        let env = StandardAtmosphere::default().with_lut(172);
        let lut = MmastLut::bake(&vehicle, &env, 35.0, 172);

        assert_eq!(lut.len(), HOUR_BINS * ALT_BINS * BATT_TEMP_BINS);
        assert_eq!(lut.module_names.len(), MODULE_COUNT);

        // Noon, 500m, warm battery — PV should be active and positive.
        let entry = lut.sample(12.0, 500.0, 293.0);
        assert!(entry.active[0], "PV should be active at noon");
        assert!(entry.power[0] > 100.0, "PV power at noon should be >100 W, got {}", entry.power[0]);

        // Night — PV should be zero.
        let night = lut.sample(2.0, 500.0, 293.0);
        assert!(!night.active[0] || night.power[0] < 1.0, "PV should be off at night");
    }

    #[test]
    fn lut_fits_in_memory() {
        let vehicle = VehicleParams::hale_solar_uav();
        let env = StandardAtmosphere::default().with_lut(172);
        let lut = MmastLut::bake(&vehicle, &env, 35.0, 172);
        let mb = lut.size_bytes() as f64 / (1024.0 * 1024.0);
        assert!(mb < 200.0, "LUT is {mb:.1} MB, should be <200 MB");
    }
}
