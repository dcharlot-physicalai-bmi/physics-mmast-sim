//! Metasurface ARC + light trapping — CVAE-designed dielectric meta-atoms
//! that push photon dwell time toward the Yablonovitch 4n² limit.
//!
//! Wei et al., Optics Express 33(1), 2025: +7.14% PCE on c-Si.
//! Ovcharenko et al., APN 4(3), 2025: <2% reflection 500–1200 nm.

use crate::MmastModule;
use sim_core::time::SimClock;
use sim_environment::EnvConditions;
use sim_vehicle::VehicleParams;

/// Metasurface antireflection + light-trapping overlay on PV cells.
#[derive(Debug, Clone)]
pub struct MetasurfaceArc {
    /// Relative PV efficiency gain (fraction, e.g. 0.06 = +6% relative).
    pub relative_gain: f64,
    /// Mass per unit area (kg/m²). ~3–5 g/m² per doc 1 §4.2.
    pub mass_per_m2: f64,
}

impl Default for MetasurfaceArc {
    fn default() -> Self {
        Self {
            relative_gain: 0.06,
            mass_per_m2: 0.004,
        }
    }
}

impl MmastModule for MetasurfaceArc {
    fn name(&self) -> &'static str {
        "Metasurface ARC"
    }

    fn applicable(&self, vehicle: &VehicleParams, env: &EnvConditions) -> bool {
        vehicle.solar_area_m2 > 0.0 && env.solar_irradiance > 0.0
    }

    fn power_w(
        &self,
        vehicle: &VehicleParams,
        env: &EnvConditions,
        _clock: &SimClock,
        _battery_temp_k: f64,
    ) -> f64 {
        // Additional harvest = baseline PV power × relative gain.
        let baseline_eff = 0.22;
        let baseline_power = env.solar_irradiance * vehicle.solar_area_m2 * baseline_eff;
        baseline_power * self.relative_gain
    }

    fn mass_kg(&self) -> f64 {
        0.0 // negligible, already in vehicle mass budget
    }

    fn description(&self) -> &'static str {
        "Sub-wavelength dielectric meta-atoms; <2% reflection, +6% relative PV gain"
    }
}
