//! Radiative cooling film — PDMS/SiO₂ overlay that dumps cell heat to the
//! 3 K sky through the 8–13 µm atmospheric transparency window.
//!
//! Zhai et al., Science 2017. ε₈₋₁₃ > 0.93, R_solar > 0.97.
//! Same film is the IR-stealth layer: negative SWaP cost.

use crate::MmastModule;
use sim_core::time::SimClock;
use sim_environment::EnvConditions;
use sim_vehicle::VehicleParams;

/// Radiative cooling film on PV surface.
#[derive(Debug, Clone)]
pub struct RadiativeCooling {
    /// Cell temperature reduction (K) from the film.
    pub delta_t_k: f64,
    /// PV temperature coefficient (fractional gain per K of cooling).
    pub pv_temp_coeff: f64,
    /// Mass per unit area (kg/m²).
    pub mass_per_m2: f64,
}

impl Default for RadiativeCooling {
    fn default() -> Self {
        Self {
            delta_t_k: 6.5,     // 5–8 K range, use midpoint
            pv_temp_coeff: 0.004,
            mass_per_m2: 0.004, // 3–5 g/m²
        }
    }
}

impl MmastModule for RadiativeCooling {
    fn name(&self) -> &'static str {
        "Radiative cooling"
    }

    fn applicable(&self, vehicle: &VehicleParams, env: &EnvConditions) -> bool {
        // Needs sky view (atmosphere or space) and solar activity.
        vehicle.solar_area_m2 > 0.0
            && env.sky_temperature_k < env.temperature_k
            && env.solar_irradiance > 0.0
    }

    fn power_w(
        &self,
        vehicle: &VehicleParams,
        env: &EnvConditions,
        _clock: &SimClock,
        _battery_temp_k: f64,
    ) -> f64 {
        // PV gain from cooling: baseline_power × temp_coeff × delta_T.
        let baseline_eff = 0.22;
        let baseline_power = env.solar_irradiance * vehicle.solar_area_m2 * baseline_eff;
        baseline_power * self.pv_temp_coeff * self.delta_t_k
    }

    fn mass_kg(&self) -> f64 {
        0.0
    }

    fn description(&self) -> &'static str {
        "PDMS/SiO₂ film: dumps heat to cold sky, cools cells 5–8 K, doubles as IR stealth"
    }
}
