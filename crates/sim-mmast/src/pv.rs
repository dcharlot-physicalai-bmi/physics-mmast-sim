//! Photovoltaic baseline — the workhorse (~95% of harvest energy).

use crate::MmastModule;
use sim_core::time::SimClock;
use sim_environment::EnvConditions;
use sim_vehicle::VehicleParams;

/// Baseline PV module: flexible mono-Si cells on the solar-facing surface.
#[derive(Debug, Clone)]
pub struct PhotovoltaicModule {
    /// Cell efficiency at STC (25°C), typically 0.22 for mono-Si.
    pub efficiency: f64,
    /// Temperature coefficient (fractional loss per K above 25°C).
    pub temp_coeff: f64,
    /// Mass per unit area (kg/m²).
    pub mass_per_m2: f64,
}

impl Default for PhotovoltaicModule {
    fn default() -> Self {
        Self {
            efficiency: 0.22,
            temp_coeff: 0.004, // -0.4%/K
            mass_per_m2: 0.5,
        }
    }
}

impl MmastModule for PhotovoltaicModule {
    fn name(&self) -> &'static str {
        "PV baseline"
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
        // Cell temperature model: ambient + 20 K (simplified NOCT).
        let cell_temp_c = env.temperature_k - 273.15 + 20.0;
        let dt_above_stc = (cell_temp_c - 25.0).max(0.0);
        let eff = self.efficiency * (1.0 - self.temp_coeff * dt_above_stc);

        env.solar_irradiance * vehicle.solar_area_m2 * eff
    }

    fn mass_kg(&self) -> f64 {
        // Mass is per-vehicle, but we report the per-m² cost.
        // Actual mass = mass_per_m2 × vehicle.solar_area_m2, computed by caller.
        0.0 // already accounted for in vehicle dry mass
    }

    fn description(&self) -> &'static str {
        "Flexible monocrystalline Si cells at 22% STC efficiency"
    }
}
