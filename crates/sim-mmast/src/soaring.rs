//! Dynamic and thermal soaring — CVAE/RL-driven atmospheric energy harvest.
//!
//! Notter et al., Nature Communications 15, 4942 (2024).
//! Park et al., AIAA SciTech 2025.

use crate::MmastModule;
use sim_core::time::SimClock;
use sim_environment::EnvConditions;
use sim_vehicle::VehicleParams;

/// Atmospheric energy harvesting via thermal and dynamic soaring.
#[derive(Debug, Clone)]
pub struct DynamicSoaring {
    /// Fraction of cruise power recoverable via thermals (0–1).
    pub thermal_recovery_fraction: f64,
    /// Fraction of cruise power recoverable via wind shear (0–1).
    pub shear_recovery_fraction: f64,
    /// Minimum wind speed for dynamic soaring to contribute (m/s).
    pub min_wind_speed: f64,
    /// Hours per day with active thermals (convective conditions).
    pub thermal_hours_per_day: f64,
}

impl Default for DynamicSoaring {
    fn default() -> Self {
        Self {
            thermal_recovery_fraction: 0.35,
            shear_recovery_fraction: 0.15,
            min_wind_speed: 3.0,
            thermal_hours_per_day: 6.0,
        }
    }
}

impl MmastModule for DynamicSoaring {
    fn name(&self) -> &'static str {
        "Soaring (CVAE-RL)"
    }

    fn applicable(&self, vehicle: &VehicleParams, env: &EnvConditions) -> bool {
        // Only works in atmosphere, with sufficient L/D for soaring.
        vehicle.medium == sim_core::Medium::Atmosphere
            && vehicle.mobility.ld_max() > 10.0
            && env.density > 0.01
    }

    fn power_w(
        &self,
        vehicle: &VehicleParams,
        env: &EnvConditions,
        clock: &SimClock,
        _battery_temp_k: f64,
    ) -> f64 {
        let cruise_mech = vehicle.mobility.cruise_power_mechanical(
            vehicle.total_mass_kg(),
            env.density,
        );

        // Thermal soaring: only active during convective hours (roughly 10:00–16:00).
        let thermal_active = clock.solar_hour > 9.0 && clock.solar_hour < 17.0;
        let thermal_power = if thermal_active {
            cruise_mech * self.thermal_recovery_fraction
        } else {
            0.0
        };

        // Dynamic soaring: proportional to wind speed above threshold.
        let wind_speed = (env.wind_velocity[0].powi(2)
            + env.wind_velocity[1].powi(2)
            + env.wind_velocity[2].powi(2))
        .sqrt();
        let shear_power = if wind_speed > self.min_wind_speed {
            let scale = ((wind_speed - self.min_wind_speed) / 5.0).min(1.0);
            cruise_mech * self.shear_recovery_fraction * scale
        } else {
            0.0
        };

        thermal_power + shear_power
    }

    fn mass_kg(&self) -> f64 {
        0.0 // firmware feature, no hardware
    }

    fn description(&self) -> &'static str {
        "CVAE-RL soaring policy harvests thermals and wind shear; 20–50% cruise recovery"
    }
}
