//! Ocean environment model — depth-dependent temperature, pressure, density,
//! current profiles, and solar attenuation through water column.

use crate::{EnvConditions, Environment};
use sim_core::time::SimClock;

/// Simple ocean model with thermocline and current.
#[derive(Debug, Clone)]
pub struct OceanEnvironment {
    /// Surface temperature (K).
    pub surface_temp_k: f64,
    /// Deep-water temperature (K), below thermocline.
    pub deep_temp_k: f64,
    /// Thermocline depth (m, positive downward).
    pub thermocline_depth_m: f64,
    /// Surface current velocity (m/s) in world frame.
    pub surface_current: [f64; 3],
    /// Diffuse attenuation coefficient for solar (1/m). ~0.05 for clear ocean.
    pub solar_kd: f64,
}

impl Default for OceanEnvironment {
    fn default() -> Self {
        Self {
            surface_temp_k: 293.0,
            deep_temp_k: 277.0,
            thermocline_depth_m: 200.0,
            surface_current: [0.2, 0.0, 0.0],
            solar_kd: 0.05,
        }
    }
}

impl Environment for OceanEnvironment {
    fn conditions(&self, position: [f64; 3], clock: &SimClock) -> EnvConditions {
        // Depth = negative y (position[1] < 0 means submerged).
        let depth = (-position[1]).max(0.0);

        // Temperature profile (simple tanh thermocline).
        let blend = ((depth - self.thermocline_depth_m) / 50.0).tanh() * 0.5 + 0.5;
        let temp = self.surface_temp_k * (1.0 - blend) + self.deep_temp_k * blend;

        // Pressure: 101325 Pa + ρgh.
        let rho = 1025.0; // seawater
        let pressure = 101_325.0 + rho * 9.81 * depth;

        // Solar attenuation through water column (Beer-Lambert).
        let surface_irradiance = if clock.solar_elevation_deg > 0.0 {
            1361.0 * 0.5 * clock.solar_elevation_deg.to_radians().sin() // rough
        } else {
            0.0
        };
        let solar = surface_irradiance * (-self.solar_kd * depth).exp();

        // Current decays with depth.
        let current_decay = (-depth / 100.0).exp();
        let current = [
            self.surface_current[0] * current_decay,
            0.0,
            self.surface_current[2] * current_decay,
        ];

        EnvConditions {
            temperature_k: temp,
            pressure_pa: pressure,
            density: rho,
            viscosity: 1.08e-3, // seawater at ~20°C
            speed_of_sound: 1500.0,
            gravity: 9.81,
            solar_irradiance: solar,
            sky_temperature_k: temp, // no radiative cooling underwater
            wind_velocity: current,
            ambient_rf_density: 0.0, // RF doesn't propagate underwater
            cloud_cover: 0.0,
        }
    }

    fn ground_height(&self, _x: f64, _z: f64) -> Option<f64> {
        Some(-4000.0) // flat seabed at 4 km depth
    }
}
