//! Ground / planetary surface environment — rolling resistance, terrain slope,
//! reduced solar flux (Mars), dust.

use crate::{EnvConditions, Environment};
use sim_core::time::SimClock;

/// Planetary surface parameters.
#[derive(Debug, Clone)]
pub struct SurfaceEnvironment {
    /// Surface gravity (m/s²). Earth = 9.81, Mars = 3.72.
    pub gravity: f64,
    /// Surface pressure (Pa). Earth SL = 101325, Mars = 636.
    pub surface_pressure: f64,
    /// Surface temperature (K). Nominal for the site.
    pub surface_temp_k: f64,
    /// Atmospheric density (kg/m³). Earth SL = 1.225, Mars = 0.020.
    pub atm_density: f64,
    /// Solar constant at this body's orbit (W/m²).
    pub solar_constant: f64,
    /// Dust optical depth (0 = clear, 1+ = dusty). Primarily for Mars.
    pub dust_tau: f64,
    /// Wind velocity (m/s).
    pub wind: [f64; 3],
}

impl SurfaceEnvironment {
    pub fn earth() -> Self {
        Self {
            gravity: 9.81,
            surface_pressure: 101_325.0,
            surface_temp_k: 293.0,
            atm_density: 1.225,
            solar_constant: 1361.0,
            dust_tau: 0.0,
            wind: [0.0; 3],
        }
    }

    pub fn mars() -> Self {
        Self {
            gravity: 3.72,
            surface_pressure: 636.0,
            surface_temp_k: 210.0,
            atm_density: 0.020,
            solar_constant: 589.0, // 1361 / 1.524²
            dust_tau: 0.5,
            wind: [0.0; 3],
        }
    }
}

impl Environment for SurfaceEnvironment {
    fn conditions(&self, _position: [f64; 3], clock: &SimClock) -> EnvConditions {
        let solar = if clock.solar_elevation_deg > 0.0 {
            let sin_e = clock.solar_elevation_deg.to_radians().sin();
            self.solar_constant * sin_e * (-self.dust_tau / sin_e).exp()
        } else {
            0.0
        };

        EnvConditions {
            temperature_k: self.surface_temp_k,
            pressure_pa: self.surface_pressure,
            density: self.atm_density,
            viscosity: 1.5e-5,
            speed_of_sound: (1.4 * 287.05 * self.surface_temp_k).sqrt(),
            gravity: self.gravity,
            solar_irradiance: solar,
            sky_temperature_k: self.surface_temp_k - 30.0,
            wind_velocity: self.wind,
            ambient_rf_density: 0.0,
            cloud_cover: 0.0,
        }
    }

    fn ground_height(&self, _x: f64, _z: f64) -> Option<f64> {
        Some(0.0)
    }
}
