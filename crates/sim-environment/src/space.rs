//! Space environment — vacuum, unattenuated solar, CMB sky temperature,
//! orbital mechanics gravity model.

use crate::{EnvConditions, Environment};
use sim_core::time::SimClock;

/// Space environment (LEO / cislunar / interplanetary).
#[derive(Debug, Clone)]
pub struct SpaceEnvironment {
    /// Orbital altitude above Earth's surface (m). Used for gravity calc.
    pub altitude_m: f64,
    /// Solar distance (AU). 1.0 for Earth orbit, 1.524 for Mars.
    pub solar_distance_au: f64,
    /// Whether the vehicle is in eclipse (planetary shadow).
    pub in_eclipse: bool,
}

impl Default for SpaceEnvironment {
    fn default() -> Self {
        Self {
            altitude_m: 400_000.0, // ISS altitude
            solar_distance_au: 1.0,
            in_eclipse: false,
        }
    }
}

impl Environment for SpaceEnvironment {
    fn conditions(&self, _position: [f64; 3], _clock: &SimClock) -> EnvConditions {
        let r_earth = 6.371e6;
        let r = r_earth + self.altitude_m;
        let g = 9.81 * (r_earth / r).powi(2);

        let solar = if self.in_eclipse {
            0.0
        } else {
            1361.0 / self.solar_distance_au.powi(2)
        };

        EnvConditions {
            temperature_k: 2.7, // CMB
            pressure_pa: 0.0,
            density: 0.0,
            viscosity: 0.0,
            speed_of_sound: 0.0,
            gravity: g,
            solar_irradiance: solar,
            sky_temperature_k: 2.7,
            wind_velocity: [0.0; 3],
            ambient_rf_density: 0.0,
            cloud_cover: 0.0,
        }
    }

    fn ground_height(&self, _x: f64, _z: f64) -> Option<f64> {
        None // no ground in space
    }
}
