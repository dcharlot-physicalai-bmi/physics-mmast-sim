//! Simulation clock and solar-position utilities.

use serde::{Deserialize, Serialize};

/// Simulation clock state.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SimClock {
    /// Elapsed simulation time (seconds from mission start).
    pub elapsed: f64,
    /// Local solar time (hours, 0.0–24.0).
    pub solar_hour: f64,
    /// Solar elevation angle (degrees; negative = below horizon).
    pub solar_elevation_deg: f64,
    /// Solar azimuth (degrees, 0 = north, 90 = east).
    pub solar_azimuth_deg: f64,
}

impl SimClock {
    /// Advance the clock by `dt` seconds and recompute solar angles.
    pub fn advance(&mut self, dt: f64, latitude_deg: f64, day_of_year: u16) {
        self.elapsed += dt;
        self.solar_hour = (self.solar_hour + dt / 3600.0) % 24.0;
        let (elev, az) = solar_position(latitude_deg, day_of_year, self.solar_hour);
        self.solar_elevation_deg = elev;
        self.solar_azimuth_deg = az;
    }

    /// Create a clock at mission start.
    pub fn new(start_hour: f64, latitude_deg: f64, day_of_year: u16) -> Self {
        let (elev, az) = solar_position(latitude_deg, day_of_year, start_hour);
        Self {
            elapsed: 0.0,
            solar_hour: start_hour,
            solar_elevation_deg: elev,
            solar_azimuth_deg: az,
        }
    }
}

/// Simple solar position model (Spencer, 1971).
/// Returns (elevation_deg, azimuth_deg).
pub fn solar_position(latitude_deg: f64, day_of_year: u16, solar_hour: f64) -> (f64, f64) {
    let lat = latitude_deg.to_radians();
    let d = day_of_year as f64;

    // Solar declination (Cooper equation).
    let decl = (23.45_f64).to_radians() * ((360.0 * (284.0 + d) / 365.0).to_radians()).sin();

    // Hour angle: 0 at solar noon, 15 deg per hour.
    let hour_angle = ((solar_hour - 12.0) * 15.0).to_radians();

    // Elevation.
    let sin_elev = lat.sin() * decl.sin() + lat.cos() * decl.cos() * hour_angle.cos();
    let elev = sin_elev.asin();

    // Azimuth (measured from north, clockwise).
    let cos_az = (decl.sin() - lat.sin() * sin_elev) / (lat.cos() * elev.cos() + 1e-12);
    let mut az = cos_az.clamp(-1.0, 1.0).acos();
    if hour_angle > 0.0 {
        az = std::f64::consts::TAU - az;
    }

    (elev.to_degrees(), az.to_degrees())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solar_noon_summer_solstice() {
        let (elev, _) = solar_position(35.0, 172, 12.0);
        // At 35 N on summer solstice, solar noon elevation ≈ 78.45 deg.
        assert!((elev - 78.45).abs() < 1.0, "elevation was {elev}");
    }

    #[test]
    fn night_time() {
        let (elev, _) = solar_position(35.0, 172, 1.0);
        assert!(elev < 0.0, "should be below horizon at 1 AM, got {elev}");
    }
}
