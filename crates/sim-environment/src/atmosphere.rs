//! International Standard Atmosphere (ISA) model with wind and solar.
//!
//! By default, this uses a pre-baked AtmosphereLut (Primitive #41 Embedding
//! Lookup, L₀) to turn every `conditions()` call into an array read instead
//! of an ISA pressure/temperature/solar chain. The LUT is baked once at
//! construction time via `with_lut()`.

use crate::atmosphere_lut::AtmosphereLut;
use crate::{EnvConditions, Environment};
use sim_core::time::SimClock;
use std::sync::Arc;

/// ISA atmosphere with configurable wind and cloud models.
#[derive(Clone)]
pub struct StandardAtmosphere {
    /// Sea-level temperature (K). ISA = 288.15.
    pub t0: f64,
    /// Sea-level pressure (Pa). ISA = 101325.
    pub p0: f64,
    /// Temperature lapse rate (K/m). ISA = -0.0065 below tropopause.
    pub lapse_rate: f64,
    /// Tropopause altitude (m). ISA = 11000.
    pub tropopause_m: f64,
    /// Cloud cover fraction (0–1), constant for now.
    pub cloud_cover: f64,
    /// Steady wind vector (m/s) in world frame.
    pub wind: [f64; 3],
    /// Ambient RF density floor (W/m²).
    pub rf_density: f64,
    /// Latitude used when baking the LUT.
    pub latitude_deg: f64,
    /// Pre-baked atmosphere LUT. Shared (Arc) so clones don't reallocate.
    pub lut: Option<Arc<AtmosphereLut>>,
}

impl Default for StandardAtmosphere {
    fn default() -> Self {
        Self {
            t0: 288.15,
            p0: 101_325.0,
            lapse_rate: -0.0065,
            tropopause_m: 11_000.0,
            cloud_cover: 0.0,
            wind: [0.0, 0.0, 0.0],
            rf_density: 1e-6,
            latitude_deg: 35.0,
            lut: None,
        }
    }
}

impl StandardAtmosphere {
    /// Bake the LUT for a specific day-of-year and attach it.
    /// Subsequent `conditions()` calls will use the LUT instead of ISA math.
    pub fn with_lut(mut self, day_of_year: u16) -> Self {
        self.lut = Some(Arc::new(AtmosphereLut::bake(day_of_year)));
        self
    }

    /// Attach a pre-baked, externally-owned LUT. Cheap Arc clone — use
    /// this when parallelizing across many atmospheres to avoid one bake
    /// per worker.
    pub fn with_shared_lut(mut self, lut: Arc<AtmosphereLut>) -> Self {
        self.lut = Some(lut);
        self
    }
}

impl StandardAtmosphere {
    /// Temperature at altitude (K).
    pub fn temperature(&self, alt_m: f64) -> f64 {
        if alt_m < self.tropopause_m {
            self.t0 + self.lapse_rate * alt_m
        } else {
            // Isothermal above tropopause (ISA stratosphere simplification).
            self.t0 + self.lapse_rate * self.tropopause_m
        }
    }

    /// Pressure at altitude (Pa).
    pub fn pressure(&self, alt_m: f64) -> f64 {
        let g = 9.81;
        let m = 0.0289644; // molar mass of air (kg/mol)
        let r = 8.31447;   // gas constant

        if alt_m < self.tropopause_m {
            self.p0 * (1.0 + self.lapse_rate * alt_m / self.t0).powf(-g * m / (r * self.lapse_rate))
        } else {
            // Above tropopause: compute p_trop directly (no recursion) and
            // apply exponential decay using the isothermal stratosphere.
            let p_trop = self.p0
                * (1.0 + self.lapse_rate * self.tropopause_m / self.t0)
                    .powf(-g * m / (r * self.lapse_rate));
            let t_trop = self.t0 + self.lapse_rate * self.tropopause_m;
            p_trop * ((-g * m * (alt_m - self.tropopause_m)) / (r * t_trop)).exp()
        }
    }

    /// Air density at altitude (kg/m³).
    pub fn density(&self, alt_m: f64) -> f64 {
        let t = self.temperature(alt_m);
        let p = self.pressure(alt_m);
        p / (287.05 * t)
    }

    /// Solar irradiance at the top of the atmosphere, attenuated by air mass
    /// and cloud cover.
    pub fn solar_irradiance(&self, alt_m: f64, solar_elev_deg: f64) -> f64 {
        if solar_elev_deg <= 0.0 {
            return 0.0;
        }
        let toa = 1361.0; // W/m² solar constant
        let sin_elev = solar_elev_deg.to_radians().sin();
        // Simple air-mass model (Kasten & Young, 1989 approx).
        let am = 1.0 / (sin_elev + 0.50572 * (6.07995 + solar_elev_deg).powf(-1.6364));
        // Extinction — less extinction at altitude (pressure ratio).
        let p_ratio = self.pressure(alt_m) / self.p0;
        let tau = 0.7_f64.powf(am * p_ratio);
        let clear_sky = toa * tau * sin_elev;
        // Cloud attenuation.
        clear_sky * (1.0 - 0.75 * self.cloud_cover)
    }

    /// Effective sky temperature for radiative cooling (K).
    /// Drops with altitude as less water vapor overhead.
    pub fn sky_temperature(&self, alt_m: f64) -> f64 {
        // Simplified: ground-level ~250 K, decreasing with altitude.
        let base = 250.0;
        let alt_factor = (alt_m / 10_000.0).min(1.0);
        base - alt_factor * 100.0 // approaches ~150 K at 10 km
    }
}

impl Environment for StandardAtmosphere {
    fn conditions(&self, position: [f64; 3], clock: &SimClock) -> EnvConditions {
        let alt = position[1].max(0.0);

        // LUT path — Primitive #41 Embedding Lookup (L₀).
        // Single trilinear array read instead of pow/exp/sin/cos.
        if let Some(lut) = &self.lut {
            let entry = lut.sample(alt, clock.solar_hour, self.latitude_deg);
            let t = entry.temperature as f64;
            return EnvConditions {
                temperature_k: t,
                pressure_pa: entry.pressure as f64,
                density: entry.density as f64,
                // Viscosity and speed-of-sound are cheap enough to compute
                // from temperature (they're not in the LUT to keep it small).
                viscosity: 1.458e-6 * t.powf(1.5) / (t + 110.4),
                speed_of_sound: (1.4 * 287.05 * t).sqrt(),
                gravity: 9.81 * (6.371e6 / (6.371e6 + alt)).powi(2),
                // Cloud attenuation applied on top of clear-sky LUT value.
                solar_irradiance: entry.solar_irradiance as f64 * (1.0 - 0.75 * self.cloud_cover),
                sky_temperature_k: entry.sky_temperature as f64,
                wind_velocity: self.wind,
                ambient_rf_density: self.rf_density,
                cloud_cover: self.cloud_cover,
            };
        }

        // Fallback ISA math path (kept for tests and verification).
        let t = self.temperature(alt);
        let p = self.pressure(alt);
        let rho = self.density(alt);

        EnvConditions {
            temperature_k: t,
            pressure_pa: p,
            density: rho,
            viscosity: 1.458e-6 * t.powf(1.5) / (t + 110.4),
            speed_of_sound: (1.4 * 287.05 * t).sqrt(),
            gravity: 9.81 * (6.371e6 / (6.371e6 + alt)).powi(2),
            solar_irradiance: self.solar_irradiance(alt, clock.solar_elevation_deg),
            sky_temperature_k: self.sky_temperature(alt),
            wind_velocity: self.wind,
            ambient_rf_density: self.rf_density,
            cloud_cover: self.cloud_cover,
        }
    }

    fn density_at(&self, position: [f64; 3], clock: &SimClock) -> f64 {
        let alt = position[1].max(0.0);
        if let Some(lut) = &self.lut {
            lut.sample(alt, clock.solar_hour, self.latitude_deg).density as f64
        } else {
            self.density(alt)
        }
    }

    fn temperature_at(&self, position: [f64; 3], clock: &SimClock) -> f64 {
        let alt = position[1].max(0.0);
        if let Some(lut) = &self.lut {
            lut.sample(alt, clock.solar_hour, self.latitude_deg).temperature as f64
        } else {
            self.temperature(alt)
        }
    }

    fn ground_height(&self, _x: f64, _z: f64) -> Option<f64> {
        Some(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isa_sea_level() {
        let atm = StandardAtmosphere::default();
        assert!((atm.temperature(0.0) - 288.15).abs() < 0.01);
        assert!((atm.pressure(0.0) - 101_325.0).abs() < 1.0);
        assert!((atm.density(0.0) - 1.225).abs() < 0.01);
    }

    #[test]
    fn isa_tropopause() {
        let atm = StandardAtmosphere::default();
        let t = atm.temperature(11_000.0);
        assert!((t - 216.65).abs() < 0.5, "tropopause temp was {t}");
    }

    #[test]
    fn density_decreases_with_altitude() {
        let atm = StandardAtmosphere::default();
        assert!(atm.density(5000.0) < atm.density(0.0));
        assert!(atm.density(10000.0) < atm.density(5000.0));
    }

    #[test]
    fn lut_matches_isa_math() {
        // LUT path must agree with analytic ISA path within interpolation tolerance.
        let atm_math = StandardAtmosphere::default();
        let atm_lut = StandardAtmosphere::default().with_lut(172);
        let clock = SimClock::new(12.0, 35.0, 172);

        for alt in [0.0, 1000.0, 5000.0, 10_000.0] {
            let c_math = atm_math.conditions([0.0, alt, 0.0], &clock);
            let c_lut = atm_lut.conditions([0.0, alt, 0.0], &clock);
            let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(1e-6);
            assert!(rel(c_math.temperature_k, c_lut.temperature_k) < 0.01,
                    "temp mismatch at {alt}: {} vs {}", c_math.temperature_k, c_lut.temperature_k);
            assert!(rel(c_math.density, c_lut.density) < 0.02,
                    "density mismatch at {alt}: {} vs {}", c_math.density, c_lut.density);
            assert!(rel(c_math.pressure_pa, c_lut.pressure_pa) < 0.05,
                    "pressure mismatch at {alt}: {} vs {}", c_math.pressure_pa, c_lut.pressure_pa);
        }
    }
}
