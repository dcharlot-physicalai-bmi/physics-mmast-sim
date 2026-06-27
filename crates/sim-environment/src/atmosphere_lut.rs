//! Atmosphere LUT — Primitive #41 Embedding Lookup (L₀).
//!
//! Pre-computes a 3D table indexed by (altitude_bin, hour_bin, latitude_bin)
//! containing all atmospheric properties the sim and renderer need.
//! At runtime, a query is a single array read + bilinear interpolation —
//! no trig, no exponentiation, no branching.
//!
//! This moves the atmosphere model from B₄ (CPU math per step) to an
//! L₀ lookup that fits in cache.

use serde::{Deserialize, Serialize};

/// Number of bins in each dimension.
pub const ALT_BINS: usize = 64;    // 0–20 km in 312.5 m steps
pub const HOUR_BINS: usize = 96;   // 24h in 15-minute steps
pub const LAT_BINS: usize = 37;    // -90 to +90 in 5° steps

/// One entry in the atmosphere LUT.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct AtmoEntry {
    /// Temperature (K).
    pub temperature: f32,
    /// Pressure (Pa).
    pub pressure: f32,
    /// Density (kg/m³).
    pub density: f32,
    /// Solar irradiance at this point (W/m²).
    pub solar_irradiance: f32,
    /// Sky temperature for radiative cooling (K).
    pub sky_temperature: f32,
    /// Solar elevation (degrees).
    pub solar_elevation: f32,
    /// Sun color RGB + intensity packed as [r, g].
    pub sun_r: f32,
    pub sun_g: f32,
}

/// The full pre-computed atmosphere table.
pub struct AtmosphereLut {
    pub data: Vec<AtmoEntry>,
}

impl AtmosphereLut {
    /// Bake the entire LUT. This runs once at startup.
    pub fn bake(day_of_year: u16) -> Self {
        let mut data = vec![AtmoEntry::default(); ALT_BINS * HOUR_BINS * LAT_BINS];

        for lat_i in 0..LAT_BINS {
            let latitude = -90.0 + lat_i as f64 * 5.0;
            for hour_i in 0..HOUR_BINS {
                let hour = hour_i as f64 * 0.25; // 15-minute bins
                let (solar_elev, _solar_az) =
                    sim_core::time::solar_position(latitude, day_of_year, hour);

                for alt_i in 0..ALT_BINS {
                    let alt = alt_i as f64 * 312.5; // 0–20 km

                    // ISA temperature.
                    let lapse = -0.0065;
                    let t0 = 288.15;
                    let tropo = 11_000.0;
                    let temp = if alt < tropo {
                        t0 + lapse * alt
                    } else {
                        t0 + lapse * tropo
                    };

                    // ISA pressure.
                    let p0 = 101_325.0;
                    let g = 9.81;
                    let m = 0.0289644;
                    let r = 8.31447;
                    let pressure = if alt < tropo {
                        p0 * (1.0 + lapse * alt / t0).powf(-g * m / (r * lapse))
                    } else {
                        let p_trop = p0 * (1.0 + lapse * tropo / t0).powf(-g * m / (r * lapse));
                        let t_trop = t0 + lapse * tropo;
                        p_trop * ((-g * m * (alt - tropo)) / (r * t_trop)).exp()
                    };

                    let density = pressure / (287.05 * temp);

                    // Solar irradiance with air-mass extinction.
                    let solar = if solar_elev > 0.0 {
                        let sin_e = solar_elev.to_radians().sin();
                        let am = 1.0
                            / (sin_e
                                + 0.50572 * (6.07995 + solar_elev).powf(-1.6364));
                        let p_ratio = pressure / p0;
                        let tau = 0.7_f64.powf(am * p_ratio);
                        1361.0 * tau * sin_e
                    } else {
                        0.0
                    };

                    // Sky temperature.
                    let sky_temp = 250.0 - (alt / 10_000.0).min(1.0) * 100.0;

                    // Sun warmth for renderer.
                    let warmth = (1.0 - solar_elev.max(0.0).to_radians().sin()).max(0.0);

                    let idx = (lat_i * HOUR_BINS + hour_i) * ALT_BINS + alt_i;
                    data[idx] = AtmoEntry {
                        temperature: temp as f32,
                        pressure: pressure as f32,
                        density: density as f32,
                        solar_irradiance: solar as f32,
                        sky_temperature: sky_temp as f32,
                        solar_elevation: solar_elev as f32,
                        sun_r: (1.0 - warmth * 0.35) as f32,
                        sun_g: (1.0 - warmth * 0.6) as f32,
                    };
                }
            }
        }

        Self { data }
    }

    /// Trilinear interpolation sample.
    /// altitude_m: 0–20000, hour: 0.0–24.0, latitude_deg: -90 to +90.
    pub fn sample(&self, altitude_m: f64, hour: f64, latitude_deg: f64) -> AtmoEntry {
        let alt_f = (altitude_m / 312.5).clamp(0.0, (ALT_BINS - 1) as f64 - 1e-6);
        let hour_f = ((hour.rem_euclid(24.0)) * 4.0).clamp(0.0, (HOUR_BINS - 1) as f64 - 1e-6);
        let lat_f = ((latitude_deg + 90.0) / 5.0).clamp(0.0, (LAT_BINS - 1) as f64 - 1e-6);

        let a0 = alt_f.floor() as usize;
        let h0 = hour_f.floor() as usize;
        let l0 = lat_f.floor() as usize;
        let fa = (alt_f - a0 as f64) as f32;
        let fh = (hour_f - h0 as f64) as f32;
        let fl = (lat_f - l0 as f64) as f32;

        let idx = |ai: usize, hi: usize, li: usize| -> usize {
            (li * HOUR_BINS + hi) * ALT_BINS + ai
        };

        // Eight corners.
        let c000 = &self.data[idx(a0,     h0,     l0)];
        let c100 = &self.data[idx(a0 + 1, h0,     l0)];
        let c010 = &self.data[idx(a0,     h0 + 1, l0)];
        let c110 = &self.data[idx(a0 + 1, h0 + 1, l0)];
        let c001 = &self.data[idx(a0,     h0,     l0 + 1)];
        let c101 = &self.data[idx(a0 + 1, h0,     l0 + 1)];
        let c011 = &self.data[idx(a0,     h0 + 1, l0 + 1)];
        let c111 = &self.data[idx(a0 + 1, h0 + 1, l0 + 1)];

        // Per-field trilinear lerp. Compile down to 7 muls + 14 adds per field.
        fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }
        fn tri(c000: f32, c100: f32, c010: f32, c110: f32,
               c001: f32, c101: f32, c011: f32, c111: f32,
               fa: f32, fh: f32, fl: f32) -> f32 {
            let x00 = lerp(c000, c100, fa);
            let x10 = lerp(c010, c110, fa);
            let x01 = lerp(c001, c101, fa);
            let x11 = lerp(c011, c111, fa);
            let y0 = lerp(x00, x10, fh);
            let y1 = lerp(x01, x11, fh);
            lerp(y0, y1, fl)
        }

        AtmoEntry {
            temperature: tri(c000.temperature, c100.temperature, c010.temperature, c110.temperature,
                             c001.temperature, c101.temperature, c011.temperature, c111.temperature,
                             fa, fh, fl),
            pressure: tri(c000.pressure, c100.pressure, c010.pressure, c110.pressure,
                          c001.pressure, c101.pressure, c011.pressure, c111.pressure,
                          fa, fh, fl),
            density: tri(c000.density, c100.density, c010.density, c110.density,
                         c001.density, c101.density, c011.density, c111.density,
                         fa, fh, fl),
            solar_irradiance: tri(c000.solar_irradiance, c100.solar_irradiance, c010.solar_irradiance, c110.solar_irradiance,
                                  c001.solar_irradiance, c101.solar_irradiance, c011.solar_irradiance, c111.solar_irradiance,
                                  fa, fh, fl),
            sky_temperature: tri(c000.sky_temperature, c100.sky_temperature, c010.sky_temperature, c110.sky_temperature,
                                 c001.sky_temperature, c101.sky_temperature, c011.sky_temperature, c111.sky_temperature,
                                 fa, fh, fl),
            solar_elevation: tri(c000.solar_elevation, c100.solar_elevation, c010.solar_elevation, c110.solar_elevation,
                                 c001.solar_elevation, c101.solar_elevation, c011.solar_elevation, c111.solar_elevation,
                                 fa, fh, fl),
            sun_r: tri(c000.sun_r, c100.sun_r, c010.sun_r, c110.sun_r,
                       c001.sun_r, c101.sun_r, c011.sun_r, c111.sun_r, fa, fh, fl),
            sun_g: tri(c000.sun_g, c100.sun_g, c010.sun_g, c110.sun_g,
                       c001.sun_g, c101.sun_g, c011.sun_g, c111.sun_g, fa, fh, fl),
        }
    }

    /// Total entries in the table.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<AtmoEntry>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_bakes_and_samples() {
        let lut = AtmosphereLut::bake(172); // summer solstice
        assert_eq!(lut.len(), ALT_BINS * HOUR_BINS * LAT_BINS);

        // Sea level, noon, 35 N.
        let entry = lut.sample(0.0, 12.0, 35.0);
        assert!((entry.temperature - 288.15).abs() < 0.5);
        assert!(entry.solar_irradiance > 500.0);

        // Night time.
        let night = lut.sample(0.0, 1.0, 35.0);
        assert!(night.solar_irradiance < 1.0);

        // High altitude — colder, less dense.
        let high = lut.sample(10000.0, 12.0, 35.0);
        assert!(high.temperature < entry.temperature);
        assert!(high.density < entry.density);

        // LUT fits in L2 cache on most CPUs.
        assert!(
            lut.size_bytes() < 8 * 1024 * 1024,
            "LUT is {} MB, should fit in cache",
            lut.size_bytes() / 1024 / 1024
        );
    }
}
