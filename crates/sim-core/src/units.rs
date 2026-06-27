//! Physical units — thin newtypes to prevent mixing watts with joules, etc.
//!
//! All internal math uses SI base units. These newtypes exist for API clarity
//! at boundaries (construction, reporting) — the solver works in raw f64.

use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Sub};

macro_rules! unit {
    ($Name:ident, $unit:expr) => {
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default, Serialize, Deserialize)]
        pub struct $Name(pub f64);

        impl $Name {
            pub const ZERO: Self = Self(0.0);

            #[inline]
            pub fn raw(self) -> f64 {
                self.0
            }
        }

        impl std::fmt::Display for $Name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{:.3} {}", self.0, $unit)
            }
        }

        impl Add for $Name {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }

        impl Sub for $Name {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }
    };
}

unit!(Watts, "W");
unit!(WattHours, "Wh");
unit!(Newtons, "N");
unit!(Kilograms, "kg");
unit!(MetersPerSecond, "m/s");
unit!(Meters, "m");
unit!(SquareMeters, "m²");
unit!(Kelvin, "K");
unit!(Celsius, "°C");
unit!(Pascals, "Pa");
unit!(KgPerCubicMeter, "kg/m³");

impl Celsius {
    pub fn to_kelvin(self) -> Kelvin {
        Kelvin(self.0 + 273.15)
    }
}

impl Kelvin {
    pub fn to_celsius(self) -> Celsius {
        Celsius(self.0 - 273.15)
    }
}

/// Power × time = energy.
impl Mul<f64> for Watts {
    type Output = WattHours;
    fn mul(self, hours: f64) -> WattHours {
        WattHours(self.0 * hours)
    }
}
