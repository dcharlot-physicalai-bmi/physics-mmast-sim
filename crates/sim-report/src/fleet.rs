//! FleetAtlas — a collection of MissionEnvelopes for multiple vehicles,
//! serialized as a single self-contained HTML page with tabs, module
//! toggles, view modes, and marginal plots.

use serde::{Deserialize, Serialize};

use crate::envelope::MissionEnvelope;
use sim_vehicle::VehicleParams;

/// A bundle of mission envelopes across a fleet of vehicles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetAtlas {
    pub envelopes: Vec<MissionEnvelope>,
}

impl FleetAtlas {
    /// Bake envelopes for a list of vehicles sequentially.
    #[allow(clippy::too_many_arguments)]
    pub fn bake(
        vehicles: Vec<VehicleParams>,
        lat_min: f64,
        lat_max: f64,
        lat_bins: usize,
        cloud_min: f64,
        cloud_max: f64,
        cloud_bins: usize,
        alt_min: f64,
        alt_max: f64,
        alt_bins: usize,
        doy_min: u16,
        doy_max: u16,
        doy_bins: usize,
        progress: impl Fn(&str, usize, usize),
    ) -> Self {
        let n = vehicles.len();
        let mut envelopes = Vec::with_capacity(n);
        for (i, vehicle) in vehicles.into_iter().enumerate() {
            progress(&vehicle.name, i, n);
            let env = MissionEnvelope::bake(
                vehicle,
                lat_min, lat_max, lat_bins,
                cloud_min, cloud_max, cloud_bins,
                alt_min, alt_max, alt_bins,
                doy_min, doy_max, doy_bins,
            );
            envelopes.push(env);
        }
        Self { envelopes }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    /// Export the full atlas as a self-contained interactive HTML page.
    pub fn to_html(&self) -> String {
        let json = serde_json::to_string(self).unwrap_or_default();
        let template = include_str!("fleet_template.html");
        template.replace("__ATLAS_JSON__", &json)
    }
}
