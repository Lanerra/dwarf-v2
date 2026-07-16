use serde::{Deserialize, Serialize};

use crate::{
    MAX_SWEEP_VERIFICATION_WORK, OracleConfig, SweepGrid, SweepRequest, TopologyError,
    VerificationReport, verify_config,
};

pub const MAX_SWEEP_CONFIGS: usize = 64;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SweepReport {
    pub oracle_kind: String,
    pub requested_configs: usize,
    pub items: Vec<SweepItem>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SweepItem {
    pub index: usize,
    pub config: OracleConfig,
    pub result: SweepResult,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum SweepResult {
    Ok { report: VerificationReport },
    InvalidConfiguration { error: String },
}

/// Expand and verify a request, consuming its configurations to avoid a second input-sized copy.
pub fn run_sweep(request: SweepRequest) -> Result<SweepReport, TopologyError> {
    let configs = expand_sweep(request)?;
    preflight_sweep_resources(&configs)?;
    let items = configs
        .into_iter()
        .enumerate()
        .map(|(index, config)| {
            let result = match verify_config(&config) {
                Ok(report) => SweepResult::Ok { report },
                Err(error) => SweepResult::InvalidConfiguration {
                    error: error.to_string(),
                },
            };
            SweepItem {
                index,
                config,
                result,
            }
        })
        .collect::<Vec<_>>();

    Ok(SweepReport {
        oracle_kind: "topology_only".to_owned(),
        requested_configs: items.len(),
        items,
    })
}

fn preflight_sweep_resources(configs: &[OracleConfig]) -> Result<(), TopologyError> {
    let mut estimated = 0_usize;
    for config in configs {
        match config.validate() {
            Ok(()) => {
                let work = config.verification_work()?;
                estimated = estimated
                    .checked_add(work)
                    .ok_or(TopologyError::SweepVerificationWorkOverflow)?;
            }
            Err(error) if error.is_resource_limit() => return Err(error),
            // Keep ordinary topology-invalid grid/list items in the deterministic report.
            Err(_) => continue,
        }
    }
    if estimated > MAX_SWEEP_VERIFICATION_WORK {
        return Err(TopologyError::SweepVerificationWorkTooLarge {
            estimated,
            limit: MAX_SWEEP_VERIFICATION_WORK,
        });
    }
    Ok(())
}

fn expand_sweep(request: SweepRequest) -> Result<Vec<OracleConfig>, TopologyError> {
    match request {
        SweepRequest::Configs { configs } => bounded(configs),
        SweepRequest::Grid { grid } => expand_grid(&grid),
    }
}

fn bounded(configs: Vec<OracleConfig>) -> Result<Vec<OracleConfig>, TopologyError> {
    if configs.is_empty() {
        return Err(TopologyError::SweepEmpty);
    }
    if configs.len() > MAX_SWEEP_CONFIGS {
        return Err(TopologyError::SweepTooLarge {
            limit: MAX_SWEEP_CONFIGS,
            requested: configs.len(),
        });
    }
    Ok(configs)
}

fn expand_grid(grid: &SweepGrid) -> Result<Vec<OracleConfig>, TopologyError> {
    let dimensions = [
        grid.sequence_lengths.len(),
        grid.valid_lengths.len(),
        grid.chunk_counts.len(),
        grid.selector_tile_lengths.len(),
        grid.local_windows.len(),
        grid.top_k_chunks.len(),
        grid.top_m_tokens.len(),
    ];
    let count = dimensions
        .into_iter()
        .try_fold(1_usize, |total, dimension| {
            total
                .checked_mul(dimension)
                .ok_or(TopologyError::SweepCardinalityOverflow)
        })?;
    if count == 0 {
        return Err(TopologyError::SweepEmpty);
    }
    if count > MAX_SWEEP_CONFIGS {
        return Err(TopologyError::SweepTooLarge {
            limit: MAX_SWEEP_CONFIGS,
            requested: count,
        });
    }

    let mut configs = Vec::with_capacity(count);
    for &sequence_length in &grid.sequence_lengths {
        for &valid_length in &grid.valid_lengths {
            for &chunk_count in &grid.chunk_counts {
                for &selector_tile_length in &grid.selector_tile_lengths {
                    for &local_window in &grid.local_windows {
                        for &top_k_chunks in &grid.top_k_chunks {
                            for &top_m_tokens in &grid.top_m_tokens {
                                configs.push(OracleConfig {
                                    sequence_length,
                                    valid_length,
                                    chunk_count,
                                    selector_tile_length,
                                    local_window,
                                    top_k_chunks,
                                    top_m_tokens,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(configs)
}
