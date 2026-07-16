use serde::{Deserialize, Serialize};

use crate::{OracleConfig, QueryTopology, TopologyError, simulate_query, topology::chunk_endpoint};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationStatus {
    Ok,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct VerificationReport {
    pub oracle_kind: String,
    pub status: VerificationStatus,
    pub config: OracleConfig,
    pub checked_queries: usize,
    pub queries_with_eligible_chunks: usize,
    pub queries_with_global_keys: usize,
}

pub fn verify_config(config: &OracleConfig) -> Result<VerificationReport, TopologyError> {
    config.validate()?;

    let mut queries_with_eligible_chunks = 0;
    let mut queries_with_global_keys = 0;
    for query in 0..config.valid_length {
        let topology = simulate_query(config, query)?;
        assert_contract(config, &topology)?;
        queries_with_eligible_chunks += usize::from(!topology.eligible_chunks.is_empty());
        queries_with_global_keys += usize::from(!topology.global_keys.is_empty());
    }

    Ok(VerificationReport {
        oracle_kind: "topology_only".to_owned(),
        status: VerificationStatus::Ok,
        config: config.clone(),
        checked_queries: config.valid_length,
        queries_with_eligible_chunks,
        queries_with_global_keys,
    })
}

fn assert_contract(config: &OracleConfig, topology: &QueryTopology) -> Result<(), TopologyError> {
    let local_start = topology.query.saturating_sub(config.local_window);
    let expected_local = (local_start..topology.query).collect::<Vec<_>>();
    if topology.local_keys != expected_local {
        return Err(TopologyError::ContractViolation(format!(
            "query {} local lane is not [{local_start}, {})",
            topology.query, topology.query
        )));
    }

    let chunk_length = config.chunk_length();
    for &chunk in &topology.eligible_chunks {
        if chunk >= config.chunk_count
            || chunk_endpoint(chunk, chunk_length)? > topology.selector_tile_start
        {
            return Err(TopologyError::ContractViolation(format!(
                "query {} includes a chunk not completed before selector tile start {}",
                topology.query, topology.selector_tile_start
            )));
        }
    }

    if topology
        .selected_chunks
        .iter()
        .any(|chunk| !topology.eligible_chunks.contains(chunk))
    {
        return Err(TopologyError::ContractViolation(format!(
            "query {} selected a non-eligible chunk",
            topology.query
        )));
    }

    if topology
        .global_keys
        .iter()
        .any(|key| *key >= local_start || *key >= topology.query || *key >= config.valid_length)
    {
        return Err(TopologyError::ContractViolation(format!(
            "query {} has a global key outside the strict causal range",
            topology.query
        )));
    }

    if topology
        .selected_keys
        .iter()
        .any(|key| *key >= topology.query || *key >= config.valid_length)
    {
        return Err(TopologyError::ContractViolation(format!(
            "query {} selected an invalid, padded, or future key",
            topology.query
        )));
    }

    if topology
        .local_keys
        .iter()
        .any(|key| topology.global_keys.contains(key))
    {
        return Err(TopologyError::ContractViolation(format!(
            "query {} overlaps local and global lanes",
            topology.query
        )));
    }

    Ok(())
}
