use dwarf_oracle::{
    OracleConfig, SweepGrid, SweepRequest, TopologyError, VerificationStatus, run_sweep,
    simulate_query, verify_config,
};

fn v16_config() -> OracleConfig {
    OracleConfig {
        sequence_length: 128,
        valid_length: 128,
        chunk_count: 4,
        selector_tile_length: 16,
        local_window: 64,
        top_k_chunks: 4,
        top_m_tokens: 64,
    }
}

#[test]
fn q_zero_has_no_selected_keys() {
    let topology = simulate_query(&v16_config(), 0).expect("q=0 is valid");

    assert!(topology.local_keys.is_empty());
    assert!(topology.global_keys.is_empty());
    assert!(topology.selected_keys.is_empty());
}

#[test]
fn partial_valid_length_excludes_padding_from_every_selected_lane() {
    let mut config = v16_config();
    config.valid_length = 100;

    let report = verify_config(&config).expect("partial valid sequence is valid");
    assert_eq!(report.status, VerificationStatus::Ok);
    assert_eq!(report.checked_queries, 100);

    let topology = simulate_query(&config, 99).expect("last valid query is valid");
    assert!(
        topology
            .selected_keys
            .iter()
            .all(|&key| key < 99 && key < 100)
    );
}

#[test]
fn first_completed_chunk_becomes_eligible_at_its_first_tile_boundary() {
    let topology = simulate_query(&v16_config(), 32).expect("query is valid");

    assert_eq!(topology.selector_tile_start, 32);
    assert_eq!(topology.eligible_chunks, vec![0]);
    assert_eq!(topology.selected_chunks, vec![0]);
}

#[test]
fn selector_tile_boundary_makes_only_prior_completed_chunks_eligible() {
    let before = simulate_query(&v16_config(), 95).expect("query is valid");
    let at_boundary = simulate_query(&v16_config(), 96).expect("query is valid");

    assert_eq!(before.selector_tile_start, 80);
    assert_eq!(before.eligible_chunks, vec![0, 1]);
    assert_eq!(at_boundary.selector_tile_start, 96);
    assert_eq!(at_boundary.eligible_chunks, vec![0, 1, 2]);
}

#[test]
fn global_keys_cannot_overlap_the_local_lane() {
    let topology = simulate_query(&v16_config(), 127).expect("query is valid");
    let local_start = 127 - 64;

    assert!(
        topology
            .local_keys
            .iter()
            .all(|&key| (local_start..127).contains(&key))
    );
    assert!(topology.global_keys.iter().all(|&key| key < local_start));
    assert!(
        topology
            .global_keys
            .iter()
            .all(|key| !topology.local_keys.contains(key))
    );
}

#[test]
fn invalid_config_is_rejected_with_a_specific_error() {
    let mut config = v16_config();
    config.valid_length = 129;

    let error = verify_config(&config).expect_err("invalid length must fail");
    assert!(matches!(
        error,
        TopologyError::ValidLengthExceedsSequence { .. }
    ));
}

#[test]
fn query_at_or_after_valid_length_is_rejected() {
    let config = v16_config();

    for query in [config.valid_length, config.valid_length + 1] {
        let error = simulate_query(&config, query).expect_err("out-of-range query must fail");
        assert!(matches!(
            error,
            TopologyError::QueryOutsideValidLength { .. }
        ));
    }
}

#[test]
fn zero_local_window_leaves_only_ordered_global_keys() {
    let config = OracleConfig {
        sequence_length: 16,
        valid_length: 16,
        chunk_count: 4,
        selector_tile_length: 4,
        local_window: 0,
        top_k_chunks: 2,
        top_m_tokens: 5,
    };

    let topology = simulate_query(&config, 12).expect("query is valid");

    assert!(topology.local_keys.is_empty());
    assert_eq!(topology.selected_chunks, vec![2, 1]);
    assert_eq!(topology.global_keys, vec![11, 10, 9, 8, 7]);
    assert_eq!(topology.selected_keys, topology.global_keys);
}

#[test]
fn top_m_zero_truncates_all_global_keys() {
    let config = OracleConfig {
        sequence_length: 16,
        valid_length: 16,
        chunk_count: 4,
        selector_tile_length: 4,
        local_window: 0,
        top_k_chunks: 2,
        top_m_tokens: 0,
    };

    let topology = simulate_query(&config, 12).expect("query is valid");

    assert_eq!(topology.selected_chunks, vec![2, 1]);
    assert!(topology.global_keys.is_empty());
    assert!(topology.selected_keys.is_empty());
}

#[test]
fn grid_expansion_preserves_declared_dimension_order() {
    let request = SweepRequest::Grid {
        grid: SweepGrid {
            sequence_lengths: vec![8, 16],
            valid_lengths: vec![8],
            chunk_counts: vec![1],
            selector_tile_lengths: vec![1],
            local_windows: vec![0],
            top_k_chunks: vec![0],
            top_m_tokens: vec![0],
        },
    };

    let report = run_sweep(request).expect("small grid is valid");

    assert_eq!(report.requested_configs, 2);
    assert_eq!(
        report
            .items
            .iter()
            .map(|item| item.config.sequence_length)
            .collect::<Vec<_>>(),
        vec![8, 16]
    );
}

#[test]
fn sweep_accepts_64_items_and_rejects_65() {
    let config = OracleConfig {
        sequence_length: 8,
        valid_length: 8,
        chunk_count: 1,
        selector_tile_length: 1,
        local_window: 0,
        top_k_chunks: 0,
        top_m_tokens: 0,
    };

    let accepted = SweepRequest::Configs {
        configs: vec![config.clone(); 64],
    };
    assert_eq!(
        run_sweep(accepted)
            .expect("64 configurations must be accepted")
            .requested_configs,
        64
    );

    let rejected = SweepRequest::Configs {
        configs: vec![config; 65],
    };
    assert!(matches!(
        run_sweep(rejected),
        Err(TopologyError::SweepTooLarge { .. })
    ));
}

#[test]
fn oversized_sequence_is_rejected_before_topology_expansion() {
    let config = OracleConfig {
        sequence_length: 8_193,
        valid_length: 8_193,
        chunk_count: 1,
        selector_tile_length: 1,
        local_window: 0,
        top_k_chunks: 0,
        top_m_tokens: 0,
    };

    assert!(config.validate().is_err());
}

#[test]
fn excessive_analytical_work_is_rejected_before_topology_expansion() {
    let config = OracleConfig {
        sequence_length: 8_192,
        valid_length: 8_192,
        chunk_count: 512,
        selector_tile_length: 1,
        local_window: 128,
        top_k_chunks: 4,
        top_m_tokens: 128,
    };

    assert!(config.validate().is_err());
}

#[test]
fn sweep_at_the_aggregate_work_limit_succeeds() {
    let max_work_config = OracleConfig {
        sequence_length: 5_000,
        valid_length: 5_000,
        chunk_count: 5,
        selector_tile_length: 1,
        local_window: 32,
        top_k_chunks: 5,
        top_m_tokens: 33,
    };
    let request = SweepRequest::Configs {
        configs: vec![max_work_config; 4],
    };

    let report = run_sweep(request).expect("four one-million-work configurations fit the limit");
    assert_eq!(report.requested_configs, 4);
    assert!(
        report
            .items
            .iter()
            .all(|item| matches!(item.result, dwarf_oracle::SweepResult::Ok { .. }))
    );
}

#[test]
fn five_production_v16_like_configs_exceed_aggregate_work_limit() {
    let production_v16_like = OracleConfig {
        sequence_length: 2_048,
        valid_length: 2_048,
        chunk_count: 32,
        selector_tile_length: 16,
        local_window: 64,
        top_k_chunks: 4,
        top_m_tokens: 64,
    };
    let request = SweepRequest::Configs {
        configs: vec![production_v16_like; 5],
    };

    let error = run_sweep(request).expect_err("five V16 configurations exceed four million work");
    assert!(matches!(
        error,
        TopologyError::SweepVerificationWorkTooLarge {
            estimated: 4_259_840,
            limit: 4_000_000,
        }
    ));
}

#[test]
fn production_v16_n2048_c32_remains_within_cpu_limits() {
    let config = OracleConfig {
        sequence_length: 2048,
        valid_length: 2048,
        chunk_count: 32,
        selector_tile_length: 16,
        local_window: 64,
        top_k_chunks: 4,
        top_m_tokens: 64,
    };

    let report = verify_config(&config).expect("production V16 point must be supported");
    assert_eq!(report.checked_queries, 2048);
}
