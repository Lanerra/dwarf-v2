use std::{fs, process::Command};

use dwarf_oracle::{OracleConfig, SweepRequest, VerificationStatus};

fn v16_config() -> OracleConfig {
    OracleConfig {
        sequence_length: 128,
        valid_length: 100,
        chunk_count: 4,
        selector_tile_length: 16,
        local_window: 64,
        top_k_chunks: 4,
        top_m_tokens: 64,
    }
}

fn write_json(name: &str, value: &impl serde::Serialize) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "dwarf-oracle-{name}-{}-{}.json",
        std::process::id(),
        name
    ));
    fs::write(
        &path,
        serde_json::to_vec(value).expect("serializable fixture"),
    )
    .expect("fixture is written");
    path
}

fn write_raw_json(name: &str, value: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "dwarf-oracle-{name}-{}-{}.json",
        std::process::id(),
        name
    ));
    fs::write(&path, value).expect("fixture is written");
    path
}

fn assert_structured_input_error(output: &std::process::Output) {
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let error: serde_json::Value = serde_json::from_slice(&output.stderr).expect("JSON error");
    assert_eq!(error["status"], "error");
    assert!(error["error"].is_string());
}

#[test]
fn verify_command_emits_a_structured_topology_report() {
    let path = write_json("verify", &v16_config());
    let output = Command::new(env!("CARGO_BIN_EXE_dwarf-oracle"))
        .args(["verify", path.to_str().expect("utf-8 path")])
        .output()
        .expect("CLI starts");
    let _ = fs::remove_file(path);

    assert!(output.status.success());
    let report: dwarf_oracle::VerificationReport =
        serde_json::from_slice(&output.stdout).expect("JSON report");
    assert_eq!(report.status, VerificationStatus::Ok);
    assert_eq!(report.oracle_kind, "topology_only");
    assert_eq!(report.checked_queries, 100);
}

#[test]
fn sweep_command_is_deterministic() {
    let request = SweepRequest::Configs {
        configs: vec![v16_config(), v16_config()],
    };
    let path = write_json("sweep", &request);
    let command = || {
        Command::new(env!("CARGO_BIN_EXE_dwarf-oracle"))
            .args(["sweep", path.to_str().expect("utf-8 path")])
            .output()
            .expect("CLI starts")
    };

    let first = command();
    let second = command();
    let _ = fs::remove_file(path);

    assert!(first.status.success());
    assert_eq!(first.stdout, second.stdout);
}

#[test]
fn verify_rejects_json_larger_than_the_documented_one_mib_input_limit() {
    // The limit is deliberately tested through the CLI so the guard remains before parsing.
    let path = write_raw_json("oversized-input", &" ".repeat(1_048_577));
    let output = Command::new(env!("CARGO_BIN_EXE_dwarf-oracle"))
        .args(["verify", path.to_str().expect("utf-8 path")])
        .output()
        .expect("CLI starts");
    let _ = fs::remove_file(path);

    assert_structured_input_error(&output);
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("JSON input exceeds maximum size of 1048576 bytes")
    );
}

#[test]
fn verify_rejects_unknown_config_fields_as_structured_json() {
    let path = write_raw_json(
        "unknown-config-field",
        r#"{
          "sequence_length": 128,
          "valid_length": 100,
          "chunk_count": 4,
          "selector_tile_length": 16,
          "local_window": 64,
          "top_k_chunks": 4,
          "top_m_tokens": 64,
          "unexpected": true
        }"#,
    );
    let output = Command::new(env!("CARGO_BIN_EXE_dwarf-oracle"))
        .args(["verify", path.to_str().expect("utf-8 path")])
        .output()
        .expect("CLI starts");
    let _ = fs::remove_file(path);

    assert_structured_input_error(&output);
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown field"));
}

#[test]
fn sweep_rejects_unknown_grid_fields_as_structured_json() {
    let path = write_raw_json(
        "unknown-grid-field",
        r#"{
          "grid": {
            "sequence_lengths": [8],
            "valid_lengths": [8],
            "chunk_counts": [1],
            "selector_tile_lengths": [1],
            "local_windows": [0],
            "top_k_chunks": [0],
            "top_m_tokens": [0],
            "unexpected": true
          }
        }"#,
    );
    let output = Command::new(env!("CARGO_BIN_EXE_dwarf-oracle"))
        .args(["sweep", path.to_str().expect("utf-8 path")])
        .output()
        .expect("CLI starts");
    let _ = fs::remove_file(path);

    assert_structured_input_error(&output);
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown field"));
}

#[test]
fn sweep_rejects_ambiguous_configs_and_grid_envelope_as_structured_json() {
    let path = write_raw_json(
        "ambiguous-sweep-envelope",
        r#"{
          "configs": [],
          "grid": {
            "sequence_lengths": [8],
            "valid_lengths": [8],
            "chunk_counts": [1],
            "selector_tile_lengths": [1],
            "local_windows": [0],
            "top_k_chunks": [0],
            "top_m_tokens": [0]
          }
        }"#,
    );
    let output = Command::new(env!("CARGO_BIN_EXE_dwarf-oracle"))
        .args(["sweep", path.to_str().expect("utf-8 path")])
        .output()
        .expect("CLI starts");
    let _ = fs::remove_file(path);

    assert_structured_input_error(&output);
    assert!(String::from_utf8_lossy(&output.stderr).contains("exactly one"));
}
