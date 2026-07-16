use std::{
    fs,
    io::{self, Read},
};

use clap::{Parser, Subcommand};
use dwarf_oracle::{OracleConfig, SweepRequest, run_sweep, verify_config};
use serde::Serialize;

/// Maximum raw JSON document size accepted by either CLI input transport.
///
/// One MiB is deliberately far above the bounded 64-configuration/grid interface while
/// preventing untrusted input from allocating or deserializing an unbounded document.
const MAX_JSON_INPUT_BYTES: usize = 1_048_576;

#[derive(Debug, Parser)]
#[command(about = "CPU-first strict-causal topology oracle for DWARF V16")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Verify one OracleConfig JSON document.
    Verify { config: String },
    /// Run a bounded config list or small Cartesian grid JSON document.
    Sweep { request: String },
}

fn main() {
    let exit_code = match Cli::parse().command {
        Command::Verify { config } => emit_result(
            read_json::<OracleConfig>(&config)
                .and_then(|input| verify_config(&input).map_err(|error| error.to_string())),
        ),
        Command::Sweep { request } => emit_result(
            read_json::<SweepRequest>(&request)
                .and_then(|input| run_sweep(input).map_err(|error| error.to_string())),
        ),
    };
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}

fn emit_result<T: Serialize>(result: Result<T, String>) -> i32 {
    match result {
        Ok(report) => {
            emit_json(&report, false);
            0
        }
        Err(error) => {
            emit_json(
                &JsonError {
                    status: "error",
                    error,
                },
                true,
            );
            2
        }
    }
}

#[derive(Serialize)]
struct JsonError<'a> {
    status: &'a str,
    error: String,
}

fn read_json<T: serde::de::DeserializeOwned>(input: &str) -> Result<T, String> {
    let content = if input == "-" {
        read_limited_json(io::stdin().lock())
            .map_err(|error| format!("failed to read JSON from stdin: {error}"))?
    } else {
        let file = fs::File::open(input)
            .map_err(|error| format!("failed to read JSON file {input:?}: {error}"))?;
        read_limited_json(file)
            .map_err(|error| format!("failed to read JSON file {input:?}: {error}"))?
    };
    if content.len() > MAX_JSON_INPUT_BYTES {
        return Err(format!(
            "JSON input exceeds maximum size of {MAX_JSON_INPUT_BYTES} bytes"
        ));
    }
    serde_json::from_slice(&content).map_err(|error| format!("invalid JSON input: {error}"))
}

fn read_limited_json(mut reader: impl Read) -> io::Result<Vec<u8>> {
    let mut content = Vec::new();
    reader
        .by_ref()
        .take((MAX_JSON_INPUT_BYTES + 1) as u64)
        .read_to_end(&mut content)?;
    Ok(content)
}

fn emit_json(value: &impl Serialize, stderr: bool) {
    let serialized = serde_json::to_string_pretty(value)
        .expect("all CLI report structures must be JSON serializable");
    if stderr {
        eprintln!("{serialized}");
    } else {
        println!("{serialized}");
    }
}
