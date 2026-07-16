//! CPU-only analytical checks for DWARF V16 strict-causal topology.
//!
//! The crate simulates deterministic candidate eligibility only. It deliberately does not
//! implement learned selector scores, PyTorch kernels, or quality prediction.

mod config;
mod report;
mod topology;
mod verification;

pub use config::{
    MAX_CHUNK_COUNT, MAX_SEQUENCE_LENGTH, MAX_SWEEP_VERIFICATION_WORK, MAX_VERIFICATION_WORK,
    OracleConfig, SweepGrid, SweepRequest,
};
pub use report::{MAX_SWEEP_CONFIGS, SweepItem, SweepReport, SweepResult, run_sweep};
pub use topology::{QueryTopology, TopologyError, simulate_query};
pub use verification::{VerificationReport, VerificationStatus, verify_config};
