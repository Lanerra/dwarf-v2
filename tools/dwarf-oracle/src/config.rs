use serde::{Deserialize, Deserializer, Serialize, de};

use crate::topology::TopologyError;

/// Maximum sequence (and therefore valid) length accepted by this CPU-only oracle.
///
/// This keeps JSON input from creating unbounded query/key vectors during analytical
/// verification. It intentionally includes the V16 production reference length of 2048.
pub const MAX_SEQUENCE_LENGTH: usize = 8_192;
/// Maximum number of uniform chunks accepted by this CPU-only oracle.
pub const MAX_CHUNK_COUNT: usize = 512;
/// Maximum conservative work estimate for one configuration verification.
pub const MAX_VERIFICATION_WORK: usize = 1_000_000;
/// Maximum combined conservative work estimate for one bounded sweep.
pub const MAX_SWEEP_VERIFICATION_WORK: usize = 4_000_000;

/// Discrete V16 topology parameters. This is an analytical configuration, not a model checkpoint.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OracleConfig {
    pub sequence_length: usize,
    pub valid_length: usize,
    pub chunk_count: usize,
    pub selector_tile_length: usize,
    pub local_window: usize,
    pub top_k_chunks: usize,
    pub top_m_tokens: usize,
}

impl OracleConfig {
    pub fn validate(&self) -> Result<(), TopologyError> {
        if self.sequence_length == 0 {
            return Err(TopologyError::SequenceLengthZero);
        }
        if self.sequence_length > MAX_SEQUENCE_LENGTH {
            return Err(TopologyError::SequenceLengthExceedsLimit {
                sequence_length: self.sequence_length,
                limit: MAX_SEQUENCE_LENGTH,
            });
        }
        if self.valid_length > self.sequence_length {
            return Err(TopologyError::ValidLengthExceedsSequence {
                valid_length: self.valid_length,
                sequence_length: self.sequence_length,
            });
        }
        if self.chunk_count == 0 {
            return Err(TopologyError::ChunkCountZero);
        }
        if self.chunk_count > MAX_CHUNK_COUNT {
            return Err(TopologyError::ChunkCountExceedsLimit {
                chunk_count: self.chunk_count,
                limit: MAX_CHUNK_COUNT,
            });
        }
        if !self.sequence_length.is_multiple_of(self.chunk_count) {
            return Err(TopologyError::UnevenChunks {
                sequence_length: self.sequence_length,
                chunk_count: self.chunk_count,
            });
        }
        if self.selector_tile_length == 0 {
            return Err(TopologyError::SelectorTileLengthZero);
        }
        if self.local_window > self.sequence_length {
            return Err(TopologyError::LocalWindowExceedsSequence {
                local_window: self.local_window,
                sequence_length: self.sequence_length,
            });
        }

        let estimated = self.verification_work()?;
        if estimated > MAX_VERIFICATION_WORK {
            return Err(TopologyError::VerificationWorkTooLarge {
                estimated,
                limit: MAX_VERIFICATION_WORK,
            });
        }
        Ok(())
    }

    pub(crate) fn chunk_length(&self) -> usize {
        self.sequence_length / self.chunk_count
    }

    /// Conservative bound for vector construction and eligibility scanning over all queries.
    pub(crate) fn verification_work(&self) -> Result<usize, TopologyError> {
        let lane_work = self
            .local_window
            .checked_add(self.top_m_tokens)
            .and_then(|lanes| lanes.checked_mul(3))
            .ok_or(TopologyError::VerificationWorkOverflow)?;
        let per_query = self
            .chunk_count
            .checked_add(lane_work)
            .ok_or(TopologyError::VerificationWorkOverflow)?;
        self.valid_length
            .checked_mul(per_query)
            .ok_or(TopologyError::VerificationWorkOverflow)
    }
}

/// A deterministic, bounded request accepted by `dwarf-oracle sweep`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(untagged)]
pub enum SweepRequest {
    Configs { configs: Vec<OracleConfig> },
    Grid { grid: SweepGrid },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SweepEnvelope {
    configs: Option<Vec<OracleConfig>>,
    grid: Option<SweepGrid>,
}

impl<'de> Deserialize<'de> for SweepRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let SweepEnvelope { configs, grid } = SweepEnvelope::deserialize(deserializer)?;
        match (configs, grid) {
            (Some(configs), None) => Ok(Self::Configs { configs }),
            (None, Some(grid)) => Ok(Self::Grid { grid }),
            (Some(_), Some(_)) => Err(de::Error::custom(
                "sweep envelope must contain exactly one of \"configs\" or \"grid\"",
            )),
            (None, None) => Err(de::Error::custom(
                "sweep envelope must contain exactly one of \"configs\" or \"grid\"",
            )),
        }
    }
}

/// Cartesian product representation for a deliberately small configuration sweep.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SweepGrid {
    pub sequence_lengths: Vec<usize>,
    pub valid_lengths: Vec<usize>,
    pub chunk_counts: Vec<usize>,
    pub selector_tile_lengths: Vec<usize>,
    pub local_windows: Vec<usize>,
    pub top_k_chunks: Vec<usize>,
    pub top_m_tokens: Vec<usize>,
}
