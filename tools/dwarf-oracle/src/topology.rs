use std::{error::Error, fmt};

use serde::{Deserialize, Serialize};

use crate::OracleConfig;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct QueryTopology {
    pub query: usize,
    pub selector_tile_start: usize,
    /// Eligible in increasing chunk index order, before deterministic top-k selection.
    pub eligible_chunks: Vec<usize>,
    /// Selected in most-recent-first order; this is a topology ordering, not learned routing.
    pub selected_chunks: Vec<usize>,
    pub local_keys: Vec<usize>,
    pub global_keys: Vec<usize>,
    pub selected_keys: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TopologyError {
    SequenceLengthZero,
    SequenceLengthExceedsLimit {
        sequence_length: usize,
        limit: usize,
    },
    ValidLengthExceedsSequence {
        valid_length: usize,
        sequence_length: usize,
    },
    ChunkCountZero,
    ChunkCountExceedsLimit {
        chunk_count: usize,
        limit: usize,
    },
    UnevenChunks {
        sequence_length: usize,
        chunk_count: usize,
    },
    SelectorTileLengthZero,
    LocalWindowExceedsSequence {
        local_window: usize,
        sequence_length: usize,
    },
    VerificationWorkTooLarge {
        estimated: usize,
        limit: usize,
    },
    VerificationWorkOverflow,
    SweepVerificationWorkTooLarge {
        estimated: usize,
        limit: usize,
    },
    SweepVerificationWorkOverflow,
    ArithmeticOverflow {
        operation: &'static str,
    },
    QueryOutsideValidLength {
        query: usize,
        valid_length: usize,
    },
    SweepEmpty,
    SweepTooLarge {
        limit: usize,
        requested: usize,
    },
    SweepCardinalityOverflow,
    ContractViolation(String),
}

impl TopologyError {
    pub(crate) const fn is_resource_limit(&self) -> bool {
        matches!(
            self,
            Self::SequenceLengthExceedsLimit { .. }
                | Self::ChunkCountExceedsLimit { .. }
                | Self::VerificationWorkTooLarge { .. }
                | Self::VerificationWorkOverflow
        )
    }
}

impl fmt::Display for TopologyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SequenceLengthZero => {
                write!(formatter, "sequence_length must be greater than zero")
            }
            Self::SequenceLengthExceedsLimit {
                sequence_length,
                limit,
            } => write!(
                formatter,
                "sequence_length ({sequence_length}) exceeds the CPU analytical limit ({limit})"
            ),
            Self::ValidLengthExceedsSequence {
                valid_length,
                sequence_length,
            } => write!(
                formatter,
                "valid_length ({valid_length}) must not exceed sequence_length ({sequence_length})"
            ),
            Self::ChunkCountZero => write!(formatter, "chunk_count must be greater than zero"),
            Self::ChunkCountExceedsLimit { chunk_count, limit } => write!(
                formatter,
                "chunk_count ({chunk_count}) exceeds the CPU analytical limit ({limit})"
            ),
            Self::UnevenChunks {
                sequence_length,
                chunk_count,
            } => write!(
                formatter,
                "sequence_length ({sequence_length}) must divide evenly by chunk_count ({chunk_count})"
            ),
            Self::SelectorTileLengthZero => {
                write!(formatter, "selector_tile_length must be greater than zero")
            }
            Self::LocalWindowExceedsSequence {
                local_window,
                sequence_length,
            } => write!(
                formatter,
                "local_window ({local_window}) must not exceed sequence_length ({sequence_length})"
            ),
            Self::VerificationWorkTooLarge { estimated, limit } => write!(
                formatter,
                "configuration requires {estimated} units of CPU analytical work; the limit is {limit}"
            ),
            Self::VerificationWorkOverflow => write!(
                formatter,
                "configuration CPU analytical work estimate overflowed usize"
            ),
            Self::SweepVerificationWorkTooLarge { estimated, limit } => write!(
                formatter,
                "sweep requires {estimated} units of CPU analytical work; the limit is {limit}"
            ),
            Self::SweepVerificationWorkOverflow => {
                write!(
                    formatter,
                    "sweep CPU analytical work estimate overflowed usize"
                )
            }
            Self::ArithmeticOverflow { operation } => {
                write!(
                    formatter,
                    "topology arithmetic overflow while computing {operation}"
                )
            }
            Self::QueryOutsideValidLength {
                query,
                valid_length,
            } => write!(
                formatter,
                "query ({query}) must be less than valid_length ({valid_length})"
            ),
            Self::SweepEmpty => write!(formatter, "sweep must contain at least one configuration"),
            Self::SweepTooLarge { limit, requested } => write!(
                formatter,
                "sweep contains {requested} configurations; the deterministic limit is {limit}"
            ),
            Self::SweepCardinalityOverflow => {
                write!(formatter, "sweep grid cardinality overflowed usize")
            }
            Self::ContractViolation(message) => {
                write!(formatter, "strict-causal contract violation: {message}")
            }
        }
    }
}

impl Error for TopologyError {}

pub fn simulate_query(config: &OracleConfig, query: usize) -> Result<QueryTopology, TopologyError> {
    config.validate()?;
    if query >= config.valid_length {
        return Err(TopologyError::QueryOutsideValidLength {
            query,
            valid_length: config.valid_length,
        });
    }

    let selector_tile_start = (query / config.selector_tile_length)
        .checked_mul(config.selector_tile_length)
        .ok_or(TopologyError::ArithmeticOverflow {
            operation: "selector tile start",
        })?;
    let chunk_length = config.chunk_length();
    let mut eligible_chunks = Vec::with_capacity(config.chunk_count);
    for chunk in 0..config.chunk_count {
        if chunk_endpoint(chunk, chunk_length)? <= selector_tile_start {
            eligible_chunks.push(chunk);
        }
    }

    let selected_chunks = eligible_chunks
        .iter()
        .rev()
        .take(config.top_k_chunks)
        .copied()
        .collect::<Vec<_>>();

    let local_start = query.saturating_sub(config.local_window);
    let local_keys = (local_start..query).collect::<Vec<_>>();
    let mut global_keys = Vec::new();
    if config.top_m_tokens > 0 {
        'chunks: for &chunk in &selected_chunks {
            let start = chunk_start(chunk, chunk_length)?;
            let end = start
                .checked_add(chunk_length)
                .ok_or(TopologyError::ArithmeticOverflow {
                    operation: "chunk end",
                })?;
            for key in (start..end).rev() {
                if key < local_start && key < config.valid_length {
                    global_keys.push(key);
                    if global_keys.len() == config.top_m_tokens {
                        break 'chunks;
                    }
                }
            }
        }
    }
    let selected_keys = local_keys
        .iter()
        .chain(&global_keys)
        .copied()
        .collect::<Vec<_>>();

    Ok(QueryTopology {
        query,
        selector_tile_start,
        eligible_chunks,
        selected_chunks,
        local_keys,
        global_keys,
        selected_keys,
    })
}

pub(crate) fn chunk_endpoint(chunk: usize, chunk_length: usize) -> Result<usize, TopologyError> {
    chunk
        .checked_add(1)
        .and_then(|next_chunk| next_chunk.checked_mul(chunk_length))
        .ok_or(TopologyError::ArithmeticOverflow {
            operation: "chunk endpoint",
        })
}

fn chunk_start(chunk: usize, chunk_length: usize) -> Result<usize, TopologyError> {
    chunk
        .checked_mul(chunk_length)
        .ok_or(TopologyError::ArithmeticOverflow {
            operation: "chunk start",
        })
}
