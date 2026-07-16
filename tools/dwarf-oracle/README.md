# dwarf-oracle

`dwarf-oracle` is a small CPU-first Rust tool for analytically checking **DWARF V16 causal topology** before spending GPU time on parity or training. It is a library crate plus a JSON CLI.

## Scope

The initial vertical slice models these discrete parameters:

- `sequence_length`, `valid_length`, `chunk_count`
- `selector_tile_length`, `local_window`
- `top_k_chunks`, `top_m_tokens`

Its deterministic candidate ordering is explicitly a **topology oracle**, not learned routing:

1. local lane is exactly `[q - local_window, q)` (saturating at zero);
2. global chunks must have completed before the selector tile starts;
3. selected global keys are strictly before `q - local_window`;
4. all selected keys are `< q` and `< valid_length`;
5. local and global lanes cannot overlap.

A configuration uses uniform chunks, so `sequence_length` must divide evenly by `chunk_count`. To keep CPU-only analytical verification bounded for untrusted JSON, each configuration is limited to `sequence_length <= 8192` (and therefore `valid_length <= 8192`), `chunk_count <= 512`, and a conservative verification-work estimate of 1,000,000; a sweep also has a combined work limit of 4,000,000. The production V16 reference point is selector tile 16, `BLOCK_Q=16`, local window 64, `C=32`, `top_k=4`, and `top_m=64`.

## JSON CLI

```bash
cargo run -- verify config.json
cargo run -- sweep sweep.json
# `-` reads JSON from stdin.
```

A verify document is one `OracleConfig`:

```json
{
  "sequence_length": 2048,
  "valid_length": 2048,
  "chunk_count": 32,
  "selector_tile_length": 16,
  "local_window": 64,
  "top_k_chunks": 4,
  "top_m_tokens": 64
}
```

A sweep is either `{"configs": [ ... ]}` (maximum 64 items) or a small Cartesian `{"grid": { ... }}` with plural parameter arrays; an envelope must contain exactly one of those fields. Output is deterministic, structured JSON. After a valid `verify` or `sweep` subcommand is selected, malformed JSON, unknown fields, or an invalid request envelope produces a structured JSON error on stderr with exit status 2; `verify` also uses that form for rejected topology configurations, while a syntactically valid sweep reports ordinary per-item topology-invalid configurations in its deterministic report.

## Deliberate non-goals

This tool is **not** a Rust rewrite of DWARF, a PyTorch/kernel parity implementation, a learned router, or a PPL/quality predictor. Passing proves only this tool's deterministic topology contract, while invalid topology input is rejected; it does not currently reject or rank dominated causal configurations and cannot replace matched quality experiments.

## Future boundary

Golden fixtures from matched PyTorch traces should be added only when their provenance and V16 contract are stable. Synthetic candidate-recall experiments belong in a separate, explicitly labeled evaluation layer; they must not be treated as routing quality or training results.
