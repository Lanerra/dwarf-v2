"""Frozen source binding for content-free semantic calibration v1."""

PROTOCOL_MANIFEST = {
    "format": "dwarf-semantic-evaluator-protocol-v1",
    "protocol": "content_free_pmi_mean_nll_v1",
    "scoring_revision": "raw_plus_content_free_pmi_v1",
    "diagnostic_only": True,
    "raw_metrics_replaced": False,
    "evaluator_source_hashes": {
        "semantic_choice_calibration.py": "3691069af19bc7f1720a6f9d3b022b7ebb399484b636790f6822bfa38275af50",
        "semantic_transfer_eval.py": "1682f76e0a78e3e7d0bafeddabddfdc181ce6f85288d1fede75b1c66758fb125",
    },
}
