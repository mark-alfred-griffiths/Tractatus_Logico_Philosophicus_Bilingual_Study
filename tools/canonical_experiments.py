from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "dsh_validation"

CANONICAL_STATUS = "canonical"

PHASE3_LAMBDA_GRID = [0.00, 0.03, 0.10, 0.30, 1.00]
PHASE3_SEEDS = list(range(10))
PHASE3_CANONICAL_CONDITIONS = ["full_model", "no_successor"]
PHASE3_CANONICAL_BATCHING = "paired"


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    phase: str
    condition: str
    batching: str | None
    lambda_language_alignment: float | None
    seeds: tuple[int, ...]
    experiment_status: str
    evidence_tier: str
    sampler_type: str | None


def lambda_tag(value: float) -> str:
    return f"align{int(round(value * 100)):03d}"


def phase3_experiment_id(batching: str, condition: str, alignment_lambda: float) -> str:
    return f"{batching}_{condition}_{lambda_tag(alignment_lambda)}"


def phase3_experiment_status(batching: str) -> str:
    if batching != PHASE3_CANONICAL_BATCHING:
        raise ValueError(f"non-canonical Phase 3 batching is not supported: {batching}")
    return CANONICAL_STATUS


def phase3_sampler_type(batching: str) -> str:
    if batching != PHASE3_CANONICAL_BATCHING:
        raise ValueError(f"non-canonical Phase 3 batching is not supported: {batching}")
    return "id_level_paired"


def canonical_phase3_ids() -> set[str]:
    return {
        phase3_experiment_id(PHASE3_CANONICAL_BATCHING, condition, alignment_lambda)
        for condition in PHASE3_CANONICAL_CONDITIONS
        for alignment_lambda in PHASE3_LAMBDA_GRID
    }


def canonical_registry() -> list[ExperimentRecord]:
    records: list[ExperimentRecord] = [
        ExperimentRecord(
            "phase1_retained_ablations",
            "phase1",
            "all_retained_ablation_conditions",
            None,
            0.0,
            tuple(PHASE3_SEEDS),
            CANONICAL_STATUS,
            CANONICAL_STATUS,
            "standard_row",
        ),
        ExperimentRecord(
            "phase2_family_holdout",
            "phase2",
            "full_model_no_successor_reconstruction_only",
            None,
            0.0,
            (0, 1, 2),
            CANONICAL_STATUS,
            CANONICAL_STATUS,
            "standard_row",
        ),
    ]
    for condition in PHASE3_CANONICAL_CONDITIONS:
        for alignment_lambda in PHASE3_LAMBDA_GRID:
            records.append(
                ExperimentRecord(
                    phase3_experiment_id(PHASE3_CANONICAL_BATCHING, condition, alignment_lambda),
                    "phase3",
                    condition,
                    PHASE3_CANONICAL_BATCHING,
                    alignment_lambda,
                    tuple(PHASE3_SEEDS),
                    CANONICAL_STATUS,
                    CANONICAL_STATUS,
                    "id_level_paired",
                )
            )
    records.append(
        ExperimentRecord(
            "phase4_frozen_case_selection",
            "phase4",
            "frozen_text_blind_cases",
            None,
            None,
            tuple(PHASE3_SEEDS),
            CANONICAL_STATUS,
            CANONICAL_STATUS,
            "paired_phase3_raw_outputs",
        )
    )
    return records
