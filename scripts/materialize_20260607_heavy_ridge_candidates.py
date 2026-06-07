from __future__ import annotations

from materialize_20260607_candidates import Candidate, materialize_candidate

CANDIDATES = [
    Candidate(
        dirname="ridge_w040_ridge042",
        kernel_id="surajranganath17/rogii-ridge-w040-ridge042",
        title="ROGII Ridge W040 Ridge042",
        code_file="ridge_w040_ridge042.py",
        profile="w040_ridge042",
        ridge_weight=0.42,
        selector_weight=0.58,
        final_seeds=32,
    ),
    Candidate(
        dirname="ridge_w040_ridge045",
        kernel_id="surajranganath17/rogii-ridge-w040-ridge045",
        title="ROGII Ridge W040 Ridge045",
        code_file="ridge_w040_ridge045.py",
        profile="w040_ridge045",
        ridge_weight=0.45,
        selector_weight=0.55,
        final_seeds=32,
    ),
    Candidate(
        dirname="ridge_w040_ridge050",
        kernel_id="surajranganath17/rogii-ridge-w040-ridge050",
        title="ROGII Ridge W040 Ridge050",
        code_file="ridge_w040_ridge050.py",
        profile="w040_ridge050",
        ridge_weight=0.50,
        selector_weight=0.50,
        final_seeds=32,
    ),
    Candidate(
        dirname="ridge_w040_ridge060",
        kernel_id="surajranganath17/rogii-ridge-w040-ridge060",
        title="ROGII Ridge W040 Ridge060",
        code_file="ridge_w040_ridge060.py",
        profile="w040_ridge060",
        ridge_weight=0.60,
        selector_weight=0.40,
        final_seeds=32,
    ),
    Candidate(
        dirname="ridge_w040_ridge070",
        kernel_id="surajranganath17/rogii-ridge-w040-ridge070",
        title="ROGII Ridge W040 Ridge070",
        code_file="ridge_w040_ridge070.py",
        profile="w040_ridge070",
        ridge_weight=0.70,
        selector_weight=0.30,
        final_seeds=32,
    ),
]


def main() -> None:
    for candidate in CANDIDATES:
        materialize_candidate(candidate)
        print(f"materialized {candidate.dirname}: {candidate.kernel_id}")


if __name__ == "__main__":
    main()
