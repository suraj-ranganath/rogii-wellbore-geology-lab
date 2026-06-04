from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rogii_wellbore.data import find_sample_submission, read_csv, scan_wells
from rogii_wellbore.features import canonicalize, prediction_mask
from rogii_wellbore.submission import parse_submission_id


def audit_dataset(data_dir: Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    pairs = scan_wells(data_dir)
    sample_submission_path = find_sample_submission(data_dir)
    sample_submission = (
        read_csv(sample_submission_path) if sample_submission_path is not None else pd.DataFrame()
    )

    split_rows = []
    for split in sorted({pair.split for pair in pairs}):
        split_pairs = [pair for pair in pairs if pair.split == split]
        horizontal_rows = 0
        prediction_rows = 0
        for pair in split_pairs:
            horizontal = canonicalize(read_csv(pair.horizontal_path))
            horizontal_rows += int(len(horizontal))
            prediction_rows += int(prediction_mask(horizontal, require_target=False).sum())
        split_rows.append(
            {
                "split": split,
                "wells": len(split_pairs),
                "horizontal_rows": horizontal_rows,
                "prediction_rows": prediction_rows,
                "missing_typewell": sum(pair.typewell_path is None for pair in split_pairs),
            }
        )

    train_pairs = {pair.well_id: pair for pair in pairs if pair.split == "train"}
    test_pairs = {pair.well_id: pair for pair in pairs if pair.split == "test"}
    overlap = sorted(set(train_pairs) & set(test_pairs))

    sample_groups = _sample_submission_groups(sample_submission)
    overlap_details = []
    for well_id in overlap:
        train_horizontal = canonicalize(read_csv(train_pairs[well_id].horizontal_path))
        test_horizontal = canonicalize(read_csv(test_pairs[well_id].horizontal_path))
        shared_columns = [column for column in test_horizontal.columns if column in train_horizontal.columns]
        equal_shared = True
        for column in shared_columns:
            left = train_horizontal[column].reset_index(drop=True)
            right = test_horizontal[column].reset_index(drop=True)
            if len(left) != len(right):
                equal_shared = False
                break
            if pd.api.types.is_numeric_dtype(left):
                if not np.allclose(left.to_numpy(), right.to_numpy(), equal_nan=True):
                    equal_shared = False
                    break
            elif not left.equals(right):
                equal_shared = False
                break

        sample_indices = sample_groups.get(well_id, [])
        train_target_available = bool(
            sample_indices
            and "tvt" in train_horizontal
            and max(sample_indices) < len(train_horizontal)
            and train_horizontal.iloc[sample_indices]["tvt"].notna().all()
        )
        overlap_details.append(
            {
                "well_id": well_id,
                "train_rows": int(len(train_horizontal)),
                "test_rows": int(len(test_horizontal)),
                "shared_test_columns_equal_train": equal_shared,
                "sample_submission_rows": int(len(sample_indices)),
                "sample_index_min": int(min(sample_indices)) if sample_indices else None,
                "sample_index_max": int(max(sample_indices)) if sample_indices else None,
                "train_target_available_for_sample_rows": train_target_available,
            }
        )

    return {
        "data_dir": str(data_dir),
        "splits": split_rows,
        "sample_submission_rows": int(len(sample_submission)),
        "sample_submission_path": str(sample_submission_path) if sample_submission_path else None,
        "train_test_overlap_wells": overlap,
        "train_test_overlap_details": overlap_details,
    }


def _sample_submission_groups(sample_submission: pd.DataFrame) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    if sample_submission.empty or "id" not in sample_submission:
        return groups
    for raw_id in sample_submission["id"]:
        well_id, row_idx = parse_submission_id(str(raw_id))
        groups.setdefault(well_id, []).append(row_idx)
    return groups
