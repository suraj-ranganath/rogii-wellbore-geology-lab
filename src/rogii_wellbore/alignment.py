from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_NCC_HALFWIDTHS = (8, 15, 25)
DEFAULT_BEAM_VARIANTS = (
    (10, 20.0, 144.0, 3, "cons"),
    (10, 8.0, 64.0, 3, "loose"),
    (10, 14.0, 90.0, 5, "sm5"),
)
BEAM_OFFSETS = (-20.0, -10.0, 0.0, 10.0, 20.0)


def prefix_ncc_features(
    gr: pd.Series | np.ndarray,
    tvt_input: pd.Series | np.ndarray,
    halfwidths: Sequence[int] = DEFAULT_NCC_HALFWIDTHS,
    stride: int = 3,
    smooth_window: int = 5,
    softmax_temperature: float = 3.0,
    min_known: int = 10,
    chunk_size: int = 2048,
) -> pd.DataFrame:
    """Build target-free GR self-correlation features against the known prefix.

    For each row, the local horizontal GR window is matched against GR windows in
    the `TVT_input` prefix. The returned TVT estimates are legal at inference
    time because they use only observed GR and the known prefix TVT.
    """
    index = gr.index if isinstance(gr, pd.Series) else pd.RangeIndex(len(gr))
    gr_arr = _to_float_array(gr)
    tvt_arr = _to_float_array(tvt_input)
    if len(gr_arr) != len(tvt_arr):
        raise ValueError("gr and tvt_input must have the same length.")

    n_rows = len(gr_arr)
    fallback_tvt = _last_finite(tvt_arr, default=0.0)
    result = pd.DataFrame(index=index)
    if n_rows == 0:
        return result

    for halfwidth in halfwidths:
        result[f"prefix_ncc_tvt_hw{halfwidth}"] = fallback_tvt
        result[f"prefix_ncc_delta_hw{halfwidth}"] = 0.0
        result[f"prefix_ncc_score_hw{halfwidth}"] = 0.0

    result["prefix_ncc_tvt_ensemble"] = fallback_tvt
    result["prefix_ncc_tvt_consensus"] = fallback_tvt
    result["prefix_ncc_delta_ensemble"] = 0.0
    result["prefix_ncc_score_mean"] = 0.0
    result["prefix_ncc_score_max"] = 0.0
    result["prefix_ncc_tvt_spread"] = 0.0
    result["prefix_ncc_known_rows"] = float(np.isfinite(tvt_arr).sum())
    result["prefix_ncc_trust"] = float(np.clip(np.isfinite(tvt_arr).sum() / 200.0, 0.0, 0.6))

    if not np.isfinite(gr_arr).any():
        return result

    known = np.isfinite(tvt_arr)
    if int(known.sum()) < min_known:
        return result

    gr_filled = _fill_signal(gr_arr)
    known_gr = gr_filled[known]
    known_tvt = tvt_arr[known]
    if len(known_gr) < min_known or not np.isfinite(known_gr).any():
        return result

    query_gr = _rolling_mean(gr_filled, smooth_window)
    prefix_gr = _rolling_mean(known_gr, smooth_window)

    tvt_columns: list[np.ndarray] = []
    score_columns: list[np.ndarray] = []
    for halfwidth in halfwidths:
        tvt_est, score = _ncc_match_scale(
            prefix_gr=prefix_gr,
            prefix_tvt=known_tvt,
            query_gr=query_gr,
            halfwidth=int(halfwidth),
            stride=int(stride),
            fallback_tvt=float(fallback_tvt),
            chunk_size=int(chunk_size),
        )
        result[f"prefix_ncc_tvt_hw{halfwidth}"] = tvt_est
        result[f"prefix_ncc_delta_hw{halfwidth}"] = tvt_est - fallback_tvt
        result[f"prefix_ncc_score_hw{halfwidth}"] = score
        tvt_columns.append(tvt_est)
        score_columns.append(score)

    tvts = np.column_stack(tvt_columns)
    scores = np.column_stack(score_columns)
    stable_scores = scores - np.nanmax(scores, axis=1, keepdims=True)
    weights = np.exp(softmax_temperature * stable_scores)
    weights /= np.sum(weights, axis=1, keepdims=True) + 1e-9

    ensemble = np.sum(tvts * weights, axis=1)
    consensus = np.mean(tvts, axis=1)
    result["prefix_ncc_tvt_ensemble"] = ensemble
    result["prefix_ncc_tvt_consensus"] = consensus
    result["prefix_ncc_delta_ensemble"] = ensemble - fallback_tvt
    result["prefix_ncc_score_mean"] = np.mean(scores, axis=1)
    result["prefix_ncc_score_max"] = np.max(scores, axis=1)
    result["prefix_ncc_tvt_spread"] = np.std(tvts, axis=1)
    return result.replace([np.inf, -np.inf], np.nan)


def typewell_beam_features(
    gr: pd.Series | np.ndarray,
    tvt_input: pd.Series | np.ndarray,
    typewell_tvt: pd.Series | np.ndarray,
    typewell_gr: pd.Series | np.ndarray,
    variants: Sequence[tuple[int, float, float, int, str]] = DEFAULT_BEAM_VARIANTS,
    offsets: Sequence[float] = BEAM_OFFSETS,
) -> pd.DataFrame:
    """Build deterministic typewell GR beam-alignment features for eval rows."""
    index = gr.index if isinstance(gr, pd.Series) else pd.RangeIndex(len(gr))
    gr_arr = _to_float_array(gr)
    tvt_arr = _to_float_array(tvt_input)
    tw_tvt = _to_float_array(typewell_tvt)
    tw_gr = _to_float_array(typewell_gr)
    if len(gr_arr) != len(tvt_arr):
        raise ValueError("gr and tvt_input must have the same length.")

    result = pd.DataFrame(index=index)
    fallback_tvt = _last_finite(tvt_arr, default=_last_finite(tw_tvt, default=0.0))
    for _, _, _, _, tag in variants:
        result[f"typewell_beam_tvt_{tag}"] = fallback_tvt
        result[f"typewell_beam_delta_{tag}"] = 0.0
    result["typewell_beam_tvt_mean"] = fallback_tvt
    result["typewell_beam_tvt_median"] = fallback_tvt
    result["typewell_beam_tvt_std"] = 0.0
    result["typewell_beam_delta_mean"] = 0.0
    result["typewell_beam_delta_std"] = 0.0
    result["typewell_beam_gr_residual"] = 0.0
    result["typewell_beam_abs_gr_residual"] = 0.0
    for offset in offsets:
        result[f"typewell_beam_gr_residual_off_{_offset_name(offset)}"] = 0.0

    eval_mask = ~np.isfinite(tvt_arr)
    eval_idx = np.flatnonzero(eval_mask)
    clean = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    if len(eval_idx) == 0 or clean.sum() < 3 or not np.isfinite(gr_arr).any():
        return result

    order = np.argsort(tw_tvt[clean])
    tw_tvt_clean = tw_tvt[clean][order].astype(np.float32)
    tw_gr_clean = tw_gr[clean][order].astype(np.float32)
    gr_filled = _fill_signal(gr_arr)
    eval_gr = gr_filled[eval_idx]

    beam_paths: dict[str, np.ndarray] = {}
    for beam_size, movement_cost, emission_scale, smooth_radius, tag in variants:
        beam_paths[tag] = _beam_search(
            gr_h=eval_gr,
            tw_tvt=tw_tvt_clean,
            tw_gr=tw_gr_clean,
            start_tvt=float(fallback_tvt),
            beam_size=int(beam_size),
            movement_cost=float(movement_cost),
            emission_scale=float(emission_scale),
            smooth_radius=int(smooth_radius),
        )
        result.iloc[eval_idx, result.columns.get_loc(f"typewell_beam_tvt_{tag}")] = beam_paths[tag]
        result.iloc[eval_idx, result.columns.get_loc(f"typewell_beam_delta_{tag}")] = (
            beam_paths[tag] - fallback_tvt
        )

    stacked = np.column_stack(list(beam_paths.values()))
    mean_path = stacked.mean(axis=1)
    median_path = np.median(stacked, axis=1)
    std_path = stacked.std(axis=1)
    result.iloc[eval_idx, result.columns.get_loc("typewell_beam_tvt_mean")] = mean_path
    result.iloc[eval_idx, result.columns.get_loc("typewell_beam_tvt_median")] = median_path
    result.iloc[eval_idx, result.columns.get_loc("typewell_beam_tvt_std")] = std_path
    result.iloc[eval_idx, result.columns.get_loc("typewell_beam_delta_mean")] = (
        mean_path - fallback_tvt
    )
    result.iloc[eval_idx, result.columns.get_loc("typewell_beam_delta_std")] = std_path

    ref_path = _beam_reference(beam_paths)
    ref_gr = np.interp(ref_path, tw_tvt_clean, tw_gr_clean)
    residual = eval_gr - ref_gr
    result.iloc[eval_idx, result.columns.get_loc("typewell_beam_gr_residual")] = residual
    result.iloc[eval_idx, result.columns.get_loc("typewell_beam_abs_gr_residual")] = np.abs(residual)
    for offset in offsets:
        column = f"typewell_beam_gr_residual_off_{_offset_name(offset)}"
        offset_gr = np.interp(ref_path + float(offset), tw_tvt_clean, tw_gr_clean)
        result.iloc[eval_idx, result.columns.get_loc(column)] = eval_gr - offset_gr

    return result.replace([np.inf, -np.inf], np.nan)


def _beam_search(
    gr_h: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    start_tvt: float,
    beam_size: int,
    movement_cost: float,
    emission_scale: float,
    smooth_radius: int,
) -> np.ndarray:
    n_steps = len(gr_h)
    n_typewell = len(tw_tvt)
    if n_steps == 0 or n_typewell == 0:
        return np.array([], dtype=np.float32)

    beam_size = max(1, min(beam_size, n_typewell))
    smoothed_gr = _smooth_for_beam(gr_h, fallback=float(np.nanmean(tw_gr)), radius=smooth_radius)
    start_idx = _nearest_index(tw_tvt, start_tvt)
    deltas = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    move_penalty = movement_cost * np.abs(deltas).astype(float)

    beam_indices = np.full(beam_size, start_idx, dtype=np.int32)
    beam_costs = np.zeros(beam_size, dtype=float)
    parent_history = np.zeros((n_steps, beam_size), dtype=np.int32)
    index_history = np.zeros((n_steps, beam_size), dtype=np.int32)

    for step, gr_value in enumerate(smoothed_gr):
        candidate_indices = np.clip(beam_indices[:, None] + deltas[None, :], 0, n_typewell - 1)
        emission = (gr_value - tw_gr[candidate_indices]) ** 2 / max(emission_scale, 1e-6)
        candidate_costs = beam_costs[:, None] + emission + move_penalty[None, :]
        flat_indices = candidate_indices.ravel()
        flat_costs = candidate_costs.ravel()
        flat_parents = np.repeat(np.arange(beam_size, dtype=np.int32), len(deltas))

        order = np.argsort(flat_costs, kind="stable")
        seen: set[int] = set()
        keep: list[int] = []
        for order_idx in order:
            typewell_idx = int(flat_indices[order_idx])
            if typewell_idx in seen:
                continue
            seen.add(typewell_idx)
            keep.append(int(order_idx))
            if len(keep) == beam_size:
                break
        if not keep:
            keep = [int(order[0])]
        while len(keep) < beam_size:
            keep.append(keep[-1])
        keep_arr = np.array(keep, dtype=np.int32)
        parent_history[step] = flat_parents[keep_arr]
        index_history[step] = flat_indices[keep_arr]
        beam_indices = flat_indices[keep_arr].astype(np.int32)
        beam_costs = flat_costs[keep_arr]

    path_idx = np.empty(n_steps, dtype=np.int32)
    current = int(np.argmin(beam_costs))
    for step in range(n_steps - 1, -1, -1):
        path_idx[step] = index_history[step, current]
        current = parent_history[step, current]
    return tw_tvt[path_idx].astype(np.float32)


def _ncc_match_scale(
    prefix_gr: np.ndarray,
    prefix_tvt: np.ndarray,
    query_gr: np.ndarray,
    halfwidth: int,
    stride: int,
    fallback_tvt: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_prefix = len(prefix_gr)
    n_query = len(query_gr)
    window = 2 * halfwidth + 1
    tvt_est = np.full(n_query, fallback_tvt, dtype=np.float32)
    score = np.zeros(n_query, dtype=np.float32)

    if window <= 1 or n_prefix < window + 1 or n_query == 0:
        return tvt_est, score

    starts = np.arange(0, n_prefix - window + 1, max(stride, 1), dtype=np.int32)
    if len(starts) == 0:
        return tvt_est, score

    window_offsets = np.arange(window, dtype=np.int32)
    candidate = prefix_gr[starts[:, None] + window_offsets[None, :]].astype(np.float32)
    candidate = _normalize_rows(candidate)
    query_padded = np.pad(query_gr.astype(np.float32), halfwidth, mode="edge")

    for start in range(0, n_query, max(chunk_size, 1)):
        stop = min(start + max(chunk_size, 1), n_query)
        query_index = np.arange(start, stop, dtype=np.int32)
        query_windows = query_padded[query_index[:, None] + window_offsets[None, :]]
        query_windows = _normalize_rows(query_windows.astype(np.float32))
        ncc = query_windows @ candidate.T / float(window)
        best = np.argmax(ncc, axis=1)
        best_score = ncc[np.arange(len(query_index)), best]
        centers = np.clip(starts[best] + halfwidth, 0, n_prefix - 1)
        tvt_est[start:stop] = prefix_tvt[centers].astype(np.float32)
        score[start:stop] = best_score.astype(np.float32)

    return tvt_est, score


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return (values - mean) / (std + 1e-6)


def _beam_reference(beam_paths: dict[str, np.ndarray]) -> np.ndarray:
    if {"cons", "sm5"}.issubset(beam_paths):
        return (beam_paths["cons"] + beam_paths["sm5"]) * 0.5
    return np.column_stack(list(beam_paths.values())).mean(axis=1)


def _smooth_for_beam(values: np.ndarray, fallback: float, radius: int) -> np.ndarray:
    series = pd.Series(values, dtype="float64").interpolate(limit_direction="both").fillna(fallback)
    if radius <= 0:
        return series.to_numpy(dtype=np.float32)
    return (
        series.rolling(radius * 2 + 1, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float32)
    )


def _nearest_index(values: np.ndarray, target: float) -> int:
    idx = int(np.searchsorted(values, target, side="left"))
    if idx >= len(values):
        return len(values) - 1
    if idx > 0 and abs(values[idx - 1] - target) <= abs(values[idx] - target):
        return idx - 1
    return idx


def _offset_name(offset: float) -> str:
    if offset < 0:
        return f"m{abs(int(offset))}"
    return f"p{int(offset)}"


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32)
    return (
        pd.Series(values)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float32)
    )


def _fill_signal(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values, dtype="float64").interpolate(limit_direction="both")
    if series.isna().any():
        fill_value = float(np.nanmean(values)) if np.isfinite(values).any() else 0.0
        series = series.fillna(fill_value)
    return series.to_numpy(dtype=np.float32)


def _last_finite(values: np.ndarray, default: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float(default)
    return float(finite[-1])


def _to_float_array(values: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
