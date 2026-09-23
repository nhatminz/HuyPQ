from __future__ import annotations

import copy
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import save_config
from .data import (
    expand_prompt_batch,
    filter_overlong_prompt_records,
    read_records,
    stable_sample_id,
    tokenize_prompts,
    validate_prompt_records,
)
from .models import load_models, load_student_tokenizer
from .resume import restore_optimizer
from .scoring import (
    generate_on_policy,
    reverse_kl_next_token,
    score_reverse_kl_rollout,
    score_student_teacher_rollout,
)
from .selectors import CMTSelector, PGTSelector
from .trainer import _make_optimizer, seed_everything


REQUIRED_STATE_COLUMNS = (
    "state_id",
    "prompt_id",
    "token_position",
    "G_t",
    "D_tilde",
    "KL_before",
    "KL_after",
    "local_gain",
    "C_before",
    "C_after",
    "downstream_gain_measured",
    "G_bin",
    "D_quantile",
)


@dataclass(frozen=True)
class _Trajectory:
    prompt_id: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_index: int


@dataclass(frozen=True)
class _Candidate:
    state_id: str
    trajectory_id: int
    prompt_id: str
    response_index: int
    token_position: int
    G_t: float
    D_tilde: float


def quantile_bin_labels(values, bins: int) -> np.ndarray:
    """Return deterministic, near-equal-frequency zero-based quantile bins.

    Ranking (with stable input-order tie breaking) avoids silently dropping
    bins when a score has many duplicate values, which is common for small
    local gains near zero.
    """
    array = np.asarray(values, dtype=np.float64)
    count = int(array.size)
    bins = int(bins)
    if count == 0:
        raise ValueError("Cannot quantile-bin an empty array")
    if bins <= 0 or count < bins:
        raise ValueError(f"Need at least {bins} values for {bins} quantile bins")
    if not np.isfinite(array).all():
        raise ValueError("Quantile-bin values must be finite")
    order = np.argsort(array, kind="stable")
    labels = np.empty(count, dtype=np.int64)
    labels[order] = np.minimum((np.arange(count) * bins) // count, bins - 1)
    return labels


def matched_quantile_sample(
    frame: pd.DataFrame,
    *,
    match_column: str,
    match_bins: int,
    d_bins: int = 5,
    samples_per_cell: int | None = None,
    seed: int = 0,
    position_bins: int = 0,
) -> pd.DataFrame:
    """Build an equal-cell matched design for one local-value covariate."""
    if match_column not in frame or "D_tilde" not in frame:
        raise ValueError(f"Missing {match_column!r} or 'D_tilde' from state table")
    result = frame.copy().reset_index(drop=True)
    result["match_bin"] = quantile_bin_labels(result[match_column], match_bins) + 1
    strata = ["match_bin"]
    if int(position_bins) > 1:
        result["position_bin"] = 0
        # Position is stratified within each local-value bin. This guarantees
        # that every G (or measured-local-gain) bin is compared over the same
        # early-to-late position mix even when position and local value are
        # strongly correlated.
        for _, indices in result.groupby("match_bin", sort=True).groups.items():
            indices = np.asarray(list(indices), dtype=np.int64)
            result.loc[indices, "position_bin"] = (
                quantile_bin_labels(
                    result.loc[indices, "token_position"], int(position_bins)
                )
                + 1
            )
        strata.append("position_bin")
    result["D_quantile_matched"] = 0
    for _, indices in result.groupby(strata, sort=True).groups.items():
        indices = np.asarray(list(indices), dtype=np.int64)
        if indices.size < int(d_bins):
            raise ValueError(
                "A matching stratum has fewer states than D quantiles; collect "
                "more candidate states or use fewer matching/position bins"
            )
        result.loc[indices, "D_quantile_matched"] = (
            quantile_bin_labels(result.loc[indices, "D_tilde"], d_bins) + 1
        )
    cell_columns = strata + ["D_quantile_matched"]
    counts = result.groupby(cell_columns, sort=True).size()
    if counts.size != math.prod([match_bins, max(1, int(position_bins)), d_bins]):
        raise ValueError("The requested matched design contains an empty cell")
    target = int(counts.min())
    if samples_per_cell is not None:
        if int(samples_per_cell) <= 0:
            raise ValueError("samples_per_cell must be positive when set")
        target = min(target, int(samples_per_cell))
    if target <= 0:
        raise ValueError("No states are available in at least one matched cell")
    rng = np.random.default_rng(int(seed))
    chosen: list[int] = []
    for _, indices in result.groupby(cell_columns, sort=True).groups.items():
        cell = np.asarray(list(indices), dtype=np.int64)
        chosen.extend(rng.choice(cell, size=target, replace=False).tolist())
    selected = result.loc[chosen].copy()
    selected = selected.sort_values(cell_columns + ["state_id"]).reset_index(drop=True)
    selected["matched_on"] = match_column
    selected["samples_per_cell"] = target
    return selected


def _spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return float("nan")
    ranked_x = x.rank(method="average")
    ranked_y = y.rank(method="average")
    if float(ranked_x.std()) == 0.0 or float(ranked_y.std()) == 0.0:
        return float("nan")
    value = ranked_x.corr(ranked_y, method="pearson")
    return float(value) if value is not None else float("nan")


def analyze_matched_states(
    frame: pd.DataFrame,
    *,
    match_bin_column: str,
    d_quantile_column: str,
    bootstrap_samples: int = 2000,
    seed: int = 0,
    position_bin_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Analyze D quintiles with equal weight for every matching stratum."""
    required = {
        "D_tilde",
        "downstream_gain_measured",
        match_bin_column,
        d_quantile_column,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Analysis table is missing columns: {sorted(missing)}")
    strata = [match_bin_column]
    if position_bin_column and position_bin_column in frame:
        strata.append(position_bin_column)
    cell_columns = strata + [d_quantile_column]
    cell_means = (
        frame.groupby(cell_columns, sort=True)["downstream_gain_measured"]
        .agg(["mean", "count"])
        .reset_index()
    )
    point = cell_means.groupby(d_quantile_column, sort=True)["mean"].mean().sort_index()
    d_levels = list(point.index)
    iterations = max(0, int(bootstrap_samples))
    draws = np.empty((iterations, len(d_levels)), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    grouped = {
        key: group["downstream_gain_measured"].to_numpy(dtype=np.float64)
        for key, group in frame.groupby(cell_columns, sort=True)
    }
    stratum_levels = sorted({key[:-1] for key in grouped})
    for iteration in range(iterations):
        for d_index, level in enumerate(d_levels):
            means = []
            for stratum in stratum_levels:
                values = grouped[(*stratum, level)]
                means.append(
                    float(rng.choice(values, size=len(values), replace=True).mean())
                )
            draws[iteration, d_index] = float(np.mean(means))
    rows = []
    for index, level in enumerate(d_levels):
        if iterations:
            low, high = np.quantile(draws[:, index], [0.025, 0.975])
        else:
            low = high = float("nan")
        rows.append(
            {
                "D_quantile": int(level),
                "mean_downstream_gain_measured": float(point.loc[level]),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "states": int((frame[d_quantile_column] == level).sum()),
                "equal_weight_strata": len(stratum_levels),
            }
        )
    aggregate = pd.DataFrame(rows)

    correlations = []
    for level, group in frame.groupby(match_bin_column, sort=True):
        rho = _spearman(group["D_tilde"], group["downstream_gain_measured"])
        correlations.append(
            {"match_bin": int(level), "spearman": rho, "states": len(group)}
        )
    correlation_frame = pd.DataFrame(correlations)
    finite_rho = correlation_frame["spearman"].to_numpy(dtype=np.float64)
    finite_rho = finite_rho[np.isfinite(finite_rho)]
    aggregate_rho = (
        float(np.tanh(np.arctanh(np.clip(finite_rho, -0.999999, 0.999999)).mean()))
        if finite_rho.size
        else float("nan")
    )
    predicted_sign = np.sign(frame["D_tilde"].to_numpy(dtype=np.float64))
    measured_sign = np.sign(
        frame["downstream_gain_measured"].to_numpy(dtype=np.float64)
    )
    nonzero = (predicted_sign != 0) & (measured_sign != 0)
    summary = {
        "states": int(len(frame)),
        "matching_bins": int(frame[match_bin_column].nunique()),
        "equal_weight_strata": len(stratum_levels),
        "aggregate_within_bin_spearman_fisher_z": aggregate_rho,
        "mean_within_bin_spearman": (
            float(finite_rho.mean()) if finite_rho.size else float("nan")
        ),
        "sign_agreement_fraction": float((predicted_sign == measured_sign).mean()),
        "sign_agreement_fraction_nonzero": (
            float((predicted_sign[nonzero] == measured_sign[nonzero]).mean())
            if nonzero.any()
            else float("nan")
        ),
        "nonzero_sign_states": int(nonzero.sum()),
    }
    return aggregate, correlation_frame, summary


def discounted_downstream_costs(
    reverse_kl: torch.Tensor, valid_mask: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Compute ``sum_{k>=1} gamma^k KL_{t+k}`` for each continuation row."""
    if reverse_kl.shape != valid_mask.shape or reverse_kl.ndim != 2:
        raise ValueError("reverse_kl and valid_mask must share shape [batch, time]")
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if reverse_kl.shape[1] <= 1:
        return torch.zeros(reverse_kl.shape[0], device=reverse_kl.device)
    discounts = torch.pow(
        reverse_kl.new_tensor(float(gamma)),
        torch.arange(1, reverse_kl.shape[1], device=reverse_kl.device),
    )
    downstream = torch.where(
        valid_mask[:, 1:].bool(), reverse_kl[:, 1:], torch.zeros_like(reverse_kl[:, 1:])
    )
    return (downstream * discounts.unsqueeze(0)).sum(dim=-1)


def _plot_quintiles(table: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    x = table["D_quantile"].to_numpy()
    y = table["mean_downstream_gain_measured"].to_numpy()
    # Percentile intervals need not contain the plug-in point estimate for a
    # skewed finite bootstrap distribution; matplotlib requires non-negative
    # error-bar lengths even in that valid case.
    lower = np.maximum(y - table["ci95_low"].to_numpy(), 0.0)
    upper = np.maximum(table["ci95_high"].to_numpy() - y, 0.0)
    axis.errorbar(x, y, yerr=np.vstack((lower, upper)), fmt="o-", capsize=4)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_xticks(x)
    axis.set_xlabel(r"LIFT $\widetilde{D}_t$ quintile (lowest $\rightarrow$ highest)")
    axis.set_ylabel("Mean measured downstream gain")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def analyze_lift_mechanism_csv(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 0,
    local_gain_bins: int = 10,
    position_bins: int = 0,
) -> dict[str, Any]:
    """Create primary and robustness artifacts from an existing state CSV."""
    csv_path = Path(csv_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(csv_path)
    missing = set(REQUIRED_STATE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(
            f"Per-state CSV is missing required columns: {sorted(missing)}"
        )

    effective_position_bins = int(position_bins)
    if effective_position_bins <= 1 and "position_bin" in frame:
        effective_position_bins = int(frame["position_bin"].nunique())
    primary, primary_correlations, primary_summary = analyze_matched_states(
        frame,
        match_bin_column="G_bin",
        d_quantile_column="D_quantile",
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        position_bin_column=("position_bin" if effective_position_bins > 1 else None),
    )
    primary.to_csv(output / "primary_quintiles.csv", index=False)
    primary_correlations.to_csv(output / "primary_within_G_spearman.csv", index=False)
    _plot_quintiles(
        primary,
        output / "primary_downstream_gain_by_D_quintile.png",
        r"Matched $G_t$: downstream gain by LIFT $\widetilde{D}_t$",
    )

    robust_matched = matched_quantile_sample(
        frame,
        match_column="local_gain",
        match_bins=int(local_gain_bins),
        d_bins=5,
        seed=int(seed) + 1,
        position_bins=effective_position_bins,
    )
    robust_matched = robust_matched.rename(
        columns={
            "match_bin": "local_gain_bin",
            "D_quantile_matched": "D_quantile_local_gain",
        }
    )
    robust_matched.to_csv(output / "local_gain_matched_states.csv", index=False)
    robust, robust_correlations, robust_summary = analyze_matched_states(
        robust_matched,
        match_bin_column="local_gain_bin",
        d_quantile_column="D_quantile_local_gain",
        bootstrap_samples=bootstrap_samples,
        seed=int(seed) + 2,
        position_bin_column=("position_bin" if effective_position_bins > 1 else None),
    )
    robust.to_csv(output / "local_gain_matched_quintiles.csv", index=False)
    robust_correlations.to_csv(
        output / "local_gain_matched_within_bin_spearman.csv", index=False
    )
    _plot_quintiles(
        robust,
        output / "local_gain_matched_downstream_gain_by_D_quintile.png",
        r"Matched measured local gain: downstream gain by $\widetilde{D}_t$",
    )
    summary = {
        "primary_matched_on_G_t": primary_summary,
        "robustness_matched_on_measured_local_gain": robust_summary,
        "bootstrap_samples": int(bootstrap_samples),
        "position_bins": effective_position_bins,
        "per_state_csv": str(csv_path),
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    return summary


def _tree_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _snapshot_parameters(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def _restore_optimizer_snapshot(optimizer, snapshot: dict[str, Any]) -> None:
    # Optimizer.load_state_dict may retain aliases to CPU scalar tensors (most
    # notably Adam's ``step``). Clone the snapshot for every intervention so an
    # optimizer step can never mutate the pristine checkpoint state.
    optimizer.load_state_dict(_tree_to_cpu(snapshot))


@torch.no_grad()
def _restore_parameters(model, snapshot: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    if parameters.keys() != snapshot.keys():
        raise ValueError("Student parameter topology changed during the experiment")
    for name, parameter in parameters.items():
        parameter.copy_(
            snapshot[name].to(device=parameter.device, dtype=parameter.dtype)
        )


def _eos_ids(tokenizer) -> int | list[int]:
    value = tokenizer.eos_token_id
    if value is None:
        raise ValueError("The student tokenizer has no EOS token ID")
    return value


def _collect_candidates(
    student,
    teacher,
    tokenizer,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[pd.DataFrame, list[_Candidate], list[_Trajectory]]:
    settings = config["mechanism_validation"]
    selector_cfg = config["selector"]
    rollout_cfg = config["rollout"]
    seed = int(settings.get("seed", config.get("experiment", {}).get("seed", 42)))
    prompt_count = min(int(settings.get("candidate_prompt_count", 16)), len(records))
    prompt_indices = random.Random(seed).sample(range(len(records)), prompt_count)
    batch_size = max(1, int(settings.get("candidate_prompt_batch_size", 4)))
    responses = max(1, int(settings.get("candidate_responses_per_prompt", 4)))
    max_positions = max(1, int(settings.get("candidate_states_per_trajectory", 16)))
    top_k = int(selector_cfg.get("top_k", 16))
    pgt_selector = PGTSelector()
    cmt_selector = CMTSelector(
        gamma=float(selector_cfg.get("cmt_gamma", 1.0)),
        successor_lambda=float(selector_cfg.get("cmt_successor_lambda", 1.0)),
    )
    trajectories: list[_Trajectory] = []
    candidates: list[_Candidate] = []
    for batch_start in range(0, prompt_count, batch_size):
        dataset_indices = prompt_indices[batch_start : batch_start + batch_size]
        batch_records = [records[index] for index in dataset_indices]
        encoded, _ = tokenize_prompts(batch_records, tokenizer, config["data"], device)
        expanded, expanded_indices, response_indices = expand_prompt_batch(
            encoded, dataset_indices, responses
        )
        rollout = generate_on_policy(
            student,
            expanded["input_ids"],
            expanded["attention_mask"],
            max_new_tokens=int(rollout_cfg["max_new_tokens"]),
            temperature=float(rollout_cfg.get("temperature", 1.0)),
            top_p=float(rollout_cfg.get("top_p", 1.0)),
            eos_token_ids=_eos_ids(tokenizer),
            pad_token_id=int(tokenizer.pad_token_id),
            seed=seed + batch_start * responses,
        )
        student_scores, teacher_scores = score_student_teacher_rollout(
            student,
            teacher,
            rollout,
            score_chunk_steps=int(selector_cfg.get("score_chunk_steps", 128)),
            top_k=top_k,
            teacher_temperature=float(
                config.get("opd", {}).get("teacher_temperature", 1.0)
            ),
            micro_batch_size=int(selector_cfg.get("score_micro_batch_size", 1)),
            trim_padding=bool(selector_cfg.get("trim_padding", True)),
            length_bucketed=bool(selector_cfg.get("length_bucketed_scoring", True)),
        )
        if any(
            value is None
            for value in (
                student_scores.top_k_ids,
                student_scores.top_k_log_probs,
                student_scores.candidate_log_probs,
                teacher_scores.top_k_ids,
                teacher_scores.top_k_log_probs,
                teacher_scores.candidate_log_probs,
            )
        ):
            raise AssertionError("Joint scoring did not return the CMT Top-K supports")
        pgt = pgt_selector.compute_scores_from_topk(
            student_scores.top_k_ids,
            teacher_scores.top_k_ids,
            student_scores.top_k_log_probs,
            teacher_scores.candidate_log_probs,
            teacher_scores.top_k_log_probs,
            student_scores.candidate_log_probs,
            rollout.valid_mask,
            token_chunk_size=int(selector_cfg.get("pgt_vocab_chunk_tokens", 2048)),
            gain_support="student_topk",
        )
        lift = cmt_selector.compute_scores(
            pgt, rollout.response_ids, rollout.valid_mask
        )
        gains = lift.diagnostics["gain"].detach().cpu()
        downstream = lift.diagnostics["sequential_gain_raw"].detach().cpu()
        lengths = rollout.valid_mask.long().sum(dim=-1).cpu().tolist()
        for row, (dataset_index, response_index, length) in enumerate(
            zip(expanded_indices, response_indices, lengths)
        ):
            prompt_mask = expanded["attention_mask"][row].bool()
            prompt_ids = tuple(
                int(item)
                for item in expanded["input_ids"][row][prompt_mask]
                .detach()
                .cpu()
                .tolist()
            )
            response_ids = tuple(
                int(item)
                for item in rollout.response_ids[row, : int(length)]
                .detach()
                .cpu()
                .tolist()
            )
            trajectory_id = len(trajectories)
            prompt_id = stable_sample_id(records[dataset_index], dataset_index)
            trajectories.append(
                _Trajectory(prompt_id, prompt_ids, response_ids, int(response_index))
            )
            # A state immediately before an observed EOS is still a valid
            # intervention: a fresh rollout can choose a non-EOS first action.
            # Exclude only the configured horizon's final state, for which no
            # downstream k>=1 state can exist under the LIFT estimand.
            eligible = list(
                range(
                    max(
                        0,
                        min(int(length), int(rollout_cfg["max_new_tokens"]) - 1),
                    )
                )
            )
            if len(eligible) > max_positions:
                local_rng = random.Random(seed + trajectory_id * 104729)
                eligible = sorted(local_rng.sample(eligible, max_positions))
            for position in eligible:
                state_id = f"{prompt_id}:i{dataset_index}:r{response_index}:t{position}"
                candidates.append(
                    _Candidate(
                        state_id=state_id,
                        trajectory_id=trajectory_id,
                        prompt_id=prompt_id,
                        response_index=int(response_index),
                        token_position=int(position),
                        G_t=float(gains[row, position]),
                        D_tilde=float(downstream[row, position]),
                    )
                )
        del rollout, student_scores, teacher_scores, pgt, lift
    frame = pd.DataFrame(
        [
            {
                "candidate_index": index,
                "state_id": item.state_id,
                "prompt_id": item.prompt_id,
                "response_index": item.response_index,
                "token_position": item.token_position,
                "G_t": item.G_t,
                "D_tilde": item.D_tilde,
            }
            for index, item in enumerate(candidates)
        ]
    )
    return frame, candidates, trajectories


def _continuation_cost(
    student,
    teacher,
    prefix: torch.Tensor,
    *,
    samples: int,
    max_new_tokens: int,
    tokenizer,
    rollout_cfg: dict[str, Any],
    selector_cfg: dict[str, Any],
    settings: dict[str, Any],
    seed: int,
    gamma: float,
    teacher_temperature: float,
) -> float:
    prompt_ids = prefix.unsqueeze(0).repeat(int(samples), 1)
    attention = torch.ones_like(prompt_ids)
    rollout = generate_on_policy(
        student,
        prompt_ids,
        attention,
        max_new_tokens=int(max_new_tokens),
        temperature=float(rollout_cfg.get("temperature", 1.0)),
        top_p=float(rollout_cfg.get("top_p", 1.0)),
        eos_token_ids=_eos_ids(tokenizer),
        pad_token_id=int(tokenizer.pad_token_id),
        seed=int(seed),
    )
    reverse_kl = score_reverse_kl_rollout(
        student,
        teacher,
        rollout,
        score_chunk_steps=int(settings.get("kl_score_chunk_steps", 32)),
        micro_batch_size=int(settings.get("kl_score_micro_batch_size", 1)),
        trim_padding=bool(selector_cfg.get("trim_padding", True)),
        length_bucketed=bool(selector_cfg.get("length_bucketed_scoring", True)),
        teacher_temperature=float(teacher_temperature),
    )
    costs = discounted_downstream_costs(reverse_kl, rollout.valid_mask, gamma)
    return float(costs.mean().item())


def run_lift_mechanism_validation(
    config: dict[str, Any], checkpoint: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Run the fixed-checkpoint LIFT/CMT causal mechanism experiment."""
    if not torch.cuda.is_available():
        raise RuntimeError("LIFT mechanism validation requires a CUDA GPU")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint is missing config.json: {checkpoint}")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    resolved = copy.deepcopy(config)
    resolved.setdefault("models", {})["student_path"] = str(checkpoint)
    settings = resolved.setdefault("mechanism_validation", {})
    seed = int(settings.get("seed", resolved.get("experiment", {}).get("seed", 42)))
    seed_everything(seed)
    save_config(resolved, output / "resolved_config.yaml")

    device = torch.device("cuda", 0)
    tokenizer_for_filter = load_student_tokenizer(resolved)
    records, data_files = read_records(
        resolved["data"]["path"], split=resolved["data"].get("split")
    )
    validate_prompt_records(records, resolved["data"])
    records, filter_summary = filter_overlong_prompt_records(
        records, tokenizer_for_filter, resolved["data"]
    )
    del tokenizer_for_filter
    student, teacher, tokenizer, model_metadata = load_models(resolved, device)
    optimizer, fused_optimizer = _make_optimizer(
        (parameter for parameter in student.parameters() if parameter.requires_grad),
        resolved["training"],
    )
    optimizer_source = "fresh_opd_configuration"
    if (checkpoint / "optimizer.pt").is_file():
        restore_optimizer(optimizer, checkpoint, device, model=student)
        optimizer_source = "checkpoint_optimizer_state"
    pristine_parameters = _snapshot_parameters(student)
    pristine_optimizer = _tree_to_cpu(optimizer.state_dict())
    pristine_cpu_rng = torch.get_rng_state().clone()
    pristine_cuda_rng = torch.cuda.get_rng_state(device).clone()
    optimizer.state.clear()

    pool, candidates, trajectories = _collect_candidates(
        student, teacher, tokenizer, records, resolved, device
    )
    pool.to_csv(output / "candidate_states.csv", index=False)
    g_bins = int(settings.get("g_bins", 10))
    if not 10 <= g_bins <= 20:
        raise ValueError("mechanism_validation.g_bins must be between 10 and 20")
    d_bins = int(settings.get("d_quantiles", 5))
    if d_bins != 5:
        raise ValueError("The mechanism-validation design requires five D quantiles")
    position_bins = int(settings.get("position_bins", 0))
    selected = matched_quantile_sample(
        pool,
        match_column="G_t",
        match_bins=g_bins,
        d_bins=d_bins,
        samples_per_cell=settings.get("samples_per_cell"),
        seed=seed + 17,
        position_bins=position_bins,
    ).rename(columns={"match_bin": "G_bin", "D_quantile_matched": "D_quantile"})
    selected.to_csv(output / "selected_states_pre_intervention.csv", index=False)

    rollout_cfg = resolved["rollout"]
    selector_cfg = resolved["selector"]
    gamma = float(selector_cfg.get("cmt_gamma", 1.0))
    teacher_temperature = float(resolved.get("opd", {}).get("teacher_temperature", 1.0))
    # This is deliberately not an independent mechanism knob: candidate
    # scoring and measured consequences must use the identical finite horizon.
    lift_horizon = int(rollout_cfg["max_new_tokens"])
    samples = int(settings.get("continuations", 8))
    if samples <= 0 or lift_horizon <= 1:
        raise ValueError("continuations must be positive and continuation_horizon > 1")
    rows: list[dict[str, Any]] = []
    selected_count = len(selected)
    for experiment_index, state_row in selected.iterrows():
        candidate = candidates[int(state_row["candidate_index"])]
        trajectory = trajectories[candidate.trajectory_id]
        prefix_values = (
            trajectory.prompt_ids + trajectory.response_ids[: candidate.token_position]
        )
        prefix = torch.tensor(prefix_values, dtype=torch.long, device=device)
        remaining = lift_horizon - candidate.token_position
        if remaining <= 1:
            raise ValueError(
                f"State {candidate.state_id} has no downstream horizon under "
                f"continuation_horizon={lift_horizon}"
            )

        _restore_parameters(student, pristine_parameters)
        _restore_optimizer_snapshot(optimizer, pristine_optimizer)
        torch.set_rng_state(pristine_cpu_rng)
        torch.cuda.set_rng_state(pristine_cuda_rng, device=device)
        student.eval()
        local_ids = prefix.unsqueeze(0)
        local_attention = torch.ones_like(local_ids)
        with torch.no_grad():
            kl_before = float(
                reverse_kl_next_token(
                    student,
                    teacher,
                    local_ids,
                    local_attention,
                    teacher_temperature=teacher_temperature,
                )
                .mean()
                .item()
            )
        c_before = _continuation_cost(
            student,
            teacher,
            prefix,
            samples=samples,
            max_new_tokens=remaining,
            tokenizer=tokenizer,
            rollout_cfg=rollout_cfg,
            selector_cfg=selector_cfg,
            settings=settings,
            seed=seed + experiment_index * 200003 + 1009,
            gamma=gamma,
            teacher_temperature=teacher_temperature,
        )

        # The intervention always starts again from the identical checkpoint,
        # including Adam moments when optimizer.pt is available.
        _restore_parameters(student, pristine_parameters)
        _restore_optimizer_snapshot(optimizer, pristine_optimizer)
        torch.set_rng_state(pristine_cpu_rng)
        torch.cuda.set_rng_state(pristine_cuda_rng, device=device)
        student.train()
        optimizer.zero_grad(set_to_none=True)
        local_loss = reverse_kl_next_token(
            student,
            teacher,
            local_ids,
            local_attention,
            teacher_temperature=teacher_temperature,
        ).mean()
        local_loss.backward()
        max_grad_norm = float(resolved["training"].get("max_grad_norm", 1.0))
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        student.eval()
        with torch.no_grad():
            kl_after = float(
                reverse_kl_next_token(
                    student,
                    teacher,
                    local_ids,
                    local_attention,
                    teacher_temperature=teacher_temperature,
                )
                .mean()
                .item()
            )
        c_after = _continuation_cost(
            student,
            teacher,
            prefix,
            samples=samples,
            max_new_tokens=remaining,
            tokenizer=tokenizer,
            rollout_cfg=rollout_cfg,
            selector_cfg=selector_cfg,
            settings=settings,
            seed=seed + experiment_index * 200003 + 100003,
            gamma=gamma,
            teacher_temperature=teacher_temperature,
        )
        row = {
            "state_id": candidate.state_id,
            "prompt_id": candidate.prompt_id,
            "response_index": candidate.response_index,
            "token_position": candidate.token_position,
            "G_t": candidate.G_t,
            "D_tilde": candidate.D_tilde,
            "tilde_D_t": candidate.D_tilde,
            "KL_before": kl_before,
            "KL_after": kl_after,
            "local_gain": kl_before - kl_after,
            "C_before": c_before,
            "C_after": c_after,
            "downstream_gain_measured": c_before - c_after,
            "G_bin": int(state_row["G_bin"]),
            "D_quantile": int(state_row["D_quantile"]),
            "continuation_samples": samples,
            "continuation_steps_including_local_action": remaining,
        }
        if "position_bin" in state_row:
            row["position_bin"] = int(state_row["position_bin"])
        rows.append(row)
        with (output / "per_state.partial.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"[LIFT mechanism] state {experiment_index + 1}/{selected_count}: "
            f"{candidate.state_id} downstream_gain={row['downstream_gain_measured']:.6g}",
            flush=True,
        )

    state_frame = pd.DataFrame(rows)
    state_path = output / "per_state.csv"
    state_frame.to_csv(state_path, index=False)
    (output / "per_state.partial.csv").unlink(missing_ok=True)
    analysis = analyze_lift_mechanism_csv(
        state_path,
        output,
        bootstrap_samples=int(settings.get("bootstrap_samples", 2000)),
        seed=seed + 31,
        local_gain_bins=int(settings.get("local_gain_bins", g_bins)),
        position_bins=position_bins,
    )
    summary = {
        "checkpoint": str(checkpoint),
        "optimizer_source": optimizer_source,
        "fused_optimizer": fused_optimizer,
        "gamma": gamma,
        "teacher_temperature": teacher_temperature,
        "lift_continuation_horizon": lift_horizon,
        "continuations_per_phase": samples,
        "candidate_states": len(pool),
        "selected_states": len(state_frame),
        "G_bins": g_bins,
        "D_quantiles": d_bins,
        "samples_per_cell_actual": int(selected["samples_per_cell"].iloc[0]),
        "position_bins": position_bins,
        "data_files": [str(path) for path in data_files],
        "data_filter": filter_summary,
        "model": model_metadata,
        "analysis": analysis,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    return summary
