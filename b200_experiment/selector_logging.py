from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import torch


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    return float(values[mask].double().mean()) if bool(mask.any()) else 0.0


@torch.no_grad()
def cmt_motivation_summary(
    diagnostics: dict[str, torch.Tensor],
    final_weights: torch.Tensor,
    global_token_indices: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize future-sensitive CMT mechanisms within one allocation group."""
    required = (
        "gain",
        "successor_excess_average",
        "marginal_flux",
        "sequential_gain_raw",
        "sequential_gain_robust",
        "learning_value_robust",
    )
    missing = [key for key in required if key not in diagnostics]
    if missing:
        raise KeyError(f"CMT motivation summary is missing diagnostics: {missing}")
    count = final_weights.numel()
    if count == 0 or global_token_indices.numel() != count:
        raise ValueError("CMT motivation summary requires one index per group token")
    values = {key: diagnostics[key].detach().double().reshape(-1) for key in required}
    weights = final_weights.detach().double().reshape(-1)
    indices = global_token_indices.detach().long().reshape(-1)
    if any(value.numel() != count for value in values.values()):
        raise ValueError("CMT motivation diagnostics must align with final weights")
    if not all(
        bool(torch.isfinite(value).all()) for value in (*values.values(), weights)
    ):
        raise FloatingPointError("CMT motivation summary received non-finite tensors")

    gain = values["gain"]
    future = values["successor_excess_average"]
    flux = values["marginal_flux"]
    d_raw = values["sequential_gain_raw"]
    d_robust = values["sequential_gain_robust"]
    l_robust = values["learning_value_robust"]
    positive_flux = flux > 0
    zero_flux = flux == 0
    log_gain = torch.log1p(gain.clamp_min(0.0))
    gain_boundaries = torch.quantile(
        log_gain,
        torch.linspace(0.0, 1.0, 21, device=log_gain.device, dtype=log_gain.dtype),
    )
    gain_bins = torch.bucketize(log_gain, gain_boundaries[1:-1], right=True)

    pair_records: list[dict[str, Any]] = []
    pair_examples: list[dict[str, Any]] = []
    for gain_bin in range(20):
        gain_mask = gain_bins.eq(gain_bin) & positive_flux
        positions = gain_mask.nonzero(as_tuple=False).flatten()
        if positions.numel() < 8:
            continue
        local_flux = flux.index_select(0, positions)
        flux_boundaries = torch.quantile(
            local_flux,
            torch.linspace(0.0, 1.0, 5, device=flux.device, dtype=flux.dtype),
        )
        flux_bins = torch.bucketize(local_flux, flux_boundaries[1:-1], right=True)
        for flux_bin in range(4):
            joint_positions = positions[flux_bins.eq(flux_bin)]
            if joint_positions.numel() < 4:
                continue
            joint_future = future.index_select(0, joint_positions)
            q25, q75 = torch.quantile(
                joint_future,
                torch.tensor([0.25, 0.75], device=future.device, dtype=future.dtype),
            )
            low_positions = joint_positions[joint_future <= q25]
            high_positions = joint_positions[joint_future >= q75]
            pair_count = min(low_positions.numel(), high_positions.numel())
            if pair_count == 0:
                continue
            # Use equally-sized extremes so every reported gap has a literal
            # pair-count interpretation even when quantile ties are uneven.
            low_order = torch.argsort(future[low_positions], stable=True)
            high_order = torch.argsort(
                future[high_positions], descending=True, stable=True
            )
            low_positions = low_positions[low_order[:pair_count]]
            high_positions = high_positions[high_order[:pair_count]]

            def gap(tensor: torch.Tensor) -> float:
                return float(
                    tensor.index_select(0, high_positions).mean()
                    - tensor.index_select(0, low_positions).mean()
                )

            record = {
                "pair_count": int(pair_count),
                "delta_g": abs(gap(gain)),
                "future_gap": gap(future),
                "D_raw_gap": gap(d_raw),
                "D_robust_gap": gap(d_robust),
                "L_robust_gap": gap(l_robust),
                "final_weight_gap": gap(weights),
            }
            pair_records.append(record)
            high_index = high_positions[torch.argmax(future[high_positions])]
            low_index = low_positions[torch.argmin(future[low_positions])]
            pair_examples.append(
                {
                    "gap": float(future[high_index] - future[low_index]),
                    "high": int(indices[high_index].item()),
                    "low": int(indices[low_index].item()),
                }
            )

    total_pairs = sum(record["pair_count"] for record in pair_records)

    def weighted_metric(key: str) -> float:
        return (
            sum(record[key] * record["pair_count"] for record in pair_records)
            / total_pairs
            if total_pairs
            else 0.0
        )

    gain_q25 = torch.quantile(gain, 0.25)
    future_q25, future_q75 = torch.quantile(
        future,
        torch.tensor([0.25, 0.75], device=future.device, dtype=future.dtype),
    )
    low_g = gain <= gain_q25
    high_future = future >= future_q75
    low_future = future <= future_q25
    cohorts = {
        "low_g_high_future_positive_flux": low_g & high_future & positive_flux,
        "low_g_low_future_positive_flux": low_g & low_future & positive_flux,
        "low_g_high_future_zero_flux": low_g & high_future & zero_flux,
    }
    cohort_payload = {
        name: {
            "count": int(mask.sum()),
            "mean_final_weight": _masked_mean(weights, mask),
        }
        for name, mask in cohorts.items()
    }
    rescued = cohorts["low_g_high_future_positive_flux"]
    baseline = cohorts["low_g_low_future_positive_flux"]
    uncontrolled = cohorts["low_g_high_future_zero_flux"]
    rescue_weight_gap = _masked_mean(weights, rescued) - _masked_mean(weights, baseline)
    future_without_control_gap = _masked_mean(weights, uncontrolled) - _masked_mean(
        weights, baseline
    )
    rescue_rate = (
        float((weights[rescued] > 1.0).double().mean()) if bool(rescued.any()) else 0.0
    )
    summary = {
        "same_g_pair_count": int(total_pairs),
        "same_g_mean_abs_delta_g": weighted_metric("delta_g"),
        "same_g_future_gap": weighted_metric("future_gap"),
        "same_g_D_raw_gap": weighted_metric("D_raw_gap"),
        "same_g_D_robust_gap": weighted_metric("D_robust_gap"),
        "same_g_L_robust_gap": weighted_metric("L_robust_gap"),
        "same_g_final_weight_gap": weighted_metric("final_weight_gap"),
        **cohort_payload,
        "rescue_weight_gap": rescue_weight_gap,
        "future_without_control_gap": future_without_control_gap,
        "rescue_rate": rescue_rate,
    }
    sparse: list[dict[str, Any]] = []
    for pair in sorted(pair_examples, key=lambda item: item["gap"], reverse=True)[:8]:
        sparse.extend(
            (
                {
                    "global_token_index": pair["high"],
                    "selection_reason": "same_g_different_future_high",
                },
                {
                    "global_token_index": pair["low"],
                    "selection_reason": "same_g_different_future_low",
                },
            )
        )
    rescued_positions = rescued.nonzero(as_tuple=False).flatten()
    if rescued_positions.numel():
        rescue_order = torch.argsort(
            weights.index_select(0, rescued_positions), descending=True, stable=True
        )
        for position in rescued_positions[rescue_order[:8]]:
            sparse.append(
                {
                    "global_token_index": int(indices[position].item()),
                    "selection_reason": "low_g_high_future_rescued",
                }
            )
    return summary, sparse


class SelectedTokenLogger:
    """Incremental, crash-safe gzip JSONL logger containing selected positions only."""

    def __init__(
        self,
        output_dir: str | Path,
        tokenizer,
        method: str,
        chunk_steps: int = 50,
        enabled: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.root = Path(output_dir) / "selector_scores"
        self.tokenizer = tokenizer
        self.method = method
        self.chunk_steps = max(1, int(chunk_steps))
        self.enabled = bool(enabled)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            manifest = {
                "format": "gzip JSONL; concatenated gzip members are valid",
                "scope": "selected/accepted response positions only",
                "method": method,
                "loss_support_definition": (
                    "student_topk" if method in {"opd", "ta", "pgt", "cmt"} else None
                ),
                "selector_support_definition": {
                    "opd": "uniform_positions_no_selector_action_support",
                    "ta": "literal_union_student_topk_teacher_topk",
                    "rac": "literal_union_student_topk_teacher_topk",
                    "pgt": "literal_union_student_topk_teacher_topk",
                    "cmt": (
                        "g_student_topk__transition_literal_union_"
                        "student_topk_teacher_topk"
                    ),
                }[method],
                "chunk_steps": self.chunk_steps,
                "distributed_world_size": self.world_size,
                "distributed_file_pattern": (
                    "selected_steps_*_rank-*.jsonl.gz"
                    if self.world_size > 1
                    else "selected_steps_*.jsonl.gz"
                ),
                "common_fields": [
                    "training_step",
                    "sample_id",
                    "dataset_index",
                    "batch_index",
                    "response_position",
                    "token_id",
                    "token_text",
                ],
                "score_fields": {
                    "opd": ["w"],
                    "ta": ["D", "C", "D_norm", "C_norm", "s_TA"],
                    "rac": ["g", "alignment", "R", "M", "V", "z", "w"],
                    "pgt": [
                        "gain",
                        "euclidean_gain",
                        "restricted_reverse_kl",
                        "student_support_mass",
                        "teacher_support_mass",
                        "teacher_tail_mass",
                        "s_PGT",
                    ],
                    "cmt": [
                        "gain",
                        "support_common_mass",
                        "conditional_support_common_mass",
                        "alignment",
                        "transition_weight",
                        "support_coverage",
                        "teacher_deficit",
                        "signed_reachability_shift",
                        "compatibility_weight",
                        "marginal_flux",
                        "downstream_effect",
                        "successor_excess",
                        "sequential_gain",
                        "learning_value",
                        "s_CMT",
                    ],
                }[method],
                "cmt_downstream_semantics": (
                    {
                        "compatibility_weight": (
                            "frozen raw-mass min(1,q(y)/p(y)) confidence"
                        ),
                        "signed_reachability_shift": (
                            "centered conditional log-ratio r_U(y)-E_pU[r_U]"
                        ),
                        "marginal_flux": (
                            "compatibility_weight * signed_reachability_shift"
                        ),
                        "downstream_effect": (
                            "gamma * marginal_flux * successor_excess"
                        ),
                        "teacher_deficit": "diagnostic only; never gates D_t",
                    }
                    if method == "cmt"
                    else None
                ),
            }
            if self.rank == 0:
                with (self.root / "manifest.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(manifest, handle, indent=2, ensure_ascii=False)
                    handle.write("\n")

    def _path(self, step: int) -> Path:
        first = ((step - 1) // self.chunk_steps) * self.chunk_steps + 1
        last = first + self.chunk_steps - 1
        suffix = f"_rank-{self.rank:05d}" if self.world_size > 1 else ""
        return self.root / (f"selected_steps_{first:06d}_{last:06d}{suffix}.jsonl.gz")

    def write(
        self,
        *,
        step: int,
        dataset_indices: list[int],
        sample_ids: list[str],
        response_ids: torch.Tensor,
        selected_mask: torch.Tensor,
        diagnostics: dict[str, Any],
        batch_index_offset: int = 0,
    ) -> int:
        if not self.enabled:
            return 0
        coordinates = selected_mask.nonzero(as_tuple=False)
        if coordinates.numel() == 0:
            return 0
        coords_cpu = coordinates.detach().cpu()
        token_ids = response_ids[selected_mask].detach().cpu().tolist()
        token_texts = self.tokenizer.convert_ids_to_tokens(token_ids)
        keys = {
            "opd": ("w",),
            "ta": ("D", "C", "D_norm", "C_norm", "s_TA"),
            "rac": ("g", "alignment", "R", "M", "V", "z", "w"),
            "pgt": (
                "gain",
                "euclidean_gain",
                "restricted_reverse_kl",
                "student_support_mass",
                "teacher_support_mass",
                "teacher_tail_mass",
                "s_PGT",
            ),
            "cmt": (
                "gain",
                "support_common_mass",
                "conditional_support_common_mass",
                "alignment",
                "transition_weight",
                "support_coverage",
                "teacher_deficit",
                "signed_reachability_shift",
                "compatibility_weight",
                "marginal_flux",
                "downstream_effect",
                "successor_excess",
                "sequential_gain",
                "learning_value",
                "s_CMT",
            ),
        }[self.method]
        values = {
            key: diagnostics[key][selected_mask].detach().float().cpu().tolist()
            for key in keys
        }
        path = self._path(step)
        with gzip.open(path, "at", encoding="utf-8", compresslevel=6) as handle:
            for offset, (batch_index_tensor, position_tensor) in enumerate(coords_cpu):
                batch_index, position = int(batch_index_tensor), int(position_tensor)
                row = {
                    "training_step": step,
                    "sample_id": sample_ids[batch_index],
                    "dataset_index": int(dataset_indices[batch_index]),
                    "batch_index": int(batch_index_offset) + batch_index,
                    "response_position": position,
                    "token_id": int(token_ids[offset]),
                    "token_text": str(token_texts[offset]),
                }
                row.update({key: float(values[key][offset]) for key in keys})
                handle.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
        return len(token_ids)


class TokenScoreStatsLogger:
    """Compact all-valid-token histograms, quantiles, and bounded samples."""

    def __init__(
        self,
        output_dir: str | Path,
        method: str,
        interval: int = 50,
        bins: int = 64,
        raw_sample_size: int = 2048,
        enabled: bool = True,
    ):
        self.root = Path(output_dir) / "token_score_stats"
        self.method = method
        self.interval = max(1, int(interval))
        self.bins = max(2, int(bins))
        self.raw_sample_size = max(0, int(raw_sample_size))
        self.enabled = bool(enabled)
        if method == "ta":
            self.ranges = {
                "D": (0.0, 10.0),
                "C": (0.0, 1.0),
                "s_TA": (0.0, 1.0),
            }
        elif method == "rac":
            self.ranges = {key: (0.0, 1.0) for key in ("g", "alignment", "V", "z", "w")}
        elif method == "opd":
            self.ranges = {"w": (0.0, 1.0)}
        elif method == "iw":
            self.ranges = {
                "w": (0.0, 1.0),
                "iw_weight": (1.0, 1.5),
            }
        elif method == "pgt":
            self.ranges = {
                "s_PGT": (0.0, 10.0),
                "gain": (0.0, 10.0),
                "euclidean_gain": (0.0, 10.0),
                "restricted_reverse_kl": (-10.0, 10.0),
                "student_support_mass": (0.0, 1.0),
                "teacher_support_mass": (0.0, 1.0),
                "teacher_tail_mass": (0.0, 1.0),
            }
        elif method == "cmt":
            self.ranges = {
                "s_CMT": (-100.0, 100.0),
                "gain": (0.0, 10.0),
                "support_reverse_kl": (0.0, 20.0),
                "support_common_mass": (0.0, 1.0),
                "conditional_support_common_mass": (0.0, 1.0),
                "alignment": (0.0, 1.0),
                "transition_weight": (0.0, 1.0),
                "support_coverage": (0.0, 1.0),
                "coverage_correction": (0.0, 1.0),
                "teacher_deficit": (0.0, 1.0),
                "signed_reachability_shift": (-100.0, 100.0),
                "compatibility_weight": (0.0, 1.0),
                "marginal_flux": (-100.0, 100.0),
                "downstream_effect": (-100.0, 100.0),
                "common_mass_derivative": (-10.0, 10.0),
                "R": (-100.0, 100.0),
                "M": (0.0, 100.0),
                "V": (-100.0, 100.0),
                "H": (-100.0, 100.0),
                "successor_excess": (-100.0, 100.0),
                "successor_return": (-100.0, 100.0),
                "successor_mass": (0.0, 100.0),
                "successor_value": (-100.0, 100.0),
                "successor_excess_total": (-100.0, 100.0),
                "successor_excess_average": (-100.0, 100.0),
                "sequential_gain": (-100.0, 100.0),
                "sequential_gain_raw": (-100.0, 100.0),
                "sequential_gain_robust": (-100.0, 100.0),
                "learning_value": (-100.0, 100.0),
                "learning_value_raw": (-100.0, 100.0),
                "learning_value_robust": (-100.0, 100.0),
                "allocation_score": (-100.0, 100.0),
                "correction_kappa": (0.0, 10.0),
                "w_raw": (0.0, 20.0),
                "w": (0.0, 20.0),
                "full_log_ratio_mean": (-20.0, 20.0),
                "full_log_ratio_variance": (0.0, 100.0),
                "full_common_mass": (0.0, 1.0),
            }
        else:
            raise ValueError(f"Unknown token-score method: {method!r}")
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            manifest = {
                "format": "one JSON file per logged optimizer step",
                "scope": "all valid response positions in the global rollout batch",
                "method": method,
                "logged_steps": "step 1, every configured interval, and final step",
                "bins": self.bins,
                "histogram_ranges": self.ranges,
                "raw_sample_size_max": self.raw_sample_size,
                "weight_semantics": {
                    "w_raw": (
                        "unbounded Gibbs reference before bounding in the legacy "
                        "posthoc mode; diagnostic only for direct_bounded_gibbs"
                    ),
                    "w": "final weight used by the OPD loss",
                },
                "cmt_downstream_semantics": (
                    "frozen compatibility times signed centered visitation shift; "
                    "teacher_deficit is diagnostic only"
                    if method == "cmt"
                    else None
                ),
                "note": "D values outside [0,10] are counted in underflow/overflow; normalized quantities use [0,1].",
            }
            (self.root / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    def should_log(self, step: int, final_step: int) -> bool:
        return self.enabled and (
            step == 1 or step == final_step or step % self.interval == 0
        )

    def write(
        self, step: int, final_step: int, diagnostics: dict[str, Any]
    ) -> Path | None:
        if not self.should_log(step, final_step):
            return None
        payload: dict[str, Any] = {
            "step": int(step),
            "method": self.method,
            "scope": "global_valid_response_tokens",
            "scores": {},
        }
        quantile_levels = torch.tensor(
            [0.05, 0.25, 0.50, 0.75, 0.95],
            device=next(
                value.device for value in diagnostics.values() if torch.is_tensor(value)
            ),
        )
        for key, (low, high) in self.ranges.items():
            if key not in diagnostics:
                continue
            values = diagnostics[key].detach().float().reshape(-1)
            values = values[torch.isfinite(values)]
            if values.numel() == 0:
                continue
            clipped = values.clamp(low, high)
            counts = torch.histc(clipped, bins=self.bins, min=low, max=high)
            edges = torch.linspace(
                low, high, self.bins + 1, device=values.device, dtype=torch.float32
            )
            quantiles = torch.quantile(values, quantile_levels)
            sample_count = min(values.numel(), self.raw_sample_size)
            if sample_count:
                indices = torch.linspace(
                    0,
                    values.numel() - 1,
                    sample_count,
                    device=values.device,
                ).long()
                sample = values.index_select(0, indices)
            else:
                sample = values.new_empty((0,))
            payload["scores"][key] = {
                "count": int(values.numel()),
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
                "quantiles": {
                    name: float(value)
                    for name, value in zip(
                        ("q05", "q25", "q50", "q75", "q95"), quantiles
                    )
                },
                "histogram": {
                    "edges": edges.cpu().tolist(),
                    "counts": counts.long().cpu().tolist(),
                    "underflow": int((values < low).sum()),
                    "overflow": int((values > high).sum()),
                },
                "sample": sample.cpu().tolist(),
            }
        destination = self.root / f"step-{step:06d}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination


class CMTTokenAuditLogger:
    """Sparse allocation-group-aware CMT audit with optional heatmaps."""

    _REASON_KEYS = (
        ("gain", "global_top_gain", "global_gain_rank"),
        ("w_raw", "global_top_w_raw", "global_raw_weight_rank"),
        ("w", "global_top_w", "global_final_weight_rank"),
    )
    _VALUE_KEYS = (
        "gain",
        "successor_return",
        "successor_mass",
        "successor_value",
        "successor_excess_total",
        "successor_excess_average",
        "signed_reachability_shift",
        "compatibility_weight",
        "marginal_flux",
        "downstream_effect",
        "sequential_gain",
        "sequential_gain_raw",
        "sequential_gain_robust",
        "learning_value",
        "learning_value_raw",
        "learning_value_robust",
        "correction_kappa",
        "successor_excess",
        "transition_weight",
        "w_raw",
        "w",
    )

    def __init__(
        self,
        output_dir: str | Path,
        tokenizer,
        *,
        enabled: bool = False,
        interval: int = 150,
        top_k: int = 50,
        context_radius: int = 32,
        heatmap_enabled: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.root = Path(output_dir) / "cmt_token_audit"
        self.important_root = self.root / "important_tokens"
        self.heatmap_root = self.root / "heatmaps"
        self.motivation_root = self.root / "motivation_summaries"
        self.tokenizer = tokenizer
        self.enabled = bool(enabled)
        self.interval = int(interval)
        self.top_k = int(top_k)
        self.context_radius = int(context_radius)
        if self.interval <= 0:
            raise ValueError("logging.cmt_token_audit_interval must be positive")
        if self.top_k <= 0:
            raise ValueError("logging.cmt_token_audit_top_k must be positive")
        if self.context_radius < 0:
            raise ValueError("logging.cmt_token_context_radius must be non-negative")
        self.heatmap_enabled = bool(heatmap_enabled)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if not self.enabled:
            return
        self.important_root.mkdir(parents=True, exist_ok=True)
        self.motivation_root.mkdir(parents=True, exist_ok=True)
        if self.heatmap_enabled:
            self.heatmap_root.mkdir(parents=True, exist_ok=True)
        if self.rank == 0:
            manifest = {
                "format": "one gzip JSONL shard per rank and audit step",
                "selection_scope": (
                    "global valid processed response tokens within the allocation "
                    "group that produced the audited optimizer step"
                ),
                "selection": (
                    "deduplicated union of allocation-group Top-K gain, w_raw, w, "
                    "and bounded motivation examples"
                ),
                "interval": self.interval,
                "top_k": self.top_k,
                "context_radius_tokens": self.context_radius,
                "distributed_world_size": self.world_size,
                "weight_semantics": {
                    "w_raw": (
                        "unbounded Gibbs reference at the active allocation KL; "
                        "diagnostic only in direct_bounded_gibbs mode"
                    ),
                    "w": "final weight used by the OPD loss",
                },
                "ranking_scope": "per allocation group; never across different beta values",
                "final_weight_tie_break": "learning_value_robust descending",
                "raw_aliases": {
                    "sequential_gain": "sequential_gain_raw",
                    "successor_R": "successor_excess_total (not successor_return)",
                },
                "downstream_semantics": {
                    "compatibility_weight": (
                        "frozen raw-mass min(1,q(y)/p(y)) confidence"
                    ),
                    "signed_reachability_shift": (
                        "centered conditional log-ratio r_U(y)-E_pU[r_U]"
                    ),
                    "marginal_flux": (
                        "compatibility_weight * signed_reachability_shift"
                    ),
                    "downstream_effect": ("gamma * marginal_flux * successor_excess"),
                    "teacher_deficit": "diagnostic only; never gates D_t",
                },
                "heatmap_enabled": self.heatmap_enabled,
            }
            (self.root / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    def audit_steps(
        self, first_step: int, last_step: int, final_step: int
    ) -> list[int]:
        """Return scheduled steps crossed by one multi-optimizer rollout."""
        if not self.enabled or last_step < first_step:
            return []
        steps: set[int] = set()
        first_multiple = (max(first_step, 1) + self.interval - 1) // self.interval
        first_multiple *= self.interval
        steps.update(range(first_multiple, last_step + 1, self.interval))
        if first_step <= 1 <= last_step:
            steps.add(1)
        if first_step <= final_step <= last_step:
            steps.add(int(final_step))
        return sorted(steps)

    @staticmethod
    def _group_top_ranks(
        values: torch.Tensor,
        group_indices: torch.Tensor,
        optimizer_steps: torch.Tensor,
        processed: torch.Tensor,
        step: int,
        top_k: int,
        *,
        tie_breaker: torch.Tensor | None = None,
    ) -> dict[int, int]:
        finite = torch.isfinite(values)
        eligible = (
            (
                finite
                & processed.bool()
                & optimizer_steps.long().eq(int(step))
                & group_indices.long().ge(0)
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        if eligible.numel() == 0:
            return {}
        ranks: dict[int, int] = {}
        for group_index in torch.unique(group_indices.index_select(0, eligible)):
            group_eligible = eligible[
                group_indices.index_select(0, eligible).eq(group_index)
            ]
            if tie_breaker is not None:
                tie_order = torch.argsort(
                    tie_breaker.index_select(0, group_eligible),
                    descending=True,
                    stable=True,
                )
                group_eligible = group_eligible.index_select(0, tie_order)
            order = torch.argsort(
                values.index_select(0, group_eligible),
                descending=True,
                stable=True,
            )
            chosen = group_eligible.index_select(0, order[: min(top_k, order.numel())])
            ranks.update(
                {int(index): rank + 1 for rank, index in enumerate(chosen.tolist())}
            )
        return ranks

    def write_motivation_summary(
        self,
        *,
        step: int,
        rollout_id: int,
        correction: dict[str, Any],
        allocation_groups: list[dict[str, Any]],
    ) -> Path | None:
        if not self.enabled or self.rank != 0:
            return None
        destination = self.motivation_root / f"step-{int(step):06d}.json"
        temporary = destination.with_suffix(".json.tmp")
        payload = {
            "step": int(step),
            "rollout_id": int(rollout_id),
            "scope": "per_allocation_group",
            "rollout_correction": correction,
            "allocation_groups": allocation_groups,
        }
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    def _decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        try:
            return str(
                self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
        except TypeError:
            return str(self.tokenizer.decode(token_ids, skip_special_tokens=False))

    def write(
        self,
        *,
        step: int,
        rollout_id: int,
        sample_ids: list[str],
        dataset_indices: list[int],
        response_indices: list[int],
        response_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        local_diagnostics: dict[str, Any],
        global_diagnostics: dict[str, Any],
        global_start: int,
        batch_index_offset: int = 0,
        max_response_length: int | None = None,
        extra_selections: list[dict[str, Any]] | None = None,
    ) -> Path | None:
        if not self.enabled:
            return None
        for key in (
            "gain",
            "w_raw",
            "w",
            "learning_value_robust",
            "allocation_group_index",
            "allocation_optimizer_step",
            "allocation_processed",
        ):
            if key not in global_diagnostics:
                raise KeyError(f"CMT audit requires global diagnostic {key!r}")
        local_count = int(valid_mask.sum().item())
        global_size = int(global_diagnostics["gain"].numel())
        global_end = int(global_start) + local_count
        if global_end > global_size:
            raise ValueError(
                "CMT audit local/global valid-token layout is inconsistent"
            )

        group_indices = global_diagnostics["allocation_group_index"].reshape(-1)
        optimizer_steps = global_diagnostics["allocation_optimizer_step"].reshape(-1)
        processed = global_diagnostics["allocation_processed"].reshape(-1).bool()
        robust_score = global_diagnostics["learning_value_robust"].reshape(-1)
        rank_maps: dict[str, dict[int, int]] = {}
        reasons: dict[int, list[str]] = {}
        for value_key, reason, rank_key in self._REASON_KEYS:
            rank_map = self._group_top_ranks(
                global_diagnostics[value_key].detach().reshape(-1),
                group_indices,
                optimizer_steps,
                processed,
                step,
                self.top_k,
                tie_breaker=robust_score if value_key == "w" else None,
            )
            rank_maps[rank_key] = rank_map
            for global_index in rank_map:
                reasons.setdefault(global_index, []).append(reason)
        for selection in extra_selections or []:
            global_index = int(selection["global_token_index"])
            if not 0 <= global_index < global_size:
                raise ValueError(
                    "Sparse CMT audit index is outside the global token layout"
                )
            if not bool(processed[global_index]):
                continue
            if int(optimizer_steps[global_index]) != int(step):
                continue
            reason = str(selection["selection_reason"])
            if reason not in reasons.setdefault(global_index, []):
                reasons[global_index].append(reason)

        local_global_indices = sorted(
            index for index in reasons if int(global_start) <= index < global_end
        )
        coordinates = valid_mask.nonzero(as_tuple=False).detach().cpu()
        response_ids_cpu = response_ids.detach().cpu()
        response_lengths = valid_mask.sum(dim=-1).detach().cpu().tolist()
        diagnostic_values: dict[str, torch.Tensor] = {}
        for key in self._VALUE_KEYS:
            value = local_diagnostics.get(key)
            if torch.is_tensor(value):
                diagnostic_values[key] = value.detach().float().cpu()
        raw_global = global_diagnostics["w_raw"].detach().double().reshape(-1)
        final_global = global_diagnostics["w"].detach().double().reshape(-1)
        output_path = self.important_root / (
            f"step-{int(step):06d}_rank-{self.rank:05d}.jsonl.gz"
        )
        with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=6) as handle:
            for global_index in local_global_indices:
                local_valid_index = global_index - int(global_start)
                row_index, position = map(int, coordinates[local_valid_index].tolist())
                response_length = int(response_lengths[row_index])
                start = max(0, position - self.context_radius)
                end = min(response_length, position + self.context_radius + 1)
                ids = response_ids_cpu[row_index]
                token_id = int(ids[position])
                token_text = self._decode([token_id])
                raw_weight = float(diagnostic_values["w_raw"][row_index, position])
                final_weight = float(diagnostic_values["w"][row_index, position])
                allocation_group_index = int(group_indices[global_index].item())
                allocation_optimizer_step = int(optimizer_steps[global_index].item())
                group_mask = (
                    processed
                    & group_indices.eq(allocation_group_index)
                    & optimizer_steps.eq(allocation_optimizer_step)
                )
                raw_sum = float(raw_global[group_mask].sum())
                final_sum = float(final_global[group_mask].sum())
                clipped = (
                    bool(response_length >= int(max_response_length))
                    if max_response_length is not None
                    else False
                )
                row: dict[str, Any] = {
                    "training_step": int(step),
                    "optimizer_step": allocation_optimizer_step,
                    "rollout_id": int(rollout_id),
                    "ppo_group_index": allocation_group_index,
                    "allocation_id": (
                        f"rollout-{int(rollout_id):06d}:"
                        f"ppo-{allocation_group_index:04d}"
                    ),
                    "rank": self.rank,
                    "sample_id": str(sample_ids[row_index]),
                    "dataset_index": int(dataset_indices[row_index]),
                    "batch_index": int(batch_index_offset) + row_index,
                    "response_index": int(response_indices[row_index]),
                    "response_position": position,
                    "response_length": response_length,
                    "response_position_fraction": position
                    / max(response_length - 1, 1),
                    "token_id": token_id,
                    "token_text": token_text,
                    "left_context_text": self._decode(ids[start:position].tolist()),
                    "token_context_text": token_text,
                    "right_context_text": self._decode(
                        ids[position + 1 : end].tolist()
                    ),
                    "selection_reason": reasons[global_index],
                    "selection_reasons": reasons[global_index],
                    "global_gain_rank": rank_maps["global_gain_rank"].get(global_index),
                    "global_raw_weight_rank": rank_maps["global_raw_weight_rank"].get(
                        global_index
                    ),
                    "global_final_weight_rank": rank_maps[
                        "global_final_weight_rank"
                    ].get(global_index),
                    "allocation_group_gain_rank": rank_maps["global_gain_rank"].get(
                        global_index
                    ),
                    "allocation_group_raw_weight_rank": rank_maps[
                        "global_raw_weight_rank"
                    ].get(global_index),
                    "allocation_group_final_weight_rank": rank_maps[
                        "global_final_weight_rank"
                    ].get(global_index),
                    "raw_token_probability": raw_weight / max(raw_sum, 1e-30),
                    "final_token_probability": final_weight / max(final_sum, 1e-30),
                    "final_weight": final_weight,
                    "response_was_clipped": clipped,
                }
                for key in self._VALUE_KEYS:
                    value = diagnostic_values.get(key)
                    row[key] = (
                        float(value[row_index, position]) if value is not None else None
                    )
                handle.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )

        if self.heatmap_enabled:
            self._write_heatmap(
                step=step,
                sample_ids=sample_ids,
                valid_mask=valid_mask,
                local_diagnostics=local_diagnostics,
                global_diagnostics=global_diagnostics,
                global_start=global_start,
                rank_maps=rank_maps,
                coordinates=coordinates,
                response_lengths=response_lengths,
            )
        return output_path

    def _write_heatmap(
        self,
        *,
        step: int,
        sample_ids: list[str],
        valid_mask: torch.Tensor,
        local_diagnostics: dict[str, Any],
        global_diagnostics: dict[str, Any],
        global_start: int,
        rank_maps: dict[str, dict[int, int]],
        coordinates: torch.Tensor,
        response_lengths: list[int],
    ) -> None:
        # Lazy import is intentional: normal training never initializes matplotlib.
        import matplotlib.pyplot as plt
        import numpy as np

        gain = local_diagnostics["gain"].detach().float().cpu()
        valid_cpu = valid_mask.detach().cpu()
        heat = torch.log1p(gain.clamp_min(0)).numpy()
        heat[~valid_cpu.numpy()] = np.nan
        global_gain = global_diagnostics["gain"].detach().float()
        global_scale = torch.log1p(global_gain.clamp_min(0))
        vmax = (
            float(torch.quantile(global_scale, 0.995)) if global_scale.numel() else 1.0
        )
        if not math.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        figure_height = max(4.0, min(16.0, 0.18 * gain.shape[0] + 2.0))
        figure, axis = plt.subplots(figsize=(14, figure_height))
        image = axis.imshow(
            heat, aspect="auto", interpolation="nearest", vmin=0.0, vmax=vmax
        )
        top_raw_positions: list[dict[str, Any]] = []
        raw_rank_map = rank_maps["global_raw_weight_rank"]
        local_end = global_start + coordinates.shape[0]
        marker_x: list[int] = []
        marker_y: list[int] = []
        for global_index, raw_rank in raw_rank_map.items():
            if global_start <= global_index < local_end:
                row, position = map(
                    int, coordinates[global_index - global_start].tolist()
                )
                marker_x.append(position)
                marker_y.append(row)
                top_raw_positions.append(
                    {
                        "row": row,
                        "response_position": position,
                        "sample_id": str(sample_ids[row]),
                        "global_raw_weight_rank": int(raw_rank),
                    }
                )
        if marker_x:
            axis.scatter(
                marker_x,
                marker_y,
                marker="s",
                facecolors="none",
                edgecolors="cyan",
                linewidths=1.2,
                s=45,
                label="global Top-K w_raw",
            )
            axis.legend(loc="upper right")
        axis.set_xlabel("response_position")
        axis.set_ylabel("local rollout / trajectory")
        axis.set_title(f"CMT log1p(gain), step={int(step)}, rank={self.rank}")
        figure.colorbar(image, ax=axis, label="log1p(gain)")
        figure.tight_layout()
        image_path = self.heatmap_root / (
            f"step-{int(step):06d}_rank-{self.rank:05d}_gain_heatmap.png"
        )
        figure.savefig(image_path, dpi=160)
        plt.close(figure)
        metadata = {
            "step": int(step),
            "rank": self.rank,
            "sample_ids": [str(value) for value in sample_ids],
            "response_lengths": [int(value) for value in response_lengths],
            "gain_color_scale": {
                "transform": "log1p(max(gain, 0))",
                "vmin": 0.0,
                "vmax": vmax,
                "vmax_quantile": 0.995,
                "quantile_scope": "global_valid_tokens",
            },
            "top_raw_weight_positions": top_raw_positions,
        }
        metadata_path = image_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
