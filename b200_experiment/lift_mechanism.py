"""Fixed-checkpoint LIFT (repository CMT) mechanism validation.

Run with ``python -m b200_experiment.lift_mechanism --help``. This deliberately
uses independent HF workers for disjoint interventions, with one globally
matched sample and no gradient synchronization or inference-engine weight cache.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

from .config import apply_overrides, load_config, resolve_runtime_paths, save_config
from .lift_mechanism_analysis import analyze, matched_sample
from .scoring import (
    generate_on_policy,
    position_ids_from_mask,
    score_student_teacher_rollout,
    supports_response_only_logits,
)
from .selectors.cmt_selector import CMTSelector, robust_cmt_correction
from .selectors.pgt_selector import PGTSelector


def stream_seed(seed: int, state_id: str, phase: str) -> int:
    payload = f"{seed}:{state_id}:{phase}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**62)


def cpu_snapshot(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return type(value)((key, cpu_snapshot(item)) for key, item in value.items())
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


class CheckpointReset:
    """An immutable CPU copy, including buffers and all optimizer moments/groups."""

    def __init__(self, student, optimizer):
        self.student, self.optimizer = student, optimizer
        self.model_state = cpu_snapshot(student.state_dict())
        self.optimizer_state = cpu_snapshot(optimizer.state_dict())

    def restore(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.student.load_state_dict(self.model_state, strict=True)
        # load_state_dict may alias CPU tensors; never hand it the saved copy.
        self.optimizer.load_state_dict(copy.deepcopy(self.optimizer_state))
        self.student.eval()


def reverse_kl(
    student_logits, teacher_logits, student_temperature=1.0, teacher_temperature=1.0
):
    """Exact vocabulary reverse KL, with gradients through p and log(p)."""
    log_p = torch.log_softmax(student_logits.float() / student_temperature, dim=-1)
    log_q = torch.log_softmax(
        teacher_logits.detach().float() / teacher_temperature, dim=-1
    )
    return (log_p.exp() * (log_p - log_q)).sum(-1)


def prefix_logits(model, prefix: torch.Tensor):
    mask = torch.ones_like(prefix)
    kwargs = {"logits_to_keep": 1} if supports_response_only_logits(model) else {}
    return model(
        input_ids=prefix,
        attention_mask=mask,
        position_ids=position_ids_from_mask(mask),
        use_cache=False,
        return_dict=True,
        **kwargs,
    ).logits[:, -1, :]


def discounted_downstream_cost(kl: torch.Tensor, valid: torch.Tensor, gamma: float):
    """Columns are states s_t,s_{t+1},...; exclude local s_t, stop at EOS.

    The state predicting EOS is included, the state *after* EOS is absent.
    A rollout with one action (immediate EOS or zero remaining horizon) costs 0.
    """
    if kl.shape != valid.shape or kl.ndim != 2:
        raise ValueError("KL and valid mask must have the same [rollout, time] shape")
    discounts = gamma ** torch.arange(1, kl.shape[1], device=kl.device, dtype=kl.dtype)
    return (torch.where(valid[:, 1:], kl[:, 1:], 0.0) * discounts).sum(-1)


class MechanismExperiment:
    def __init__(self, student, teacher, tokenizer, optimizer, config):
        self.student, self.teacher, self.tokenizer, self.optimizer = (
            student,
            teacher,
            tokenizer,
            optimizer,
        )
        self.config = config
        self.settings = config["mechanism"]
        self.device = next(student.parameters()).device
        self.seed = int(self.settings.get("seed", config["experiment"].get("seed", 42)))
        self.gamma = float(config["selector"].get("cmt_gamma", 1.0))
        self.horizon = int(config["rollout"]["max_new_tokens"])
        self.student_temperature = float(config["rollout"].get("temperature", 1.0))
        self.teacher_temperature = float(
            config.get("opd", {}).get("teacher_temperature", 1.0)
        )
        self.top_k = int(config["selector"].get("top_k", 16))
        if self.student_temperature <= 0 or self.teacher_temperature <= 0:
            raise ValueError(
                "Scoring and stochastic rollout temperatures must be positive"
            )
        if float(config["rollout"].get("top_p", 1.0)) != 1.0:
            raise ValueError(
                "LIFT requires full-policy continuations: rollout.top_p must be 1"
            )
        if self.horizon < 2 or not 0 <= self.gamma <= 1:
            raise ValueError("Need max_new_tokens >= 2 and gamma in [0,1]")
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        self.student.eval()  # Disable dropout, including in the differentiable intervention.
        generation_config = getattr(student, "generation_config", None)
        self.eos = getattr(generation_config, "eos_token_id", None)
        if self.eos is None:
            self.eos = tokenizer.eos_token_id
        if self.eos is None:
            raise ValueError("Student must define EOS token IDs")
        self.reset = CheckpointReset(student, optimizer)

    def generate(self, prefix, *, steps, count, seed):
        return generate_on_policy(
            self.student,
            prefix.expand(count, -1),
            torch.ones_like(prefix).expand(count, -1),
            max_new_tokens=steps,
            temperature=self.student_temperature,
            top_p=1.0,
            eos_token_ids=self.eos,
            pad_token_id=self.tokenizer.pad_token_id,
            seed=seed,
        )

    def score(self, rollout, *, full_kl=False):
        return score_student_teacher_rollout(
            self.student,
            self.teacher,
            rollout,
            top_k=self.top_k,
            student_temperature=self.student_temperature,
            teacher_temperature=self.teacher_temperature,
            score_chunk_steps=int(
                self.config["selector"].get("score_chunk_steps", 128)
            ),
            micro_batch_size=1,
            compute_full_vocab_metrics=full_kl,
        )

    def collect_candidates(self, records):
        from .data import stable_sample_id, tokenize_prompts

        candidates, gains, raw_scores = [], [], []
        rng = np.random.default_rng(self.seed)
        responses = int(self.settings.get("candidate_responses", 2))
        states_per_response = int(self.settings.get("states_per_response", 32))
        selector = CMTSelector(
            gamma=self.gamma,
            successor_lambda=float(
                self.config["selector"].get("cmt_successor_lambda", 1.0)
            ),
        )
        offset = 0
        for prompt_index, record in enumerate(records):
            encoded, _ = tokenize_prompts(
                [record], self.tokenizer, self.config["data"], self.device
            )
            prompt_id = stable_sample_id(record, prompt_index)
            for response in range(responses):
                trajectory_id = f"p{prompt_index}:r{response}"
                rollout = self.generate(
                    encoded["input_ids"],
                    steps=self.horizon,
                    count=1,
                    seed=stream_seed(self.seed, trajectory_id, "candidate"),
                )
                p, q = self.score(rollout)
                support = PGTSelector().compute_scores_from_topk(
                    p.top_k_ids,
                    q.top_k_ids,
                    p.top_k_log_probs,
                    q.candidate_log_probs,
                    q.top_k_log_probs,
                    p.candidate_log_probs,
                    rollout.valid_mask,
                    gain_support="student_topk",
                    token_chunk_size=int(
                        self.config["selector"].get("pgt_vocab_chunk_tokens", 2048)
                    ),
                )
                scored = selector.compute_scores(
                    support, rollout.response_ids, rollout.valid_mask
                )
                gain = scored.diagnostics["gain"][rollout.valid_mask].cpu()
                raw = scored.diagnostics["sequential_gain_raw"][
                    rollout.valid_mask
                ].cpu()
                gains.append(gain)
                raw_scores.append(raw)
                positions = rng.choice(
                    len(gain), min(states_per_response, len(gain)), replace=False
                )
                for position in sorted(positions.tolist()):
                    candidates.append(
                        {
                            "state_id": f"{trajectory_id}:t{position}",
                            "prompt_id": prompt_id,
                            "prompt_index": prompt_index,
                            "response_index": response,
                            "token_position": position,
                            "G_t": float(gain[position]),
                            "D_raw": float(raw[position]),
                            "score_index": offset + position,
                            "downstream_horizon": self.horizon - position - 1,
                            "prefix_ids": rollout.input_ids[
                                0, : rollout.prompt_width + position
                            ].tolist(),
                        }
                    )
                offset += len(gain)
            print(
                f"Scored prompts {prompt_index + 1}/{len(records)}; candidate states={len(candidates)}",
                flush=True,
            )
        cfg = self.config["selector"]
        corrected, _, kappa, _ = robust_cmt_correction(
            torch.cat(gains),
            torch.cat(raw_scores),
            mode=cfg.get("cmt_correction_mode", "none"),
            quantile=float(cfg.get("cmt_correction_quantile", 0.99)),
        )
        for state in candidates:
            state["D_tilde"] = float(corrected[state.pop("score_index")])
            state["correction_kappa"] = kappa
        return pd.DataFrame(candidates)

    @torch.no_grad()
    def rollout_costs(self, prefix, *, horizon: int, seed: int):
        count = int(self.settings.get("rollouts", 8))
        batch = int(self.settings.get("rollout_batch_size", 1))
        costs = []
        for start in range(0, count, batch):
            # H downstream decision states require H+1 sampled actions, counting
            # the root action. This preserves LIFT's original response cap.
            rollout = self.generate(
                prefix,
                steps=horizon + 1,
                count=min(batch, count - start),
                seed=seed + start,
            )
            p, _ = self.score(rollout, full_kl=True)
            if p.full_log_ratio_mean is None:
                raise AssertionError("Full-vocabulary KL scoring was not returned")
            values = discounted_downstream_cost(
                -p.full_log_ratio_mean, rollout.valid_mask, self.gamma
            )
            costs.extend(values.double().cpu().tolist())
        if not np.isfinite(costs).all():
            raise FloatingPointError("Non-finite downstream rollout cost")
        return costs

    def intervene(self, state):
        prefix = torch.tensor(
            [state["prefix_ids"]], device=self.device, dtype=torch.long
        )
        before_seed = stream_seed(self.seed, state["state_id"], "before")
        after_seed = stream_seed(self.seed, state["state_id"], "after")
        horizon = int(state["downstream_horizon"])
        self.reset.restore()
        try:
            with torch.no_grad():
                teacher_logits = prefix_logits(self.teacher, prefix).detach()
                kl_before = float(
                    reverse_kl(
                        prefix_logits(self.student, prefix),
                        teacher_logits,
                        self.student_temperature,
                        self.teacher_temperature,
                    ).item()
                )
            before = self.rollout_costs(prefix, horizon=horizon, seed=before_seed)
            self.reset.restore()
            loss = reverse_kl(
                prefix_logits(self.student, prefix),
                teacher_logits,
                self.student_temperature,
                self.teacher_temperature,
            ).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite local intervention loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.student.parameters(),
                float(self.config["training"].get("max_grad_norm", 1.0)),
                error_if_nonfinite=True,
            )
            self.optimizer.step()  # Exactly one update, solely on local reverse KL.
            self.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                kl_after = float(
                    reverse_kl(
                        prefix_logits(self.student, prefix),
                        teacher_logits,
                        self.student_temperature,
                        self.teacher_temperature,
                    ).item()
                )
            after = self.rollout_costs(prefix, horizon=horizon, seed=after_seed)
            result = {key: value for key, value in state.items() if key != "prefix_ids"}
            result.update(
                KL_before=kl_before,
                KL_after=kl_after,
                local_gain=kl_before - kl_after,
                C_before=float(np.mean(before)),
                C_after=float(np.mean(after)),
                downstream_gain_measured=float(np.mean(before) - np.mean(after)),
                before_seed=before_seed,
                after_seed=after_seed,
                M=len(before),
                gradient_norm=float(gradient_norm),
                C_before_std=float(np.std(before, ddof=1)) if len(before) > 1 else 0.0,
                C_after_std=float(np.std(after, ddof=1)) if len(after) > 1 else 0.0,
            )
            if not np.isfinite([kl_before, kl_after]).all():
                raise FloatingPointError("Non-finite local KL")
            return result, {
                "state_id": state["state_id"],
                "before": before,
                "after": after,
            }
        finally:
            self.reset.restore()


def load_experiment(config, checkpoint: Path, device: torch.device):
    """Load the identical frozen starting point in the coordinator or a worker."""
    from .models import load_models
    from .resume import restore_optimizer
    from .trainer import _make_optimizer

    student, teacher, tokenizer, assets = load_models(config, device)
    training = dict(config["training"])
    if device.type != "cuda":
        training["fused_optimizer"] = False
    optimizer, fused = _make_optimizer(
        [p for p in student.parameters() if p.requires_grad], training
    )
    restored = (
        restore_optimizer(optimizer, checkpoint, device, model=student)
        if (checkpoint / "optimizer.pt").exists()
        else None
    )
    return (
        MechanismExperiment(student, teacher, tokenizer, optimizer, config),
        assets,
        fused,
        restored,
    )


def write_states(path: Path, states: list[dict]):
    # Preserve scores and integer IDs/seeds exactly across worker boundaries.
    with path.open("w", encoding="utf-8") as handle:
        for state in states:
            handle.write(json.dumps(state, allow_nan=False) + "\n")


def read_states(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def execute_interventions(experiment, states: list[dict], output: Path, *, label=""):
    results = []
    with (output / "rollout_costs.jsonl").open("w", encoding="utf-8") as costs_file:
        for index, state in enumerate(states):
            result, costs = experiment.intervene(state)
            results.append(result)
            pd.DataFrame([result]).to_csv(
                output / "per_state.csv", mode="a", header=index == 0, index=False
            )
            costs_file.write(json.dumps(costs, allow_nan=False) + "\n")
            costs_file.flush()
            print(
                f"{label}Interventions {index + 1}/{len(states)}; "
                f"downstream gain={result['downstream_gain_measured']:.6g}",
                flush=True,
            )
    return pd.DataFrame(results)


def intervention_worker(
    rank: int, config, checkpoint: Path, output: Path, devices: list[str]
):
    """One model/optimizer replica per GPU; no DDP or gradient collectives."""
    device = torch.device(devices[rank])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    states = read_states(output / "selected_states.jsonl")[rank :: len(devices)]
    worker_output = output / "workers" / f"rank_{rank:03d}"
    worker_output.mkdir(parents=True, exist_ok=False)
    experiment, _, _, restored = load_experiment(config, checkpoint, device)
    execute_interventions(experiment, states, worker_output, label=f"[worker {rank}] ")
    # Written last: incomplete or failed shards must never enter final analysis.
    (worker_output / "complete.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "device": str(device),
                "n_states": len(states),
                "optimizer_step": restored.step if restored else None,
            }
        )
        + "\n"
    )


def merge_intervention_shards(
    output: Path, states: list[dict], workers: int
) -> pd.DataFrame:
    """Validate exact coverage and merge in original selection order, once."""
    ordered_ids = [state["state_id"] for state in states]
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("Selected states contain duplicate IDs")
    frames, costs_by_id = [], {}
    for rank in range(workers):
        worker_output = output / "workers" / f"rank_{rank:03d}"
        completed = json.loads((worker_output / "complete.json").read_text())
        expected = set(ordered_ids[rank::workers])
        if completed["rank"] != rank or completed["n_states"] != len(expected):
            raise ValueError(
                f"Worker {rank} completion manifest does not match its assignment"
            )
        frame = pd.read_csv(
            worker_output / "per_state.csv",
            dtype={"state_id": str, "prompt_id": str},
            float_precision="round_trip",
            keep_default_na=False,
        )
        costs = read_states(worker_output / "rollout_costs.jsonl")
        cost_ids = [item["state_id"] for item in costs]
        if (
            len(frame) != len(expected)
            or frame.state_id.duplicated().any()
            or set(frame.state_id) != expected
            or len(cost_ids) != len(expected)
            or set(cost_ids) != expected
        ):
            raise ValueError(
                f"Worker {rank} has missing, duplicate, or unassigned states"
            )
        frames.append(frame)
        costs_by_id.update({item["state_id"]: item for item in costs})
    merged = (
        pd.concat(frames, ignore_index=True)
        .set_index("state_id")
        .loc[ordered_ids]
        .reset_index()
    )
    merged.to_csv(output / "per_state.csv", index=False)
    write_states(
        output / "rollout_costs.jsonl",
        [costs_by_id[state_id] for state_id in ordered_ids],
    )
    return merged


def parallel_interventions(config, checkpoint: Path, output: Path, devices: list[str]):
    states = read_states(output / "selected_states.jsonl")
    if not devices or len(devices) > len(states):
        raise ValueError(
            "Worker count must be between one and the number of selected states"
        )
    # spawn propagates failures and terminates other workers; never wait forever
    # for a filesystem barrier or merge a partial experiment.
    mp.spawn(
        intervention_worker,
        args=(config, checkpoint, output, devices),
        nprocs=len(devices),
        join=True,
    )
    return merge_intervention_shards(output, states, len(devices))


def run(config, checkpoint: Path, output: Path, device: torch.device):
    from .data import filter_overlong_prompt_records, read_records

    settings = config["mechanism"]
    for key in (
        "candidate_prompts",
        "candidate_responses",
        "states_per_response",
        "states_per_cell",
        "rollouts",
        "rollout_batch_size",
        "bootstrap",
        "position_bins",
    ):
        if int(settings[key]) < 1:
            raise ValueError(f"mechanism.{key} must be positive")
    if not 10 <= int(settings["g_bins"]) <= 20:
        raise ValueError("mechanism.g_bins must be between 10 and 20")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Launch with --workers N, not multi-process torchrun")
    workers = int(settings.get("workers", 1))
    if workers < 1 or workers > int(settings["g_bins"]) * 5 * int(
        settings["states_per_cell"]
    ):
        raise ValueError(
            "Worker count must be positive and no greater than the selected-state count"
        )
    if workers > 1:
        if device.type != "cuda" or device.index not in (None, 0):
            raise ValueError(
                "With multiple workers, use --device cuda:0 and CUDA_VISIBLE_DEVICES to select GPUs"
            )
        if torch.cuda.device_count() < workers:
            raise ValueError(
                f"Requested {workers} workers but only {torch.cuda.device_count()} CUDA GPUs are visible"
            )
    if (
        config["training"].get("use_lora", False)
        or (checkpoint / "adapter_config.json").exists()
    ):
        raise ValueError(
            "Use a full/merged student checkpoint and training.use_lora=false"
        )
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Missing HF checkpoint config: {checkpoint}")
    output.mkdir(parents=True, exist_ok=False)
    config["models"]["student_path"] = str(checkpoint)
    save_config(config, output / "resolved_config.yaml")
    experiment, assets, fused, restored = load_experiment(config, checkpoint, device)
    seed = experiment.seed
    records, files = read_records(
        config["data"]["path"], split=config["data"].get("split")
    )
    records, filtering = filter_overlong_prompt_records(
        records, experiment.tokenizer, config["data"]
    )
    indices = np.random.default_rng(experiment.seed).choice(
        len(records),
        min(len(records), int(settings["candidate_prompts"])),
        replace=False,
    )
    records = [records[int(index)] for index in indices]
    metadata = {
        "checkpoint": str(checkpoint),
        "assets": assets,
        "data_files": list(map(str, files)),
        "filtering": filtering,
        "optimizer_restored": restored is not None,
        "optimizer_step": restored.step if restored else None,
        "optimizer": "AdamW",
        "fused": fused,
        "optimizer_groups": [
            {key: value for key, value in group.items() if key != "params"}
            for group in experiment.optimizer.param_groups
        ],
        "gamma": experiment.gamma,
        "max_response_tokens": experiment.horizon,
        "horizon": "remaining original response cap: max_new_tokens - token_position - 1; EOS terminates",
        "score": "CMTSelector gain and sequential_gain_raw; robust_cmt_correction over all candidate-rollout valid states",
        "D_tilde": "configured corrected sequential gain; uncorrected score saved as D_raw",
        "loss_and_cost": "full-vocabulary reverse KL, configured student/teacher temperatures",
        "intervention": "one local reverse-KL AdamW step; student dropout disabled; teacher frozen",
        "optimizer_fallback": "fresh OPD-config AdamW state"
        if restored is None
        else None,
        "rollout_backend": "HF, current student weights; full policy top_p=1",
        "seed": experiment.seed,
        "tie_policy": "equal-count ranks; seeded random tie breaking",
        "workers": workers,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "parallelism": "candidate scoring and matching once on coordinator; disjoint state interventions on GPU replicas",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )
    candidates = experiment.collect_candidates(records)
    write_states(output / "candidate_states.jsonl", candidates.to_dict("records"))
    selected = matched_sample(
        candidates,
        bins=int(settings["g_bins"]),
        per_cell=int(settings["states_per_cell"]),
        seed=experiment.seed,
    )
    states = selected.to_dict("records")
    write_states(output / "selected_states.jsonl", states)
    if workers == 1:
        results = execute_interventions(experiment, states, output)
    else:
        # Release coordinator weights, moments and immutable CPU snapshots before
        # worker 0 loads its own replica on that same GPU.
        del experiment
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"Distributing {len(states)} selected states across {workers} GPU workers",
            flush=True,
        )
        results = parallel_interventions(
            config, checkpoint, output, [f"cuda:{rank}" for rank in range(workers)]
        )
    reports = analyze(
        results,
        output / "analysis",
        bins=int(settings["g_bins"]),
        position_bins=int(settings["position_bins"]),
        bootstrap=int(settings["bootstrap"]),
        seed=seed,
    )
    (output / "complete.json").write_text(
        json.dumps({"n_states": len(results), "analyses": list(reports)}) + "\n"
    )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/lift_mechanism.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--workers",
        type=int,
        help="GPU workers for one experiment; defaults to mechanism.workers",
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = load_config(args.config)
    defaults = load_config(
        Path(__file__).resolve().parents[1] / "configs/lift_mechanism.yaml"
    )["mechanism"]
    config["mechanism"] = {**defaults, **config.get("mechanism", {})}
    config = resolve_runtime_paths(apply_overrides(config, args.set))
    if args.workers is not None:
        config["mechanism"]["workers"] = args.workers
    run(
        config,
        args.checkpoint.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        torch.device(args.device),
    )


if __name__ == "__main__":
    main()
