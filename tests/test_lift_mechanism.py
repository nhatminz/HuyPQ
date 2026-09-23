from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from b200_experiment.config import load_config
from b200_experiment.lift_mechanism import (
    MechanismExperiment,
    discounted_downstream_cost,
    execute_interventions,
    merge_intervention_shards,
    parallel_interventions,
    read_states,
    reverse_kl,
    run,
    write_states,
)
from b200_experiment.lift_mechanism_analysis import (
    analyze,
    matched_sample,
    quantile_labels,
    spearman,
    summarize,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyLM(torch.nn.Module):
    """Trainable bigram model; accepts the production rollout/scoring interface."""

    def __init__(self, seed):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(7, 7, generator=generator) * 0.5)
        self.register_buffer("fixed_buffer", torch.tensor(3.0))
        self.config = SimpleNamespace(model_type="tiny")
        self.generation_config = SimpleNamespace(eos_token_id=6)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.weight[input_ids], past_key_values=None)


def setup_experiment():
    student, teacher = TinyLM(1), TinyLM(2)
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.02, weight_decay=0.0)
    # Non-empty optimizer moments are part of the treatment baseline.
    student.weight.square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    config = {
        "experiment": {"seed": 19},
        "selector": {"top_k": 3, "cmt_gamma": 0.7},
        "rollout": {"max_new_tokens": 8, "temperature": 0.8, "top_p": 1.0},
        "opd": {"teacher_temperature": 1.2},
        "training": {
            "max_grad_norm": 1.0,
            "learning_rate": 0.02,
            "fused_optimizer": False,
        },
        "mechanism": {
            "rollouts": 4,
            "rollout_batch_size": 2,
            "candidate_responses": 2,
            "states_per_response": 8,
        },
    }
    tokenizer = SimpleNamespace(eos_token_id=6, pad_token_id=0)
    return MechanismExperiment(student, teacher, tokenizer, optimizer, config)


def synthetic_states(n=1000):
    rng = np.random.default_rng(9)
    d = rng.normal(size=n)
    return pd.DataFrame(
        {
            "state_id": [f"s{i}" for i in range(n)],
            "prompt_id": [f"p{i // 4}" for i in range(n)],
            "token_position": rng.integers(0, 100, n),
            "G_t": np.arange(n, dtype=float),
            "D_tilde": d,
            "local_gain": rng.normal(size=n),
            "downstream_gain_measured": d * 2,
        }
    )


def test_reverse_kl_value_and_true_gradient():
    student = torch.tensor([[0.2, -0.8, 1.0]], requires_grad=True)
    teacher = torch.tensor([[0.7, 0.3, -0.2]], requires_grad=True)
    loss = reverse_kl(student, teacher)
    p, q = student.softmax(-1), teacher.softmax(-1)
    expected = (p * (p.log() - q.log())).sum(-1)
    assert torch.allclose(loss, expected)
    loss.sum().backward()
    expected_grad = p.detach() * (
        (p.log() - q.log()).detach() - expected.detach()[:, None]
    )
    assert torch.allclose(student.grad, expected_grad, atol=1e-7)
    assert teacher.grad is None


def test_cost_excludes_local_and_respects_discount_and_eos_mask():
    kl = torch.tensor([[100.0, 2.0, 4.0, 999.0], [100.0, 999.0, 999.0, 999.0]])
    valid = torch.tensor([[True, True, True, False], [True, False, False, False]])
    assert torch.equal(
        discounted_downstream_cost(kl, valid, 0.5), torch.tensor([2.0, 0.0])
    )
    assert torch.equal(discounted_downstream_cost(kl, valid, 0), torch.zeros(2))
    assert discounted_downstream_cost(kl[:, :1], valid[:, :1], 1).tolist() == [0.0, 0.0]


def test_snapshot_restores_weights_buffers_and_optimizer_without_aliasing():
    experiment = setup_experiment()
    reset = experiment.reset
    original = copy.deepcopy(reset.optimizer_state)
    for _ in range(2):
        experiment.student.weight.sum().backward()
        experiment.optimizer.step()
        experiment.student.fixed_buffer.add_(5)
        reset.restore()
        assert torch.equal(experiment.student.weight, reset.model_state["weight"])
        assert experiment.student.fixed_buffer.item() == 3
        for key, value in experiment.optimizer.state_dict()["state"][0].items():
            assert torch.equal(value, original["state"][0][key])
        assert experiment.student.weight.grad is None


def test_one_update_restores_every_state_and_rerolls_updated_student(monkeypatch):
    experiment = setup_experiment()
    baseline = experiment.student.weight.detach().clone()
    teacher = experiment.teacher.weight.detach().clone()
    states_seen, seeds_seen, step_calls = [], [], []
    generate, step = experiment.generate, experiment.optimizer.step

    def recording_generate(*args, **kwargs):
        states_seen.append(experiment.student.weight.detach().clone())
        seeds_seen.append(kwargs["seed"])
        return generate(*args, **kwargs)

    def recording_step(*args, **kwargs):
        step_calls.append(1)
        return step(*args, **kwargs)

    monkeypatch.setattr(experiment, "generate", recording_generate)
    monkeypatch.setattr(experiment.optimizer, "step", recording_step)
    state = {
        "state_id": "test",
        "prefix_ids": [1, 2],
        "downstream_horizon": 4,
        "D_tilde": 100.0,
    }
    first, costs = experiment.intervene(state)
    assert len(step_calls) == 1
    assert first["local_gain"] > 0
    assert first["downstream_gain_measured"] == pytest.approx(
        np.mean(costs["before"]) - np.mean(costs["after"])
    )
    assert first["local_gain"] == pytest.approx(first["KL_before"] - first["KL_after"])
    assert torch.equal(states_seen[0], baseline) and torch.equal(
        states_seen[1], baseline
    )
    assert not torch.equal(states_seen[2], baseline)
    assert torch.equal(states_seen[2], states_seen[3])
    assert len(set(seeds_seen)) == 4
    assert torch.equal(experiment.student.weight, baseline)
    assert torch.equal(experiment.teacher.weight, teacher)
    assert experiment.teacher.weight.grad is None
    state["D_tilde"] = -100.0
    second, second_costs = experiment.intervene(state)
    for field in ("KL_before", "KL_after", "C_before", "C_after", "local_gain"):
        assert first[field] == second[field]
    assert costs == second_costs  # Score never enters the update.
    assert len(step_calls) == 2


def test_reset_on_failed_after_rollout(monkeypatch):
    experiment = setup_experiment()
    baseline = experiment.student.weight.detach().clone()
    original = experiment.rollout_costs
    calls = []

    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("rollout failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(experiment, "rollout_costs", failing)
    with pytest.raises(RuntimeError, match="rollout failed"):
        experiment.intervene(
            {"state_id": "test", "prefix_ids": [1, 2], "downstream_horizon": 2}
        )
    assert torch.equal(experiment.student.weight, baseline)


def test_rollout_costs_equal_direct_full_vocabulary_cost():
    experiment = setup_experiment()
    prefix = torch.tensor([[1, 2]])
    rollout = experiment.generate(prefix, steps=4, count=4, seed=100)
    expected = []
    for row in range(4):
        value = 0.0
        for k in range(1, int(rollout.valid_mask[row].sum())):
            token = rollout.input_ids[row, rollout.prompt_width + k - 1]
            kl = reverse_kl(
                experiment.student.weight[token],
                experiment.teacher.weight[token],
                0.8,
                1.2,
            )
            value += 0.7**k * float(kl.detach())
        expected.append(value)
    assert experiment.rollout_costs(prefix, horizon=3, seed=100) == pytest.approx(
        expected, abs=1e-6
    )


def test_immediate_eos_has_no_downstream_cost():
    experiment = setup_experiment()
    with torch.no_grad():
        experiment.student.weight.fill_(-100)
        experiment.student.weight[:, 6] = 100
    assert (
        experiment.rollout_costs(torch.tensor([[1, 2]]), horizon=5, seed=10)
        == [0.0] * 4
    )


def test_matching_balances_all_cells_and_is_outcome_independent():
    frame = synthetic_states()
    matched = matched_sample(frame, per_cell=7, seed=12)
    assert len(matched) == 350
    assert matched.groupby(["G_bin", "D_quantile"]).size().eq(7).all()
    changed = frame.assign(downstream_gain_measured=-frame.downstream_gain_measured)
    assert matched_sample(changed, per_cell=7, seed=12).state_id.equals(
        matched.state_id
    )
    for _, group in matched.groupby("G_bin"):
        assert group.groupby("D_quantile").D_tilde.mean().is_monotonic_increasing
    with pytest.raises(ValueError, match="smallest cell"):
        matched_sample(frame, per_cell=1000)


def test_ties_are_seeded_and_constant_correlations_are_undefined():
    labels = quantile_labels(np.zeros(100), 5, np.random.default_rng(1))
    assert np.array_equal(np.bincount(labels)[1:], [20] * 5)
    assert np.array_equal(
        labels, quantile_labels(np.zeros(100), 5, np.random.default_rng(1))
    )
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert spearman([1, 1, 3], [1, 1, 3]) == pytest.approx(1)
    frame = matched_sample(synthetic_states().assign(D_tilde=0.0), per_cell=2)
    result = summarize(frame, bootstrap=30)
    assert result["spearman_equal_bin_mean"] is None
    assert result["spearman_defined_bins"] == 0
    assert result["sign_agreement_fraction"] == 0


def test_equal_bin_weighting_bootstrap_and_spearman():
    frame = matched_sample(synthetic_states(), per_cell=4)
    result = summarize(frame, bootstrap=100)
    assert result["spearman_equal_bin_mean"] == pytest.approx(1)
    assert result["sign_agreement_fraction"] == 1
    assert result == summarize(frame, bootstrap=100)
    # Duplicating one bin should not change the point estimates.
    uneven = pd.concat([frame, frame[frame.G_bin == 1]] * 1, ignore_index=True)
    second = summarize(uneven, bootstrap=100)
    for first, other in zip(result["quintiles"], second["quintiles"]):
        assert first["mean_downstream_gain"] == pytest.approx(
            other["mean_downstream_gain"]
        )
        assert first["ci_low"] <= first["ci_high"]


def test_robustness_rematches_on_actual_gain_and_position(tmp_path):
    frame = matched_sample(synthetic_states(3000), per_cell=20)
    reports = analyze(frame, tmp_path, position_bins=2, bootstrap=30)
    assert set(reports) == {
        "matched_G",
        "matched_local_gain",
        "matched_G_position",
        "matched_local_gain_position",
    }
    for name in reports:
        assert (tmp_path / name / "downstream_gain.png").is_file()
        assert (tmp_path / name / "balance.csv").is_file()
    local = pd.read_csv(tmp_path / "matched_local_gain" / "matched_states.csv")
    assert local.groupby("match_bin").local_gain.mean().is_monotonic_increasing
    assert local.groupby(["match_bin", "D_quantile"]).size().nunique() == 1
    position = pd.read_csv(tmp_path / "matched_G_position" / "matched_states.csv")
    assert (
        position.groupby(["match_bin", "position_bin", "D_quantile"]).size().nunique()
        == 1
    )
    assert not np.array_equal(local.match_bin, local.G_bin)


def test_end_to_end_outputs_with_restored_optimizer(tmp_path, monkeypatch):
    from b200_experiment import data, models

    experiment = setup_experiment()
    config = copy.deepcopy(experiment.config)
    config.update(models={}, data={"path": "unused"})
    config["mechanism"].update(
        candidate_prompts=20,
        states_per_cell=1,
        g_bins=10,
        bootstrap=20,
        position_bins=1,
        rollouts=2,
    )
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    torch.save(
        {"step": 1, "optimizer": experiment.optimizer.state_dict()},
        checkpoint / "optimizer.pt",
    )
    monkeypatch.setattr(
        models,
        "load_models",
        lambda *_: (experiment.student, experiment.teacher, experiment.tokenizer, {}),
    )
    monkeypatch.setattr(
        data, "read_records", lambda *_, **__: ([{"id": i} for i in range(20)], [])
    )
    monkeypatch.setattr(
        data, "filter_overlong_prompt_records", lambda records, *_: (records, {})
    )
    monkeypatch.setattr(
        data, "tokenize_prompts", lambda *_: ({"input_ids": torch.tensor([[1, 2]])}, [])
    )
    output = tmp_path / "result"
    run(config, checkpoint, output, torch.device("cpu"))
    frame = pd.read_csv(output / "per_state.csv")
    assert len(frame) == 50
    assert {
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
    } <= set(frame)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["optimizer_restored"] and metadata["optimizer_step"] == 1
    assert (output / "complete.json").exists()
    assert (output / "analysis/matched_G/downstream_gain.png").exists()
    candidates = pd.read_json(output / "candidate_states.jsonl", lines=True)
    assert (candidates.downstream_horizon == 7 - candidates.token_position).all()
    assert np.allclose(candidates.D_tilde, candidates.D_raw)
    assert candidates.prefix_ids.map(len).equals(candidates.token_position + 2)
    with pytest.raises(FileExistsError):
        run(config, checkpoint, output, torch.device("cpu"))


def test_default_config():
    config = load_config("configs/lift_mechanism.yaml")
    assert config["mechanism"]["rollouts"] == 8
    assert config["selector"]["cmt_gamma"] == 1.0
    assert config["rollout"]["max_new_tokens"] == 4096
    assert config["mechanism"]["workers"] == 1


def _tiny_process_worker(rank, config, checkpoint, output, devices):
    """Executed in a real spawned interpreter; replace only model loading."""
    from unittest.mock import patch

    from b200_experiment.lift_mechanism import intervention_worker

    torch.set_num_threads(1)
    with patch(
        "b200_experiment.lift_mechanism.load_experiment",
        side_effect=lambda *_: (setup_experiment(), {}, False, None),
    ):
        intervention_worker(rank, config, checkpoint, output, devices)


def _failed_process_worker(rank, config, checkpoint, output, devices):
    if rank == 1:
        raise RuntimeError("synthetic worker failure")


def test_four_process_interventions_match_single_process_and_merge_once(
    tmp_path, monkeypatch
):
    from b200_experiment import lift_mechanism

    states = [
        {
            "state_id": f"state_{i}",
            "prompt_id": "NA",
            "prefix_ids": [1, 2 + i % 3],
            "downstream_horizon": 3,
            "G_t": 0.12345678901234567,
            "D_tilde": i * 0.01,
            "token_position": i,
        }
        for i in range(9)
    ]
    sequential = tmp_path / "sequential"
    sequential.mkdir()
    expected = execute_interventions(setup_experiment(), states, sequential)
    parallel = tmp_path / "parallel"
    parallel.mkdir()
    write_states(parallel / "selected_states.jsonl", states)
    assert read_states(parallel / "selected_states.jsonl") == states
    monkeypatch.setattr(lift_mechanism, "intervention_worker", _tiny_process_worker)
    actual = parallel_interventions({}, tmp_path, parallel, ["cpu"] * 4)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
    assert read_states(parallel / "rollout_costs.jsonl") == read_states(
        sequential / "rollout_costs.jsonl"
    )
    assert len(actual) == 9 and actual.state_id.is_unique
    counts = []
    for rank in range(4):
        shard = parallel / "workers" / f"rank_{rank:03d}"
        done = json.loads((shard / "complete.json").read_text())
        counts.append(done["n_states"])
        ids = pd.read_csv(shard / "per_state.csv").state_id.tolist()
        assert ids == [state["state_id"] for state in states[rank::4]]
    assert counts == [3, 2, 2, 2]

    # A corrupt/duplicated cost shard must be rejected even if its CSV is intact.
    costs_path = parallel / "workers/rank_000/rollout_costs.jsonl"
    costs = read_states(costs_path)
    write_states(costs_path, [costs[0]] * len(costs))
    with pytest.raises(ValueError, match="missing, duplicate, or unassigned"):
        merge_intervention_shards(parallel, states, 4)


def test_failed_worker_propagates_without_merged_outputs(tmp_path, monkeypatch):
    from b200_experiment import lift_mechanism

    write_states(
        tmp_path / "selected_states.jsonl", [{"state_id": f"s{i}"} for i in range(4)]
    )
    monkeypatch.setattr(lift_mechanism, "intervention_worker", _failed_process_worker)
    with pytest.raises(
        torch.multiprocessing.ProcessRaisedException, match="synthetic worker failure"
    ):
        parallel_interventions({}, tmp_path, tmp_path, ["cpu"] * 2)
    assert not (tmp_path / "per_state.csv").exists()
    assert not (tmp_path / "complete.json").exists()


def test_parallel_launch_rejects_missing_gpus_before_creating_output(
    tmp_path, monkeypatch
):
    config = load_config("configs/lift_mechanism.yaml")
    config["mechanism"]["workers"] = 4
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="only 2 CUDA GPUs"):
        run(config, tmp_path, tmp_path / "output", torch.device("cuda:0"))
    assert not (tmp_path / "output").exists()
