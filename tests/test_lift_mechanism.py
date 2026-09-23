import numpy as np
import pandas as pd
import pytest
import torch
from types import SimpleNamespace

from b200_experiment.lift_mechanism import (
    _restore_optimizer_snapshot,
    _tree_to_cpu,
    analyze_matched_states,
    discounted_downstream_costs,
    matched_quantile_sample,
    quantile_bin_labels,
)
from b200_experiment.scoring import (
    RolloutBatch,
    reverse_kl_next_token,
    score_reverse_kl_rollout,
)


class _TinyNextTokenModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, input_ids, **_kwargs):
        batch, width = input_ids.shape
        logits = self.logits.reshape(1, 1, -1).expand(batch, width, -1)
        return SimpleNamespace(logits=logits)


def test_quantile_bins_are_complete_and_balanced_with_ties():
    labels = quantile_bin_labels(np.zeros(103), 10)
    counts = np.bincount(labels, minlength=10)
    assert set(labels) == set(range(10))
    assert counts.max() - counts.min() <= 1


def test_matched_design_has_equal_g_by_d_cells():
    rows = []
    for g_bin in range(10):
        for offset in range(20):
            rows.append(
                {
                    "state_id": f"{g_bin}-{offset}",
                    "G_t": g_bin + offset / 100.0,
                    "D_tilde": ((offset * 7) % 20) + g_bin / 1000.0,
                    "token_position": offset,
                }
            )
    selected = matched_quantile_sample(
        pd.DataFrame(rows),
        match_column="G_t",
        match_bins=10,
        d_bins=5,
        samples_per_cell=3,
        seed=7,
    )
    counts = selected.groupby(["match_bin", "D_quantile_matched"]).size()
    assert len(counts) == 50
    assert counts.nunique() == 1
    assert counts.iloc[0] == 3


def test_analysis_equal_weights_bins_instead_of_states():
    rows = []
    # Bin 1 has many high-gain states; bin 2 has one low-gain state per D group.
    # Equal-bin weighting must return (10 + 0) / 2, not a state-weighted value.
    for d_quantile in range(1, 6):
        for index in range(10):
            rows.append(
                {
                    "G_bin": 1,
                    "D_quantile": d_quantile,
                    "D_tilde": d_quantile + index / 100,
                    "downstream_gain_measured": 10.0,
                }
            )
        rows.append(
            {
                "G_bin": 2,
                "D_quantile": d_quantile,
                "D_tilde": float(d_quantile),
                "downstream_gain_measured": 0.0,
            }
        )
    aggregate, correlations, summary = analyze_matched_states(
        pd.DataFrame(rows),
        match_bin_column="G_bin",
        d_quantile_column="D_quantile",
        bootstrap_samples=50,
        seed=3,
    )
    assert aggregate["mean_downstream_gain_measured"].tolist() == [5.0] * 5
    assert len(correlations) == 2
    assert summary["equal_weight_strata"] == 2


def test_discounted_cost_excludes_local_state_and_starts_at_gamma_one():
    reverse_kl = torch.tensor([[100.0, 2.0, 3.0, 4.0]])
    valid = torch.tensor([[True, True, True, False]])
    cost = discounted_downstream_costs(reverse_kl, valid, gamma=0.5)
    assert cost.item() == pytest.approx(0.5 * 2.0 + 0.25 * 3.0)


def test_matched_design_can_add_position_strata():
    rows = []
    for position_bin in range(2):
        for g_bin in range(2):
            for item in range(10):
                rows.append(
                    {
                        "state_id": f"{position_bin}-{g_bin}-{item}",
                        "G_t": g_bin * 100 + position_bin * 10 + item,
                        "D_tilde": item,
                        "token_position": position_bin * 100 + item,
                    }
                )
    selected = matched_quantile_sample(
        pd.DataFrame(rows),
        match_column="G_t",
        match_bins=2,
        d_bins=5,
        samples_per_cell=1,
        seed=11,
        position_bins=2,
    )
    counts = selected.groupby(
        ["match_bin", "position_bin", "D_quantile_matched"]
    ).size()
    assert len(counts) == 20
    assert counts.eq(1).all()


def test_next_token_reverse_kl_is_exact_and_differentiable():
    student = _TinyNextTokenModel([1.0, -1.0])
    teacher = _TinyNextTokenModel([0.0, 0.0])
    teacher.requires_grad_(False)
    ids = torch.tensor([[4, 5]])
    mask = torch.ones_like(ids)
    loss = reverse_kl_next_token(student, teacher, ids, mask).mean()
    log_p = torch.log_softmax(student.logits, dim=-1)
    log_q = torch.log_softmax(teacher.logits, dim=-1)
    expected = (log_p.exp() * (log_p - log_q)).sum()
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert student.logits.grad is not None
    assert teacher.logits.grad is None


def test_rollout_reverse_kl_is_exact_and_zero_on_padding():
    student = _TinyNextTokenModel([1.0, -1.0])
    teacher = _TinyNextTokenModel([0.0, 0.0])
    teacher.requires_grad_(False)
    rollout = RolloutBatch(
        input_ids=torch.tensor([[8, 9, 1, 2], [8, 9, 1, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        response_ids=torch.tensor([[1, 2], [1, 0]]),
        valid_mask=torch.tensor([[True, True], [True, False]]),
        rollout_log_probs=torch.zeros(2, 2),
        prompt_width=2,
    )
    scored = score_reverse_kl_rollout(student, teacher, rollout, micro_batch_size=1)
    log_p = torch.log_softmax(student.logits, dim=-1)
    log_q = torch.log_softmax(teacher.logits, dim=-1)
    expected = (log_p.exp() * (log_p - log_q)).sum().item()
    torch.testing.assert_close(
        scored, torch.tensor([[expected, expected], [expected, 0.0]])
    )


def test_optimizer_intervention_does_not_mutate_pristine_snapshot():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    snapshot = _tree_to_cpu(optimizer.state_dict())
    original_steps = [state["step"].item() for state in snapshot["state"].values()]

    optimizer.state.clear()
    _restore_optimizer_snapshot(optimizer, snapshot)
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()

    assert [
        state["step"].item() for state in snapshot["state"].values()
    ] == original_steps
