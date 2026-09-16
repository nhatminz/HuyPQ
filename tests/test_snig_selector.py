from __future__ import annotations

import torch

from b200_experiment.selectors.cmt_selector import CMTSelector
from b200_experiment.selectors.pgt_selector import PGTOutput
from b200_experiment.selectors.rac_selector import (
    bellman_parallel_scan,
    bellman_reference_scan,
)
from b200_experiment.selectors.snig_selector import SNIGSelector


def _support(
    student_cond: torch.Tensor,
    teacher_cond: torch.Tensor,
    *,
    student_mass: float = 1.0,
    teacher_mass: float = 1.0,
    gain: torch.Tensor | None = None,
    ids: torch.Tensor | None = None,
) -> PGTOutput:
    batch, time, width = student_cond.shape
    if gain is None:
        p = student_cond.exp()
        r = teacher_cond - student_cond
        mean = (p * r).sum(dim=-1)
        gain = (p * (r - mean.unsqueeze(-1)).square()).sum(dim=-1)
    if ids is None:
        ids = torch.arange(width).reshape(1, 1, width).expand(batch, time, width)
    support = torch.ones_like(student_cond, dtype=torch.bool)
    diagnostics = {
        "gain": gain,
        "s_PGT": gain,
        "student_union_mass": torch.full_like(gain, student_mass),
        "teacher_union_mass": torch.full_like(gain, teacher_mass),
        "teacher_tail_mass": torch.full_like(gain, 1.0 - teacher_mass),
        "support_width": torch.full_like(gain, float(width)),
    }
    return PGTOutput(gain, diagnostics, ids, student_cond, teacher_cond, support)


def test_snig_lambda_zero_is_exactly_pgt_gain_and_support_is_preserved():
    p = torch.log(torch.tensor([[[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]]]))
    q = torch.log(torch.tensor([[[0.2, 0.5, 0.3], [0.2, 0.5, 0.3]]]))
    output = _support(p, q, ids=torch.tensor([[[10, 11, 12], [10, 11, 12]]]))
    result = SNIGSelector(successor_lambda=0.0).compute_scores(
        output, torch.tensor([[10, 11]]), torch.ones(1, 2, dtype=torch.bool)
    )
    assert torch.equal(result.candidate_ids, output.candidate_ids)
    assert torch.equal(result.support_mask, output.support_mask)
    assert torch.allclose(result.scores, output.diagnostics["gain"])
    transition = result.diagnostics["transition_weight"]
    assert bool(((transition >= 0.0) & (transition <= 1.0)).all())


def test_snig_terminal_and_padding_boundaries_are_zero_successor():
    p = torch.log(torch.tensor([[[0.5, 0.5]] * 4]))
    q = torch.log(torch.tensor([[[0.8, 0.2]] * 4]))
    valid = torch.tensor([[True, True, False, False]])
    output = _support(p, q, gain=torch.full((1, 4), 1.0))
    result = SNIGSelector().compute_scores(output, torch.tensor([[0, 0, 0, 0]]), valid)
    assert result.diagnostics["successor_utility"][0, 1].item() == 0.0
    assert torch.all(result.diagnostics["successor_utility"][~valid] == 0)
    assert torch.all(result.scores[~valid] == 0)


def test_snig_constant_gain_is_length_neutral_with_survival_less_than_one():
    p = torch.log(torch.tensor([[[0.5, 0.5]] * 64]))
    q = torch.log(torch.tensor([[[0.25, 0.75]] * 64]))
    valid = torch.ones(1, 64, dtype=torch.bool)
    output = _support(p, q, gain=torch.full((1, 64), 2.0))
    # Action 0 has q/p < 1, so all sampled transitions survive with a
    # bounded factor but a constant-gain suffix must have zero utility.
    result = SNIGSelector().compute_scores(output, torch.zeros(1, 64, dtype=torch.long), valid)
    assert torch.allclose(
        result.diagnostics["successor_utility"], torch.zeros_like(result.scores), atol=1e-5
    )
    assert torch.allclose(result.scores, output.diagnostics["gain"], atol=1e-5)


def test_snig_constant_gain_long_suffix_has_no_length_reward():
    length = 4096
    p = torch.log(torch.tensor([[[0.5, 0.5]] * length]))
    q = torch.log(torch.tensor([[[0.2, 0.8]] * length]))
    valid = torch.ones(1, length, dtype=torch.bool)
    output = _support(p, q, gain=torch.full((1, length), 0.5))
    result = SNIGSelector().compute_scores(
        output, torch.ones(1, length, dtype=torch.long), valid
    )
    utility = result.diagnostics["successor_utility"][valid]
    assert float(utility.abs().max()) < 1.0e-4
    assert torch.allclose(result.scores, output.diagnostics["gain"], atol=1e-4)


def test_snig_parallel_recurrence_matches_reference_with_padding_boundaries():
    torch.manual_seed(7)
    length = 37
    gain = torch.rand(2, length)
    transition = torch.rand(2, length)
    valid = torch.ones(2, length, dtype=torch.bool)
    valid[0, 18:] = False
    valid[1, 7:9] = False
    valid[1, 24:] = False
    parallel = bellman_parallel_scan(gain, transition, valid, gamma=0.93)
    reference = bellman_reference_scan(gain, transition, valid, gamma=0.93)
    for actual, expected in zip(parallel, reference):
        assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_snig_truncated_common_mass_estimator_has_exact_one_step_expectation():
    # The fourth action is outside U.  Under Y~p, its estimator contribution is
    # zero, while the three actions in U recover sum_a min(p_raw(a), q_raw(a)).
    p_cond = torch.tensor([0.50, 0.30, 0.20])
    q_cond = torch.tensor([0.20, 0.50, 0.30])
    student_mass, teacher_mass = 0.70, 0.60
    p_raw = student_mass * p_cond
    q_raw = teacher_mass * q_cond
    sampled_ids = torch.tensor([[0, 1, 2, 99]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    sampled = SNIGSelector(successor_lambda=0.0).compute_scores(
        _support(
            p_cond.log().repeat(1, 4, 1),
            q_cond.log().repeat(1, 4, 1),
            student_mass=student_mass,
            teacher_mass=teacher_mass,
        ),
        sampled_ids,
        valid,
    )
    factors = sampled.diagnostics["transition_weight"][0]
    expected = (p_raw * factors[:3]).sum()
    exact = torch.minimum(p_raw, q_raw).sum()
    assert torch.all((factors >= 0.0) & (factors <= 1.0))
    assert factors[3].item() == 0.0
    assert torch.allclose(expected, exact, atol=1e-6)


def test_nonzero_kernel_derivative_is_only_on_accepted_common_mass_branch():
    p = torch.log(torch.tensor([[[0.5, 0.5]]]))
    q = torch.log(torch.tensor([[[0.2, 0.8]]]))
    output = _support(p, q, gain=torch.tensor([[1.0]]))
    valid = torch.ones(1, 1, dtype=torch.bool)
    accepted = SNIGSelector().compute_scores(output, torch.tensor([[1]]), valid)
    rejected = SNIGSelector().compute_scores(output, torch.tensor([[0]]), valid)
    assert accepted.diagnostics["kernel_derivative"].item() != 0.0
    assert accepted.diagnostics["transition_weight"].item() == 1.0
    assert rejected.diagnostics["kernel_derivative"].item() == 0.0


def test_snig_successor_formula_matches_cmt_derivative_normalization():
    p = torch.log(torch.tensor([[[0.50, 0.30, 0.20], [0.50, 0.30, 0.20]]]))
    q = torch.log(torch.tensor([[[0.20, 0.50, 0.30], [0.20, 0.50, 0.30]]]))
    gain = torch.tensor([[0.6, 1.4]])
    ids = torch.tensor([[[10, 11, 12], [10, 11, 12]]])
    output = _support(p, q, gain=gain, ids=ids)
    sampled = torch.tensor([[11, 11]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    cmt = CMTSelector().compute_scores(output, sampled, valid)
    snig = SNIGSelector().compute_scores(output, sampled, valid)
    denominator = snig.diagnostics["M"] * (
        snig.diagnostics["M"] + snig.diagnostics["R"]
    )
    assert torch.allclose(
        snig.diagnostics["successor_utility"] * denominator,
        cmt.diagnostics["sequential_gain"],
        atol=1e-6,
    )


def test_snig_frozen_descendant_finite_difference_on_three_actions():
    # This verifies the derivative of Phi for a fixed action-indexed successor
    # return/mass pair. It is deliberately an offline diagnostic; the training
    # selector uses the one-rollout plug-in quantities.
    p = torch.tensor([0.50, 0.30, 0.20], dtype=torch.float64)
    q = torch.tensor([0.20, 0.50, 0.30], dtype=torch.float64)
    m_p, m_q = 0.7, 0.6
    g = torch.tensor(0.7, dtype=torch.float64)
    gamma = 1.0
    r = q.log() - p.log()
    mean_r = (p * r).sum()
    child_r = torch.tensor([0.20, 1.70, 0.90], dtype=torch.float64)
    child_m = torch.tensor([1.20, 0.80, 1.10], dtype=torch.float64)
    # At the base point, action 1 is the only q°-surplus branch.
    original_p = m_p * p
    original_q = m_q * q
    dk = torch.where(
        original_p < original_q,
        original_p * (r - mean_r),
        torch.zeros_like(p),
    )
    R = g + gamma * torch.minimum(original_p, original_q).mul(child_r).sum()
    M = 1.0 + gamma * torch.minimum(original_p, original_q).mul(child_m).sum()
    analytic = gamma * (dk.mul(child_r).sum() * M - R * dk.mul(child_m).sum()) / (M * (M + R))

    def phi(x: torch.Tensor) -> torch.Tensor:
        x = x / x.sum()
        rr = torch.minimum(m_p * x, original_q)
        rx = g + gamma * rr.mul(child_r).sum()
        mx = 1.0 + gamma * rr.mul(child_m).sum()
        return torch.log1p(rx / mx)

    h = 1e-6
    p_plus = p * torch.exp(h * r)
    p_minus = p * torch.exp(-h * r)
    finite_difference = (phi(p_plus) - phi(p_minus)) / (2.0 * h)
    assert torch.allclose(finite_difference, analytic, atol=2e-6, rtol=2e-5)
