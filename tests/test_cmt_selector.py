import math

import torch
import pytest

from b200_experiment.selectors.cmt_selector import (
    CMTSelector,
    bounded_mean_one_weights,
    cmt_allocation,
    direct_bounded_gibbs_allocation,
    kl_constrained_allocation,
    robust_cmt_correction,
)
from b200_experiment.diagnostics import selector_summary
from b200_experiment.selectors.pgt_selector import PGTOutput


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
        "gain_support_definition": "student_topk",
        "transition_support_definition": ("literal_union_student_topk_teacher_topk"),
        "student_support_mass": torch.full_like(gain, student_mass),
        "teacher_support_mass": torch.full_like(gain, teacher_mass),
        "teacher_tail_mass": torch.full_like(gain, 1.0 - teacher_mass),
        "support_width": torch.full_like(gain, float(width)),
    }
    return PGTOutput(
        gain,
        diagnostics,
        ids,
        student_cond,
        teacher_cond,
        support,
    )


def test_local_excess_removes_constant_length_bias():
    p = torch.log(torch.tensor([[[0.5], [0.5], [0.5], [0.5]]]))
    q = p.clone()
    output = _support(p, q, gain=torch.ones(1, 4))
    result = CMTSelector().compute_scores(
        output,
        torch.zeros(1, 4, dtype=torch.long),
        torch.ones(1, 4, dtype=torch.bool),
    )
    assert torch.allclose(result.diagnostics["H"], torch.zeros(1, 4))
    assert torch.allclose(result.scores, torch.ones(1, 4))


def test_padding_is_a_hard_boundary_for_local_excess_value():
    p = torch.log(torch.tensor([[[0.5], [0.5], [0.5], [0.5]]]))
    output = _support(p, p.clone(), gain=torch.ones(1, 4))
    valid = torch.tensor([[True, True, False, False]])
    result = CMTSelector().compute_scores(
        output,
        torch.zeros(1, 4, dtype=torch.long),
        valid,
    )
    assert torch.all(result.diagnostics["H"][~valid] == 0)
    assert torch.all(result.scores[~valid] == 0)
    assert torch.allclose(result.scores[valid], torch.ones(2))


def test_constant_gain_is_length_neutral_even_when_coupling_survival_is_below_one():
    p = torch.log(torch.tensor([[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]]))
    q = torch.log(torch.tensor([[[0.25, 0.75], [0.25, 0.75], [0.25, 0.75]]]))
    output = _support(p, q, gain=torch.ones(1, 3))
    result = CMTSelector().compute_scores(
        output,
        torch.tensor([[0, 0, 0]]),
        torch.ones(1, 3, dtype=torch.bool),
    )
    assert torch.allclose(result.diagnostics["H"], torch.zeros(1, 3))
    assert torch.allclose(result.scores, torch.ones(1, 3))


def test_constant_gain_is_length_neutral_under_raw_truncated_mass():
    p = torch.log(torch.tensor([[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]]))
    q = torch.log(torch.tensor([[[0.75, 0.25], [0.75, 0.25], [0.75, 0.25]]]))
    output = _support(p, q, student_mass=0.4, teacher_mass=0.7, gain=torch.ones(1, 3))
    result = CMTSelector().compute_scores(
        output,
        torch.tensor([[0, 0, 0]]),
        torch.ones(1, 3, dtype=torch.bool),
    )
    assert torch.allclose(result.diagnostics["H"], torch.zeros(1, 3))
    assert torch.allclose(result.scores, torch.ones(1, 3))


def test_truncated_kernel_keeps_transition_bounded_without_inverse_coverage():
    p = torch.log(torch.tensor([[[0.5, 0.5], [0.5, 0.5]]]))
    q = torch.log(torch.tensor([[[0.8, 0.2], [0.8, 0.2]]]))
    ids = torch.tensor([[[10, 11], [10, 11]]])
    output = _support(p, q, student_mass=0.5, teacher_mass=0.25, ids=ids)
    result = CMTSelector().compute_scores(
        output,
        torch.tensor([[10, 99]]),
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert torch.allclose(
        result.diagnostics["coverage_correction"], torch.tensor([[1.0, 0.0]])
    )
    assert torch.allclose(
        result.diagnostics["transition_weight"], torch.tensor([[0.8, 0.0]])
    )
    assert torch.equal(
        result.diagnostics["teacher_deficit"], torch.tensor([[0.0, 0.0]])
    )


def test_truncated_kernel_has_the_raw_common_mass_expectation():
    # Full student masses are [.1, .3, .6_tail], and teacher masses on U are
    # [.3, .1].  For successor values [2,-1], the exact raw expectation is
    # .1*2 + .1*(-1) = .1.
    p = torch.log(torch.tensor([[[0.25, 0.75]]]))
    q = torch.log(torch.tensor([[[0.75, 0.25]]]))
    ids = torch.tensor([[[10, 11]]])
    output = _support(p, q, student_mass=0.4, teacher_mass=0.4, ids=ids)
    result = CMTSelector().compute_scores(
        output,
        torch.tensor([[10]]),
        torch.ones(1, 1, dtype=torch.bool),
    )
    assert torch.allclose(
        result.diagnostics["transition_weight"], torch.tensor([[1.0]])
    )
    assert torch.allclose(
        result.diagnostics["support_common_mass"], torch.tensor([[0.2]])
    )
    assert torch.allclose(
        result.diagnostics["conditional_support_common_mass"], torch.tensor([[0.5]])
    )
    full_p = torch.tensor([0.10, 0.30, 0.60], dtype=torch.float64)
    full_q = torch.tensor([0.30, 0.10, 0.60], dtype=torch.float64)
    successor = torch.tensor([2.0, -1.0], dtype=torch.float64)
    sampled = torch.tensor(
        [
            min(1.0, (full_q[0] / full_p[0]).item()) * successor[0].item(),
            min(1.0, (full_q[1] / full_p[1]).item()) * successor[1].item(),
            0.0,
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(
        (full_p * sampled).sum(), full_p[0] * 2.0 + min(full_p[1], full_q[1]) * (-1.0)
    )
    # The production selector gives the same bounded factor for every possible
    # sampled ID, including the killed tail event, without a full-vocabulary
    # lookup.
    production_estimates = []
    for token_id in (10, 11, 99):
        token_result = CMTSelector().compute_scores(
            output,
            torch.tensor([[token_id]]),
            torch.ones(1, 1, dtype=torch.bool),
        )
        transition = token_result.diagnostics["transition_weight"].item()
        successor_value = (
            successor[len(production_estimates)].item() if token_id != 99 else 0.0
        )
        production_estimates.append(transition * successor_value)
    assert torch.allclose(
        (full_p * torch.tensor(production_estimates)).sum(),
        full_p[0] * 2.0 + min(full_p[1], full_q[1]) * (-1.0),
        atol=1e-7,
    )


def test_truncated_transition_factor_is_bounded():
    p = torch.log(torch.tensor([[[0.99, 0.01]]]))
    q = torch.log(torch.tensor([[[0.01, 0.99]]]))
    output = _support(p, q, student_mass=0.01, teacher_mass=0.99)
    result = CMTSelector().compute_scores(
        output,
        torch.tensor([[0]]),
        torch.ones(1, 1, dtype=torch.bool),
    )
    transition = result.diagnostics["transition_weight"]
    assert bool((transition >= 0).all())
    assert bool((transition <= 1).all())


def test_cmt_recurrence_uses_local_baseline_excess_opportunity():
    p = torch.log(torch.tensor([[[0.5], [0.5], [0.5]]]))
    q = torch.log(torch.tensor([[[0.5], [0.5], [0.5]]]))
    gain = torch.tensor([[1.0, 2.0, 3.0]])
    output = _support(p, q, gain=gain)
    result = CMTSelector().compute_scores(
        output,
        torch.zeros(1, 3, dtype=torch.long),
        torch.ones(1, 3, dtype=torch.bool),
    )
    # H = R - g_t M = [3, 1, 0] under a unit transition.  The current score
    # has no sequential term because r=0, but H exposes future opportunity in
    # excess of the current state's own local value.
    assert torch.allclose(result.diagnostics["H"], torch.tensor([[3.0, 1.0, 0.0]]))
    assert torch.allclose(result.scores, gain)


def test_soft_signed_effect_survives_when_student_is_slightly_above_teacher():
    p = torch.log(torch.tensor([[[0.60, 0.40], [0.50, 0.50]]]))
    q = torch.log(torch.tensor([[[0.59, 0.41], [0.50, 0.50]]]))
    output = _support(p, q, gain=torch.tensor([[1.0, 2.0]]))
    result = CMTSelector(gamma=0.75, successor_lambda=1.5).compute_scores(
        output,
        torch.tensor([[0, 0]]),
        torch.ones(1, 2, dtype=torch.bool),
    )
    expected_c = torch.tensor(0.59 / 0.60)
    signed_shift = result.diagnostics["signed_reachability_shift"][0, 0]
    assert torch.allclose(
        result.diagnostics["transition_weight"][0, 0], expected_c, atol=1e-6
    )
    assert torch.allclose(
        result.diagnostics["compatibility_weight"][0, 0], expected_c, atol=1e-6
    )
    assert signed_shift < 0
    assert torch.allclose(
        result.diagnostics["marginal_flux"][0, 0],
        expected_c * signed_shift,
        atol=1e-6,
    )
    assert result.diagnostics["marginal_flux"][0, 0] != 0
    assert result.diagnostics["sequential_gain"][0, 0] != 0
    assert torch.allclose(
        result.diagnostics["downstream_effect"][0, 0],
        0.75
        * result.diagnostics["marginal_flux"][0, 0]
        * result.diagnostics["successor_excess"][0, 0],
    )
    assert torch.allclose(
        result.diagnostics["sequential_gain"][0, 0],
        1.5 * result.diagnostics["downstream_effect"][0, 0],
    )
    # Kept only as a backward-compatible raw-ratio diagnostic.  It must not
    # gate the soft signed downstream effect.
    assert result.diagnostics["teacher_deficit"][0, 0] == 0
    summary = selector_summary(
        "cmt",
        result.diagnostics,
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert summary["marginal_flux_positive_fraction"] == 0.0
    assert summary["marginal_flux_negative_fraction"] == 0.5


def test_soft_signed_effect_has_unit_compatibility_when_student_is_below_teacher():
    p = torch.log(torch.tensor([[[0.40, 0.60], [0.50, 0.50]]]))
    q = torch.log(torch.tensor([[[0.60, 0.40], [0.50, 0.50]]]))
    output = _support(p, q, gain=torch.tensor([[1.0, 2.0]]))
    result = CMTSelector().compute_scores(
        output,
        torch.tensor([[0, 0]]),
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert torch.allclose(
        result.diagnostics["compatibility_weight"][0, 0], torch.tensor(1.0)
    )
    assert torch.allclose(
        result.diagnostics["marginal_flux"][0, 0],
        result.diagnostics["signed_reachability_shift"][0, 0],
    )


def test_shift_sign_is_not_inferred_from_raw_teacher_deficit():
    # Raw p(y)=.8*.4=.32 is above raw q(y)=.4*.7=.28, but within the
    # conditional support token 0 moves upward relative to the normalized
    # mean.  The hard teacher-deficit diagnostic is therefore false while the
    # signed visitation effect is positive.
    p = torch.log(torch.tensor([[[0.40, 0.60], [0.50, 0.50]]]))
    q = torch.log(torch.tensor([[[0.70, 0.30], [0.50, 0.50]]]))
    output = _support(
        p,
        q,
        student_mass=0.8,
        teacher_mass=0.4,
        gain=torch.tensor([[1.0, 2.0]]),
    )
    result = CMTSelector().compute_scores(
        output,
        torch.tensor([[0, 0]]),
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert result.diagnostics["teacher_deficit"][0, 0] == 0
    assert result.diagnostics["signed_reachability_shift"][0, 0] > 0
    assert result.diagnostics["marginal_flux"][0, 0] > 0
    assert torch.allclose(
        result.diagnostics["compatibility_weight"][0, 0],
        torch.tensor(0.28 / 0.32),
        atol=1e-6,
    )


def test_soft_signed_effect_is_attenuated_not_zero_when_student_dominates():
    p = torch.log(torch.tensor([[[0.90, 0.10], [0.50, 0.50]]]))
    q = torch.log(torch.tensor([[[0.01, 0.99], [0.50, 0.50]]]))
    output = _support(p, q, gain=torch.tensor([[1.0, 2.0]]))
    result = CMTSelector().compute_scores(
        output,
        torch.tensor([[0, 0]]),
        torch.ones(1, 2, dtype=torch.bool),
    )
    compatibility = result.diagnostics["compatibility_weight"][0, 0]
    shift = result.diagnostics["signed_reachability_shift"][0, 0]
    flux = result.diagnostics["marginal_flux"][0, 0]
    assert torch.allclose(compatibility, torch.tensor(0.01 / 0.90), atol=1e-6)
    assert flux != 0
    assert torch.allclose(flux.abs(), compatibility * shift.abs(), atol=1e-6)


def test_soft_signed_effect_is_zero_outside_truncated_support():
    p = torch.log(torch.tensor([[[0.60, 0.40], [0.50, 0.50]]]))
    q = torch.log(torch.tensor([[[0.59, 0.41], [0.50, 0.50]]]))
    ids = torch.tensor([[[10, 11], [10, 11]]])
    output = _support(p, q, gain=torch.tensor([[1.0, 2.0]]), ids=ids)
    result = CMTSelector().compute_scores(
        output,
        torch.tensor([[99, 10]]),
        torch.ones(1, 2, dtype=torch.bool),
    )
    for key in (
        "compatibility_weight",
        "signed_reachability_shift",
        "marginal_flux",
        "downstream_effect",
        "sequential_gain",
    ):
        assert result.diagnostics[key][0, 0] == 0


def test_frozen_compatibility_directional_formula_matches_finite_difference():
    p = torch.tensor([0.50, 0.30, 0.20], dtype=torch.float64)
    q = torch.tensor([0.20, 0.50, 0.30], dtype=torch.float64)
    child = torch.tensor([0.20, 1.70, 0.90], dtype=torch.float64)
    r = q.log() - p.log()
    mean = (p * r).sum()
    compatibility = torch.minimum(torch.ones_like(p), q / p).detach()
    analytic = (p * compatibility * (r - mean) * child).sum()

    def path(eta):
        value = p * torch.exp(eta * r)
        return value / value.sum()

    def frozen_compatibility_access(x):
        return (compatibility * x * child).sum()

    eta = 1e-5
    finite_difference = (
        frozen_compatibility_access(path(eta)) - frozen_compatibility_access(path(-eta))
    ) / (2.0 * eta)
    assert torch.allclose(finite_difference, analytic, atol=1e-9)


def test_production_score_is_single_surrogate_derivative():
    p = torch.tensor([0.50, 0.30, 0.20], dtype=torch.float64)
    q = torch.tensor([0.20, 0.50, 0.30], dtype=torch.float64)
    child_return = torch.tensor([0.20, 1.70, 0.90], dtype=torch.float64)
    child_mass = torch.tensor([1.20, 0.80, 1.10], dtype=torch.float64)
    r = q.log() - p.log()
    mean = (p * r).sum()
    local_gain = (p * (r - mean).square()).sum()
    compatibility = torch.minimum(torch.ones_like(p), q / p).detach()
    child_excess = child_return - local_gain * child_mass
    d_excess = (p * compatibility * (r - mean) * child_excess).sum()
    expected = local_gain + d_excess

    def objective(x):
        local_improvement = (p * (p.log() - q.log())).sum() - (
            x * (x.log() - q.log())
        ).sum()
        compatible_visitation = compatibility * x
        excess = (
            local_gain
            + (compatible_visitation * child_return).sum()
            - local_gain * (1.0 + (compatible_visitation * child_mass).sum())
        )
        return local_improvement + excess

    eta = 1e-5
    p_plus = p * torch.exp(eta * r)
    p_plus /= p_plus.sum()
    p_minus = p * torch.exp(-eta * r)
    p_minus /= p_minus.sum()
    finite_difference = (objective(p_plus) - objective(p_minus)) / (2.0 * eta)
    assert torch.allclose(finite_difference, expected, atol=1e-9)


def test_truncated_frozen_compatibility_uses_original_probability_masses():
    p_u = torch.tensor([0.50, 0.30, 0.20], dtype=torch.float64)
    q_u = torch.tensor([0.20, 0.50, 0.30], dtype=torch.float64)
    student_mass = 0.40
    teacher_mass = 0.70
    child_excess = torch.tensor([0.20, 1.70, 0.90], dtype=torch.float64)
    r = q_u.log() - p_u.log()
    mean = (p_u * r).sum()
    original_p = student_mass * p_u
    original_q = teacher_mass * q_u
    compatibility = torch.minimum(torch.ones_like(original_p), original_q / original_p)
    analytic = (original_p * compatibility * (r - mean) * child_excess).sum()

    def frozen_compatibility_access(x):
        return (student_mass * compatibility * x * child_excess).sum()

    eta = 1e-5
    p_plus = p_u * torch.exp(eta * r)
    p_plus /= p_plus.sum()
    p_minus = p_u * torch.exp(-eta * r)
    p_minus /= p_minus.sum()
    finite_difference = (
        frozen_compatibility_access(p_plus) - frozen_compatibility_access(p_minus)
    ) / (2.0 * eta)
    assert torch.allclose(finite_difference, analytic, atol=1e-9)


def test_truncated_excess_estimator_is_unbiased_by_enumeration():
    # Full student probabilities are [.1, .3, .6_tail], while p_U=[.25,.75].
    # Original teacher masses on U are [.3,.1].  Compatibility uses these raw
    # masses, while direction uses the conditional Student-Top-K tangent.
    p_u = torch.tensor([0.25, 0.75], dtype=torch.float64)
    q_u = torch.tensor([0.75, 0.25], dtype=torch.float64)
    full_p = torch.tensor([0.10, 0.30, 0.60], dtype=torch.float64)
    m_p = 0.40
    m_q = 0.40
    child_excess = torch.tensor([2.0, -1.0], dtype=torch.float64)
    r = q_u.log() - p_u.log()
    mean = (p_u * r).sum()
    original_p = m_p * p_u
    original_q = m_q * q_u
    compatibility = torch.minimum(torch.ones_like(original_p), original_q / original_p)
    sampled = torch.tensor(
        [
            compatibility[0].item() * (r[0] - mean).item() * child_excess[0].item(),
            compatibility[1].item() * (r[1] - mean).item() * child_excess[1].item(),
            0.0,
        ],
        dtype=torch.float64,
    )
    enumerated_expectation = (full_p * sampled).sum()
    exact = (original_p * compatibility * (r - mean) * child_excess).sum()
    assert torch.allclose(enumerated_expectation, exact, atol=1e-12)


def test_kl_allocation_hits_budget_and_is_affine_scale_invariant():
    values = torch.tensor([-1.0, 0.0, 0.5, 3.0, 4.0])
    weights, inverse_temperature, achieved = kl_constrained_allocation(values, 0.4)
    assert torch.all(weights > 0)
    assert torch.allclose(weights.mean(), torch.tensor(1.0), atol=1e-6)
    assert abs(achieved - 0.4) < 1e-5
    assert inverse_temperature > 0
    assert torch.equal(torch.argsort(weights), torch.argsort(values))
    transformed, _, transformed_kl = kl_constrained_allocation(7.0 * values + 9.0, 0.4)
    assert torch.allclose(weights, transformed, atol=1e-5)
    assert abs(transformed_kl - 0.4) < 1e-5


def test_zero_kl_budget_is_uniform():
    weights, inverse_temperature, achieved = kl_constrained_allocation(
        torch.tensor([0.0, 2.0, 5.0]), 0.0
    )
    assert torch.equal(weights, torch.ones(3))
    assert inverse_temperature == 0.0
    assert achieved == 0.0


def test_bounded_uniform_weights_remain_exactly_uniform():
    raw = torch.ones(17)
    final = bounded_mean_one_weights(raw, 0.5, 2.0)
    assert torch.equal(final, raw)


def test_bounded_allocation_handles_extreme_outliers_zeros_and_tiny_values():
    for raw in (
        torch.tensor([1.0e-12, 0.0, 1.0, 1.0e4]),
        torch.tensor([1.0e-30, 0.0, 1.0, 1.0e20], dtype=torch.float64),
    ):
        final = bounded_mean_one_weights(raw, 0.5, 2.0)
        assert torch.isfinite(final).all()
        assert abs(float(final.double().mean()) - 1.0) <= 1e-6
        assert float(final.min()) >= 0.5
        assert float(final.max()) <= 2.0
        order = torch.argsort(raw, stable=True)
        assert bool((final[order][1:] >= final[order][:-1]).all())


def test_bounded_half_to_two_has_normalized_ess_at_least_two_thirds():
    raw = torch.logspace(-20, 20, 1001)
    final = bounded_mean_one_weights(raw, 0.5, 2.0)
    normalized_ess = final.sum().square() / (final.numel() * final.square().sum())
    assert float(normalized_ess) >= 2.0 / 3.0 - 1e-6


def test_gibbs_mode_is_exactly_the_legacy_solver_output():
    values = torch.tensor([-3.0, -0.5, 0.0, 4.0, 7.0])
    legacy, legacy_beta, _ = kl_constrained_allocation(values, 0.4)
    raw, final, beta, metrics = cmt_allocation(values, 0.4, mode="gibbs")
    assert torch.equal(raw, legacy)
    assert torch.equal(final, legacy)
    assert beta == legacy_beta
    assert metrics["allocation_kl_pre_bound"] == metrics["allocation_kl_post_bound"]


def test_bounded_mode_returns_distinct_raw_and_final_weights():
    values = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
    raw, final, _, metrics = cmt_allocation(
        values,
        1.0,
        mode="bounded_gibbs",
        weight_min=0.5,
        weight_max=2.0,
    )
    assert not torch.equal(raw, final)
    assert float(raw.max()) > 2.0
    assert float(final.max()) <= 2.0
    assert abs(float(final.mean()) - 1.0) <= 1e-6
    assert metrics["allocation_kl_post_bound"] <= metrics["allocation_kl_pre_bound"]


def test_bounded_scaling_is_global_not_independently_normalized_per_rank():
    raw_global = torch.tensor([0.1, 0.1, 0.1, 3.7])
    final_global = bounded_mean_one_weights(raw_global, 0.5, 2.0)
    assert abs(float(final_global.mean()) - 1.0) <= 1e-6
    # A correct global solve does not force each artificial rank shard to mean 1.
    assert not torch.isclose(final_global[:2].mean(), torch.tensor(1.0), atol=1e-6)
    assert not torch.isclose(final_global[2:].mean(), torch.tensor(1.0), atol=1e-6)


def test_bounded_allocation_rejects_invalid_bounds_and_infeasible_zero_support():
    with pytest.raises(ValueError, match="0 <= min <= 1 <= max"):
        bounded_mean_one_weights(torch.ones(3), 1.1, 2.0)
    with pytest.raises(ValueError, match="No scale c"):
        bounded_mean_one_weights(torch.tensor([0.0, 0.0, 0.0, 4.0]), 0.5, 2.0)


def test_tanh_correction_zero_sign_bound_and_small_signal_limit():
    gain = torch.tensor([2.0, 2.0, 2.0, 2.0])
    raw = torch.tensor([0.0, -100.0, 100.0, 2.0e-5])
    robust, learning, kappa, _ = robust_cmt_correction(gain, raw, mode="tanh_q99")
    assert robust[0] == 0
    assert torch.equal(torch.sign(robust), torch.sign(raw))
    assert bool((robust.abs() <= kappa + 1e-6).all())
    assert torch.isclose(robust[-1], raw[-1], rtol=1e-5, atol=1e-9)
    assert torch.allclose(learning, gain + robust)


def test_none_correction_preserves_raw_score_exactly():
    gain = torch.tensor([0.1, 3.0, 2.0])
    raw = torch.tensor([-5.0, 0.0, 7.0])
    robust, learning, _, _ = robust_cmt_correction(gain, raw, mode="none")
    assert torch.equal(robust, raw)
    assert torch.equal(learning, gain + raw)


def test_direct_bounded_solver_is_finite_mean_one_bounded_and_monotone():
    values = torch.tensor([-1.0e30, -1.0, 0.0, 0.5, 3.0, 1.0e30], dtype=torch.float64)
    weights, beta, log_c, status, _ = direct_bounded_gibbs_allocation(
        values, 0.02, 0.5, 2.0
    )
    assert status == "kl_active"
    assert torch.isfinite(weights).all()
    assert abs(float(weights.double().mean()) - 1.0) <= 1e-6
    assert float(weights.min()) >= 0.5 - 1e-6
    assert float(weights.max()) <= 2.0 + 1e-6
    assert float((weights.double() * weights.double().log()).mean()) <= 0.02 + 1e-6
    order = torch.argsort(values, stable=True)
    assert bool((weights[order][1:] >= weights[order][:-1]).all())
    reconstructed = torch.exp(
        (log_c + beta * values.double()).clamp(math.log(0.5), math.log(2.0))
    )
    assert torch.allclose(weights.double(), reconstructed, atol=1e-6, rtol=1e-6)


def test_direct_bounded_solver_constant_and_zero_kl_are_uniform():
    constant, *_ = direct_bounded_gibbs_allocation(
        torch.full((9,), 42.0), 0.02, 0.5, 2.0
    )
    zero_kl, *_ = direct_bounded_gibbs_allocation(torch.arange(9.0), 0.0, 0.5, 2.0)
    assert torch.equal(constant, torch.ones(9))
    assert torch.equal(zero_kl, torch.ones(9))


def test_direct_bounded_solver_large_target_returns_box_optimum():
    values = torch.arange(7.0)
    weights, beta, _, status, maximum_kl = direct_bounded_gibbs_allocation(
        values, 10.0, 0.5, 2.0
    )
    assert status == "box_optimal_kl_inactive"
    assert beta == float("inf")
    assert abs(float(weights.mean()) - 1.0) <= 1e-6
    assert float((weights.double() * weights.double().log()).mean()) == pytest.approx(
        maximum_kl
    )


def test_direct_bounded_solution_is_not_posthoc_clipped_unbounded_gibbs():
    values = torch.tensor([-5.0, -1.0, -0.2, 0.0, 0.3, 1.0, 9.0])
    raw, _, _ = kl_constrained_allocation(values, 0.08)
    posthoc = bounded_mean_one_weights(raw, 0.5, 1.4)
    direct, *_ = direct_bounded_gibbs_allocation(values, 0.08, 0.5, 1.4)
    assert not torch.allclose(direct, posthoc, rtol=1e-5, atol=1e-6)


def test_direct_mode_keeps_unbounded_reference_diagnostic_only():
    values = torch.tensor([-3.0, 0.0, 1.0, 8.0])
    raw, final, _, metrics = cmt_allocation(
        values,
        0.5,
        mode="direct_bounded_gibbs",
        weight_min=0.5,
        weight_max=2.0,
        final_epsilon=0.02,
    )
    expected_raw, _, _ = kl_constrained_allocation(values, 0.02)
    assert torch.equal(raw, expected_raw)
    assert not torch.equal(raw, final)
    assert metrics["allocation_kl_target"] == 0.02
    assert metrics["allocation_solver_status"] == "kl_active"


def test_direct_bounded_allocation_rejects_zero_lower_bound():
    with pytest.raises(ValueError, match="weight_min > 0"):
        direct_bounded_gibbs_allocation(torch.arange(3.0), 0.02, 0.0, 2.0)
