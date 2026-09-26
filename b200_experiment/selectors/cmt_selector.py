from __future__ import annotations

import math

import torch

from .pgt_selector import PGTOutput
from .rac_selector import bellman_parallel_scan


@torch.no_grad()
def kl_constrained_allocation(
    values: torch.Tensor,
    epsilon: float,
    *,
    iterations: int = 64,
    tolerance: float = 1e-7,
) -> tuple[torch.Tensor, float, float]:
    """Solve max_w E_w[value] with KL(w || uniform) <= epsilon."""
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("KL allocation expects a non-empty one-dimensional tensor")
    if epsilon < 0.0:
        raise ValueError("KL allocation epsilon must be non-negative")
    scores = values.detach().float()
    if not torch.isfinite(scores).all():
        raise FloatingPointError("KL allocation received non-finite values")
    count = scores.numel()
    if count == 1 or epsilon <= tolerance or bool(scores.eq(scores[0]).all()):
        return torch.ones_like(scores), 0.0, 0.0

    centered = scores - scores.max()
    maxima = centered.eq(0).sum().item()
    maximum_kl = math.log(count / max(int(maxima), 1))
    target = min(float(epsilon), maximum_kl)
    if target >= maximum_kl - tolerance:
        probabilities = centered.eq(0).float() / float(maxima)
        return probabilities * count, math.inf, maximum_kl

    log_count = math.log(count)

    def distribution_and_kl(inverse_temperature: float):
        logits = centered * float(inverse_temperature)
        log_normalizer = torch.logsumexp(logits, dim=0)
        probabilities = torch.exp(logits - log_normalizer)
        kl = (
            float(inverse_temperature) * (probabilities * centered).sum()
            - log_normalizer
            + log_count
        )
        return probabilities, float(kl.item())

    low, high = 0.0, 1.0
    _, high_kl = distribution_and_kl(high)
    while high_kl < target and high < 1e12:
        high *= 2.0
        _, high_kl = distribution_and_kl(high)
    for _ in range(max(1, int(iterations))):
        midpoint = 0.5 * (low + high)
        _, midpoint_kl = distribution_and_kl(midpoint)
        if midpoint_kl < target:
            low = midpoint
        else:
            high = midpoint
    inverse_temperature = 0.5 * (low + high)
    probabilities, achieved_kl = distribution_and_kl(inverse_temperature)
    return probabilities * count, inverse_temperature, achieved_kl


def validate_cmt_allocation(
    mode: str, weight_min: float, weight_max: float
) -> tuple[str, float, float]:
    """Validate the two supported CMT allocation policies and their bounds."""
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in {"gibbs", "bounded_gibbs", "direct_bounded_gibbs"}:
        raise ValueError(
            "selector.cmt_allocation_mode must be exactly 'gibbs', "
            "'bounded_gibbs', or 'direct_bounded_gibbs'"
        )
    lower, upper = float(weight_min), float(weight_max)
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise ValueError("CMT weight bounds must be finite")
    if not 0.0 <= lower <= 1.0 <= upper:
        raise ValueError("CMT weight bounds must satisfy 0 <= min <= 1 <= max")
    if resolved_mode == "direct_bounded_gibbs" and lower <= 0.0:
        raise ValueError("Direct bounded Gibbs requires cmt_weight_min > 0")
    return resolved_mode, lower, upper


def validate_cmt_correction(mode: str, quantile: float) -> tuple[str, float]:
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in {"none", "tanh_q99"}:
        raise ValueError("selector.cmt_correction_mode must be 'none' or 'tanh_q99'")
    resolved_quantile = float(quantile)
    if not math.isfinite(resolved_quantile) or not 0.0 < resolved_quantile <= 1.0:
        raise ValueError("selector.cmt_correction_quantile must be in (0, 1]")
    return resolved_mode, resolved_quantile


@torch.no_grad()
def robust_cmt_correction(
    gain: torch.Tensor,
    sequential_gain_raw: torch.Tensor,
    *,
    mode: str = "none",
    quantile: float = 0.99,
    numerical_epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, float, dict[str, float]]:
    """Apply the rollout-global bounded-influence correction to raw CMT D."""
    resolved_mode, resolved_quantile = validate_cmt_correction(mode, quantile)
    if gain.ndim != 1 or sequential_gain_raw.shape != gain.shape or gain.numel() == 0:
        raise ValueError("CMT correction expects aligned non-empty 1-D tensors")
    gain64 = gain.detach().double()
    raw64 = sequential_gain_raw.detach().double()
    if not bool(torch.isfinite(gain64).all() and torch.isfinite(raw64).all()):
        raise FloatingPointError("CMT correction received non-finite values")
    epsilon = float(numerical_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("CMT correction numerical epsilon must be positive")
    kappa = max(float(torch.quantile(gain64, resolved_quantile)), epsilon)
    robust64 = raw64 if resolved_mode == "none" else kappa * torch.tanh(raw64 / kappa)
    learning64 = gain64 + robust64
    if not bool(torch.isfinite(robust64).all() and torch.isfinite(learning64).all()):
        raise FloatingPointError("CMT correction produced non-finite values")
    raw_abs = raw64.abs()
    robust_abs = robust64.abs()
    metrics = {
        "correction_kappa": kappa,
        "sequential_gain_raw_abs_q95": float(torch.quantile(raw_abs, 0.95)),
        "sequential_gain_raw_abs_q99": float(torch.quantile(raw_abs, 0.99)),
        "sequential_gain_robust_abs_q95": float(torch.quantile(robust_abs, 0.95)),
        "sequential_gain_robust_abs_q99": float(torch.quantile(robust_abs, 0.99)),
        "correction_saturation_rate_1kappa": float((raw_abs >= kappa).double().mean()),
        "correction_saturation_rate_2kappa": float(
            (raw_abs >= 2.0 * kappa).double().mean()
        ),
    }
    return (
        robust64.to(dtype=sequential_gain_raw.dtype),
        learning64.to(dtype=gain.dtype),
        kappa,
        metrics,
    )


def _mean_w_log_w(weights: torch.Tensor) -> float:
    values = weights.detach().double()
    positive = values > 0
    terms = torch.zeros_like(values)
    terms[positive] = values[positive] * values[positive].log()
    return float(terms.mean())


@torch.no_grad()
def bounded_mean_one_weights(
    raw_weights: torch.Tensor,
    weight_min: float,
    weight_max: float,
    *,
    iterations: int = 96,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    """Compute ``clip(c * raw, min, max)`` whose arithmetic mean is one.

    Scaling is solved before clipping by monotone bisection.  In particular,
    the result is never divided by its mean after clipping, because that would
    invalidate the configured bounds.
    """
    _, lower, upper = validate_cmt_allocation("bounded_gibbs", weight_min, weight_max)
    if raw_weights.ndim != 1 or raw_weights.numel() == 0:
        raise ValueError("Bounded Gibbs expects a non-empty one-dimensional tensor")
    raw = raw_weights.detach().double()
    if not bool(torch.isfinite(raw).all()):
        raise FloatingPointError("Bounded Gibbs received non-finite raw weights")
    if bool((raw < 0).any()):
        raise ValueError("Bounded Gibbs raw weights must be non-negative")
    if not bool((raw > 0).any()):
        raise ValueError("Bounded Gibbs requires at least one positive raw weight")
    if lower == upper == 1.0:
        return torch.ones_like(raw_weights)
    maximum_reachable_mean = float(
        torch.where(raw > 0, raw.new_tensor(upper), raw.new_tensor(lower)).mean()
    )
    if maximum_reachable_mean < 1.0 - float(tolerance):
        raise ValueError(
            "No scale c can make clip(c * raw, min, max) mean one: too many "
            "raw weights are exactly zero for the configured bounds"
        )

    def transformed(scale: float) -> torch.Tensor:
        return (raw * scale).clamp(min=lower, max=upper)

    low, high = 0.0, 1.0
    while float(transformed(high).mean()) < 1.0:
        high *= 2.0
        if not math.isfinite(high):
            raise FloatingPointError("Could not bracket the bounded Gibbs scale")
    for _ in range(max(1, int(iterations))):
        midpoint = 0.5 * (low + high)
        if float(transformed(midpoint).mean()) < 1.0:
            low = midpoint
        else:
            high = midpoint
    final64 = transformed(0.5 * (low + high))
    final = final64.to(dtype=raw_weights.dtype)
    if not bool(torch.isfinite(final).all()):
        raise FloatingPointError("Bounded Gibbs produced non-finite weights")
    mean_error = abs(float(final.double().mean()) - 1.0)
    if mean_error > float(tolerance):
        raise AssertionError(
            f"Bounded Gibbs mean-one constraint failed: error={mean_error:.3e}"
        )
    if float(final.min()) < lower - 1e-7 or float(final.max()) > upper + 1e-7:
        raise AssertionError("Bounded Gibbs violated its configured bounds")
    return final


@torch.no_grad()
def cmt_weight_metrics(
    raw_weights: torch.Tensor,
    final_weights: torch.Tensor,
    *,
    weight_min: float,
    weight_max: float,
) -> dict[str, float]:
    """Return allocation diagnostics, using final weights for risk metrics."""
    if raw_weights.shape != final_weights.shape or raw_weights.numel() == 0:
        raise ValueError("Raw and final CMT weights must have the same non-empty shape")
    raw = raw_weights.detach().double()
    final = final_weights.detach().double()
    if not bool(torch.isfinite(raw).all() and torch.isfinite(final).all()):
        raise FloatingPointError("CMT allocation metrics received non-finite weights")
    total = final.sum()
    normalized_ess = total.square() / (
        final.numel() * final.square().sum().clamp_min(1e-30)
    )
    atol = 1e-6
    return {
        "allocation_kl_pre_bound": _mean_w_log_w(raw),
        "allocation_kl_post_bound": _mean_w_log_w(final),
        "weight_raw_max": float(raw.max()),
        "weight_final_min": float(final.min()),
        "weight_final_max": float(final.max()),
        "fraction_at_weight_min": float(
            torch.isclose(
                final, final.new_tensor(float(weight_min)), rtol=0.0, atol=atol
            )
            .double()
            .mean()
        ),
        "fraction_at_weight_max": float(
            torch.isclose(
                final, final.new_tensor(float(weight_max)), rtol=0.0, atol=atol
            )
            .double()
            .mean()
        ),
        "normalized_ess": float(normalized_ess),
        "max_token_probability": float(final.max() / total.clamp_min(1e-30)),
    }


def _box_optimal_weights(
    scores: torch.Tensor, lower: float, upper: float
) -> torch.Tensor:
    """Linear-objective optimum on the mean-one box, with stable score order."""
    count = scores.numel()
    weights = torch.full_like(scores, lower, dtype=torch.float64)
    capacity = upper - lower
    remaining = float(count) * (1.0 - lower)
    if remaining <= 0.0 or capacity <= 0.0:
        return torch.ones_like(scores, dtype=torch.float64)
    order = torch.argsort(scores, descending=True, stable=True)
    full = min(int(math.floor(remaining / capacity + 1e-14)), count)
    if full:
        weights[order[:full]] = upper
        remaining -= full * capacity
    if full < count and remaining > 1e-14:
        weights[order[full]] = lower + min(remaining, capacity)
    return weights


def _bounded_exponential_for_temperature(
    normalized_centered_scores: torch.Tensor,
    temperature: float,
    lower: float,
    upper: float,
    *,
    iterations: int = 96,
) -> tuple[torch.Tensor, float]:
    """Inner KKT solve for log(c) at one non-negative scaled beta."""
    log_lower, log_upper = math.log(lower), math.log(upper)
    theta = normalized_centered_scores * float(temperature)

    def weights(log_c: float) -> torch.Tensor:
        return torch.exp((theta + float(log_c)).clamp(log_lower, log_upper))

    # normalized_centered_scores is in [-2, 0], so these bounds force all
    # weights to the lower/upper box face without exponentiating large scores.
    low = log_lower - 2.0 * float(temperature) - 2.0
    high = log_upper + 2.0 * float(temperature) + 2.0
    for _ in range(max(1, int(iterations))):
        midpoint = 0.5 * (low + high)
        if float(weights(midpoint).mean()) < 1.0:
            low = midpoint
        else:
            high = midpoint
    log_c = 0.5 * (low + high)
    return weights(log_c), log_c


@torch.no_grad()
def direct_bounded_gibbs_allocation(
    values: torch.Tensor,
    epsilon: float,
    weight_min: float,
    weight_max: float,
    *,
    outer_iterations: int = 96,
    inner_iterations: int = 96,
    tolerance: float = 1e-7,
) -> tuple[torch.Tensor, float, float, str, float]:
    """Solve the KL-constrained bounded Gibbs KKT system directly in float64."""
    _, lower, upper = validate_cmt_allocation(
        "direct_bounded_gibbs", weight_min, weight_max
    )
    target = float(epsilon)
    if not math.isfinite(target) or target < 0.0:
        raise ValueError("selector.cmt_final_allocation_kl must be finite and >= 0")
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("Direct bounded Gibbs expects a non-empty 1-D score tensor")
    scores = values.detach().double()
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("Direct bounded Gibbs received non-finite scores")
    count = scores.numel()
    uniform = torch.ones_like(scores)
    if count == 1 or target <= tolerance or bool(scores.eq(scores[0]).all()):
        return uniform.to(dtype=values.dtype), 0.0, 0.0, "uniform", 0.0

    box_optimal = _box_optimal_weights(scores, lower, upper)
    maximum_kl = _mean_w_log_w(box_optimal)
    if target >= maximum_kl - tolerance:
        # The KL constraint is inactive. The stable score ordering makes the
        # selected box optimum deterministic even when several scores tie.
        return (
            box_optimal.to(dtype=values.dtype),
            math.inf,
            0.0,
            "box_optimal_kl_inactive",
            maximum_kl,
        )

    scale = float(scores.abs().max())
    if not math.isfinite(scale) or scale <= 0.0:
        return uniform.to(dtype=values.dtype), 0.0, 0.0, "uniform", 0.0
    normalized = scores / scale
    normalized_centered = normalized - normalized.max()

    def solve_temperature(temperature: float) -> tuple[torch.Tensor, float, float]:
        weights, log_c = _bounded_exponential_for_temperature(
            normalized_centered,
            temperature,
            lower,
            upper,
            iterations=inner_iterations,
        )
        return weights, log_c, _mean_w_log_w(weights)

    low_temperature, high_temperature = 0.0, 1.0
    high_weights, high_log_c, high_kl = solve_temperature(high_temperature)
    while high_kl < target and high_temperature < 1e12:
        high_temperature *= 2.0
        high_weights, high_log_c, high_kl = solve_temperature(high_temperature)
    if high_kl < target - tolerance:
        # Numerical convergence reached the limiting box face before the
        # requested KL. This is the same inactive-constraint solution.
        return (
            box_optimal.to(dtype=values.dtype),
            high_temperature / scale,
            high_log_c - (high_temperature / scale) * float(scores.max()),
            "box_optimal_kl_inactive",
            maximum_kl,
        )

    final_weights = uniform
    final_log_c = 0.0
    final_kl = 0.0
    for _ in range(max(1, int(outer_iterations))):
        midpoint = 0.5 * (low_temperature + high_temperature)
        candidate, candidate_log_c, candidate_kl = solve_temperature(midpoint)
        if candidate_kl <= target:
            low_temperature = midpoint
            final_weights = candidate
            final_log_c = candidate_log_c
            final_kl = candidate_kl
        else:
            high_temperature = midpoint
    final = final_weights.to(dtype=values.dtype)
    mean_error = abs(float(final.double().mean()) - 1.0)
    if not bool(torch.isfinite(final).all()):
        raise FloatingPointError("Direct bounded Gibbs produced non-finite weights")
    if mean_error > 1e-6:
        raise AssertionError(
            f"Direct bounded Gibbs mean-one error exceeds tolerance: {mean_error}"
        )
    if float(final.min()) < lower - 1e-6 or float(final.max()) > upper + 1e-6:
        raise AssertionError("Direct bounded Gibbs violated configured bounds")
    if final_kl > target + 1e-6:
        raise AssertionError("Direct bounded Gibbs exceeded its final KL budget")
    beta = low_temperature / scale
    # The stable exponent uses beta * (L - max(L)); report log(c) in the
    # requested uncentered KKT form clip(c * exp(beta * L), lower, upper).
    reported_log_c = final_log_c - beta * float(scores.max())
    return final, beta, reported_log_c, "kl_active", maximum_kl


@torch.no_grad()
def cmt_allocation(
    values: torch.Tensor,
    epsilon: float,
    *,
    mode: str = "gibbs",
    weight_min: float = 0.5,
    weight_max: float = 2.0,
    final_epsilon: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, float, dict[str, float]]:
    """Allocate one PPO group under a legacy or direct bounded Gibbs policy.

    In direct mode ``raw`` is only the unbounded reference at the same final KL
    target. ``final`` is solved independently from the bounded KKT system and is
    the only weight tensor consumed by the training loss.
    """
    resolved_mode, lower, upper = validate_cmt_allocation(mode, weight_min, weight_max)
    allocation_target = (
        float(final_epsilon)
        if resolved_mode == "direct_bounded_gibbs"
        else float(epsilon)
    )
    if not math.isfinite(allocation_target) or allocation_target < 0.0:
        raise ValueError("CMT allocation KL target must be finite and non-negative")
    raw, raw_inverse_temperature, _raw_solver_kl = kl_constrained_allocation(
        values, allocation_target
    )
    allocation_log_c = 0.0
    maximum_feasible_kl = float("nan")
    if resolved_mode == "gibbs":
        final = raw
        inverse_temperature = raw_inverse_temperature
        solver_status = "legacy_gibbs"
    elif resolved_mode == "bounded_gibbs":
        final = bounded_mean_one_weights(raw, lower, upper)
        inverse_temperature = raw_inverse_temperature
        solver_status = "legacy_posthoc_bounded"
    else:
        (
            final,
            inverse_temperature,
            allocation_log_c,
            solver_status,
            maximum_feasible_kl,
        ) = direct_bounded_gibbs_allocation(
            values,
            allocation_target,
            lower,
            upper,
        )
    metrics = cmt_weight_metrics(raw, final, weight_min=lower, weight_max=upper)
    if resolved_mode == "gibbs":
        metrics["fraction_at_weight_min"] = 0.0
        metrics["fraction_at_weight_max"] = 0.0
    metrics.update(
        allocation_beta=float(inverse_temperature),
        allocation_log_c=float(allocation_log_c),
        allocation_kl_target=allocation_target,
        allocation_kl_final=metrics["allocation_kl_post_bound"],
        allocation_mean_weight_error=abs(float(final.double().mean()) - 1.0),
        allocation_solver_status=solver_status,
        allocation_maximum_feasible_kl=maximum_feasible_kl,
    )
    return raw, final, inverse_temperature, metrics


class CMTSelector:
    """Student-Top-K-local, local-excess Coupled Marginal Teachability.

    The local gain action space is exactly the Student Top-K set S. Student and
    teacher probabilities are conditionalized on S by ``PGTSelector`` and
    ``g_t = Var_{p_S}[log q_S - log p_S]``. Teacher-only Top-K actions therefore
    cannot affect ``g_t``. Sequential accessibility is a distinct object: it
    retains the original probability mass on the student/teacher Top-K union U,
    yielding the truncated sub-Markov kernel

        K_tilde f(s) = sum_{a in U} min(p(a|s), q(a|s)) f(sa).

    The strict full-policy student rollout therefore needs no inverse-coverage
    correction: for ``Y in U``, ``min(1, m_q*q_U(Y)/(m_p*p_U(Y)))`` is an
    unbiased one-sample transition factor; for ``Y`` outside ``U`` it is zero.
    The factor is bounded by one and requires no tail probability lookup.

    The sequential quantity is a local-baseline excess opportunity, not an
    episodic average-reward return.  For a root state with local opportunity
    ``g_t``, the frozen-descendant quantity is

        E_t = R_t - g_t M_t
            = gamma * K_tilde (R_{t+1} - g_t M_{t+1}),

    and the current-action effect uses the successor contrast
    ``R_{t+1} - g_t M_{t+1}``.  This baseline is the current state's own
    Student-Top-K local value, so constant-gain suffixes cancel exactly.  The
    common-support overlap is retained as a frozen compatibility weight, while
    the directional factor is the signed first-order change in student
    visitation probability under the teacher-directed perturbation.
    With ``successor_lambda=1``, ``g_t + D_t`` is the derivative of a single
    frozen-descendant surrogate consisting of local reverse-KL improvement plus
    the baseline-subtracted successor opportunity; lambda=0 is only a named
    local-only ablation.
    Descendant values are frozen in this categorical surrogate; no causal
    shared-neural-network claim is made. This union never becomes the OPD loss
    support: optimization is separately restricted to the configured Student
    Top-K.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        successor_lambda: float = 1.0,
        ablation_arm: str = "canonical",
    ):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("CMT gamma must be in [0, 1]")
        if successor_lambda < 0.0:
            raise ValueError("CMT successor lambda must be non-negative")
        arm = str(ablation_arm).strip().lower()
        aliases = {
            "canonical": "canonical",
            "cmt": "canonical",
            "g": "g",
            "g_x": "g_x",
            "gx": "g_x",
            "g_d": "g_d",
            "gd": "g_d",
            "d_only": "d_only",
            "d-only": "d_only",
            "d": "d_only",
        }
        if arm not in aliases:
            raise ValueError(
                "CMT ablation_arm must be one of canonical, g, g_x, g_d, or "
                "d_only; "
                f"got {ablation_arm!r}"
            )
        self.gamma = float(gamma)
        self.successor_lambda = float(successor_lambda)
        self.ablation_arm = aliases[arm]

    @torch.no_grad()
    def compute_scores(
        self,
        pgt_support: PGTOutput,
        sampled_token_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> PGTOutput:
        """Compute CMT scores from local conditional and raw transition quantities.

        ``sampled_token_ids`` are drawn from the full student policy.  Local
        gain geometry is conditional on the Student Top-K set S.  The separate
        successor kernel retains the original masses p(a), q(a) on the union U.
        Tokens outside U are intentionally killed rather than reweighted into
        the conditional simplex.
        """
        expected_shape = valid_mask.shape
        if valid_mask.ndim != 2:
            raise ValueError("CMT valid_mask must have shape [batch, time]")
        if sampled_token_ids.shape != expected_shape:
            raise ValueError("CMT sampled token IDs must align with valid_mask")
        if pgt_support.candidate_ids.shape[:2] != expected_shape:
            raise ValueError("CMT support must align with valid_mask")

        valid = valid_mask.bool()
        candidate_ids = pgt_support.candidate_ids
        support_mask = pgt_support.support_mask.bool()
        student_cond = pgt_support.student_candidate_log_probs.detach().float()
        teacher_cond = pgt_support.teacher_candidate_log_probs.detach().float()
        support_p = torch.where(
            support_mask, student_cond.exp(), torch.zeros_like(student_cond)
        )
        support_q = torch.where(
            support_mask, teacher_cond.exp(), torch.zeros_like(teacher_cond)
        )
        support_r = torch.where(
            support_mask, teacher_cond - student_cond, torch.zeros_like(student_cond)
        )
        mean_r = (support_p * support_r).sum(dim=-1)
        if pgt_support.diagnostics.get("gain_support_definition") != "student_topk":
            raise ValueError(
                "CMT requires g_t computed on the exact Student Top-K support"
            )
        g = pgt_support.diagnostics["gain"].detach().float().clamp_min(0.0)
        g = torch.where(valid, g, torch.zeros_like(g))

        student_mass = pgt_support.diagnostics["student_support_mass"].detach().float()
        teacher_mass = pgt_support.diagnostics["teacher_support_mass"].detach().float()
        tiny = torch.finfo(torch.float32).tiny
        student_mass = student_mass.clamp_min(tiny)
        teacher_mass = teacher_mass.clamp_min(tiny)
        log_mass_ratio = torch.log(teacher_mass) - torch.log(student_mass)
        original_p = support_p * student_mass.unsqueeze(-1)
        original_q = support_q * teacher_mass.unsqueeze(-1)
        original_r = support_r + log_mass_ratio.unsqueeze(-1)

        sampled_ids = sampled_token_ids.long().unsqueeze(-1)
        in_support_matrix = candidate_ids.eq(sampled_ids) & support_mask
        in_support = in_support_matrix.any(dim=-1) & valid
        # IDs are unique on support, so summing selects exactly one slot.
        sampled_cond_student = torch.where(
            in_support_matrix, student_cond, torch.zeros_like(student_cond)
        ).sum(dim=-1)
        sampled_cond_teacher = torch.where(
            in_support_matrix, teacher_cond, torch.zeros_like(teacher_cond)
        ).sum(dim=-1)
        sampled_cond_r = sampled_cond_teacher - sampled_cond_student
        sampled_r = sampled_cond_r + torch.where(
            in_support, log_mass_ratio, torch.zeros_like(log_mass_ratio)
        )
        acceptance = torch.where(
            in_support,
            torch.exp(sampled_r.clamp(max=0.0)),
            torch.zeros_like(sampled_r),
        )
        # The raw truncated kernel is estimated directly under Y~p.  There is
        # deliberately no 1 / P_student(U) importance factor.
        transition_weight = acceptance
        # Kept as a compatibility diagnostic for older JSON consumers.  It is
        # identically one on an in-support transition and is never applied as
        # an inverse-coverage correction.
        coverage_correction = torch.where(
            in_support, torch.ones_like(student_mass), torch.zeros_like(student_mass)
        )
        # Backward-compatible diagnostic only: the production downstream term
        # is no longer gated by this raw teacher-deficit indicator.
        teacher_deficit = in_support & sampled_r.gt(0.0)
        signed_reachability_shift = torch.where(
            in_support,
            sampled_cond_r - mean_r,
            torch.zeros_like(sampled_cond_r),
        )
        # Freeze the raw common-support compatibility c_t at eta=0.  Direction
        # comes only from the centered conditional mirror tangent above; c_t is
        # a bounded confidence factor, not a differentiated min(p_eta, q).
        compatibility_weight = acceptance
        marginal_flux = compatibility_weight * signed_reachability_shift

        # Support-level overlap diagnostics can be summed exactly with no
        # successor evaluations.  The production action-conditioned future
        # term cannot be summed over U without evaluating counterfactual
        # successors.
        support_common_mass = torch.minimum(original_p, original_q).sum(dim=-1)
        conditional_support_common_mass = torch.minimum(support_p, support_q).sum(
            dim=-1
        )
        # Historical exact derivative of raw common mass.  It remains useful
        # for audit comparisons but does not enter marginal_flux or D_t.
        common_mass_derivative = torch.where(
            support_mask & original_p.lt(original_q),
            original_p * (support_r - mean_r.unsqueeze(-1)),
            torch.zeros_like(original_p),
        ).sum(dim=-1)

        # Raw cumulative return is retained only as a diagnostic.  The score
        # uses the local-baseline excess derivative below to remove
        # constant-opportunity length bias without episodic reward centering.
        cumulative_return, masses, cumulative_value = bellman_parallel_scan(
            g, transition_weight, valid, gamma=self.gamma
        )
        # E_t = R_t - g_t M_t is the expected suffix opportunity in excess of
        # the current state's own local opportunity.  It is deliberately not
        # implemented as reward centering by a batch average: that construction
        # is canonical for continuing average-reward problems, but can change
        # policy ordering in episodic problems with termination.
        local_excess = torch.where(
            valid,
            cumulative_return - g * masses,
            torch.zeros_like(cumulative_return),
        )
        successor_return = torch.zeros_like(cumulative_return)
        successor_mass = torch.zeros_like(masses)
        if cumulative_return.shape[1] > 1:
            successor_return[:, :-1] = torch.where(
                valid[:, 1:],
                cumulative_return[:, 1:],
                torch.zeros_like(cumulative_return[:, 1:]),
            )
            successor_mass[:, :-1] = torch.where(
                valid[:, 1:],
                masses[:, 1:],
                torch.zeros_like(masses[:, 1:]),
            )
        successor_excess = successor_return - g * successor_mass
        successor_value = torch.where(
            valid,
            successor_return / (successor_mass + 1e-8),
            torch.zeros_like(successor_return),
        )
        successor_excess_average = torch.where(
            valid,
            successor_excess / (successor_mass + 1e-8),
            torch.zeros_like(successor_excess),
        )
        downstream_effect = self.gamma * marginal_flux * successor_excess
        sequential_gain = self.successor_lambda * downstream_effect
        canonical_learning_value = torch.where(
            valid, g + sequential_gain, torch.zeros_like(g)
        )
        # Ablation arms deliberately reuse the exact CMT intermediates and the
        # same downstream KL/Gibbs allocator and weighted OPD objective.  The
        # default is byte-for-byte equivalent to the canonical score.  X is
        # the semantic successor excess (not R/M/V/H); D is the canonical
        # sequential marginal gain.
        if self.ablation_arm == "g":
            learning_value = g
            score_definition = "ablation_local_gain_g"
        elif self.ablation_arm == "g_x":
            learning_value = torch.where(
                valid, g + successor_excess, torch.zeros_like(g)
            )
            score_definition = "ablation_local_gain_plus_successor_excess_g_x"
        elif self.ablation_arm == "d_only":
            learning_value = torch.where(
                valid, sequential_gain, torch.zeros_like(sequential_gain)
            )
            score_definition = "ablation_sequential_gain_d_only"
        else:
            learning_value = canonical_learning_value
            score_definition = (
                "canonical_cmt_g_plus_sequential_gain"
                if self.ablation_arm == "g_d"
                else "support_matched_pgt_plus_local_baseline_excess_successor_derivative"
            )
        diagnostics = dict(pgt_support.diagnostics)
        diagnostics.update(
            gain=g,
            s_PGT=g,
            support_reverse_kl=(-mean_r).clamp_min(0.0),
            support_common_mass=torch.where(
                valid, support_common_mass, torch.zeros_like(support_common_mass)
            ),
            sampled_log_ratio=torch.where(
                in_support, sampled_r, torch.zeros_like(sampled_r)
            ),
            sampled_conditional_log_ratio=torch.where(
                in_support, sampled_cond_r, torch.zeros_like(sampled_cond_r)
            ),
            alignment=acceptance,
            transition_weight=transition_weight,
            support_coverage=torch.where(
                valid, student_mass, torch.zeros_like(student_mass)
            ),
            coverage_correction=coverage_correction,
            teacher_support_mass=torch.where(
                valid, teacher_mass, torch.zeros_like(teacher_mass)
            ),
            conditional_support_common_mass=torch.where(
                valid,
                conditional_support_common_mass,
                torch.zeros_like(conditional_support_common_mass),
            ),
            teacher_deficit=teacher_deficit.float(),
            signed_reachability_shift=signed_reachability_shift,
            compatibility_weight=compatibility_weight,
            marginal_flux=marginal_flux,
            downstream_effect=downstream_effect,
            common_mass_derivative=common_mass_derivative,
            R=cumulative_return,
            M=masses,
            V=cumulative_value,
            H=local_excess,
            successor_excess=successor_excess,
            successor_return=successor_return,
            successor_mass=successor_mass,
            successor_value=successor_value,
            successor_excess_total=successor_excess,
            successor_excess_average=successor_excess_average,
            # Compatibility alias retained for older selector JSON readers;
            # this is no longer raw successor R, but the baseline-subtracted
            # successor excess used by the production derivative.
            successor_R=successor_excess,
            sequential_gain=sequential_gain,
            sequential_gain_raw=sequential_gain,
            learning_value=learning_value,
            learning_value_raw=(
                learning_value
                if self.ablation_arm == "d_only"
                else canonical_learning_value
            ),
            s_CMT=learning_value,
            score_definition=score_definition,
            ablation_arm=self.ablation_arm,
            ablation_score=learning_value,
            transition_definition=(
                "truncated_original_union_common_mass_without_inverse_coverage"
            ),
            gamma=self.gamma,
            successor_lambda=self.successor_lambda,
        )
        for value in (
            g,
            mean_r,
            support_p,
            original_p,
            original_q,
            original_r,
            acceptance,
            transition_weight,
            signed_reachability_shift,
            compatibility_weight,
            marginal_flux,
            cumulative_return,
            masses,
            cumulative_value,
            local_excess,
            successor_return,
            successor_mass,
            successor_value,
            successor_excess,
            successor_excess_average,
            downstream_effect,
            sequential_gain,
            canonical_learning_value,
            learning_value,
        ):
            if value.requires_grad or value.grad_fn is not None:
                raise AssertionError("CMT statistics must be detached")
        return PGTOutput(
            learning_value,
            diagnostics,
            pgt_support.candidate_ids,
            pgt_support.student_candidate_log_probs,
            pgt_support.teacher_candidate_log_probs,
            pgt_support.support_mask,
        )
