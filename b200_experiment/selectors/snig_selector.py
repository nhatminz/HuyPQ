from __future__ import annotations

import torch

from .cmt_selector import _shared_kernel_samples
from .pgt_selector import PGTOutput
from .rac_selector import bellman_parallel_scan


class SNIGSelector:
    """Support-matched SNIG successor information-geometric selector.

    SNIG keeps the PGT local gain and the CMT truncated common-mass transition
    kernel, but scores the successor effect through the normalized log
    potential ``Phi=log1p(R/M)``.  The resulting local surrogate is

        score_t = g_t + lambda * gamma * delta_t
            * (R_{t+1} - g_t M_{t+1}) / (M_t (M_t + R_t)).

    ``R`` and ``M`` are finite-horizon suffix recurrences with hard padding
    boundaries.  This is a frozen-descendant categorical surrogate; it is not
    a claim about the exact effect of updating shared neural parameters.
    The support/kernel construction is shared with CMT so conditional local
    geometry and unnormalized Top-K accessibility cannot diverge.
    """

    def __init__(self, gamma: float = 1.0, successor_lambda: float = 1.0):
        if not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("SNIG gamma must be in [0, 1]")
        if float(successor_lambda) < 0.0:
            raise ValueError("SNIG successor lambda must be non-negative")
        self.gamma = float(gamma)
        self.successor_lambda = float(successor_lambda)

    @torch.no_grad()
    def compute_scores(
        self,
        pgt_support: PGTOutput,
        sampled_token_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> PGTOutput:
        shared = _shared_kernel_samples(
            pgt_support, sampled_token_ids, valid_mask
        )
        valid = shared["valid"]
        g = shared["g"]
        transition_weight = shared["transition_weight"]
        delta = shared["marginal_flux"]

        cumulative_return, masses, _ = bellman_parallel_scan(
            g, transition_weight, valid, gamma=self.gamma
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
        denominator = torch.where(
            valid,
            masses * (masses + cumulative_return),
            torch.ones_like(masses),
        )
        # M>=1 and R>=0 on valid positions by construction.  Clamp only as a
        # numerical guard for floating-point roundoff, not as a tunable scale.
        denominator = denominator.clamp_min(1.0)
        successor_utility = torch.where(
            valid,
            self.gamma * delta * successor_excess / denominator,
            torch.zeros_like(g),
        )
        phi = torch.where(
            valid,
            torch.log1p(cumulative_return / masses.clamp_min(1.0)),
            torch.zeros_like(g),
        )
        score = torch.where(
            valid,
            g + self.successor_lambda * successor_utility,
            torch.zeros_like(g),
        )

        diagnostics = dict(pgt_support.diagnostics)
        diagnostics.update(
            gain=g,
            s_PGT=g,
            support_common_mass=torch.where(
                valid,
                shared["support_common_mass"],
                torch.zeros_like(g),
            ),
            conditional_support_common_mass=torch.where(
                valid,
                shared["conditional_support_common_mass"],
                torch.zeros_like(g),
            ),
            sampled_log_ratio=torch.where(
                shared["in_support"],
                shared["sampled_r"],
                torch.zeros_like(g),
            ),
            sampled_conditional_log_ratio=torch.where(
                shared["in_support"],
                shared["sampled_cond_r"],
                torch.zeros_like(g),
            ),
            alignment=shared["acceptance"],
            transition_weight=transition_weight,
            support_coverage=torch.where(
                valid, shared["student_mass"], torch.zeros_like(g)
            ),
            coverage_correction=shared["coverage_correction"],
            teacher_deficit=shared["teacher_deficit"].float(),
            marginal_flux=delta,
            kernel_derivative=delta,
            common_mass_derivative=shared["common_mass_derivative"],
            R=cumulative_return,
            M=masses,
            R_next=successor_return,
            M_next=successor_mass,
            successor_excess=successor_excess,
            Phi=phi,
            successor_utility=successor_utility,
            learning_value=score,
            s_SNIG=score,
            score_definition=(
                "snig_pgt_gain_plus_normalized_log_potential_successor_derivative"
            ),
            transition_definition=(
                "truncated_original_union_common_mass_without_inverse_coverage"
            ),
            gamma=self.gamma,
            successor_lambda=self.successor_lambda,
        )
        for name, value in (
            ("gain", g),
            ("transition_weight", transition_weight),
            ("kernel_derivative", delta),
            ("R", cumulative_return),
            ("M", masses),
            ("R_next", successor_return),
            ("M_next", successor_mass),
            ("Phi", phi),
            ("successor_utility", successor_utility),
            ("score", score),
        ):
            if value.requires_grad or value.grad_fn is not None:
                raise AssertionError(f"SNIG {name} must be detached")
            if not torch.isfinite(value[valid]).all():
                raise FloatingPointError(f"SNIG {name} contains non-finite values")
        if not bool(
            ((transition_weight[valid] >= 0.0)
             & (transition_weight[valid] <= 1.0)).all()
        ):
            raise AssertionError("SNIG transition weights must lie in [0, 1]")

        return PGTOutput(
            score,
            diagnostics,
            pgt_support.candidate_ids,
            pgt_support.student_candidate_log_probs,
            pgt_support.teacher_candidate_log_probs,
            pgt_support.support_mask,
        )
