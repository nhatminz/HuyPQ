# Fixed-checkpoint LIFT mechanism validation

This experiment tests whether the existing downstream score predicts the actual
downstream effect of a single local training update, conditional on local value.
The implementation lives in `b200_experiment/lift_mechanism.py`; it uses the
repository's `CMTSelector` implementation of LIFT's local/sequential scores.
It does not assume a positive result.

## Run

Use the project's training environment with its installed requirements. The
student must be a full Hugging Face checkpoint with the student tokenizer; the
teacher and dataset come from the config. A single GPU must fit both models and
the student's gradients/optimizer. Immutable model and optimizer copies occupy
CPU RAM. Launch one process (no `torchrun` or vLLM).

```bash
python -m b200_experiment.lift_mechanism \
  --checkpoint /absolute/path/to/checkpoint-000600 \
  --output-dir /absolute/path/to/lift-mechanism-step600 \
  --set paths.storage_root=/workspace/storage-shared \
  --set mechanism.rollouts=16 \
  --set mechanism.position_bins=2
```

Equivalent shell entry point: `bash scripts/run_lift_mechanism.sh` with the same
arguments (`PYTHON` optionally selects the interpreter). Output directories must
be new, to avoid mixing checkpoints or overwriting completed interventions.

Benchmark EVAL is disabled in the experiment config and enforced by the shell
launcher, including with a custom training config. The mechanism runner does
not call benchmark evaluation; its before/after rollouts are required experiment
measurements and still run.

Defaults in `configs/lift_mechanism.yaml`:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `mechanism.candidate_prompts` | 64 | Random dataset prompts after length filtering |
| `mechanism.candidate_responses` | 2 | Original-student trajectories per prompt |
| `mechanism.states_per_response` | 32 | Uniformly sampled visited decision states per trajectory, or all if shorter |
| `mechanism.g_bins` | 10 | Local-value rank quantiles; allowed range 10–20 |
| `mechanism.states_per_cell` | 4 | States per local-value/downstream-quintile cell |
| `mechanism.rollouts` | 8 | Independent continuations per state **per phase** |
| `mechanism.rollout_batch_size` | 1 | Generation batch size; scoring uses one trajectory at a time |
| `mechanism.bootstrap` | 2000 | Bootstrap replicates |
| `mechanism.position_bins` | 1 | Additional position stratification when greater than one |

The default primary sample has 10 × 5 × 4 = 200 interventions. Each intervention
has eight before and eight after rollouts. Increase states per cell and the
candidate pool for more precise estimates and position-stratified checks.
Insufficient primary cells fail explicitly before any update; a sparse optional
robustness design is marked `unavailable` with its reason in the analysis JSON.

Use the checkpoint's original scoring/training configuration when it differs
from the provided CMT defaults. Pass `--config /path/to/resolved_config.yaml`
and supply experiment settings via `--set mechanism.KEY=VALUE` as needed.
Restoring `optimizer.pt` takes precedence over config optimizer hyperparameters,
including learning rate. Without it, the experiment uses fresh AdamW moments
with the configured OPD learning rate, betas, and weight decay. The metadata
records which case occurred. An invalid existing optimizer file is an error,
not a silent fallback. Standard and repository `fsdp_full_v1` optimizer exports
are restored through the existing resume loader. LoRA-only checkpoints are not
supported; use a full checkpoint and `training.use_lora=false`.

## Score and intervention definitions

All candidate scoring happens before the first update. `G_t` is the existing
Student-Top-K conditional log-ratio variance (`gain`), and `D_raw` is the existing
`sequential_gain_raw`, including `cmt_successor_lambda` and `cmt_gamma`.
`D_tilde` is the downstream term after the configured `robust_cmt_correction`.
Its scale is computed once over **all valid states in the original candidate
trajectories**, before subsampling states or matching. This follows the existing
rollout-population correction and does not use intervention outcomes. To study
the raw score directly, set `selector.cmt_correction_mode=none` before running.

First form equal-count `G_t` bins; within each, form five equal-count `D_tilde`
bins. Uniformly sample the configured number from every cell. Seeded random tie
breaking balances cells even when many downstream scores are zero; tied scores
can span quintiles and need not represent distinct score ranges. Inspect the
saved balance tables. Matching is approximate within local-value bins, not exact
equality of every state's local value.

For every selected state:

1. Restore the same original student parameters, buffers, and optimizer state.
   Measure the exact vocabulary reverse KL at the prefix.
2. Generate M independent original-student continuations and measure
   `C_before = mean(sum_{k=1..H} gamma**k * KL(p || q) at s_(t+k))`.
3. Restore again and perform exactly one optimizer step on **only the local
   full-vocabulary reverse KL**. The teacher is frozen. Neither `G_t` nor
   `D_tilde` is part of the loss or a sample weight. Gradient clipping and AdamW
   settings come from OPD training; its Top-K PPO surrogate is not used here.
   Existing optimizer momentum and weight decay, if any, remain part of the
   specified intervention.
4. Measure `KL_after`, then draw M fresh continuations from the **updated**
   student using a separate random stream. Score `C_after` with the same cost.
5. Record `local_gain = KL_before - KL_after` and
   `downstream_gain_measured = C_before - C_after`. Restore the original model
   and optimizer even if the intervention raises an exception.

Student dropout is disabled for scores, local gradients, and rollouts. The
student sampling temperature and teacher scoring temperature are taken from
the training config; the same distributions are used for all KL measurements.
Sampling requires `rollout.top_p=1`, as the LIFT estimator assumes full-policy
student visitation.

The horizon follows the existing finite-response LIFT implementation:
`H = rollout.max_new_tokens - token_position - 1`, where positions are zero-based
response decision states. Re-rollouts use this remaining response cap, **not**
the realized original trajectory's EOS length. A trajectory includes the state
that predicts EOS but no state after EOS. Immediate EOS has downstream cost zero.
The root KL is never part of `C`; the first successor is discounted by `gamma`.
No common-mass acceptance weights or local-gain terms enter the measured KL cost.
The default CMT config uses `gamma=1` and a 4096-token total response cap.

## Outputs and analysis

- `resolved_config.yaml`, `metadata.json`: model/teacher paths, actual optimizer
  settings and restore status, scoring definitions, discount, horizon, seeds.
- `candidate_states.jsonl`, `selected_states.jsonl`: scores and exact prefix IDs
  captured before any intervention.
- `per_state.csv`: `state_id`, `prompt_id`, `token_position`, `G_t`, `D_tilde`,
  `D_raw`, `KL_before`, `KL_after`, `local_gain`, `C_before`, `C_after`,
  `downstream_gain_measured`, `G_bin`, `D_quantile`, horizon, rollout seeds,
  rollout count, cost sample standard deviations, and gradient norm.
- `rollout_costs.jsonl`: all M individual costs in each phase, keyed by state ID.
- `analysis/matched_G/`: primary quintile plot (PNG/PDF), quintile statistics,
  within-bin correlations, local-value/position balance table, and summary JSON.
- `analysis/matched_local_gain/`: re-bin the measured states on **actual
  local_gain**, recompute within-bin downstream quintiles, and balance cells anew.
  This is a post-intervention robustness check on the primary selected sample.
- `analysis/matched_G_position/` and `matched_local_gain_position/`: optional
  analyses crossing local-value bins with position quantiles before forming
  downstream quintiles. Every local-value/position stratum has equal weight.
- `complete.json`: written only after interventions and analysis complete.

The primary plotted mean is the arithmetic mean of the within-G-bin means for
each downstream quintile. The 95% percentile bootstrap resamples **states within
each fixed (matching-bin, downstream-quintile) cell**, retaining equal bin weight.
These are state-level conditional intervals; multiple states from the same
prompt can be dependent, and the intervals do not adjust for prompt clustering
or candidate-selection uncertainty. M rollout draws are not counted as M
independent interventions. A cell with one state yields a degenerate bootstrap
distribution and is suitable only for smoke testing.

Spearman uses average ranks for ties. Constant-score or constant-outcome bins
have undefined correlations (`null`); the aggregate is the equal-weight mean
over defined bins and reports their count. Sign agreement uses exact three-way
signs: positive with positive, negative with negative, zero only with zero.
The report includes overall and equal-bin sign agreement and both zero rates.

Re-run analysis without loading any model:

```bash
python -m b200_experiment.lift_mechanism_analysis \
  --csv /absolute/path/to/lift-mechanism-step600/per_state.csv \
  --output-dir /absolute/path/to/reanalysis \
  --g-bins 10 --position-bins 2 --bootstrap 5000 --seed 42
```

An increasing quintile curve, positive within-bin correlations, and similar
trends after measured-local-gain/position matching support the proposed
mechanism. A flat or reversed trend is a valid falsifying result. This experiment
measures a shared-network update's effect on student-visited distillation costs,
not downstream answer accuracy or a guaranteed causal interpretation of the
categorical LIFT surrogate.
