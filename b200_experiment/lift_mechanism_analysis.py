"""Matched-state analysis for the fixed-checkpoint LIFT intervention experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def quantile_labels(values, bins: int, rng: np.random.Generator) -> np.ndarray:
    """Equal-count rank bins, with seeded random tie breaking independent of outcomes."""
    values = np.asarray(values, dtype=float)
    if bins < 1 or len(values) < bins or not np.isfinite(values).all():
        raise ValueError(f"Need at least {bins} finite values for quantile bins")
    order = np.lexsort((rng.random(len(values)), values))
    labels = np.empty(len(values), dtype=int)
    labels[order] = np.arange(len(values)) * bins // len(values) + 1
    return labels


def matched_sample(
    frame: pd.DataFrame,
    *,
    match_on: str = "G_t",
    bins: int = 10,
    per_cell: int | None = None,
    position_bins: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Match on local value, optionally crossed with position, then balance all cells.

    Quintiles are relative to each local-value/position stratum. Tied values may
    span adjacent quantiles; the balance tables expose this rather than inventing
    separated score thresholds. No outcome enters primary selection.
    """
    if not 10 <= bins <= 20 or position_bins < 1:
        raise ValueError("Use 10--20 local-value bins and positive position_bins")
    if per_cell is not None and per_cell < 1:
        raise ValueError("per_cell must be positive")
    result = frame.copy().reset_index(drop=True)
    if result.state_id.duplicated().any():
        raise ValueError("state_id must be unique")
    rng = np.random.default_rng(seed)
    result["match_bin"] = quantile_labels(result[match_on], bins, rng)
    result["position_bin"] = 1
    if position_bins > 1:
        result["position_bin"] = quantile_labels(
            result.token_position, position_bins, rng
        )
    result["D_quantile"] = 0
    for _, group in result.groupby(["match_bin", "position_bin"]):
        result.loc[group.index, "D_quantile"] = quantile_labels(group.D_tilde, 5, rng)
    groups = list(result.groupby(["match_bin", "position_bin", "D_quantile"]))
    if len(groups) != bins * position_bins * 5:
        raise ValueError(
            "Empty matched cell; increase the state pool or reduce position_bins"
        )
    available = min(len(group) for _, group in groups)
    count = available if per_cell is None else per_cell
    if count > available:
        raise ValueError(
            f"Need {count} states per cell, but smallest cell has {available}; "
            "increase candidate prompts/states or lower states_per_cell"
        )
    selected = pd.concat(
        [
            group.loc[rng.choice(group.index, count, replace=False)]
            for _, group in groups
        ]
    ).sort_values(["match_bin", "position_bin", "D_quantile", "state_id"])
    if match_on == "G_t":
        selected["G_bin"] = selected.match_bin
    return selected.reset_index(drop=True)


def spearman(x, y) -> float | None:
    x, y = pd.Series(x).rank(method="average"), pd.Series(y).rank(method="average")
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return None
    return float(np.corrcoef(x.to_numpy(), y.to_numpy())[0, 1])


def summarize(frame: pd.DataFrame, *, bootstrap: int = 2000, seed: int = 42) -> dict:
    """Equal-stratum means and a stratified, within-cell state bootstrap.

    This is a conditional state-level interval: it does not treat the M rollout
    samples as M independent interventions, nor claim prompt-cluster coverage.
    """
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    numeric = frame[["D_tilde", "downstream_gain_measured", "G_t", "local_gain"]]
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Analysis requires finite scores and outcomes")
    rng = np.random.default_rng(seed)
    strata = [group for _, group in frame.groupby(["match_bin", "position_bin"])]
    correlations = []
    for group in strata:
        agreement = np.sign(group.D_tilde) == np.sign(group.downstream_gain_measured)
        correlations.append(
            {
                "match_bin": int(group.match_bin.iloc[0]),
                "position_bin": int(group.position_bin.iloc[0]),
                "n": len(group),
                "spearman": spearman(group.D_tilde, group.downstream_gain_measured),
                "sign_agreement": float(agreement.mean()),
            }
        )
    estimates = []
    for quintile in range(1, 6):
        cells = [group[group.D_quantile == quintile] for group in strata]
        if any(cell.empty for cell in cells):
            raise ValueError("Every matching stratum must contain all five D quintiles")
        samples = np.zeros(bootstrap)
        for cell in cells:
            values = cell.downstream_gain_measured.to_numpy()
            samples += rng.choice(
                values, size=(bootstrap, len(values)), replace=True
            ).mean(1)
        samples /= len(cells)
        low, high = np.quantile(samples, [0.025, 0.975])
        estimates.append(
            {
                "D_quantile": quintile,
                "n": sum(len(cell) for cell in cells),
                "mean_downstream_gain": float(
                    np.mean([cell.downstream_gain_measured.mean() for cell in cells])
                ),
                "ci_low": float(low),
                "ci_high": float(high),
                "mean_G_t": float(np.mean([cell.G_t.mean() for cell in cells])),
                "mean_local_gain": float(
                    np.mean([cell.local_gain.mean() for cell in cells])
                ),
                "mean_D_tilde": float(np.mean([cell.D_tilde.mean() for cell in cells])),
                "mean_token_position": float(
                    np.mean([cell.token_position.mean() for cell in cells])
                ),
            }
        )
    defined = [row["spearman"] for row in correlations if row["spearman"] is not None]
    agreement = np.sign(frame.D_tilde) == np.sign(frame.downstream_gain_measured)
    return {
        "n_states": len(frame),
        "n_prompts": int(frame.prompt_id.nunique()),
        "n_strata": len(strata),
        "quintiles": estimates,
        "within_bin": correlations,
        "spearman_equal_bin_mean": float(np.mean(defined)) if defined else None,
        "spearman_defined_bins": len(defined),
        "sign_agreement_fraction": float(agreement.mean()),
        "sign_agreement_equal_bin_mean": float(
            np.mean([row["sign_agreement"] for row in correlations])
        ),
        "zero_D_fraction": float((frame.D_tilde == 0).mean()),
        "zero_measured_gain_fraction": float(
            (frame.downstream_gain_measured == 0).mean()
        ),
        "bootstrap_replicates": bootstrap,
        "bootstrap_unit": "state, resampled within each fixed matching/D cell",
        "ci_scope": "Conditional on scored checkpoint and matched sample; does not adjust for prompt clustering",
        "sign_convention": "Exact three-way sign; zero agrees only with zero",
        "spearman_aggregation": "Arithmetic mean over defined within-stratum correlations; constant bins are null",
    }


def save_analysis(
    frame: pd.DataFrame, directory: Path, *, bootstrap: int, seed: int
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    summary = summarize(frame, bootstrap=bootstrap, seed=seed)
    frame.to_csv(directory / "matched_states.csv", index=False)
    pd.DataFrame(summary["within_bin"]).to_csv(
        directory / "within_bin.csv", index=False
    )
    values = pd.DataFrame(summary["quintiles"])
    values.to_csv(directory / "quintiles.csv", index=False)
    balance = frame.groupby(["match_bin", "position_bin", "D_quantile"])[
        ["G_t", "local_gain", "D_tilde", "token_position"]
    ].agg(["count", "mean", "min", "max", "std"])
    balance.to_csv(directory / "balance.csv")
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(values.D_quantile, values.mean_downstream_gain, "o-", color="#2563eb")
    ax.vlines(
        values.D_quantile, values.ci_low, values.ci_high, color="#2563eb", linewidth=2
    )
    ax.axhline(0, color="0.5", linewidth=1, linestyle="--")
    ax.set(
        xticks=range(1, 6),
        xlabel=r"$\widetilde{D}_t$ quintile (lowest → highest)",
        ylabel="Mean measured downstream gain",
        title="Equal matching-bin weight; 95% bootstrap CI",
    )
    fig.tight_layout()
    fig.savefig(directory / "downstream_gain.png", dpi=180)
    fig.savefig(directory / "downstream_gain.pdf")
    plt.close(fig)
    return summary


def analyze(
    frame: pd.DataFrame,
    directory: Path,
    *,
    bins: int = 10,
    position_bins: int = 1,
    bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    primary = frame.copy()
    primary["match_bin"] = primary.G_bin
    primary["position_bin"] = 1
    reports = {
        "matched_G": save_analysis(
            primary, directory / "matched_G", bootstrap=bootstrap, seed=seed
        )
    }
    checks = [("matched_local_gain", "local_gain", 1)]
    if position_bins > 1:
        checks += [
            ("matched_G_position", "G_t", position_bins),
            ("matched_local_gain_position", "local_gain", position_bins),
        ]
    for name, field, positions in checks:
        try:
            matched = matched_sample(
                frame, match_on=field, bins=bins, position_bins=positions, seed=seed
            )
        except ValueError as error:
            reports[name] = {"status": "unavailable", "reason": str(error)}
            continue
        reports[name] = save_analysis(
            matched, directory / name, bootstrap=bootstrap, seed=seed
        )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(
        json.dumps(reports, indent=2, allow_nan=False) + "\n"
    )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--g-bins", type=int, default=10)
    parser.add_argument("--position-bins", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.csv, dtype={"state_id": str, "prompt_id": str})
    analyze(
        frame,
        args.output_dir,
        bins=args.g_bins,
        position_bins=args.position_bins,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
