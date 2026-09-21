from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import torch

from b200_experiment.selector_logging import (
    CMTTokenAuditLogger,
    SelectedTokenLogger,
    TokenScoreStatsLogger,
    cmt_d_only_comparison_summary,
    cmt_motivation_summary,
)


class FakeTokenizer:
    @staticmethod
    def convert_ids_to_tokens(ids):
        return [f"tok-{item}" for item in ids]

    @staticmethod
    def decode(ids, **_kwargs):
        return " ".join(f"tok-{item}" for item in ids)


class SelectorLoggingTests(unittest.TestCase):
    def test_ta_selected_tokens_are_incrementally_gzipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            mask = torch.tensor([[True, False], [False, True]])
            diagnostics = {
                key: torch.arange(4, dtype=torch.float32).reshape(2, 2)
                for key in ("D", "C", "D_norm", "C_norm", "s_TA")
            }
            logger = SelectedTokenLogger(
                temporary, FakeTokenizer(), "ta", chunk_steps=2
            )
            count = logger.write(
                step=1,
                dataset_indices=[10, 11],
                sample_ids=["a", "b"],
                response_ids=torch.tensor([[5, 6], [7, 8]]),
                selected_mask=mask,
                diagnostics=diagnostics,
            )
            self.assertEqual(count, 2)
            path = next((Path(temporary) / "selector_scores").glob("*.jsonl.gz"))
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(rows[0]["token_text"], "tok-5")
            self.assertEqual(rows[1]["sample_id"], "b")
            self.assertIn("s_TA", rows[0])

    def test_distributed_ranks_use_independent_gzip_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            mask = torch.tensor([[True]])
            diagnostics = {
                key: torch.ones(1, 1) for key in ("D", "C", "D_norm", "C_norm", "s_TA")
            }
            for rank in (0, 1):
                logger = SelectedTokenLogger(
                    temporary,
                    FakeTokenizer(),
                    "ta",
                    rank=rank,
                    world_size=2,
                )
                logger.write(
                    step=1,
                    dataset_indices=[10 + rank],
                    sample_ids=[str(rank)],
                    response_ids=torch.tensor([[5 + rank]]),
                    selected_mask=mask,
                    diagnostics=diagnostics,
                    batch_index_offset=rank,
                )
            paths = sorted((Path(temporary) / "selector_scores").glob("*.jsonl.gz"))
            self.assertEqual(len(paths), 2)
            self.assertIn("rank-00000", paths[0].name)
            self.assertIn("rank-00001", paths[1].name)

    def test_cmt_selected_tokens_include_bounded_kernel_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            mask = torch.tensor([[True, False]])
            keys = (
                "gain",
                "support_common_mass",
                "conditional_support_common_mass",
                "alignment",
                "transition_weight",
                "support_coverage",
                "teacher_deficit",
                "marginal_flux",
                "successor_excess",
                "sequential_gain",
                "learning_value",
                "s_CMT",
            )
            diagnostics = {key: torch.ones(1, 2) for key in keys}
            logger = SelectedTokenLogger(
                temporary, FakeTokenizer(), "cmt", chunk_steps=2
            )
            count = logger.write(
                step=1,
                dataset_indices=[10],
                sample_ids=["cmt"],
                response_ids=torch.tensor([[5, 6]]),
                selected_mask=mask,
                diagnostics=diagnostics,
            )
            self.assertEqual(count, 1)
            path = next((Path(temporary) / "selector_scores").glob("*.jsonl.gz"))
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                row = json.loads(next(handle))
            self.assertEqual(row["token_text"], "tok-5")
            self.assertEqual(row["transition_weight"], 1.0)
            self.assertEqual(row["conditional_support_common_mass"], 1.0)

    def test_compact_rac_stats_cover_all_values_and_bound_raw_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            diagnostics = {
                key: torch.linspace(0, 1, 100)
                for key in ("g", "alignment", "V", "z", "w")
            }
            logger = TokenScoreStatsLogger(
                temporary, "rac", interval=50, bins=10, raw_sample_size=7
            )
            path = logger.write(1, 100, diagnostics)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["scores"]["V"]["count"], 100)
            self.assertEqual(sum(payload["scores"]["V"]["histogram"]["counts"]), 100)
            self.assertEqual(len(payload["scores"]["V"]["sample"]), 7)
            self.assertIsNone(logger.write(2, 100, diagnostics))

    def test_compact_opd_stats_record_uniform_all_token_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            logger = TokenScoreStatsLogger(temporary, "opd", interval=50, bins=10)
            path = logger.write(1, 100, {"w": torch.ones(17)})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(tuple(payload["scores"]), ("w",))
            self.assertEqual(payload["scores"]["w"]["count"], 17)
            self.assertEqual(payload["scores"]["w"]["mean"], 1.0)

    def test_compact_cmt_stats_accept_all_selector_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            keys = (
                "s_CMT",
                "gain",
                "support_reverse_kl",
                "support_common_mass",
                "alignment",
                "transition_weight",
                "support_coverage",
                "coverage_correction",
                "teacher_deficit",
                "marginal_flux",
                "common_mass_derivative",
                "R",
                "M",
                "V",
                "H",
                "sequential_gain",
                "learning_value",
                "w_raw",
                "w",
            )
            diagnostics = {key: torch.linspace(0, 1, 9) for key in keys}
            logger = TokenScoreStatsLogger(temporary, "cmt", interval=10, bins=5)
            path = logger.write(1, 10, diagnostics)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(payload["scores"]), set(keys))
            self.assertEqual(payload["scores"]["learning_value"]["count"], 9)
            manifest = json.loads(
                (Path(temporary) / "token_score_stats" / "manifest.json").read_text()
            )
            self.assertIn("before bounding", manifest["weight_semantics"]["w_raw"])

    def test_cmt_audit_uses_global_topk_and_deduplicates_reasons(self):
        with tempfile.TemporaryDirectory() as temporary:
            global_diagnostics = {
                "gain": torch.tensor([9.0, 1.0, 2.0, 3.0]),
                "w_raw": torch.tensor([1.0, 2.0, 10.0, 3.0]),
                # The final top token overlaps the gain top token.
                "w": torch.tensor([4.0, 1.0, 2.0, 3.0]),
                "learning_value_robust": torch.tensor([9.0, 1.0, 2.0, 3.0]),
                "allocation_group_index": torch.zeros(4, dtype=torch.long),
                "allocation_optimizer_step": torch.full((4,), 10, dtype=torch.long),
                "allocation_processed": torch.ones(4, dtype=torch.bool),
            }
            all_rows = []
            for rank, start in ((0, 0), (1, 2)):
                local = {
                    key: torch.tensor([[float(rank * 2 + 1), float(rank * 2 + 2)]])
                    for key in (
                        "gain",
                        "successor_return",
                        "successor_mass",
                        "successor_value",
                        "successor_excess_total",
                        "successor_excess_average",
                        "sequential_gain",
                        "sequential_gain_raw",
                        "sequential_gain_robust",
                        "learning_value",
                        "learning_value_raw",
                        "learning_value_robust",
                        "correction_kappa",
                        "successor_excess",
                        "marginal_flux",
                        "transition_weight",
                        "w_raw",
                        "w",
                    )
                }
                # Preserve selection-driving local values for the three keys.
                for key in ("gain", "w_raw", "w"):
                    local[key] = global_diagnostics[key][start : start + 2].reshape(
                        1, 2
                    )
                logger = CMTTokenAuditLogger(
                    temporary,
                    FakeTokenizer(),
                    enabled=True,
                    interval=10,
                    top_k=1,
                    context_radius=1,
                    rank=rank,
                    world_size=2,
                )
                path = logger.write(
                    step=10,
                    rollout_id=3,
                    sample_ids=[f"sample-{rank}"],
                    dataset_indices=[100 + rank],
                    response_indices=[rank],
                    response_ids=torch.tensor([[10 + start, 11 + start]]),
                    valid_mask=torch.ones(1, 2, dtype=torch.bool),
                    local_diagnostics=local,
                    global_diagnostics=global_diagnostics,
                    global_start=start,
                    batch_index_offset=rank,
                    max_response_length=2,
                )
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    all_rows.extend(json.loads(line) for line in handle)
            self.assertEqual(len(all_rows), 2)
            by_token = {row["token_id"]: row for row in all_rows}
            self.assertEqual(
                set(by_token[10]["selection_reasons"]),
                {"global_top_gain", "global_top_w"},
            )
            self.assertEqual(by_token[12]["selection_reasons"], ["global_top_w_raw"])
            self.assertEqual(len({row["token_id"] for row in all_rows}), len(all_rows))

    def test_cmt_audit_ranks_per_group_and_breaks_weight_caps_by_robust_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            logger = CMTTokenAuditLogger(
                temporary,
                FakeTokenizer(),
                enabled=True,
                interval=10,
                top_k=1,
            )
            local = {key: torch.zeros(1, 4) for key in CMTTokenAuditLogger._VALUE_KEYS}
            local.update(
                gain=torch.tensor([[1.0, 2.0, 100.0, 200.0]]),
                w_raw=torch.tensor([[1.0, 1.0, 9.0, 10.0]]),
                w=torch.tensor([[2.0, 2.0, 2.0, 2.0]]),
                learning_value_robust=torch.tensor([[1.0, 8.0, 100.0, 200.0]]),
                allocation_score=torch.tensor([[1.0, 8.0, 100.0, 200.0]]),
            )
            global_diagnostics = {
                key: value.reshape(-1) for key, value in local.items()
            }
            global_diagnostics.update(
                allocation_group_index=torch.tensor([0, 0, 1, 1]),
                allocation_optimizer_step=torch.tensor([10, 10, 11, 11]),
                allocation_processed=torch.ones(4, dtype=torch.bool),
            )
            path = logger.write(
                step=10,
                rollout_id=0,
                sample_ids=["sample"],
                dataset_indices=[1],
                response_indices=[0],
                response_ids=torch.tensor([[10, 11, 12, 13]]),
                valid_mask=torch.ones(1, 4, dtype=torch.bool),
                local_diagnostics=local,
                global_diagnostics=global_diagnostics,
                global_start=0,
            )
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            # Step 11/group 1 is excluded. Among equal capped weights in group 0,
            # robust score selects token 11 rather than stable tensor order 10.
            final_top = [
                row for row in rows if "global_top_w" in row["selection_reasons"]
            ]
            self.assertEqual([row["token_id"] for row in final_top], [11])
            self.assertTrue(all(row["optimizer_step"] == 10 for row in rows))

    def test_motivation_summary_uses_one_allocation_group_and_rescue_cohorts(self):
        count = 400
        gain = torch.linspace(0.01, 0.2, count)
        future = torch.tensor([0.0, 10.0] * (count // 2))
        flux = torch.ones(count)
        d_raw = flux * future
        diagnostics = {
            "gain": gain,
            "successor_excess_average": future,
            "marginal_flux": flux,
            "sequential_gain_raw": d_raw,
            "sequential_gain_robust": torch.tanh(d_raw),
            "learning_value_robust": gain + torch.tanh(d_raw),
        }
        weights = torch.where(future > 0, torch.tensor(1.5), torch.tensor(0.5))
        summary, sparse = cmt_motivation_summary(
            diagnostics, weights, torch.arange(count)
        )
        self.assertGreater(summary["same_g_pair_count"], 0)
        self.assertGreater(summary["same_g_future_gap"], 0)
        self.assertGreater(summary["same_g_final_weight_gap"], 0)
        self.assertGreater(summary["rescue_weight_gap"], 0)
        self.assertLessEqual(
            sum(
                item["selection_reason"] == "low_g_high_future_rescued"
                for item in sparse
            ),
            8,
        )

    def test_d_only_summary_compares_corrected_score_and_canonical_weights(self):
        d_robust = torch.tensor([-2.0, -0.5, 0.5, 2.0])
        gain = torch.tensor([3.0, 2.0, 1.0, 0.0])
        actual = torch.tensor([0.5, 0.75, 1.25, 1.5])
        canonical = torch.tensor([1.5, 1.25, 0.75, 0.5])
        diagnostics = {
            "gain": gain,
            "marginal_flux": torch.tensor([1.0, 1.0, 0.0, 1.0]),
            "successor_excess": torch.tensor([-2.0, -0.5, 0.0, 2.0]),
            "sequential_gain_raw": 10.0 * d_robust,
            "sequential_gain_robust": d_robust,
            "canonical_score_robust": gain + d_robust,
            "allocation_score": d_robust.clone(),
        }
        summary, sparse = cmt_d_only_comparison_summary(
            diagnostics, actual, canonical, torch.arange(10, 14)
        )
        self.assertTrue(summary["allocation_score_matches_D_robust"])
        self.assertEqual(summary["D_robust_positive_rate"], 0.5)
        self.assertEqual(summary["D_robust_negative_rate"], 0.5)
        self.assertGreater(summary["mean_abs_weight_delta_vs_canonical"], 0)
        reasons = {item["selection_reason"] for item in sparse}
        self.assertIn("d_only_top_positive_D_robust", reasons)
        self.assertIn("d_only_largest_weight_change_vs_canonical", reasons)


if __name__ == "__main__":
    unittest.main()
