"""Offline checks for the standalone source layout and data contracts."""
import importlib
import json
import re
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from common.cli import parse_args, build_config
from common.io import RunOutput
from common.paths import ROOT
from plotting.loaders import load_records
from plotting.placement import error_bars_overlap, position_error_bars
from plotting.statistics import summarize
from simulation.environment import MRSUEnvironment


ENTRIES = ["main", "run_thompson_sampling", "run_cp_sat", "train_dqn", "test_dqn",
           "run_direct_llm", "run_open_loop", "run_static_mrsu", "run_no_llm",
           "compare_llm_models", "measure_llm_overhead", "scan_vehicle_count"]


class ReleaseTests(unittest.TestCase):
    def args(self, *args):
        return parse_args("test", {"seeds": [42], "capacities": [200]}, list(args))

    def record(self, config):
        logs = [{"round": i, "metrics": {"hit_count": 6, "not_cached_count": 4},
                 "round_delay_ms": 40, "tool_decision_error": None}
                for i in range(config.decision_rounds)]
        return {"summary": {"method": "tool_agent", "local_rsu_achr": .6, "average_delay_ms": 40},
                "round_logs": logs}

    def test_every_experiment_defaults_to_one_seed(self):
        for name in ENTRIES:
            module = importlib.import_module(name)
            args = parse_args("test", module.HYPERPARAMETERS, [])
            self.assertEqual(len(args.seeds), 1, name)

    def test_script_hyperparameters_support_more_than_one_seed(self):
        args = parse_args("test", {"seeds": [7, 42, 2026]}, [])
        self.assertEqual(args.seeds, [7, 42, 2026])
        self.assertEqual(parse_args("test", {"seeds": [7, 42]}, ["--seed", "9"]).seeds, [9])

    def test_capacity_parameters_and_default_config(self):
        self.assertEqual(self.args("--rsu-cache", "50").capacities, [50])
        self.assertEqual(self.args("--capacities", "50,100").capacities, [50, 100])
        self.assertEqual(self.args().latency_content_size_kbit, 800.)

    def test_dataset_loading_does_not_depend_on_working_directory(self):
        import os
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            try:
                os.chdir(temp)
                env = MRSUEnvironment(build_config(self.args(), 42, 200))
                self.assertTrue(env.train_data)
                self.assertTrue(env.test_data)
            finally:
                os.chdir(original)

    def test_timestamped_method_outputs_and_canonical_method_in_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args("--output-dir", temp)
            paths = []
            for _ in range(2):
                with RunOutput("fd_emc", args) as output:
                    config = build_config(args, 42, 200, output.path)
                    output.append(self.record(config), config, controller_mode="mock")
                    paths.append(output.path)
            self.assertNotEqual(*paths)
            for p in paths:
                self.assertEqual(p.parent, Path(temp)/"fd_emc")
                self.assertRegex(p.name, r"^\d{8}_\d{6}_\d{6}$")
                self.assertTrue(all((p/n).is_file() for n in ("config.json", "manifest.json", "results.json", "summary.csv")))
                self.assertEqual(json.loads((p/"manifest.json").read_text())["status"], "complete")
                self.assertIn("fd_emc", (p/"summary.csv").read_text(encoding="utf-8-sig"))

    def test_failed_batches_are_not_plotted(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args("--output-dir", temp)
            with self.assertRaisesRegex(RuntimeError, "test error"):
                with RunOutput("fd_emc", args):
                    raise RuntimeError("test error")
            records, _, skipped = load_records(temp)
            self.assertEqual(records, [])
            self.assertEqual(skipped[0]["reason"], "incomplete batch")

    def test_loader_selects_latest_complete_batch_without_pooling_separate_batches(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args("--output-dir", temp)
            for seed in (7, 42):
                with RunOutput("fd_emc", args) as output:
                    config = build_config(args, seed, 200, output.path)
                    output.append(self.record(config), config, controller_mode="mock")
            records, _, _ = load_records(temp)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["seed"], 42)

    def test_single_run_has_no_fabricated_uncertainty(self):
        stats = summarize([50])
        self.assertIsNone(stats["sd"])
        self.assertIsNone(stats["ci95_half"])

    def test_mock_and_llm_records_are_not_pooled_into_one_curve(self):
        from plotting.figures import select_group
        with tempfile.TemporaryDirectory() as temp:
            args = self.args("--output-dir", temp)
            for mode, seed in [("mock", 7), ("llm", 42)]:
                with RunOutput("fd_emc", args) as output:
                    config = build_config(args, seed, 200, output.path)
                    output.append(self.record(config), config, model="qwen3.7-flash", controller_mode=mode)
            rows, _, _ = load_records(temp)
            self.assertEqual(len(rows), 2)
            selected, _ = select_group(rows, "comparison")
            self.assertEqual([r["controller_mode"] for r in selected], ["llm"])

    def test_incompatible_simulation_settings_are_not_combined(self):
        from plotting.figures import select_group
        with tempfile.TemporaryDirectory() as temp:
            args = self.args("--output-dir", temp)
            for method in ("fd_emc", "thompson_sampling"):
                with RunOutput(method, args) as output:
                    config = build_config(args, 42, 200, output.path)
                    if method != "fd_emc":
                        config.vehicle_num = 60
                    output.append(self.record(config), config, controller_mode="algorithm")
            rows, _, _ = load_records(temp)
            selected, skipped = select_group(rows, "comparison")
            self.assertEqual([r["method"] for r in selected], ["fd_emc"])
            self.assertEqual(len(skipped), 1)

    def test_plot_statistics_use_runs_not_window_count_as_n(self):
        from plotting.figures import series_for
        args = self.args()
        rows = []
        for seed, value in [(7, .5), (42, .6), (2026, .7)]:
            config = build_config(args, seed, 200)
            result = self.record(config)
            result["summary"]["local_rsu_achr"] = value
            rows.append({**result, "method": "fd_emc", "seed": seed, "capacity": 200,
                         "config": asdict(config), "model": "qwen3.7-flash"})
        point = series_for(rows, "ACHR")[0]["points"][0]
        self.assertEqual(point["n"], 3)
        self.assertAlmostEqual(point["mean"], 60.)
        self.assertAlmostEqual(point["sd"], 10.)

    def test_explicit_legacy_main_result_can_be_read_without_private_paths(self):
        from common.io import write_json
        with tempfile.TemporaryDirectory() as temp:
            config = build_config(self.args(), 42, 200)
            source = Path(temp)/"old_result.json"
            write_json(source, {"config_by_capacity": {"200": asdict(config)},
                               "results_by_capacity": {"200": {"tool_agent": self.record(config)}}})
            rows, _, _ = load_records(temp, [str(source)])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["seed"], 42)
            self.assertEqual(rows[0]["controller_mode"], "legacy_unverified")

    def test_run_level_student_t_statistics(self):
        stats = summarize([1, 2, 3])
        self.assertEqual(stats["mean"], 2)
        self.assertEqual(stats["sd"], 1)
        self.assertAlmostEqual(stats["ci95_half"], 2.48413771175)

    def test_main_error_bar_stays_fixed_and_separate_bar_does_not_move(self):
        rows = [dict(method=m, capacity=200, original_x_svg=400, displayed_x_svg=400,
                     displayed_bar_low=1, displayed_y_top_svg=y, displayed_y_bottom_svg=y+20, cap_half_svg=2.5)
                for m, y in [("qwen", 100), ("other", 110), ("separate", 160)]]
        position_error_bars(rows)
        self.assertEqual([r["visual_x_offset_svg"] for r in rows], [0, 8, 0])
        self.assertTrue(all(not error_bars_overlap(a, b) for i, a in enumerate(rows) for b in rows[i+1:]))

    def test_legend_moves_as_a_whole_inside_axes_away_from_large_error_bars(self):
        from plotting.figures import draw, plt
        fig, ax = plt.subplots(figsize=(9.8, 5.6), dpi=100)
        series = [{"label": label, "style_key": key, "color": color, "marker": "o",
                   "points": [{"x": x, "mean": mean, "n": 3, "sd": 10,
                               "ci95_half": 25, "ci95_low": mean-25, "ci95_high": mean+25}
                              for x, mean in [(20, 63), (40, 60)]]}
                  for label, key, color in [("FD-EMC(Qwen3.7-flash)", "qwen", "#f97316"),
                                            ("Direct LLM", "direct_llm", "#22c55e")]]
        draw(ax, series, "ACTD", "capacity", {"errorbar_width": "method-mean", "bar_step_pixels": 8})
        layout = ax._legend_layout
        self.assertEqual(layout["obstacle_overlap_count"], 0)
        left, bottom, right, top = layout["axes_bounds_pixels"]
        x0, y0, x1, y1 = layout["bounds_pixels"]
        self.assertTrue(left <= x0 < x1 <= right and bottom <= y0 < y1 <= top)
        plt.close(fig)

    def test_empty_results_does_not_create_figures_or_start_experiments(self):
        import plot_results
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/"figures"
            result = plot_results.main(["--results-root", str(Path(temp)/"results"), "--output-dir", str(out)])
            self.assertIsNone(result)
            self.assertFalse(out.exists())

    def test_readme_scope_and_excluded_files(self):
        text = (ROOT/"README.md").read_text(encoding="utf-8")
        self.assertIn("unpublished", text)
        self.assertIn("Feedback-Driven Embodied LLM for Coupled", text)
        self.assertEqual(re.findall(r"^## (.*)$", text, re.M), ["Installation", "Data"])
        self.assertFalse((ROOT/".env.example").exists())
        self.assertFalse((ROOT/"experiments/common.py").exists())


if __name__ == "__main__":
    unittest.main()
