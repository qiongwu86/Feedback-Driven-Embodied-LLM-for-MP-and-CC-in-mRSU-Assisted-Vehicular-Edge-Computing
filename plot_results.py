"""Plot all available comparison and ablation results; never run experiments."""
import argparse
import json
from datetime import datetime, timedelta, timezone

from common.io import write_csv, write_json
from common.paths import ROOT, RESULTS, FIGURES, project_path
from plotting.loaders import load_records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(RESULTS))
    parser.add_argument("--output-dir", default=str(FIGURES))
    parser.add_argument("--config", default=str(ROOT/"configs/plots.json"))
    parser.add_argument("--groups", default=None)
    args = parser.parse_args(argv)
    settings = json.loads(project_path(args.config).read_text(encoding="utf-8"))
    if args.groups:
        settings["groups"] = args.groups.split(",")
    if settings["errorbar_width"] not in ("method-mean", "pointwise"):
        raise ValueError("Unknown error-bar width definition.")
    if not 0 <= settings["bar_step_pixels"] <= 16:
        raise ValueError("bar_step_pixels must be between 0 and 16.")
    if not set(settings["formats"]).issubset({"svg", "pdf", "png"}):
        raise ValueError("Supported export formats: svg, pdf, png.")
    records, provenance, skipped = load_records(args.results_root, settings.get("sources"))
    if not records:
        print("No complete experimental results found. No figures were created and no experiment was started.")
        return None
    from plotting.figures import export_all
    output = project_path(args.output_dir)/datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    report = export_all(records, output, settings)
    write_csv(output/"statistics.csv", report.pop("statistics"))
    write_json(output/"plot_manifest.json", {"settings": settings, "sources": provenance,
        "input_skipped": skipped, **report,
        "controller_modes": sorted({r["controller_mode"] for r in records}),
        "metric_definition": "ACHR is local RSU hit/(hit+not_cached); ACTD is covered-request average delay in ms",
        "errorbar_definition": ("Per-curve mean of pointwise 95% t-CI half-widths; not pointwise confidence intervals"
                                if settings["errorbar_width"] == "method-mean" else "Pointwise run-mean 95% t-CI"),
        "placement": "Qwen primary fixed on its data point; other bars move horizontally only on overlap",
        "single_run_uncertainty": "undefined; no error bar is fabricated"})
    print("Figures:", output)
    return output


if __name__ == "__main__":
    main()
