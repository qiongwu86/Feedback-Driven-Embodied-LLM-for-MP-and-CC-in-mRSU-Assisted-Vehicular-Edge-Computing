"""Unified capacity/round/vehicle figures from saved run-level data."""
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting.loaders import compatibility
from plotting.placement import position_error_bars
from plotting.statistics import metric_value, round_values, summarize
from plotting.styles import label_and_style

plt.rcParams.update({"font.family": "Arial", "font.size": 15,
                     "axes.labelsize": 20, "axes.labelweight": "bold",
                     "xtick.labelsize": 16, "ytick.labelsize": 16,
                     "legend.fontsize": 16, "svg.fonttype": "none",
                     "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})

GROUPS = {
    "comparison": {"fd_emc", "llm_models", "thompson_sampling", "cp_sat", "dqn"},
    "ablation": {"fd_emc", "direct_llm", "open_loop", "static_mrsu", "no_llm"},
    "models": {"fd_emc", "llm_models"},
    "vehicles": {"vehicle_count"},
}
ORDER = {"qwen": 0, "deepseek": 1, "thompson_sampling": 2, "direct_llm": 2,
         "cp_sat": 3, "open_loop": 3, "dqn": 4, "static_mrsu": 4, "no_llm": 5}


def place_legend(ax, legend, rows, series, metric):
    """Translate the intact legend to the nearest clear, fully in-axis position."""
    ax.figure.canvas.draw()
    box = legend.get_window_extent(ax.figure.canvas.get_renderer())
    left, bottom, right, top = ax.bbox.extents
    width, height = box.width, box.height
    if width > right-left-4 or height > top-bottom-4:
        raise ValueError("Legend cannot fit inside the axes at the approved font size.")
    obstacles = []
    for row in rows:
        if row["displayed_bar_low"] is not None:
            x = row["displayed_x_svg"]
            obstacles.append((x-row["cap_half_svg"]-.8, row["displayed_y_top_svg"]-.8,
                              x+row["cap_half_svg"]+.8, row["displayed_y_bottom_svg"]+.8))
    for item in series:
        for point in item["points"]:
            x, y = ax.transData.transform((point["x"], point["mean"]))
            obstacles.append((x-6, y-6, x+6, y+6))
    clamp_x = lambda x: max(left+2, min(right-width-2, x))
    clamp_y = lambda y: max(bottom+2, min(top-height-2, y))
    preferred = (clamp_x(box.x0), clamp_y(box.y0))
    xs = {preferred[0], left+2, right-width-2}
    ys = {preferred[1], bottom+2, top-height-2}
    for x0, y0, x1, y1 in obstacles:
        xs.update((clamp_x(x0-width-2), clamp_x(x1+2)))
        ys.update((clamp_y(y0-height-2), clamp_y(y1+2)))
    def collisions(position):
        x, y = position
        return sum(x < x1 and x+width > x0 and y < y1 and y+height > y0
                   for x0, y0, x1, y1 in obstacles)
    candidates = sorted(((x, y) for x in xs for y in ys),
                        key=lambda p: (collisions(p), (p[0]-preferred[0])**2+(p[1]-preferred[1])**2))
    x, y = candidates[0]
    anchor = (x+width, y if metric == "ACHR" else y+height)
    legend.set_bbox_to_anchor(ax.transAxes.inverted().transform(anchor), transform=ax.transAxes)
    ax.figure.canvas.draw()
    ax._legend_layout = {"bounds_pixels": [x, y, x+width, y+height],
                         "axes_bounds_pixels": [float(left), float(bottom), float(right), float(top)],
                         "preferred_translation_pixels": [x-preferred[0], y-preferred[1]],
                         "obstacle_overlap_count": collisions((x, y))}


def select_group(records, group):
    candidates = [r for r in records if r["method"] in GROUPS[group]]
    if not candidates:
        return [], []
    anchors = [r for r in candidates if r["method"] == "fd_emc" and "qwen" in (r.get("model") or "").lower()]
    if not anchors:
        anchors = [r for r in candidates if r["method"] in ("fd_emc", "llm_models")] or candidates
    anchor = max(anchors, key=lambda r: r["batch_order"])
    signature = compatibility(anchor, group == "vehicles")
    compatible, skipped = [], []
    for row in candidates:
        if compatibility(row, group == "vehicles") == signature:
            compatible.append(row)
        else:
            skipped.append({"group": group, "source": row["source"], "reason": "different simulation/latency configuration"})
    # A curve may not pool Mock with LLM, or combine separate result batches.
    choices = {}
    for row in compatible:
        label = label_and_style(row)[0]
        rank = (row["controller_mode"] == "llm", row["method"] == "fd_emc", row["batch_order"])
        if label not in choices or rank > choices[label][0]:
            choices[label] = (rank, row["source"], row["controller_mode"])
    selected = [r for r in compatible
                if (r["source"], r["controller_mode"]) == choices[label_and_style(r)[0]][1:]]
    return selected, skipped


def series_for(records, metric, view="capacity", round_capacity=200):
    grouped, styles = defaultdict(lambda: defaultdict(list)), {}
    for row in records:
        if view == "round" and row["capacity"] != round_capacity:
            continue
        if view == "vehicles" and row["capacity"] != round_capacity:
            continue
        label, color, marker, style_key = label_and_style(row)
        styles[label] = (color, marker, style_key)
        if view == "round":
            for i, value in enumerate(round_values(row, metric)):
                grouped[label][i+1].append((row["seed"], value))
        else:
            x = row["config"]["vehicle_num"] if view == "vehicles" else row["capacity"]
            grouped[label][x].append((row["seed"], metric_value(row, metric)))
    series = []
    for label in sorted(grouped, key=lambda name: (ORDER.get(styles[name][2], 99), name)):
        points = []
        for x, values in sorted(grouped[label].items()):
            seeds = [s for s, v in values]
            if len(seeds) != len(set(seeds)):
                raise ValueError("Duplicate seed in curve point: "+str((label, x)))
            points.append({"x": x, "seeds": seeds, **summarize([v for s, v in values])})
        series.append({"label": label, "color": styles[label][0], "marker": styles[label][1],
                       "style_key": styles[label][2], "points": points})
    return series


def draw(ax, series, metric, view, settings):
    xs = sorted({p["x"] for s in series for p in s["points"]})
    values = [p["mean"] for s in series for p in s["points"]]
    maximum_half = max((p["ci95_half"] or 0 for s in series for p in s["points"]), default=0)
    span = max(max(values)-min(values), maximum_half*2, 1.)
    ax.set_ylim(min(values)-maximum_half-.12*span, max(values)+maximum_half+.12*span)
    if len(xs) > 1:
        ax.set_xlim(min(xs), max(xs))
    else:
        margin = max(abs(xs[0])*.1, 1.)
        ax.set_xlim(xs[0]-margin, xs[0]+margin)
    ax.set_xticks(xs)
    ax.grid(True, color="#bfc1c5", linestyle="--", linewidth=.7)
    ax.set_axisbelow(True)
    xlabel = {"capacity": "mRSU/sRSU Cache Capacity", "round": "Decision Window", "vehicles": "Number of Vehicles"}[view]
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average Cache Hit Ratio (%)" if metric == "ACHR" else "Average Delay (ms)")
    ax.figure.canvas.draw()
    rows = []
    primary = next((s["label"] for s in series if s["style_key"] == "qwen"), series[0]["label"])
    for index, s in enumerate(series):
        halves = [p["ci95_half"] for p in s["points"] if p["ci95_half"] is not None]
        mean_half = sum(halves)/len(halves) if halves else None
        for p in s["points"]:
            half = mean_half if settings["errorbar_width"] == "method-mean" else p["ci95_half"]
            if p["n"] < 2:
                half = None
            low = None if half is None else p["mean"]-half
            high = None if half is None else p["mean"]+half
            px = float(ax.transData.transform((p["x"], p["mean"]))[0])
            yy = [float(ax.transData.transform((p["x"], v))[1]) for v in (low, high)] if half is not None else [None, None]
            rows.append({"label": s["label"], "method": "qwen" if s["label"] == primary else f"series_{index}",
                         "capacity": p["x"], "metric": metric, "n": p["n"], "mean": p["mean"],
                         "sd": p["sd"], "ci95_low": p["ci95_low"], "ci95_high": p["ci95_high"],
                         "displayed_halfwidth": half, "displayed_bar_low": low, "displayed_bar_high": high,
                         "displayed_y_top_svg": min(yy) if half is not None else None,
                         "displayed_y_bottom_svg": max(yy) if half is not None else None,
                         "original_x_svg": px, "displayed_x_svg": px, "cap_half_svg": 2.5,
                         "color": s["color"], "width_policy": settings["errorbar_width"]})
    position_error_bars(rows, settings["bar_step_pixels"], x_bounds=(ax.bbox.x0, ax.bbox.x1))
    for row in rows:
        if row["displayed_bar_low"] is None:
            continue
        px = row["displayed_x_svg"]
        x = ax.transData.inverted().transform((px, 0))[0]
        left = ax.transData.inverted().transform((px-row["cap_half_svg"], 0))[0]
        right = ax.transData.inverted().transform((px+row["cap_half_svg"], 0))[0]
        ax.vlines(x, row["displayed_bar_low"], row["displayed_bar_high"], color=row["color"], linewidth=1.15, zorder=2)
        ax.hlines([row["displayed_bar_low"], row["displayed_bar_high"]], left, right, color=row["color"], linewidth=1.15, zorder=2)
    for s in series:
        ax.plot([p["x"] for p in s["points"]], [p["mean"] for p in s["points"]],
                color=s["color"], marker=s["marker"], label=s["label"], linewidth=2.8,
                markersize=8, markeredgecolor="white", markeredgewidth=.8, zorder=3)
    location = "lower right" if metric == "ACHR" else "upper right"
    legend = ax.legend(loc=location, bbox_to_anchor=(.998, .002 if metric == "ACHR" else .998),
                       borderaxespad=0, framealpha=.86, prop={"size": 16, "weight": "bold"})
    place_legend(ax, legend, rows, series, metric)
    return rows


def export_all(records, output, settings):
    stats, skipped, used, panels, generated, legend_layout = [], [], set(), {}, [], {}
    for group in settings["groups"]:
        if group not in GROUPS:
            raise ValueError("Unknown figure group: "+group)
        selected, rejected = select_group(records, group)
        skipped.extend(rejected)
        if not selected:
            print("No compatible data for:", group)
            continue
        if group == "models" and len({label_and_style(r)[0] for r in selected}) < 2:
            print("Model comparison requires at least two available model backbones.")
            continue
        used.update(r["source"] for r in selected)
        views = ["vehicles"] if group == "vehicles" else ["capacity", "round"]
        for view in views:
            for metric in ("ACHR", "ACTD"):
                series = series_for(selected, metric, view, settings["round_capacity"])
                if not series:
                    continue
                stem = f"{group}_{view}_{metric.lower()}"
                fig, ax = plt.subplots(figsize=(9.8, 5.6), dpi=100)
                fig.subplots_adjust(left=.105, right=.975, top=.94, bottom=.18)
                rows = draw(ax, series, metric, view, settings)
                legend_layout[stem] = ax._legend_layout
                stats.extend(dict(figure=stem, **row) for row in rows)
                for ext in settings["formats"]:
                    fig.savefig(output/(stem+"."+ext), dpi=200)
                plt.close(fig)
                generated.append(stem)
                panels[stem] = series
    names = ["comparison_capacity_achr", "comparison_capacity_actd", "ablation_capacity_actd"]
    if all(n in panels for n in names):
        fig, axes = plt.subplots(1, 3, figsize=(29.4, 5.6), dpi=100)
        fig.subplots_adjust(left=.035, right=.992, bottom=.18, top=.94, wspace=.25)
        for ax, name, metric in zip(axes, names, ("ACHR", "ACTD", "ACTD")):
            draw(ax, panels[name], metric, "capacity", settings)
            legend_layout["combined_"+name] = ax._legend_layout
        for ext in settings["formats"]:
            fig.savefig(output/("figure3_combined."+ext), dpi=200)
        plt.close(fig)
        generated.append("figure3_combined")
    return {"statistics": stats, "skipped": skipped, "used_sources": sorted(used),
            "figures": generated, "legend_layout": legend_layout}
