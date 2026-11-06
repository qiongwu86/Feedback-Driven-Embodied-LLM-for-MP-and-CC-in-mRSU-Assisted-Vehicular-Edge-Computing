"""Placement; algorithm logic preserved from the research implementation."""
from __future__ import annotations
import math


def error_bars_overlap(first, second, stroke_width=1.6):
    """Detect touching/overlap of error-bar glyph bounds, including caps/strokes."""
    horizontal = (abs(first["displayed_x_svg"]-second["displayed_x_svg"]) <=
                  first["cap_half_svg"]+second["cap_half_svg"]+stroke_width)
    vertical = (max(first["displayed_y_top_svg"], second["displayed_y_top_svg"]) <=
                min(first["displayed_y_bottom_svg"], second["displayed_y_bottom_svg"])+stroke_width)
    return horizontal and vertical


def position_error_bars(records, bar_step=8.0, primary_method="qwen", x_bounds=(82.0, 932.0)):
    """Keep the primary bar on its point; move other bars only on collision.

    Work independently at each capacity, prioritising Qwen then legend order.
    Try the original x first, then the nearest free columns, inside the axes.
    No point values, interval endpoints, cap sizes, or y positions are changed.
    """
    for cap in dict.fromkeys(row["capacity"] for row in records):
        rows = [row for row in records if row["capacity"] == cap]
        rows.sort(key=lambda row: row["method"] != primary_method)
        placed = []
        for row in rows:
            row["displayed_x_svg"] = row["original_x_svg"]
            row["visual_x_offset_svg"] = 0.0
            row["placement_policy"] = "primary_fixed_overlap_only"
            if row["displayed_bar_low"] is None:
                row["placement_reason"] = "no_bar"
                continue
            if row["method"] == primary_method:
                row["placement_reason"] = "fixed_primary"
            elif not bar_step:
                row["placement_reason"] = "dodging_disabled"
            elif not any(error_bars_overlap(row, other) for other in placed):
                row["placement_reason"] = "no_overlap"
            else:
                row["placement_reason"] = "overlap_dodged"
                # Positive first breaks equal-distance ties deterministically;
                # endpoint groups naturally move only towards the plot interior.
                count = int(math.ceil((x_bounds[1]-x_bounds[0])/bar_step))
                for offset in (sign*distance*bar_step for distance in range(1, count+1)
                               for sign in (1, -1)):
                    candidate = row["original_x_svg"]+offset
                    if not x_bounds[0] <= candidate <= x_bounds[1]:
                        continue
                    row["displayed_x_svg"] = candidate
                    if not any(error_bars_overlap(row, other) for other in placed):
                        row["visual_x_offset_svg"] = offset
                        break
                else:
                    raise ValueError("No collision-free error-bar position: " + str((row["method"], cap)))
            placed.append(row)
    return records

