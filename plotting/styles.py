"""The approved line colours/markers and in-axis corner legends."""
COLORS = {"qwen": "#f97316", "deepseek": "#1d4ed8", "thompson_sampling": "#f0134d",
          "cp_sat": "#8b5cf6", "dqn": "#06b6d4", "direct_llm": "#22c55e",
          "open_loop": "#f0134d", "static_mrsu": "#8b5cf6", "no_llm": "#06b6d4"}
MARKERS = {"qwen": "o", "deepseek": "D", "thompson_sampling": "*", "cp_sat": "D",
           "dqn": "s", "direct_llm": "^", "open_loop": "*", "static_mrsu": "D", "no_llm": "s"}
LABELS = {"thompson_sampling": "Thompson Sampling", "cp_sat": "CP-SAT", "dqn": "DQN",
          "direct_llm": "Direct LLM", "open_loop": "Open-loop FD-EMC",
          "static_mrsu": "Static mRSU", "no_llm": "FD-EMC w/o LLM"}


def label_and_style(record):
    method, model = record["method"], record.get("model") or "unspecified"
    key = method
    if method in ("fd_emc", "llm_models", "vehicle_count"):
        key = "deepseek" if "deepseek" in model.lower() else ("qwen" if "qwen" in model.lower() else "other_model")
        display = {"qwen3.7-flash": "Qwen3.7-flash", "deepseek-v4-flash": "DeepSeek-v4-flash"}.get(model, model)
        label = "FD-EMC("+display+")"
    else:
        label = LABELS.get(method, method)
    return label, COLORS.get(key, "#555555"), MARKERS.get(key, "o"), key
