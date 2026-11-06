"""Llm overhead; algorithm logic preserved from the research implementation."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import math
import random
import re
import statistics
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
import numpy as np
from agents.llm_tool_agent import LLMToolAgent, _extract_json_object
from agents.mock_tool_agent import MockToolAgent


MEASUREMENT_VERSION = "llm_api_only_v2_3"


API_LATENCY = "端到端LLM API请求时延_ms"


API_PERIOD_RATIO = "端到端LLM API请求时延占周期比例"


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")


def now():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def to_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(vars(value))


def token_values(usage):
    usage = to_dict(usage)
    prompt_details = to_dict(usage.get("prompt_tokens_details"))
    values = {
        "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
        "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": prompt_details.get("cached_tokens", usage.get("prompt_cache_hit_tokens")),
    }
    # Missing provider fields remain null, not an invented zero or character estimate.
    return {key: int(value) if isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0 else None for key, value in values.items()}


def recorded_usage(usage):
    data = dict(to_dict(usage))
    data.pop("reasoning_tokens", None)
    for key in ("completion_tokens_details", "output_tokens_details"):
        if data.get(key) is not None:
            details = dict(to_dict(data[key]))
            details.pop("reasoning_tokens", None)
            data[key] = details
    return data or None


def exception_info(exc):
    return {"异常类型": type(exc).__name__, "HTTP状态码": getattr(exc, "status_code", None),
            "请求ID": getattr(exc, "request_id", None)}


def valid_json_object(text):
    if not isinstance(text, str):
        return False
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            if isinstance(json.loads(candidate), dict):
                return True
        except (ValueError, TypeError):
            pass
    return False


def retryable(exc):
    from openai import APIConnectionError
    headers = getattr(getattr(exc, "response", None), "headers", {})
    directive = headers.get("x-should-retry")
    if directive in ("true", "false"):
        return directive == "true"
    status = getattr(exc, "status_code", None)
    return isinstance(exc, APIConnectionError) or status in (408, 409, 429) or (status is not None and status >= 500)


def retry_delay(exc, attempt, base_delay):
    headers = getattr(getattr(exc, "response", None), "headers", {})
    delay = None
    try:
        if "retry-after-ms" in headers:
            delay = float(headers["retry-after-ms"]) / 1000
        elif "retry-after" in headers:
            try:
                delay = float(headers["retry-after"])
            except ValueError:
                delay = parsedate_to_datetime(headers["retry-after"]).timestamp() - time.time()
    except (ValueError, TypeError, OverflowError):
        pass
    if delay is not None and 0 < delay <= 60:
        return delay
    # OS randomness does not consume the simulation's seeded random sequence.
    return min(base_delay * (2 ** min(attempt, 20)), 8.0) * random.SystemRandom().uniform(0.75, 1.0)


class TelemetryStore:
    def __init__(self, output):
        self.output = Path(output)
        self.calls, self.attempts, self.windows = [], [], []

    def checkpoint(self, status):
        from common.io import write_json
        write_json(self.output / "llm_telemetry.json", {
            "status": status, "calls": self.calls, "attempts": self.attempts,
            "windows": self.windows, "api_wait_injected_into_simulation": False,
        })
        write_json(self.output / "llm_overhead_summary.json", {
            "status": status,
            "end_to_end_api_latency_ms": describe_values(row.get(API_LATENCY) for row in self.calls),
            "llm_decision_success_count": sum(bool(row.get("LLM决策成功")) for row in self.calls),
            "api_request_attempt_count": sum(row.get("实际API请求次数", 0) for row in self.calls),
            "window_count": len(self.windows),
            "api_wait_injected_into_simulation": False,
            "latency_scope": "non-streaming SDK call until full response; not pure server inference",
        })


class MeasuredAPIError(RuntimeError):
    pass


class MeasuredLLMAgent(LLMToolAgent):
    def __init__(self, args, model, api_key, store, identity):
        super().__init__(api_key=api_key or "offline-test", base_url=args.base_url, model_name=model,
                         max_context_chars=args.max_context_chars, api_key_env=args.api_key_env)
        self.args = args
        self.store = store
        self.identity = identity
        self.round_index = None
        self.current = None
        self.mock_decision = None
        self.started_calls = 0
        self.candidate_ids = set()
        self.enable_thinking = None if args.enable_thinking is None else args.enable_thinking == "true"

    def decide_tools(self, context):
        self.current = {
            **self.identity, "决策窗口": self.round_index, "开始时间": now(),
            "本运行首次调用": self.started_calls == 0, "本进程首次调用": not self.store.calls,
            "模式": "模拟测试" if self.args.agent == "mock" else "真实API",
            "LLM决策成功": False, "LLM调用执行成功": False, "API响应成功": False,
            "响应JSON有效": False, "决策字段有效": False, "使用兼容默认或转换动作": False,
            "实际API请求次数": 0, "重试次数": 0, "重试等待耗时_ms": 0.0,
            API_LATENCY: None,
            "API累计请求耗时_ms": None, "API总耗时含重试_ms": None,
            "context字符数": len(json.dumps(context, ensure_ascii=False)),
            "streaming": False, "enable_thinking": self.enable_thinking,
            "temperature": self.args.temperature, "top_p": self.args.top_p,
            **{key: None for key in TOKEN_FIELDS},
        }
        self.started_calls += 1
        self.store.calls.append(self.current)
        self.candidate_ids = {int(item["hotspot_id"]) for item in context.get("candidate_hotspots", [])}
        try:
            if self.args.agent == "mock":
                self.mock_decision = MockToolAgent().decide_tools(context).to_dict()
            decision = super().decide_tools(context)
            self.current["LLM调用执行成功"] = True
            self.current["LLM决策成功"] = self.current["响应JSON有效"] and self.current["决策字段有效"]
            self.current["使用兼容默认或转换动作"] = not self.current["LLM决策成功"]
            return decision
        except Exception as exc:
            self.current.setdefault("错误信息", exception_info(exc))
            # Do not propagate response bodies or accidentally log authentication secrets.
            raise MeasuredAPIError("LLM决策失败: " + self.current["错误信息"]["异常类型"]) from None
        finally:
            self.current["结束时间"] = now()

    def _build_prompt(self, context):
        prompt = super()._build_prompt(context)
        self.current.update({
            "完整prompt字符数": len(prompt), "完整prompt字节数": len(prompt.encode("utf-8")),
            "prompt_SHA256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "context是否截断": self.current["context字符数"] > self.max_context_chars,
        })
        return prompt

    def _client_or_raise(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  max_retries=0, timeout=self.args.api_timeout)
        return self._client

    def _call_json(self, prompt):
        if self.args.agent == "mock":
            self.current.update({"API响应成功": None, "响应JSON有效": True, "决策字段有效": True,
                                 "输出字符数": len(json.dumps(self.mock_decision))})
            return self.mock_decision
        client = self._client_or_raise()
        kwargs = dict(model=self.model, messages=[{"role": "user", "content": prompt}],
                      temperature=self.args.temperature, top_p=self.args.top_p, stream=False)
        if self.enable_thinking is not None:
            kwargs["extra_body"] = {"enable_thinking": self.enable_thinking}
        if self.args.max_output_tokens is not None:
            kwargs["max_tokens"] = self.args.max_output_tokens
        start = time.perf_counter()
        cumulative = 0.0
        response = None
        try:
            for index in range(self.args.max_retries + 1):
                attempt = {**self.identity, "决策窗口": self.round_index, "尝试序号": index + 1,
                           "开始时间": now(), "请求成功": False, "重试等待耗时_ms": 0.0}
                self.current["实际API请求次数"] += 1
                self.current["重试次数"] = index
                attempt_start = time.perf_counter()
                error = None
                try:
                    response = client.chat.completions.create(**kwargs)
                    request_ms = (time.perf_counter() - attempt_start) * 1000
                    self.current[API_LATENCY] = request_ms
                    attempt["请求成功"] = True
                    attempt["请求ID"] = getattr(response, "_request_id", None)
                    attempt.update(token_values(getattr(response, "usage", None)))
                except BaseException as exc:
                    request_ms = (time.perf_counter() - attempt_start) * 1000
                    error = exc
                    attempt.update(exception_info(exc))
                finally:
                    attempt["请求耗时_ms"] = request_ms
                    attempt["结束时间"] = now()
                    cumulative += attempt["请求耗时_ms"]
                    self.store.attempts.append(attempt)
                if error is None:
                    break
                self.current["错误信息"] = exception_info(error)
                if not isinstance(error, Exception) or index >= self.args.max_retries or not retryable(error):
                    raise error
                delay = retry_delay(error, index, self.args.retry_base_delay)
                attempt["计划重试等待_s"] = delay
                wait_start = time.perf_counter()
                try:
                    time.sleep(delay)
                finally:
                    waited = (time.perf_counter() - wait_start) * 1000
                    attempt["重试等待耗时_ms"] = waited
                    self.current["重试等待耗时_ms"] += waited
        finally:
            self.current["API累计请求耗时_ms"] = cumulative
            self.current["API总耗时含重试_ms"] = (time.perf_counter() - start) * 1000
        self.current["API响应成功"] = True
        self.current.pop("错误信息", None)
        self.current.update(token_values(getattr(response, "usage", None)))
        self.current["usage"] = recorded_usage(getattr(response, "usage", None))
        self.current["返回模型名称"] = getattr(response, "model", None)
        self.current["响应ID"] = getattr(response, "id", None)
        self.current["请求ID"] = getattr(response, "_request_id", None)
        self.current["system_fingerprint"] = getattr(response, "system_fingerprint", None)
        choice = response.choices[0]
        self.current["finish_reason"] = getattr(choice, "finish_reason", None)
        content = choice.message.content
        self.current["输出字符数"] = len(content) if isinstance(content, str) else None
        self.current["输出字节数"] = len(content.encode("utf-8")) if isinstance(content, str) else None
        data = _extract_json_object(content)
        self.current["响应JSON有效"] = valid_json_object(content)
        self.current["决策字段有效"] = (
            isinstance(data, dict) and type(data.get("selected_hotspot_id")) is int
            and data["selected_hotspot_id"] in self.candidate_ids
            and type(data.get("update_mrsu_cache")) is bool
            and type(data.get("update_frsu_cache")) is bool
        )
        return data

    def close(self):
        if self._client is not None:
            self._client.close()


class WindowObserver:
    def __init__(self, config, agent, store, identity):
        self.config = config
        self.agent = agent
        self.store = store
        self.identity = identity
        self.info = {}

    def __call__(self, event, **payload):
        if event == "round_start":
            self.info = {**self.identity, "决策窗口": payload["round_index"], "开始时间": now(),
                         "本窗口模拟时长_s": payload["window_ticks"] * self.config.dt}
            self.agent.round_index = payload["round_index"]
            self.agent.current = None
        if event in ("path_end", "estimate_path_end"):
            prefix = "实际路径" if event == "path_end" else "预评估路径"
            self.info[prefix + "求解器"] = payload["solver"]
            self.info[prefix + "状态"] = payload["status"]
        if event == "agent_end":
            self.info["LLM是否回退"] = payload["fallback"]
            if (payload["fallback"] or not (self.agent.current or {}).get("LLM决策成功")) and self.agent.args.on_api_error == "abort":
                raise RuntimeError("LLM失败，按--on-api-error abort停止；本次API记录将保存。")
        if event == "round_complete":
            self.finish(payload["log"])

    def finish(self, log):
        row = dict(self.info)
        period_ms = row["本窗口模拟时长_s"] * 1000
        call = self.agent.current or {}
        for key, value in call.items():
            if key not in self.identity and key not in {"开始时间", "结束时间", "usage"}:
                row[key] = value
        row["结束时间"] = now()
        request_ms = call.get(API_LATENCY)
        row[API_PERIOD_RATIO] = None if request_ms is None else request_ms / period_ms
        row["LLM API请求是否超过决策周期"] = None if request_ms is None else request_ms > period_ms
        # Retry-inclusive waiting is diagnostic, not the single-request latency above.
        row["API耗时占周期比例"] = None if call.get("API总耗时含重试_ms") is None else call["API总耗时含重试_ms"] / period_ms
        row["实际路径是否回退"] = str(log.get("path_plan_status", "")).startswith("fallback")
        row["更新mRSU缓存"] = log["tool_decision"]["update_mrsu_cache"]
        row["更新sRSU缓存"] = log["tool_decision"]["update_frsu_cache"]
        row["实际是否调用缓存更新"] = log["cache_update_used"]
        metrics = log["metrics"]
        row.update({"本地缓存命中率": metrics["local_rsu_chr"], "系统缓存命中率": metrics["chr"],
                    "内容传输时延_ms": log["round_delay_ms"], "请求数": metrics["request_count"],
                    "命中数": metrics["hit_count"], "未覆盖请求数": metrics["not_covered_count"],
                    "覆盖内未缓存请求数": metrics["not_cached_count"], "LLM开销是否注入仿真": False})
        self.store.windows.append(row)
        # Checkpoint after each completed window, outside the API measurement.
        with (self.store.output / "逐窗口记录.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"开销": row, "原始窗口记录": log}, ensure_ascii=False, allow_nan=False) + "\n")
        self.store.checkpoint("运行中")
        if not self.agent.args.quiet:
            latency = "UNKNOWN" if request_ms is None else f"{request_ms:.2f}ms"
            print(f"窗口{row['决策窗口']:02d}: 端到端LLM API请求时延={latency}; "
                  f"输入token={row.get('prompt_tokens')}; API输出token={row.get('completion_tokens')}; "
                  f"重试={row.get('重试次数', 0)}; 回退={row['LLM是否回退']}", flush=True)


def describe_values(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"有效样本数": 0, "均值": None, "中位数": None, "P95": None, "最大值": None, "最小值": None, "标准差": None}
    return {"有效样本数": len(values), "均值": statistics.mean(values), "中位数": statistics.median(values),
            "P95": float(np.percentile(values, 95)), "最大值": max(values), "最小值": min(values),
            "标准差": statistics.stdev(values) if len(values) > 1 else None}

