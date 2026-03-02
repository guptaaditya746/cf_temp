from __future__ import annotations

import json
import sys
from csv import DictReader
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, request

import numpy as np
from dash import Dash, Input, Output, State, dcc, html
import plotly.graph_objects as go
import torch

try:
    from utils.metrics import CounterfactualMetrics
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from utils.metrics import CounterfactualMetrics


@dataclass
class DemoResult:
    status: str
    score_cf: float
    x_cf: np.ndarray
    meta: Dict[str, Any]
    reason: str = ""
    message: str = ""


@dataclass
class ExampleArtifact:
    x: np.ndarray
    x_cf: np.ndarray
    score_before: float
    score_after: float
    method: str
    test_index: int
    run_dir: str
    threshold: Optional[float] = None
    normal_core: Optional[np.ndarray] = None


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_np_load(path: Path) -> np.ndarray:
    try:
        return np.load(path)
    except ValueError as exc:
        msg = str(exc)
        if "allow_pickle" in msg or "pickled" in msg:
            # Fallback for notebook-generated object arrays.
            return np.load(path, allow_pickle=True)
        raise


def _find_latest_run(results_root: str) -> Optional[Path]:
    root = Path(results_root)
    if not root.exists() or not root.is_dir():
        return None

    run_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0]


def _list_example_indices(results_root: str, method: str) -> list[int]:
    run_dir = _find_latest_run(results_root)
    if run_dir is None:
        return []

    log_path = run_dir / "evaluation" / "counterfactual_log.csv"
    if not log_path.exists():
        return []

    seen = set()
    with log_path.open("r", encoding="utf-8") as fh:
        for row in DictReader(fh):
            if row.get("method") != method:
                continue
            test_idx = row.get("test_index")
            try:
                idx = int(test_idx)
            except (TypeError, ValueError):
                continue
            seen.add(idx)
    return sorted(seen)


def _load_example_artifact(results_root: str, method: str, preferred_index: Optional[int]) -> ExampleArtifact:
    run_dir = _find_latest_run(results_root)
    if run_dir is None:
        raise ValueError(f"No run_* folders found under {results_root!r}")

    eval_dir = run_dir / "evaluation"
    log_path = eval_dir / "counterfactual_log.csv"
    x_test_path = run_dir / "X_test.npy"
    x_train_path = run_dir / "X_train.npy"
    cf_dir = eval_dir / "cf_arrays"

    if not log_path.exists():
        raise ValueError(f"Missing artifact: {log_path}")
    if not x_test_path.exists():
        raise ValueError(f"Missing artifact: {x_test_path}")
    if not x_train_path.exists():
        raise ValueError(f"Missing artifact: {x_train_path}")
    if not cf_dir.exists():
        raise ValueError(f"Missing artifact: {cf_dir}")

    threshold = None
    threshold_path = eval_dir / "window_threshold.npy"
    if threshold_path.exists():
        threshold = float(_safe_np_load(threshold_path))

    x_test = _safe_np_load(x_test_path)
    x_train = _safe_np_load(x_train_path)
    rows = []
    with log_path.open("r", encoding="utf-8") as fh:
        for row in DictReader(fh):
            if row.get("method") == method:
                rows.append(row)

    if not rows:
        raise ValueError(f"No rows in {log_path.name} for method {method!r}")

    if preferred_index is not None:
        rows = [r for r in rows if str(preferred_index) == str(r.get("test_index"))] or rows

    picked = rows[-1]
    idx = int(picked["test_index"])
    if idx < 0 or idx >= x_test.shape[0]:
        raise ValueError(f"test_index {idx} out of bounds for X_test with {x_test.shape[0]} windows")

    x = np.asarray(x_test[idx], dtype=np.float64)
    score_before = _coerce_float(picked.get("original_score"), default=float(np.mean((x - np.mean(x, axis=0, keepdims=True)) ** 2)))

    if picked.get("status") == "success":
        cf_file = str(picked.get("cf_array_file", "")).strip()
        cf_path = cf_dir / cf_file
        if not cf_file or cf_file == "N/A" or not cf_path.exists():
            raise ValueError(f"Counterfactual file not found for method={method}, test_index={idx}: {cf_path}")
        x_cf = np.asarray(_safe_np_load(cf_path), dtype=np.float64)
        score_after = _coerce_float(picked.get("cf_score"), default=score_before)
    else:
        x_cf = x.copy()
        score_after = score_before

    return ExampleArtifact(
        x=x,
        x_cf=x_cf,
        score_before=score_before,
        score_after=score_after,
        method=method,
        test_index=idx,
        run_dir=str(run_dir),
        threshold=threshold,
        normal_core=np.asarray(x_train, dtype=np.float64),
    )


def _compute_dashboard_metrics(
    x: np.ndarray,
    x_cf: np.ndarray,
    score_cf: Optional[float],
    threshold: Optional[float],
    normal_core: Optional[np.ndarray],
) -> Dict[str, Optional[float]]:
    metrics = {
        "MarginToThr": None,
        "TrimmedRMSE_scaled": None,
        "TVL1D1": None,
        "MahalMean_edited": None,
    }

    thr = _coerce_float(threshold, default=np.nan)
    if not np.isfinite(thr):
        return metrics

    x_t = torch.as_tensor(x, dtype=torch.float32)
    x_cf_t = torch.as_tensor(x_cf, dtype=torch.float32)
    cf_result = {"x_cf": x_cf_t, "score": _coerce_float(score_cf, default=np.nan), "meta": {}}

    nc_t = None
    if normal_core is not None:
        nc_t = torch.as_tensor(normal_core, dtype=torch.float32)

    evaluator = CounterfactualMetrics()
    out = evaluator.compute(
        x=x_t,
        cf_result=cf_result,
        threshold=float(thr),
        normal_core=nc_t,
    )
    metrics["MarginToThr"] = out.get("margin_to_thr")
    metrics["TrimmedRMSE_scaled"] = out.get("dist_trimmed_rmse_scaled")
    metrics["TVL1D1"] = out.get("tv_l1_d1")
    metrics["MahalMean_edited"] = out.get("mahal_mean_edited")
    return metrics


def _build_demo_data(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    length = 48
    t = np.linspace(0, 2 * np.pi, length, endpoint=False)

    x_normal = np.zeros((length, 3), dtype=np.float64)
    x_normal[:, 0] = np.sin(t)
    x_normal[:, 1] = 0.7 * np.cos(1.7 * t + 0.3)
    x_normal[:, 2] = 0.4 * np.sin(2.1 * t - 0.2)
    x_normal += rng.normal(0.0, 0.06, size=x_normal.shape)

    x_anom = x_normal.copy()
    x_anom[16:22, 0] += 1.5
    x_anom[27:34, 1] -= 1.2
    x_anom[38:, 2] += 0.8

    core = np.stack([x_normal + rng.normal(0.0, 0.04, x_normal.shape) for _ in range(120)], axis=0)
    return core, x_anom, x_normal


def _mock_counterfactual(x: np.ndarray, strength: float = 0.65) -> DemoResult:
    baseline = x.copy()
    x_cf = baseline.copy()

    for col in range(x_cf.shape[1]):
        kernel = np.array([0.2, 0.6, 0.2], dtype=np.float64)
        padded = np.pad(x_cf[:, col], pad_width=1, mode="edge")
        smooth = np.convolve(padded, kernel, mode="valid")
        x_cf[:, col] = (1.0 - strength) * x_cf[:, col] + strength * smooth

    score_cf = float(np.mean((x_cf - np.mean(x_cf, axis=0, keepdims=True)) ** 2))

    feature_delta = np.mean(np.abs(x_cf - baseline), axis=0)
    time_delta = np.mean(np.abs(x_cf - baseline), axis=1)
    meta = {
        "top_changed_features": np.argsort(feature_delta)[::-1][:3].tolist(),
        "top_changed_timesteps": np.argsort(time_delta)[::-1][:6].tolist(),
        "delta_by_feature": feature_delta.tolist(),
    }
    return DemoResult(status="success", score_cf=score_cf, x_cf=x_cf, meta=meta)


def _call_api(method: str, api_url: str, normal_core: np.ndarray, x: np.ndarray) -> DemoResult:
    payload = {
        "method": method,
        "model": {"type": "moving_average", "kernel_size": 5},
        "normal_core": normal_core.tolist(),
        "x": x.tolist(),
        "threshold": None,
        "method_kwargs": {"nearest_top_k": 8, "nearest_alpha_steps": 11},
    }

    req = request.Request(
        api_url.rstrip("/") + "/v1/explain",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        return DemoResult(status="failure", score_cf=0.0, x_cf=x, meta={}, reason="http_error", message=details)
    except Exception as exc:
        return DemoResult(status="failure", score_cf=0.0, x_cf=x, meta={}, reason="connection_error", message=str(exc))

    if body.get("status") == "success":
        return DemoResult(
            status="success",
            score_cf=float(body.get("score_cf", 0.0)),
            x_cf=np.asarray(body.get("x_cf", x.tolist()), dtype=np.float64),
            meta=body.get("meta", {}),
        )

    return DemoResult(
        status="failure",
        score_cf=0.0,
        x_cf=x,
        meta={},
        reason=str(body.get("reason", "unknown_error")),
        message=str(body.get("message", "Counterfactual generation failed")),
    )


def _make_series_figure(x: np.ndarray, x_cf: np.ndarray, feature_idx: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=x[:, feature_idx], mode="lines", name="Anomalous", line={"color": "#f45d48", "width": 2}))
    fig.add_trace(go.Scatter(y=x_cf[:, feature_idx], mode="lines", name="Counterfactual", line={"color": "#06a77d", "width": 2}))

    fig.update_layout(
        template="plotly_white",
        margin={"l": 26, "r": 20, "t": 20, "b": 28},
        legend={"orientation": "h", "y": 1.1, "x": 0.0},
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
    )
    return fig


def _make_delta_figure(x: np.ndarray, x_cf: np.ndarray) -> go.Figure:
    delta = np.mean(np.abs(x_cf - x), axis=0)
    fig = go.Figure(
        data=[go.Bar(x=[f"Feature {i}" for i in range(x.shape[1])], y=delta, marker={"color": ["#ff7f51", "#2a9d8f", "#4f6d7a"]})]
    )
    fig.update_layout(
        template="plotly_white",
        margin={"l": 26, "r": 20, "t": 20, "b": 28},
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        yaxis_title="Mean |delta|",
    )
    return fig


def _chip(text: str, tone: str = "neutral"):
    palette = {
        "good": {"bg": "#dcfff3", "fg": "#0b6a4f"},
        "warn": {"bg": "#fff3db", "fg": "#9a6200"},
        "bad": {"bg": "#ffe6e2", "fg": "#9e2f1f"},
        "neutral": {"bg": "#ebf1ff", "fg": "#2f4a8a"},
    }
    colors = palette[tone]
    return html.Span(
        text,
        style={
            "padding": "6px 10px",
            "borderRadius": "999px",
            "fontSize": "12px",
            "fontWeight": "600",
            "background": colors["bg"],
            "color": colors["fg"],
            "display": "inline-block",
        },
    )


def create_app() -> Dash:
    app = Dash(__name__)

    app.layout = html.Div(
        [
            html.Link(
                rel="stylesheet",
                href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@400;600&display=swap",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("Counterfactual Results Studio", style={"margin": "0", "fontSize": "34px", "letterSpacing": "-0.6px"}),
                            html.P(
                                "Demo dashboard for presenting anomaly-to-counterfactual transitions.",
                                style={"margin": "6px 0 0", "opacity": "0.78", "fontSize": "14px"},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="data-source",
                                options=[
                                    {"label": "Example Artifacts", "value": "example"},
                                    {"label": "Mock Demo", "value": "mock"},
                                    {"label": "Live API", "value": "api"},
                                ],
                                value="example",
                                clearable=False,
                                style={"width": "180px", "fontFamily": "IBM Plex Mono, monospace", "fontSize": "13px"},
                            ),
                            dcc.Dropdown(
                                id="method",
                                options=[{"label": v.title(), "value": v} for v in ["nearest", "segment", "motif", "genetic"]],
                                value="nearest",
                                clearable=False,
                                style={"width": "180px", "fontFamily": "IBM Plex Mono, monospace", "fontSize": "13px"},
                            ),
                            dcc.Input(
                                id="api-url",
                                type="text",
                                value="http://localhost:8000",
                                debounce=True,
                                style={
                                    "width": "220px",
                                    "padding": "10px 12px",
                                    "borderRadius": "10px",
                                    "border": "1px solid #d5deea",
                                    "fontFamily": "IBM Plex Mono, monospace",
                                    "fontSize": "13px",
                                },
                            ),
                            dcc.Input(
                                id="results-root",
                                type="text",
                                value="results",
                                debounce=True,
                                style={
                                    "width": "140px",
                                    "padding": "10px 12px",
                                    "borderRadius": "10px",
                                    "border": "1px solid #d5deea",
                                    "fontFamily": "IBM Plex Mono, monospace",
                                    "fontSize": "13px",
                                },
                            ),
                            dcc.Dropdown(
                                id="example-index",
                                options=[],
                                value=None,
                                placeholder="example index",
                                clearable=False,
                                style={"width": "150px", "fontFamily": "IBM Plex Mono, monospace", "fontSize": "13px"},
                            ),
                            html.Button(
                                "Run Demo",
                                id="run-btn",
                                n_clicks=0,
                                style={
                                    "background": "linear-gradient(135deg, #1947e5 0%, #20a4f3 100%)",
                                    "color": "#fff",
                                    "border": "0",
                                    "padding": "11px 18px",
                                    "borderRadius": "10px",
                                    "fontWeight": "700",
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                        style={"display": "flex", "gap": "10px", "alignItems": "center", "flexWrap": "wrap"},
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "gap": "16px",
                    "alignItems": "end",
                    "flexWrap": "wrap",
                    "marginBottom": "18px",
                },
            ),
            dcc.Store(id="result-store"),
            html.Div(id="status-banner", style={"marginBottom": "14px"}),
            html.Div(
                [
                    html.Div(id="kpi-cards", style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "12px"}),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Feature View", style={"margin": "0 0 8px"}),
                                    dcc.RadioItems(
                                        id="feature-radio",
                                        options=[{"label": f"Feature {i}", "value": i} for i in range(3)],
                                        value=0,
                                        inline=True,
                                        labelStyle={"marginRight": "10px", "fontSize": "13px"},
                                    ),
                                    dcc.Graph(id="series-graph", config={"displayModeBar": False}),
                                ],
                                className="fade-in",
                                style={"background": "#ffffffd9", "borderRadius": "16px", "padding": "14px", "boxShadow": "0 10px 30px rgba(18,33,77,0.08)"},
                            ),
                            html.Div(
                                [
                                    html.H3("Change Magnitude", style={"margin": "0 0 8px"}),
                                    dcc.Graph(id="delta-graph", config={"displayModeBar": False}),
                                ],
                                className="fade-in",
                                style={"background": "#ffffffd9", "borderRadius": "16px", "padding": "14px", "boxShadow": "0 10px 30px rgba(18,33,77,0.08)"},
                            ),
                        ],
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
                            "gap": "12px",
                            "marginTop": "12px",
                        },
                    ),
                ]
            ),
        ],
        style={
            "minHeight": "100vh",
            "padding": "26px",
            "background": "radial-gradient(1300px 500px at 5% -20%, #dce7ff 0%, rgba(220,231,255,0) 60%), linear-gradient(160deg, #f6f9ff 0%, #edf4ff 45%, #f6fbf6 100%)",
            "fontFamily": "Space Grotesk, sans-serif",
            "color": "#1d2840",
        },
    )

    @app.callback(
        Output("example-index", "options"),
        Output("example-index", "value"),
        Input("method", "value"),
        Input("results-root", "value"),
    )
    def refresh_example_indices(method: str, results_root: str):
        indices = _list_example_indices(results_root=results_root or "results", method=method)
        options = [{"label": str(i), "value": i} for i in indices]
        value = indices[0] if indices else None
        return options, value

    @app.callback(
        Output("result-store", "data"),
        Output("status-banner", "children"),
        Input("run-btn", "n_clicks"),
        State("data-source", "value"),
        State("method", "value"),
        State("api-url", "value"),
        State("results-root", "value"),
        State("example-index", "value"),
        prevent_initial_call=False,
    )
    def run_demo(
        n_clicks: int,
        data_source: str,
        method: str,
        api_url: str,
        results_root: str,
        example_index: Optional[int],
    ):
        normal_core, x_anom, x_normal = _build_demo_data(seed=42 + int(n_clicks or 0))
        del x_normal

        if data_source == "api":
            result = _call_api(method=method, api_url=api_url, normal_core=normal_core, x=x_anom)
            score_before = float(np.mean((x_anom - np.mean(x_anom, axis=0, keepdims=True)) ** 2))
            x_for_display = x_anom
            source_label = "Live API"
            threshold_for_metrics = None
            normal_core_for_metrics = normal_core
            extra = {}
        elif data_source == "example":
            try:
                artifact = _load_example_artifact(
                    results_root=results_root or "results",
                    method=method,
                    preferred_index=example_index,
                )
                result = DemoResult(status="success", score_cf=artifact.score_after, x_cf=artifact.x_cf, meta={})
                score_before = artifact.score_before
                x_for_display = artifact.x
                source_label = "Example Artifacts"
                threshold_for_metrics = artifact.threshold
                normal_core_for_metrics = artifact.normal_core
                extra = {
                    "example_index": artifact.test_index,
                    "run_dir": artifact.run_dir,
                    "threshold": artifact.threshold,
                }
            except Exception as exc:
                x_for_display = x_anom
                score_before = float(np.mean((x_anom - np.mean(x_anom, axis=0, keepdims=True)) ** 2))
                result = DemoResult(
                    status="failure",
                    score_cf=score_before,
                    x_cf=x_anom,
                    meta={},
                    reason="example_load_failed",
                    message=str(exc),
                )
                source_label = "Example Artifacts"
                threshold_for_metrics = None
                normal_core_for_metrics = None
                extra = {}
        else:
            result = _mock_counterfactual(x_anom)
            score_before = float(np.mean((x_anom - np.mean(x_anom, axis=0, keepdims=True)) ** 2))
            x_for_display = x_anom
            source_label = "Mock"
            threshold_for_metrics = score_before
            normal_core_for_metrics = normal_core
            extra = {}

        quality_metrics = _compute_dashboard_metrics(
            x=x_for_display,
            x_cf=result.x_cf,
            score_cf=result.score_cf,
            threshold=threshold_for_metrics,
            normal_core=normal_core_for_metrics,
        )

        payload = {
            "x": x_for_display.tolist(),
            "x_cf": result.x_cf.tolist(),
            "status": result.status,
            "score_before": score_before,
            "score_cf": result.score_cf,
            "quality_metrics": quality_metrics,
            "meta": result.meta,
            "reason": result.reason,
            "message": result.message,
            "method": method,
            "source_mode": data_source,
            "source_label": source_label,
            **extra,
        }

        if result.status == "success":
            details = []
            if payload.get("example_index") is not None:
                details.append(f"index={payload['example_index']}")
            if payload.get("threshold") is not None:
                details.append(f"threshold={float(payload['threshold']):.4f}")
            suffix = f" ({', '.join(details)})" if details else ""
            banner = html.Div(
                [_chip("SUCCESS", "good"), html.Span(f"{source_label}: method {method} produced a valid counterfactual{suffix}.", style={"marginLeft": "10px", "fontSize": "13px"})],
                style={"padding": "10px 12px", "background": "#ffffffd9", "borderRadius": "12px", "boxShadow": "0 8px 24px rgba(18,33,77,0.08)"},
            )
        else:
            banner = html.Div(
                [_chip("FAILURE", "bad"), html.Span(f"{source_label}: {result.reason}: {result.message}", style={"marginLeft": "10px", "fontSize": "13px"})],
                style={"padding": "10px 12px", "background": "#ffffffd9", "borderRadius": "12px", "boxShadow": "0 8px 24px rgba(18,33,77,0.08)"},
            )
        return payload, banner

    @app.callback(
        Output("kpi-cards", "children"),
        Output("feature-radio", "options"),
        Output("feature-radio", "value"),
        Output("series-graph", "figure"),
        Output("delta-graph", "figure"),
        Input("result-store", "data"),
        Input("feature-radio", "value"),
    )
    def refresh_charts(data: Optional[dict], feature_idx: int):
        if not data:
            x = np.zeros((48, 3), dtype=np.float64)
            options = [{"label": f"Feature {i}", "value": i} for i in range(x.shape[1])]
            return [], options, 0, _make_series_figure(x, x, 0), _make_delta_figure(x, x)

        x = np.asarray(data["x"], dtype=np.float64)
        x_cf = np.asarray(data["x_cf"], dtype=np.float64)
        options = [{"label": f"Feature {i}", "value": i} for i in range(x.shape[1])]
        selected_feature = int(feature_idx or 0)
        if selected_feature < 0 or selected_feature >= x.shape[1]:
            selected_feature = 0

        score_before = float(data.get("score_before", np.mean((x - np.mean(x, axis=0, keepdims=True)) ** 2)))
        score_after = float(data.get("score_cf", 0.0))

        delta = np.mean(np.abs(x_cf - x), axis=0)
        top_feature = int(np.argmax(delta))

        cards = [
            _kpi_card("Method", data.get("method", "n/a"), "neutral"),
            _kpi_card("Score Before", f"{score_before:.4f}", "warn"),
            _kpi_card("Score After", f"{score_after:.4f}", "good" if score_after <= score_before else "warn"),
            _kpi_card("Top Changed Feature", f"Feature {top_feature}", "neutral"),
            _kpi_card("Mode", data.get("source_label", "Mock"), "neutral"),
        ]
        quality = data.get("quality_metrics") or {}
        cards.extend(
            [
                _kpi_card("MarginToThr", _fmt_metric(quality.get("MarginToThr")), "neutral"),
                _kpi_card("TrimmedRMSE_scaled", _fmt_metric(quality.get("TrimmedRMSE_scaled")), "neutral"),
                _kpi_card("TVL1D1", _fmt_metric(quality.get("TVL1D1")), "neutral"),
                _kpi_card("MahalMean_edited", _fmt_metric(quality.get("MahalMean_edited")), "neutral"),
            ]
        )

        return cards, options, selected_feature, _make_series_figure(x, x_cf, selected_feature), _make_delta_figure(x, x_cf)

    return app


def _kpi_card(title: str, value: str, tone: str):
    return html.Div(
        [
            html.Div(title, style={"fontSize": "12px", "opacity": "0.75", "marginBottom": "4px"}),
            html.Div(value, style={"fontSize": "24px", "fontWeight": "700", "lineHeight": "1.1", "marginBottom": "8px"}),
            _chip(tone.upper(), tone=tone),
        ],
        style={"background": "#ffffffd9", "padding": "14px", "borderRadius": "14px", "boxShadow": "0 10px 30px rgba(18,33,77,0.08)"},
    )


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(f):
        return "N/A"
    return f"{f:.4f}"


def run() -> None:
    app = create_app()
    app.run(host="0.0.0.0", port=8050, debug=False)


if __name__ == "__main__":
    run()
