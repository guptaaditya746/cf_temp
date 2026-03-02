from __future__ import annotations

import importlib
from typing import Annotated, Any, Dict, Literal, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from cftsad import CFFailure, CFResult, CounterfactualExplainer


class IdentityModel:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)


class MovingAverageModel:
    def __init__(self, kernel_size: int = 5):
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 3")
        self.kernel_size = int(kernel_size)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 3 and x_arr.shape[0] == 1:
            x_arr = x_arr[0]
        if x_arr.ndim != 2:
            raise ValueError(f"expected (L, F), got {x_arr.shape}")

        pad = self.kernel_size // 2
        kernel = np.ones(self.kernel_size, dtype=np.float64) / self.kernel_size
        x_hat = np.empty_like(x_arr)
        for feat_idx in range(x_arr.shape[1]):
            padded = np.pad(x_arr[:, feat_idx], pad_width=pad, mode="edge")
            x_hat[:, feat_idx] = np.convolve(padded, kernel, mode="valid")
        return x_hat


class IdentityModelSpec(BaseModel):
    type: Literal["identity"]


class MovingAverageModelSpec(BaseModel):
    type: Literal["moving_average"]
    kernel_size: int = 5


class PythonCallableModelSpec(BaseModel):
    type: Literal["python_callable"]
    import_path: str = Field(
        ...,
        description="Callable import path in module:attribute format",
    )


ModelSpec = Annotated[
    Union[IdentityModelSpec, MovingAverageModelSpec, PythonCallableModelSpec],
    Field(discriminator="type"),
]


class ExplainRequest(BaseModel):
    method: Literal["nearest", "segment", "motif", "genetic"]
    model: ModelSpec
    normal_core: list[list[list[float]]]
    x: list[list[float]]
    threshold: Optional[float] = None
    method_kwargs: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class SuccessResponse(BaseModel):
    status: Literal["success"] = "success"
    score_cf: float
    x_cf: list[list[float]]
    meta: Dict[str, Any]


class FailureResponse(BaseModel):
    status: Literal["failure"] = "failure"
    reason: str
    message: str
    diagnostics: Dict[str, Any]


ExplainResponse = Annotated[
    Union[SuccessResponse, FailureResponse],
    Field(discriminator="status"),
]


def _load_callable(import_path: str):
    if ":" not in import_path:
        raise ValueError("import_path must be in module:attribute format")

    module_name, attr_name = import_path.split(":", 1)
    if not module_name or not attr_name:
        raise ValueError("import_path must be in module:attribute format")

    module = importlib.import_module(module_name)
    target = getattr(module, attr_name)

    if isinstance(target, type):
        target = target()

    if callable(target):
        return target

    raise ValueError(f"Imported object {import_path!r} is not callable")


def _build_model(spec: ModelSpec):
    if isinstance(spec, IdentityModelSpec):
        return IdentityModel()
    if isinstance(spec, MovingAverageModelSpec):
        return MovingAverageModel(kernel_size=spec.kernel_size)
    if isinstance(spec, PythonCallableModelSpec):
        return _load_callable(spec.import_path)
    raise ValueError("Unsupported model spec")


def create_app() -> FastAPI:
    app = FastAPI(
        title="cftsad API",
        version="0.1.0",
        description="HTTP service wrapper for cftsad counterfactual explanations",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/explain", response_model=ExplainResponse)
    def explain(req: ExplainRequest):
        try:
            model = _build_model(req.model)
            explainer = CounterfactualExplainer(
                method=req.method,
                model=model,
                normal_core=np.asarray(req.normal_core, dtype=np.float64),
                threshold=req.threshold,
                **req.method_kwargs,
            )
            result = explainer.explain(np.asarray(req.x, dtype=np.float64))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Service error: {exc}") from exc

        if isinstance(result, CFResult):
            return SuccessResponse(
                score_cf=float(result.score_cf),
                x_cf=np.asarray(result.x_cf, dtype=np.float64).tolist(),
                meta=result.meta,
            )

        assert isinstance(result, CFFailure)
        return FailureResponse(
            reason=result.reason,
            message=result.message,
            diagnostics=result.diagnostics,
        )

    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("cftsad.service:app", host="0.0.0.0", port=8000, reload=False)
