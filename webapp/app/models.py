from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    url: str = Field(..., min_length=1)
    mode: str | None = Field(default=None, description="Model mode: rf, dt, dl")
    recheck: bool = Field(default=False, description="Bypass cached external checks and re-fetch live data")
    qr_metadata: dict[str, Any] | None = Field(default=None, description="Optional metadata extracted from QR code")


class AnalyzeQrRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded QR image or data URL")
    mode: str | None = Field(default=None, description="Model mode: rf, dt, dl")
    recheck: bool = Field(default=False, description="Bypass cached external checks")


class AnalyzeResponse(BaseModel):
    url: str
    normalized_url: str
    mode: str
    model_type: str
    model_prediction: str
    final_prediction: str
    base_malicious_probability: float
    final_malicious_probability: float
    risk_tier: str
    thresholds: dict[str, float]
    risk_triggers: list[str]
    top_debug_features: list[dict[str, Any]]
    features: dict[str, Any]
    fusion_reasons: list[str]
    checks: dict[str, Any]
    policy: dict[str, Any]
    qr_metadata: dict[str, Any] | None = None
