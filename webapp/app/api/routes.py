import base64

from fastapi import APIRouter, HTTPException, Request

from url_ml.qr_engineering import extract_qr_metadata
from webapp.app.models import AnalyzeQrRequest, AnalyzeRequest, AnalyzeResponse

router = APIRouter(prefix="/api", tags=["analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_url(payload: AnalyzeRequest, request: Request):
    analyzers = request.app.state.url_analyzers
    default_mode = request.app.state.default_model_mode
    requested_mode = (payload.mode or default_mode).strip().lower()

    analyzer = analyzers.get(requested_mode)
    if analyzer is None:
        available_modes = ", ".join(sorted(analyzers.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported mode '{requested_mode}'. Available modes: {available_modes}",
        )

    url = payload.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    try:
        result = analyzer.analyze(url, qr_metadata=payload.qr_metadata, recheck=payload.recheck)
        result["mode"] = requested_mode
        result["model_type"] = analyzer.model_service.model_type
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to analyze URL: {exc}") from exc


@router.post("/analyze-qr", response_model=AnalyzeResponse)
def analyze_qr_image(payload: AnalyzeQrRequest, request: Request):
    analyzers = request.app.state.url_analyzers
    default_mode = request.app.state.default_model_mode
    requested_mode = (payload.mode or default_mode).strip().lower()

    analyzer = analyzers.get(requested_mode)
    if analyzer is None:
        available_modes = ", ".join(sorted(analyzers.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported mode '{requested_mode}'. Available modes: {available_modes}",
        )

    try:
        raw_b64 = payload.image_base64.strip()
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(raw_b64)
        qr_meta = extract_qr_metadata(image_bytes)
        if not qr_meta.get("valid") or not qr_meta.get("url"):
            err_msg = qr_meta.get("error") or "Không thể tìm thấy hoặc giải mã URL từ ảnh QR này."
            raise HTTPException(status_code=400, detail=err_msg)

        decoded_url = qr_meta["url"].strip()
        result = analyzer.analyze(decoded_url, qr_metadata=qr_meta, recheck=payload.recheck)
        result["mode"] = requested_mode
        result["model_type"] = analyzer.model_service.model_type
        result["qr_metadata"] = qr_meta
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process QR image: {exc}") from exc
