import io
import os
import uuid
import time
import base64
import math
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from inference import predict, get_available_models, DEVICE

app = FastAPI(
    title="MetaGaze API",
    description="Gaze estimation powered by GazeSymCAT (ETH-XGaze).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MAX_RESULTS = 50
_results: dict = {}

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()


def _store_result(annotated_img: Image.Image) -> str:
    rid = uuid.uuid4().hex[:12]
    _results[rid] = {"image": annotated_img, "ts": time.time()}
    if len(_results) > _MAX_RESULTS:
        oldest = min(_results, key=lambda k: _results[k]["ts"])
        del _results[oldest]
    return rid


def _read_upload(file: UploadFile) -> Image.Image:
    try:
        contents = file.file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "message": "MetaGaze API is running",
        "device": DEVICE.type,
        "is_cpu": DEVICE.type == "cpu"
    }


@app.get("/api/model-info")
async def model_info():
    return {
        "model": "GazeSymCAT",
        "backbone": "ResNet50 + DCA",
        "dataset": "ETH-XGaze",
        "input_size": 224,
        "output": "pitch, yaw (radians) → 3D gaze vector",
        "head_pose": True,
    }


@app.get("/api/models")
async def list_models():
    return get_available_models()


@app.post("/api/predict")
async def predict_endpoint(file: UploadFile = File(...), model: Optional[str] = None):
    pil_img = _read_upload(file)

    try:
        result = predict(pil_img, model_name=model)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    annotated: Image.Image = result.pop("annotated_image")
    rid = _store_result(annotated)
    result["result_id"] = rid
    result["annotated_image_url"] = f"/api/result/{rid}"
    return result


@app.post("/api/predict/json")
async def predict_json_only(file: UploadFile = File(...), model: Optional[str] = None):
    pil_img = _read_upload(file)
    try:
        result = predict(pil_img, model_name=model)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    result.pop("annotated_image", None)
    return result


@app.get("/api/result/{result_id}")
async def get_result_image(result_id: str):
    entry = _results.get(result_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Result not found or expired")
    img_bytes = _pil_to_bytes(entry["image"], "PNG")
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")


@app.post("/api/predict/annotated")
async def predict_annotated(file: UploadFile = File(...), model: Optional[str] = None):
    pil_img = _read_upload(file)
    try:
        result = predict(pil_img, model_name=model)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    annotated = result["annotated_image"]
    img_bytes = _pil_to_bytes(annotated, "PNG")
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")


@app.websocket("/api/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            frame_b64 = data.get("frame")
            model_name = data.get("model") or None

            if not frame_b64:
                await ws.send_json({"error": "No frame data"})
                continue

            try:
                if "," in frame_b64:
                    frame_b64 = frame_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(frame_b64)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                result = predict(pil_img, model_name=model_name)
                annotated: Image.Image = result.pop("annotated_image")

                quality = 60 if DEVICE.type == 'cpu' else 85
                buf = io.BytesIO()
                annotated.save(buf, format="JPEG", quality=quality)
                annotated_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

                await ws.send_json({
                    "pitch_deg": result["pitch_deg"],
                    "yaw_deg": result["yaw_deg"],
                    "head_pitch_deg": result["head_pitch_deg"],
                    "head_yaw_deg": result["head_yaw_deg"],
                    "gaze_vector": result["gaze_vector"],
                    "annotated_frame": f"data:image/jpeg;base64,{annotated_b64}",
                })

            except Exception as e:
                await ws.send_json({"error": str(e)})

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
