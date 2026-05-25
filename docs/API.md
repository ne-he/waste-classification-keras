# API

A FastAPI service exposing the trained model for classification and Grad-CAM.

## Run

```bash
export MODEL_PATH=models/final/<run>/model.keras   # Windows: set MODEL_PATH=...
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

The model path is read from the `MODEL_PATH` environment variable and the
model is loaded lazily on the first prediction request. CORS origins are read
from `configs/inference.yaml` (default: `*`).

## Endpoints

### `GET /health`

Returns service status and whether the model is loaded.

```json
{ "status": "ok", "model_loaded": false }
```

### `GET /classes`

```json
{ "classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"] }
```

### `POST /predict`

Multipart upload of an image file (`file` field). Returns:

```json
{
  "predicted_class": "glass",
  "confidence": 0.84,
  "probabilities": { "cardboard": 0.02, "glass": 0.84, "...": 0.0 }
}
```

```bash
curl -F "file=@photo.jpg" http://localhost:8000/predict
```

### `POST /gradcam`

Multipart upload of an image file. Returns the prediction plus a base64-encoded
PNG of the Grad-CAM overlay:

```json
{
  "predicted_class": "glass",
  "confidence": 0.84,
  "heatmap_png_base64": "iVBORw0KGgo..."
}
```

## Errors

| Status | Meaning |
|--------|---------|
| 400 | Uploaded file could not be decoded as an image |
| 500 | Prediction or Grad-CAM failed |
| 503 | Model file unavailable (`MODEL_PATH` not found) |

Interactive docs are available at `/docs` (Swagger UI) when the server runs.
