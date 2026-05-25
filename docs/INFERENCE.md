# Inference

## CLI

```bash
python scripts/predict.py --image photo.jpg --model models/final/<run>/model.keras
```

Output is JSON:

```json
{
  "predicted_class": "metal",
  "confidence": 0.91,
  "probabilities": {
    "cardboard": 0.01, "glass": 0.04, "metal": 0.91,
    "paper": 0.01, "plastic": 0.02, "trash": 0.01
  }
}
```

## Programmatic use

```python
import cv2
from src.api.inference import InferenceEngine

engine = InferenceEngine("models/final/<run>/model.keras")
image = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)
result = engine.predict(image)
print(result.predicted_class, result.confidence)
```

`InferenceEngine` loads the model lazily on first use and is thread-safe.

## Class names

The engine reads `class_indices.json` (written next to the model during
training) to map output indices to class names. If the file is absent, it
falls back to the default TrashNet ordering.

## Grad-CAM explanations

```python
result = engine.explain(image)          # GradCAMResponse
# result.heatmap_png_base64 -> base64 PNG of the overlaid heatmap
```

Or generate overlays for a directory of images:

```bash
python scripts/generate_gradcam.py --model-path models/final/<run>/model.keras \
    --image-dir data/dataset-resized/metal --output outputs/gradcam
```
