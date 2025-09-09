---
inclusion: fileMatch
fileMatchPattern: "src/**/*"
---

# Project Structure Guidelines

## Module Organization

### src/vision/
- `camera.py` - Video capture and frame processing
- `hand_detector.py` - MediaPipe hand landmark detection
- `preprocessor.py` - Image preprocessing (grayscale, normalization)
- `gesture_tracker.py` - Track gestures across frames

### src/ml/
- `model.py` - Neural network architecture definition
- `trainer.py` - Model training pipeline
- `predictor.py` - Real-time gesture prediction
- `data_loader.py` - Dataset loading and preprocessing

### src/speech/
- `tts_engine.py` - Text-to-speech synthesis
- `text_processor.py` - Text formatting and processing

### src/ui/
- `main_window.py` - Main application interface
- `gesture_display.py` - Visual feedback for recognized gestures
- `settings_panel.py` - Configuration options

## Naming Conventions
- Classes: PascalCase (e.g., `GestureClassifier`)
- Functions: snake_case (e.g., `detect_hand_landmarks`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_HANDS`)
- Files: snake_case (e.g., `hand_detector.py`)

## Import Structure
```python
# Standard library imports
import os
import sys

# Third-party imports
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Local imports
from src.vision import hand_detector
from src.ml import model
```
