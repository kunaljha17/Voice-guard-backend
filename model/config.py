import os
from pathlib import Path

# Base directory (model/ root)
BASE_DIR = Path(__file__).resolve().parent

# Model weights path
VOICE_CLONE_PT = BASE_DIR / "saved_models" / "ssl_spoof.pt"

# Environment-controlled model configuration
DEFAULT_MODEL_PATH = str(VOICE_CLONE_PT) if VOICE_CLONE_PT.exists() else str(BASE_DIR / "models" / "voiceguard_v1.onnx")
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
MODEL_VERSION = os.environ.get("MODEL_VERSION", "2.0.0-Wav2Vec2")

# Resolve path
if not os.path.exists(MODEL_PATH):
    alt_path = str(BASE_DIR / MODEL_PATH.lstrip("/"))
    if os.path.exists(alt_path):
        MODEL_PATH = alt_path
    elif VOICE_CLONE_PT.exists():
        MODEL_PATH = str(VOICE_CLONE_PT)
