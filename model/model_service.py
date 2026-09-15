"""
VoiceGuard ML Model Service Layer (Isolated Module)
This is the ONLY module allowed to import ML libraries (onnx/torch/tensorflow)
and manage the loaded deepfake acoustic model.
"""

import os
import sys

# Prevent unauthenticated network round-trips to Hugging Face Hub if weights are locally cached
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import time
import json
import math
import struct
import wave
from pathlib import Path
from config import MODEL_PATH, MODEL_VERSION

# Optional ML library imports - isolated strictly within model_service.py
_onnxruntime = None
try:
    import onnxruntime as _onnxruntime
except ImportError:
    _onnxruntime = None

_torch = None
try:
    import torch as _torch
except ImportError:
    _torch = None

# Dynamically locate voice-clone-detection package
# model_service.py now lives inside model/ alongside src/
VOICE_CLONE_ROOT = Path(__file__).resolve().parent
if (VOICE_CLONE_ROOT / "src").exists():
    sys.path.insert(0, str(VOICE_CLONE_ROOT / "src"))

_SSLSpoofDetector = None
_audio_file_to_waveform = None
_device = None

if _torch is not None:
    try:
        from model import SSLSpoofDetector as _SSLSpoofDetector
        from preprocess import audio_file_to_waveform as _audio_file_to_waveform
        _device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    except Exception as _import_err:
        print(f"⚠️  [ModelService] Note: SSLSpoofDetector import deferred or skipped: {_import_err}", file=sys.stderr)

# Global in-memory model instance loaded ONCE at startup
_LOADED_MODEL = None
_MODEL_METADATA = {}
_MODEL_LOAD_ERROR = None


def _load_model():
    """Loads the model ONCE at startup and keeps it in memory."""
    global _LOADED_MODEL, _MODEL_METADATA, _MODEL_LOAD_ERROR

    # 1. First priority: Trained PyTorch Wav2Vec2 model from voice-clone-detection
    pt_candidates = [
        Path(MODEL_PATH) if MODEL_PATH.endswith(".pt") else None,
        VOICE_CLONE_ROOT / "saved_models" / "ssl_spoof.pt",
    ]
    pt_path = next((p for p in pt_candidates if p and p.exists()), None)

    if pt_path is not None and _torch is not None and _SSLSpoofDetector is not None:
        try:
            detector = _SSLSpoofDetector().to(_device)
            detector.load_state_dict(_torch.load(str(pt_path), map_location=_device))
            detector.eval()
            _LOADED_MODEL = detector
            _MODEL_METADATA["type"] = "pytorch_wav2vec2"
            _MODEL_METADATA["path"] = str(pt_path)
            _MODEL_METADATA["device"] = str(_device)
            print(f"✅ Loaded PyTorch SSLSpoofDetector from {pt_path} on {_device}", file=sys.stderr)
            return True
        except Exception as pt_err:
            print(f"⚠️  [ModelService] Failed loading PyTorch model: {pt_err}", file=sys.stderr)

    if not os.path.exists(MODEL_PATH):
        _MODEL_LOAD_ERROR = f"Model file not found at '{MODEL_PATH}'. Please verify MODEL_PATH configuration."
        print(f"⚠️  [ModelService] Failed to load model: {_MODEL_LOAD_ERROR}", file=sys.stderr)
        return False

    try:
        file_size = os.path.getsize(MODEL_PATH)
        if file_size == 0:
            _MODEL_LOAD_ERROR = f"Model file at '{MODEL_PATH}' is empty (0 bytes)."
            print(f"⚠️  [ModelService] {_MODEL_LOAD_ERROR}", file=sys.stderr)
            return False

        # Attempt loading with ONNX Runtime if available
        if _onnxruntime is not None:
            try:
                session = _onnxruntime.InferenceSession(
                    MODEL_PATH, providers=["CPUExecutionProvider"]
                )
                _LOADED_MODEL = session
                _MODEL_METADATA["type"] = "onnx_session"
            except Exception as onnx_err:
                _LOADED_MODEL = {"path": MODEL_PATH, "size": file_size, "fallback_onnx_err": str(onnx_err)}
                _MODEL_METADATA["type"] = "onnx_binary"
        else:
            with open(MODEL_PATH, "rb") as f:
                header = f.read(256)
            _LOADED_MODEL = {
                "path": MODEL_PATH,
                "size": file_size,
                "header_signature": header[:16].hex(),
            }
            _MODEL_METADATA["type"] = "acoustic_binary"

        print(f"✅ Loaded model v{MODEL_VERSION} from {MODEL_PATH}", file=sys.stderr)
        return True
    except Exception as e:
        _MODEL_LOAD_ERROR = f"Failed to initialize model from '{MODEL_PATH}': {str(e)}"
        print(f"❌ [ModelService] {_MODEL_LOAD_ERROR}", file=sys.stderr)
        return False


# Execute model load once at module startup
_load_model()


def _load_audio_waveform(audio_file_path: str, target_sr: int = 16000):
    """
    Robust audio loader supporting WAV, MP3, MP4/AAC (WhatsApp), OGG, WebM, FLAC.
    Returns (1D numpy float32 array, sample_rate).
    """
    # 1. Try PyAV first (native support for WhatsApp mp4, aac, webm, mp3, ogg, wav)
    try:
        import av
        import numpy as np
        container = av.open(str(audio_file_path))
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is not None:
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
            frames = []
            for frame in container.decode(audio_stream):
                for rf in resampler.resample(frame):
                    frames.append(rf.to_ndarray())
            if frames:
                raw_audio = np.concatenate(frames, axis=1).squeeze(0)
                if len(raw_audio) > 0:
                    return raw_audio.astype(np.float32), target_sr
    except Exception:
        pass

    # 2. Fallback to librosa
    try:
        import librosa
        raw_audio, sr = librosa.load(audio_file_path, sr=target_sr, mono=True)
        return raw_audio.astype(np.float32), target_sr
    except Exception:
        pass

    return None, target_sr


def _preprocess_audio(audio_file_path: str) -> dict:
    """
    Prepares audio file for inference by extracting acoustic features
    and validating audio integrity.
    """
    if not os.path.exists(audio_file_path):
        raise ValueError(f"Audio file does not exist: {audio_file_path}")

    file_size = os.path.getsize(audio_file_path)
    if file_size < 44:  # Less than minimal WAV header or valid audio container
        raise ValueError(f"Invalid or corrupt audio file: file is empty or unreadable ({file_size} bytes)")

    samples = []
    sample_rate = 16000

    # 1. Try robust audio decoder (handles WhatsApp mp4, aac, webm, mp3, ogg, wav)
    decoded_wave, sr = _load_audio_waveform(audio_file_path, 16000)
    if decoded_wave is not None and len(decoded_wave) > 0:
        samples = [float(s) for s in decoded_wave]
        sample_rate = sr
    else:
        # 2. Fallback to standard wave container
        try:
            with wave.open(audio_file_path, "rb") as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()

                if n_frames == 0:
                    raise ValueError("Invalid or corrupt audio file: recording contains 0 audio frames")

                raw_bytes = wf.readframes(n_frames)
                if sampwidth == 2:  # 16-bit PCM
                    count = len(raw_bytes) // 2
                    raw_samples = struct.unpack(f"<{count}h", raw_bytes)
                    if n_channels == 2:
                        samples = [(raw_samples[i] + raw_samples[i + 1]) / 2.0 for i in range(0, count - 1, 2)]
                    else:
                        samples = list(raw_samples)
                elif sampwidth == 1:  # 8-bit PCM
                    raw_samples = struct.unpack(f"<{len(raw_bytes)}B", raw_bytes)
                    samples = [s - 128 for s in raw_samples]
                else:
                    samples = [float(b) for b in raw_bytes[:10000]]
        except (wave.Error, EOFError, struct.error):
            # Non-WAV container fallback
            with open(audio_file_path, "rb") as f:
                raw_bytes = f.read()

            if len(raw_bytes) < 64:
                raise ValueError("Invalid or corrupt audio file: stream buffer too small or corrupt")

            payload = raw_bytes[64:]
            if not payload:
                raise ValueError("Invalid or corrupt audio file: no audio payload detected")

            aligned_len = (len(payload) // 2) * 2
            try:
                samples = list(struct.unpack(f"<{aligned_len // 2}h", payload[:aligned_len]))
            except Exception:
                samples = [float(b - 128) for b in payload]

    if not samples:
        raise ValueError("Invalid or corrupt audio file: unable to decode audio samples")

    # 1. Normalization (Peak amplitude scaling to [-1.0, 1.0])
    max_val = max(max(abs(s) for s in samples), 1.0)
    normalized_samples = [s / max_val for s in samples]

    # 2. Acoustic Feature Extraction
    # Zero Crossing Rate (ZCR)
    zero_crossings = 0
    for i in range(1, len(normalized_samples)):
        if (normalized_samples[i] >= 0 and normalized_samples[i - 1] < 0) or \
           (normalized_samples[i] < 0 and normalized_samples[i - 1] >= 0):
            zero_crossings += 1
    zcr = zero_crossings / max(len(normalized_samples), 1)

    # RMS Energy & Variance
    sum_sq = sum(s * s for s in normalized_samples)
    rms_energy = math.sqrt(sum_sq / max(len(normalized_samples), 1))

    # Frame-by-frame energy variance (AI voices exhibit unnatural steady-state flatness or micro-jitter)
    frame_size = 512
    frame_energies = []
    for i in range(0, len(normalized_samples), frame_size):
        chunk = normalized_samples[i : i + frame_size]
        if chunk:
            chunk_rms = math.sqrt(sum(c * c for c in chunk) / len(chunk))
            frame_energies.append(chunk_rms)

    mean_energy = (sum(frame_energies) / len(frame_energies)) if frame_energies else 0.0
    energy_variance = (
        sum((e - mean_energy) ** 2 for e in frame_energies) / len(frame_energies)
        if frame_energies
        else 0.0
    )

    # High frequency ratio & unnatural harmonics (spectral centroid estimation)
    diff_sum = sum(abs(normalized_samples[i] - normalized_samples[i - 1]) for i in range(1, len(normalized_samples)))
    high_freq_ratio = diff_sum / max(len(normalized_samples), 1)

    return {
        "sample_count": len(normalized_samples),
        "sample_rate": sample_rate,
        "rms_energy": rms_energy,
        "energy_variance": energy_variance,
        "zcr": zcr,
        "high_freq_ratio": high_freq_ratio,
    }


def predict(audio_file_path: str) -> dict:
    """
    Main entry point for deepfake audio classification.
    Only function exposed to external callers / API routes.

    Returns:
        {
            "label": "real" | "cloned",
            "confidence": float,   # 0.0 - 1.0
            "processing_time_ms": int,
            "model_version": str
        }
    """
    start_time = time.perf_counter()

    if _MODEL_LOAD_ERROR:
        raise RuntimeError(f"Model service unready: {_MODEL_LOAD_ERROR}")

    # Step 1: Preprocessing & Validation (fails fast on corrupt audio)
    features = _preprocess_audio(audio_file_path)

    # Step 2: Model Inference
    synthetic_prob = 0.5
    model_name = "Acoustic Heuristic Discriminator"

    # Priority A: Trained PyTorch Wav2Vec2 + Attentive Pooling model
    if _LOADED_MODEL is not None and _MODEL_METADATA.get("type") == "pytorch_wav2vec2" and _audio_file_to_waveform is not None:
        try:
            import numpy as np
            raw_audio, _ = _load_audio_waveform(audio_file_path, target_sr=16000)
            if raw_audio is None or len(raw_audio) == 0:
                raise ValueError("Could not decode audio waveform for model inference")
            chunk_len = 16000 * 4  # 4 seconds

            if len(raw_audio) <= chunk_len:
                norm_w = (raw_audio - raw_audio.mean()) / (raw_audio.std() + 1e-6)
                tensor = _torch.from_numpy(norm_w.astype(np.float32)).unsqueeze(0).to(_device)
                with _torch.no_grad():
                    prob_tensor = _LOADED_MODEL.predict_spoof_prob(tensor)
                    synthetic_prob = float(prob_tensor.item())
            else:
                # Active vocal speech multi-window consensus
                step = chunk_len // 2
                windows = []
                for start in range(0, len(raw_audio) - chunk_len + 1, step):
                    w = raw_audio[start : start + chunk_len]
                    w_rms = float(np.sqrt(np.mean(w ** 2)))
                    windows.append((w_rms, w))

                # Focus evaluation on the top speech-dense segments
                windows.sort(key=lambda x: x[0], reverse=True)
                top_windows = [w for _, w in windows[:3]] if windows else [raw_audio[:chunk_len]]

                probs = []
                for w in top_windows:
                    norm_w = (w - w.mean()) / (w.std() + 1e-6)
                    tensor = _torch.from_numpy(norm_w.astype(np.float32)).unsqueeze(0).to(_device)
                    with _torch.no_grad():
                        probs.append(float(_LOADED_MODEL.predict_spoof_prob(tensor).item()))

                synthetic_prob = float(np.median(probs)) if probs else 0.5

            model_name = "SSLSpoofDetector (Wav2Vec2 + Attentive Pooling)"
        except Exception as infer_err:
            print(f"⚠️  [ModelService] PyTorch inference error: {infer_err}, using acoustic fallback", file=sys.stderr)
            synthetic_prob = _calculate_acoustic_score(features)
    # Priority B: ONNX Runtime
    elif _LOADED_MODEL is not None and isinstance(_LOADED_MODEL, dict) is False and _onnxruntime is not None:
        try:
            input_name = _LOADED_MODEL.get_inputs()[0].name
            import numpy as np
            tensor_data = np.array(
                [[features["zcr"], features["rms_energy"], features["energy_variance"], features["high_freq_ratio"]]],
                dtype=np.float32,
            )
            raw_output = _LOADED_MODEL.run(None, {input_name: tensor_data})
            synthetic_prob = float(raw_output[0][0][1] if len(raw_output[0][0]) > 1 else raw_output[0][0][0])
            model_name = "ONNX Deepfake Acoustic Classifier"
        except Exception:
            synthetic_prob = _calculate_acoustic_score(features)
    # Step 3: Hybrid Domain Calibration & Consensus
    # ASVspoof 2019 LA models suffer an acoustic domain shift on mobile/compressed audio (AAC, MP4, Opus, WebM),
    # misinterpreting lossy codec framing and high-frequency cutoff (>7kHz) as synthetic vocoder artifacts.
    # Cross-reference with biological acoustic indicators (vocal tract attenuation, natural prosody, speech dynamics):
    acoustic_score = _calculate_acoustic_score(features)
    ext = os.path.splitext(audio_file_path)[1].lower()
    is_compressed = ext in [".mp4", ".m4a", ".aac", ".ogg", ".opus", ".webm", ".mp3"]

    if _MODEL_METADATA.get("type") == "pytorch_wav2vec2":
        if acoustic_score <= 0.42:
            if is_compressed:
                # Compressed mobile speech with confirmed biological human vocal dynamics:
                # Calibrate probability so genuine human voice notes are correctly classified as real
                synthetic_prob = min(synthetic_prob * 0.25 + acoustic_score * 0.5, 0.22)
            else:
                # Uncompressed PCM with low acoustic score and human dynamics
                if synthetic_prob > 0.60 and features.get("high_freq_ratio", 1.0) < 0.10 and features.get("rms_energy", 0) > 0.03:
                    synthetic_prob = min(synthetic_prob, 0.32)
        elif acoustic_score >= 0.55 and synthetic_prob >= 0.50:
            # Neural vocoder flat-energy or unnatural harmonic signature
            synthetic_prob = max(synthetic_prob, 0.88)

    # Step 4: Post-processing into requested schema
    synthetic_prob = max(0.0, min(1.0, synthetic_prob))

    # Determine classification label
    label = "cloned" if synthetic_prob >= 0.5 else "real"
    confidence = synthetic_prob if label == "cloned" else (1.0 - synthetic_prob)
    confidence = round(confidence, 4)

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    return {
        "label": label,
        "confidence": confidence,
        "spoof_prob": round(synthetic_prob, 4),
        "processing_time_ms": elapsed_ms,
        "model_version": MODEL_VERSION,
        "model_name": model_name,
        "features": features,
    }



def _calculate_acoustic_score(features: dict) -> float:
    """Acoustic feature scoring for vocal synthesis / clone artifacts."""
    zcr = features["zcr"]
    var = features["energy_variance"]
    hf = features["high_freq_ratio"]
    rms = features["rms_energy"]

    # Neural vocoders (HiFi-GAN, WaveGlow, Diffusion) produce distinct signature artifacts:
    # 1. Very low energy variance relative to human natural speech micro-pauses
    # 2. Elevated high-frequency transition smoothness or periodic buzz
    # 3. High zero-crossing regularity in unvoiced segments
    score = 0.5

    if var < 0.005 and rms > 0.02:
        score += 0.22  # Unnatural robotic energy leveling
    elif var > 0.04:
        score -= 0.18  # Natural human dynamic range

    if hf > 0.35:
        score += 0.18  # Neural vocoder phase artifact
    elif hf < 0.15 and rms > 0.03:
        score -= 0.12  # Natural vocal tract attenuation

    if zcr > 0.25:
        score += 0.12
    elif zcr < 0.08:
        score -= 0.10

    return max(0.05, min(0.98, score))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python model_service.py <audio_file_path> | --worker"}))
        sys.exit(1)

    # 1. Persistent High-Performance IPC Worker Mode
    if "--worker" in sys.argv:
        # Signal readiness to parent Node.js process immediately
        ready_payload = {
            "status": "ready",
            "model_type": _MODEL_METADATA.get("type", "acoustic_fallback"),
            "model_path": _MODEL_METADATA.get("path", MODEL_PATH),
        }
        print(json.dumps(ready_payload), flush=True)

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            if line == "PING":
                print(json.dumps({"status": "pong"}), flush=True)
                continue

            req_id = None
            try:
                if line.startswith("{"):
                    data = json.loads(line)
                    target_path = data.get("path", "")
                    req_id = data.get("id")
                else:
                    target_path = line

                res = predict(target_path)
                if req_id is not None:
                    res["id"] = req_id
                print(json.dumps(res), flush=True)
            except ValueError as ve:
                err_payload = {"error": str(ve), "code": 400}
                if req_id is not None:
                    err_payload["id"] = req_id
                print(json.dumps(err_payload), flush=True)
            except Exception as exc:
                err_payload = {"error": str(exc), "code": 500}
                if req_id is not None:
                    err_payload["id"] = req_id
                print(json.dumps(err_payload), flush=True)
        sys.exit(0)

    # 2. Standalone Single-shot CLI execution
    target_path = sys.argv[1]
    try:
        result = predict(target_path)
        print(json.dumps(result))
    except ValueError as ve:
        print(json.dumps({"error": str(ve), "code": 400}), file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(json.dumps({"error": str(exc), "code": 500}), file=sys.stderr)
        sys.exit(3)
