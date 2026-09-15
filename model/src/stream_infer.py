"""
Live real-time detection from the microphone, using the trained SSL
(Wav2Vec2) model instead of the earlier CNN.

Usage:
    python stream_infer.py --model_path ../saved_models/ssl_spoof.pt
    python stream_infer.py --model_path ../saved_models/ssl_spoof.pt --threshold 0.35
"""

import argparse
import collections
import time

import numpy as np
import sounddevice as sd
import torch

from preprocess import waveform_to_model_input, SAMPLE_RATE
from model import SSLSpoofDetector

WINDOW_SEC = 3.0        # a bit longer than the CNN version — Wav2Vec2 benefits
                        # from more context per window
STEP_SEC = 1.0          # how often we run inference
SMOOTH_N = 3            # rolling average over last N scores
ALERT_CONSECUTIVE = 2   # need this many consecutive smoothed scores above threshold


class RingBuffer:
    def __init__(self, max_samples: int):
        self.buffer = np.zeros(max_samples, dtype=np.float32)
        self.max_samples = max_samples

    def write(self, chunk: np.ndarray):
        n = len(chunk)
        if n >= self.max_samples:
            self.buffer[:] = chunk[-self.max_samples:]
        else:
            self.buffer = np.roll(self.buffer, -n)
            self.buffer[-n:] = chunk

    def read(self) -> np.ndarray:
        return self.buffer.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../saved_models/ssl_spoof.pt")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Lower this (e.g. 0.35) to catch more clones at the "
                              "cost of more false alarms — see evaluate.py results.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (CPU is fine for occasional inference, just slower per window)")

    model = SSLSpoofDetector().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    ring = RingBuffer(int(WINDOW_SEC * SAMPLE_RATE))
    score_history = collections.deque(maxlen=SMOOTH_N)
    consecutive_alerts = 0

    def audio_callback(indata, frames, time_info, status):
        ring.write(indata[:, 0])

    print("Listening... (Ctrl+C to stop)")
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, callback=audio_callback,
        blocksize=int(STEP_SEC * SAMPLE_RATE),
    ):
        try:
            while True:
                time.sleep(STEP_SEC)
                audio = ring.read()

                if np.abs(audio).mean() < 0.001:
                    continue  # skip near-silent windows

                features = waveform_to_model_input(audio)
                tensor = torch.from_numpy(features).unsqueeze(0).to(device)
                spoof_prob = model.predict_spoof_prob(tensor).item()

                score_history.append(spoof_prob)
                smoothed = float(np.mean(score_history))

                if smoothed > args.threshold:
                    consecutive_alerts += 1
                    status_str = "SUSPECTED CLONE"
                else:
                    consecutive_alerts = 0
                    status_str = "REAL"

                alert = consecutive_alerts >= ALERT_CONSECUTIVE
                flag = "  <<< ALERT: possible voice cloning attack" if alert else ""
                print(f"spoof_prob={smoothed:.3f}  [{status_str}]{flag}")

        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
