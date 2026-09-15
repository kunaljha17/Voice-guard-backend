"""
Standalone diagnostic: records a few seconds from your mic, saves it as a
.wav file, and scores it with the model directly — no browser, no
WebSocket, no live-streaming buffering involved. This isolates whether a
wrong result comes from the live-demo pipeline or from the model itself.

Usage:
    python record_and_test.py --seconds 5 --model_path ../saved_models/ssl_spoof.pt
"""

import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

from preprocess import waveform_to_model_input, SAMPLE_RATE
from model import SSLSpoofDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--model_path", type=str, default="../saved_models/ssl_spoof.pt")
    parser.add_argument("--save_wav", type=str, default="test_recording.wav")
    args = parser.parse_args()

    print(f"Recording {args.seconds} seconds... speak now.")
    audio = sd.rec(int(args.seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()

    sf.write(args.save_wav, audio, SAMPLE_RATE)
    print(f"Saved recording to {args.save_wav} (you can play this back to double check it recorded correctly)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSLSpoofDetector().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    features = waveform_to_model_input(audio, sr=SAMPLE_RATE)
    tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    spoof_prob = model.predict_spoof_prob(tensor).item()

    print(f"\nspoof_prob = {spoof_prob:.4f}")
    print("(0 = confidently real, 1 = confidently spoof/clone)")


if __name__ == "__main__":
    main()
