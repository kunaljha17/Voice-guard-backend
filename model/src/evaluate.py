"""
Evaluate the trained SSL model on the ASVspoof eval split, or your own
data/custom/ folder (real + cloned recordings).

Usage:
    python evaluate.py --data_root ../data/LA --split eval --model_path ../saved_models/ssl_spoof.pt
    python evaluate.py --custom_root ../data/custom --model_path ../saved_models/ssl_spoof.pt
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve

from dataset import ASVspoofWaveformDataset, CustomFolderWaveformDataset
from model import SSLSpoofDetector


def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="eval")
    parser.add_argument("--custom_root", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="../saved_models/ssl_spoof.pt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Spoof-probability cutoff. Lower catches more clones "
                              "at the cost of more false alarms on real voices.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSLSpoofDetector().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    if args.custom_root:
        ds = CustomFolderWaveformDataset(args.custom_root)
    else:
        ds = ASVspoofWaveformDataset(args.data_root, split=args.split)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    all_labels, all_scores, all_preds = [], [], []
    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            probs = model.predict_spoof_prob(waveforms).cpu().numpy()
            preds = (probs > args.threshold).astype(int)
            all_labels.extend(labels.numpy())
            all_scores.extend(probs)
            all_preds.extend(preds)

    all_labels, all_scores, all_preds = map(np.array, (all_labels, all_scores, all_preds))

    print(f"Samples evaluated: {len(all_labels)}")
    print(f"Threshold used: {args.threshold}")
    print(f"EER: {compute_eer(all_labels, all_scores):.4f}")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print("Confusion matrix (rows=true, cols=pred) [0=real, 1=spoof]:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
