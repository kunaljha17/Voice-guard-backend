"""Quick sanity check that the ASVspoof dataset is laid out correctly."""

from dataset import ASVspoofDataset

if __name__ == "__main__":
    for split in ["train", "dev", "eval"]:
        try:
            ds = ASVspoofDataset("../data/LA", split=split)
            n_bonafide = sum(1 for _, label in ds.items if label == 0)
            n_spoof = sum(1 for _, label in ds.items if label == 1)
            print(f"[{split}] total={len(ds)}  bonafide={n_bonafide}  spoof={n_spoof}")
        except FileNotFoundError as e:
            print(f"[{split}] NOT FOUND — check data/README.md for setup steps. ({e})")
