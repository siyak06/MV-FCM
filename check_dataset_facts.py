"""Run from the folder that holds mvfcm_multiseed_v2.py (and CUB_200_2011/images, cub_features_v2.npz).
Prints the facts the paper states about CUB-200-2011, MNIST+USPS and Reuters-21578."""
import os, numpy as np

# ---- CUB: folder counts vs cached features -------------------------------
root = "CUB_200_2011/images"
dirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))[:10]
tot = 0
for d in dirs:
    n = len([f for f in os.listdir(os.path.join(root, d)) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    tot += n; print(f"{d:35s} {n}")
print("images in first ten class folders:", tot)
if os.path.exists("cub_features_v2.npz"):
    z = np.load("cub_features_v2.npz")
    print("cached samples:", len(z["y"]), "| per-class:", np.bincount(z["y"]).tolist(), "| view dims:", z["v1"].shape[1], z["v2"].shape[1])
    print("MATCH" if len(z["y"]) == tot else "MISMATCH: delete cub_features_v2.npz, rerun, and rerun all experiments")

# ---- Reuters: categories and overlap --------------------------------------
import nltk
nltk.download("reuters", quiet=True)
from nltk.corpus import reuters as rc
cats = sorted(rc.categories(), key=lambda t: -len(rc.fileids(t)))[:4]
sel = [f for c in cats for f in rc.fileids(c)[:200]]
print("\nReuters categories:", cats, [len(rc.fileids(c)) for c in cats])
print("selected:", len(sel), "unique:", len(set(sel)), "-> documents used under two labels:", len(sel) - len(set(sel)))

# ---- MNIST+USPS: which path ran ---------------------------------------------
import mvfcm_multiseed_v2 as base
views, y = base.load_mnist_usps()
print("\nMNIST+USPS views:", [v.shape for v in views], "labels:", np.bincount(y).tolist())
print("real USPS used" if views[1].shape[1] == 256 else "FALLBACK (HOG) was used")