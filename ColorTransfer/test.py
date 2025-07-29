import os
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from utils import transform_SW  # Your function
import imageio
import tqdm

# List of (source, target) image paths
image_pairs = [
    ("images/img1.jpg", "images/img2.jpg"),
    ("images/imageA.jpg", "images/imageB.jpg"),
    ("images/autumn.jpg", "images/green_forest.jpg"),
    ("images/forest_road.jpg", "images/ocean_day.jpg"),
    ("images/ocean_day.jpg", "images/ocean_sunset.jpg"),
    ("images/starry_night.jpg", "images/ste_victoire.jpg"),
    ("images/s1.bmp", "images/s2.bmp"),
    ("images/s3.bmp", "images/s4.bmp"),
    ("images/s5.bmp", "images/s6.bmp"),
    ("images/fallingwater.jpg", "images/autumn.jpg"),
]

methods = [
    'sw', 'qsw', 'rqsw', 'sqsw',
    'bosw', 'rbosw',
    'rnqsw', 'rrnqsw',
    'odqsw', 'rodqsw',
    'ocqsw', 'rocqsw'
]

num_seeds = 3
L = 100  # Number of projections
num_iter = 500

results = {method: [] for method in methods}

for method in methods:
    print(f"\n🔍 Evaluating {method.upper()}")
    for src_path, tgt_path in tqdm.tqdm(image_pairs):
        src = imageio.imread(src_path)
        tgt = imageio.imread(tgt_path)
        src_flat = src.reshape(-1, 3)
        tgt_flat = tgt.reshape(-1, 3)

        W2s, SSIMs = [], []

        for seed in range(num_seeds):
            # Run the transfer
            transferred_flat, output_img = transform_SW(
                src_flat, tgt_flat, src_label=np.arange(len(src_flat)),
                origin=src, sw_type=method, L=L, num_iter=num_iter
            )

            # Compute W₂
            w2 = torch.sqrt(torch.mean(torch.sum(
                (torch.tensor(transferred_flat) - torch.tensor(tgt_flat)) ** 2, dim=1
            ))).item()
            W2s.append(w2)

            # SSIM (optional)
            if src.shape == tgt.shape:
                ssim_val = ssim(output_img, tgt, channel_axis=-1)
                SSIMs.append(ssim_val)

        results[method].append({
            'W2_mean': np.mean(W2s),
            'W2_std': np.std(W2s),
            'SSIM_mean': np.mean(SSIMs),
            'SSIM_std': np.std(SSIMs),
        })

# --- Print Summary ---
print("\n📊 Summary:")
for method in methods:
    w2_vals = [r['W2_mean'] for r in results[method]]
    ssim_vals = [r['SSIM_mean'] for r in results[method]]
    print(f"{method.upper():<6} | W₂ avg: {np.mean(w2_vals):.4f} | SSIM avg: {np.mean(ssim_vals):.4f}")
