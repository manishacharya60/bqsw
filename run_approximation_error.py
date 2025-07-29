# # file: run_approximation_error_avg.py

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

# Reproducibility
SEED = 1024
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

sys.path.append('./PointCloudAE')
from bayesian_sw import (
    bq_sliced_wasserstein,
    vanilla_bq_sliced_wasserstein,
    get_coulomb_projections,
    get_sobol_projections
)
from GradientFlow.utils import SW as random_sw
from GradientFlow.utils import one_dimensional_Wasserstein_prod
from dataset.shapenet_core55 import ShapeNetCore55XyzOnlyDataset

def run_experiment(num_trials=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading ShapeNet data...")
    data_file_path = os.path.join(
        "/Users/manishacharya/Developer/Research/Quasi-SW/PointcloudAE/dataset/shapenet_core55/",
        "shapenet57448xyzonly.npz"
    )
    dataset = ShapeNetCore55XyzOnlyDataset(root=data_file_path, num_points=2048, phase="test")
    pc1 = torch.from_numpy(dataset[0]).float().unsqueeze(0).to(device)
    pc2 = torch.from_numpy(dataset[1]).float().unsqueeze(0).to(device)
    print("Data loaded.")

    print("Calculating ground truth with L=100,000...")
    gt_sw = random_sw(pc1[0], pc2[0], L=100000, p=2, device=device).item()
    print(f"Ground Truth SW: {gt_sw:.6f}")

    L_values = [10, 50, 100, 200, 500, 1000, 2000]
    methods = ['vanilla_qsw', 'adv_qsw', 'vanilla_bq', 'adv_bq']
    errors = {m: np.zeros((len(L_values), num_trials)) for m in methods}

    for t in range(num_trials):
        print(f"\n🔁 Trial {t+1}/{num_trials}")
        for i, L in enumerate(tqdm(L_values, desc="Evaluating Ls")):
            thetas_sobol = get_sobol_projections(L, device)
            sq = one_dimensional_Wasserstein_prod(pc1[0], pc2[0], thetas_sobol, p=2).mean().item()
            errors['vanilla_qsw'][i, t] = np.abs(np.sqrt(sq) - gt_sw)

            thetas_coulomb = get_coulomb_projections(L, device)
            sq = one_dimensional_Wasserstein_prod(pc1[0], pc2[0], thetas_coulomb, p=2).mean().item()
            errors['adv_qsw'][i, t] = np.abs(np.sqrt(sq) - gt_sw)

            vbq = vanilla_bq_sliced_wasserstein(pc1[0], pc2[0], L, device)
            errors['vanilla_bq'][i, t] = np.abs(vbq - gt_sw)

            abq = bq_sliced_wasserstein(pc1[0], pc2[0], L, device)
            errors['adv_bq'][i, t] = np.abs(abq - gt_sw)

    fig, ax = plt.subplots(figsize=(6, 5))

    def plot_line(method, label, color, marker):
        means = errors[method].mean(axis=1)
        stds = errors[method].std(axis=1)
        ax.plot(L_values, means, label=label, color=color, marker=marker, linewidth=2, markersize=6)

    plot_line('vanilla_qsw', 'QSW', 'green', 'o')
    plot_line('adv_qsw', 'Adv QSW', 'blue', 'D')
    plot_line('vanilla_bq', 'BQ', 'gray', 'v')
    plot_line('adv_bq', 'Adv BQ', 'red', 's')

    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xticks([0, 1000, 1500, 2000])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("L", fontsize=14, fontweight='bold')
    ax.set_ylabel("Absolute Error", fontsize=14, fontweight='bold')
    ax.set_title("1 vs 2", fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("figure1_style_avg_error_1_vs_2.png")
    print("\n✅ Saved: figure1_style_avg_error_1_vs_2.png")
    plt.show()

if __name__ == "__main__":
    run_experiment(num_trials=5)

# file: run_approximation_error_avg.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

sys.path.append('./PointCloudAE')

from bayesian_sw import (
    bq_sliced_wasserstein,
    vanilla_bq_sliced_wasserstein,
    get_coulomb_projections,
    get_sobol_projections
)
from GradientFlow.utils import SW as random_sw
from GradientFlow.utils import one_dimensional_Wasserstein_prod
from dataset.shapenet_core55 import ShapeNetCore55XyzOnlyDataset

def run_experiment(num_trials=5, show_error_bars=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load point cloud data
    print("Loading ShapeNet data...")
    data_file_path = os.path.join(
        "/Users/manishacharya/Developer/Research/Quasi-SW/PointcloudAE/dataset/shapenet_core55/",
        "shapenet57448xyzonly.npz"
    )
    dataset = ShapeNetCore55XyzOnlyDataset(root=data_file_path, num_points=2048, phase="test")
    pcs = [torch.from_numpy(dataset[i]).float().unsqueeze(0).to(device) for i in range(5)]
    print("Data loaded.")

    pair_indices = [(1, 2), (1, 3), (2, 4), (3, 4)]
    L_values = [10, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 10000]
    methods = ['vanilla_qsw', 'adv_qsw', 'vanilla_bq', 'adv_bq']

    for pi, pj in pair_indices:
        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()
        pair_label = f"{pi+1} vs {pj+1}"

        print(f"\n📊 Plotting error curve for Pair {pair_label}")
        pair_errors = {m: [[] for _ in L_values] for m in methods}

        A = pcs[pi]
        B = pcs[pj]
        print("Calculating ground truth with L=100,000...")
        gt_sw = random_sw(A[0], B[0], L=100000, p=2, device=device).item()
        print(f"Ground Truth SW: {gt_sw:.6f}")

        for t in range(num_trials):
            print(f"\n🔁 Trial {t+1}/{num_trials} for Pair {pair_label}")
            for i, L in enumerate(tqdm(L_values, desc=f"Evaluating L for {pair_label}")):
                # Vanilla QSW
                sobol_thetas = get_sobol_projections(L, device)
                sq = one_dimensional_Wasserstein_prod(A[0], B[0], sobol_thetas, p=2).mean().item()
                pair_errors['vanilla_qsw'][i].append(np.abs(np.sqrt(sq) - gt_sw))

                # Advanced QSW
                coulomb_thetas = get_coulomb_projections(L, device)
                sq = one_dimensional_Wasserstein_prod(A[0], B[0], coulomb_thetas, p=2).mean().item()
                pair_errors['adv_qsw'][i].append(np.abs(np.sqrt(sq) - gt_sw))

                # Vanilla BQ
                vbq = vanilla_bq_sliced_wasserstein(A[0], B[0], L, device)
                pair_errors['vanilla_bq'][i].append(np.abs(vbq - gt_sw))

                # Advanced BQ
                abq = bq_sliced_wasserstein(A[0], B[0], L, device)
                pair_errors['adv_bq'][i].append(np.abs(abq - gt_sw))

        def plot_curve(errs, label, color, style, marker):
            means = [np.mean(errs[i]) for i in range(len(L_values))]
            if show_error_bars:
                stds = [np.std(errs[i]) for i in range(len(L_values))]
                ax.errorbar(L_values, means, yerr=stds, fmt=marker+style, color=color,
                            label=label, capsize=3, markersize=6, linewidth=2)
            else:
                ax.plot(L_values, means, marker+style, color=color, label=label,
                        markersize=6, linewidth=2)

        plot_curve(pair_errors['vanilla_qsw'], 'QSW', 'green', '-', 'o')
        plot_curve(pair_errors['adv_qsw'], 'Adv QSW', 'blue', '-', 'D')
        plot_curve(pair_errors['vanilla_bq'], 'BQ', 'grey', '-', 'v')
        plot_curve(pair_errors['adv_bq'], 'Adv BQ', 'red', '-', 's')

        ax.set_xscale('linear')  # 📌 Linear X axis
        ax.set_yscale('log')     # 📌 Log Y axis
        ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        ax.set_xlabel("L", fontsize=14, fontweight='bold')
        ax.set_ylabel("Absolute Error", fontsize=14, fontweight='bold')
        ax.set_title(f"{pair_label}", fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=12, frameon=True)
        ax.grid(True, which="both", ls="--", alpha=0.6)

if __name__ == "__main__":
    run_experiment(num_trials=5, show_error_bars=False)

