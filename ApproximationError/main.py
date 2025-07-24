# # At the top
# import torch
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from collections import OrderedDict
# from scipy.stats import qmc
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# sys.path.append('..')

# # Your QSW and dataset imports
# from qsw import get_eqsw_projections, get_gqsw_projections, get_sqsw_projections, get_dqsw_projections, get_cqsw_projections
# from utils import rand_projections, one_dimensional_Wasserstein_prod
# from PointcloudAE.dataset.shapenet_core55 import ShapeNetCore55XyzOnlyDataset

# # BO components
# def normalize_safe(vectors):
#     norms = np.linalg.norm(vectors, axis=1, keepdims=True)
#     norms[norms == 0] = 1
#     return vectors / norms

# def get_projection(pc, theta):
#     return pc @ theta.T

# def sliced_wasserstein_1d(proj_A, proj_B):
#     proj_A_sorted = np.sort(proj_A, axis=0)
#     proj_B_sorted = np.sort(proj_B, axis=0)
#     return np.mean((proj_A_sorted - proj_B_sorted) ** 2)

# def evaluate_direction(theta, A, B):
#     proj_A = get_projection(A, theta)
#     proj_B = get_projection(B, theta)
#     return sliced_wasserstein_1d(proj_A, proj_B)

# def bo_sliced_wasserstein(A, B, L=20, eval_grid_size=1000):
#     dim = 3
#     sobol = qmc.Sobol(d=dim, scramble=False)
#     theta_init = normalize_safe(sobol.random(n=8))
#     y_init = np.array([evaluate_direction(th, A, B) for th in theta_init])
#     kernel = C(1.0) * RBF(length_scale=1.0)
#     gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
#     gp.fit(theta_init, y_init)

#     for _ in range(L - len(theta_init)):
#         candidate_thetas = normalize_safe(np.random.randn(eval_grid_size, dim))
#         mu, sigma = gp.predict(candidate_thetas, return_std=True)
#         beta = 2.0
#         ucb = mu - beta * sigma
#         next_theta = candidate_thetas[np.argmin(ucb)]
#         y_next = evaluate_direction(next_theta, A, B)
#         theta_init = np.vstack([theta_init, next_theta])
#         y_init = np.append(y_init, y_next)
#         gp.fit(theta_init, y_init)

#     test_thetas = normalize_safe(np.random.randn(1000, dim))
#     pred_mu = gp.predict(test_thetas)
#     return np.mean(pred_mu)

# def get_bosw_error(L, device, pc1, pc2):
#     A = pc1.cpu().numpy()
#     B = pc2.cpu().numpy()
#     return bo_sliced_wasserstein(A, B, L=L)

# # Main experiment
# def run_full_replication():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     dataset_path = "/Users/manishacharya/Developer/Research/Quasi-SW/PointcloudAE/dataset/shapenet_core55/shapenet57448xyzonly.npz"
#     dataset = ShapeNetCore55XyzOnlyDataset(root=dataset_path, num_points=2048, phase="test")
#     pcs = [torch.from_numpy(dataset[i]).float().to(device) for i in range(5)]
    
#     L_values = [10, 50, 100, 200, 500, 1000]
#     methods = {
#         'SW': lambda L, dev: rand_projections(dim=3, num_projections=L, device=dev),
#         # 'EQSW': get_eqsw_projections,
#         # 'GQSW': get_gqsw_projections,
#         # 'SQSW': get_sqsw_projections,
#         # 'DQSW': get_dqsw_projections,
#         'CQSW': get_cqsw_projections,
#         'BOSW': lambda L, dev: "SPECIAL_CASE",
#     }

#     plot_config = [
#         ('SW', {'label': 'SW', 'color': 'red', 'marker': 'o'}),
#         # ('GQSW', {'label': 'GQSW', 'color': 'orange', 'marker': 'd'}),
#         # ('EQSW', {'label': 'EQSW', 'color': 'green', 'marker': 'v'}),
#         # ('SQSW', {'label': 'SQSW', 'color': 'deepskyblue', 'marker': '|'}),
#         # ('DQSW', {'label': 'DQSW', 'color': 'olive', 'marker': 'h'}),
#         ('CQSW', {'label': 'CQSW', 'color': 'blue', 'marker': 's'}),
#         ('BOSW', {'label': 'BOSW', 'color': 'purple', 'marker': '*'}),
#     ]

#     pair_indices = [(0, 1)]
#     fig, axes = plt.subplots(1, 4, figsize=(24, 6))
#     axes = axes.flatten()

#     for idx, (pi, pj) in enumerate(pair_indices):
#         ax = axes[idx]
#         print(f"\n--- Pair {pi+1} vs {pj+1} ---")
#         pc1, pc2 = pcs[pi], pcs[pj]

#         gt_proj = rand_projections(dim=3, num_projections=100000, device=device)
#         gt_sw_sq = one_dimensional_Wasserstein_prod(pc1, pc2, gt_proj, p=2).mean().item()
#         gt_sw = np.sqrt(gt_sw_sq)
#         print(f"Ground truth: {gt_sw:.6f}")

#         errors = {name: [] for name in methods}
#         for name, proj_func in methods.items():
#             for L in tqdm(L_values, desc=f"{name}"):
#                 if name == 'BOSW':
#                     sw_sq = get_bosw_error(L, device, pc1, pc2)
#                 else:
#                     projections = proj_func(L, device)
#                     sw_sq = one_dimensional_Wasserstein_prod(pc1, pc2, projections, p=2).mean().item()
#                 error = np.abs(np.sqrt(sw_sq) - gt_sw)
#                 errors[name].append(error)

#         for name, style in plot_config:
#             ax.plot(L_values, errors[name], label=style['label'], marker=style['marker'], linewidth=2)

#         ax.set_xscale('linear')
#         ax.set_yscale('log')
#         ax.set_xticks(L_values)
#         ax.set_title(f"{pi+1} vs {pj+1}", fontsize=14)
#         ax.set_xlabel("L")
#         ax.set_ylabel("Absolute Error")
#         ax.grid(True, which="both", linestyle="--", alpha=0.6)

#     handles, labels = ax.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05), fontsize=12)
#     fig.suptitle("Approximation Error of SW Estimators", fontsize=20)
#     plt.tight_layout(rect=[0, 0.06, 1, 0.96])
#     plt.savefig("figure1_full_replication.png")
#     print("✅ Saved as figure1_full_replication.png")
#     plt.show()

# if __name__ == "__main__":
#     run_full_replication()

# main.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from collections import OrderedDict

sys.path.append('..')

# Make sure qsw.py is in the same directory or Python path
from qsw import (
    get_eqsw_projections,
    get_gqsw_projections,
    get_sqsw_projections,
    get_dqsw_projections, 
    get_cqsw_projections,
    get_smart_bo_coverage,
    get_bosw_coverage_plus,
    get_bosw_coverage_focused,  # <-- Import the new method
    one_dimensional_Wasserstein_prod
)
from PointcloudAE.dataset.shapenet_core55 import ShapeNetCore55XyzOnlyDataset

def rand_projections(dim, num_projections=1000, device='cpu'):
    projections = torch.randn((num_projections, dim), device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections

def run_full_replication():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data ---
    print("Loading ShapeNet data...")
    # NOTE: Update this path to where your data is stored
    data_file_path = "/Users/manishacharya/Developer/Research/Quasi-SW/PointcloudAE/dataset/shapenet_core55/shapenet57448xyzonly.npz"
    try:
        dataset = ShapeNetCore55XyzOnlyDataset(root=data_file_path, num_points=2048, phase="test")
    except FileNotFoundError:
        print(f"FATAL: Data file not found at {data_file_path}")
        return
    
    pcs = [torch.from_numpy(dataset[i]).float().to(device) for i in range(5)]
    print("Data loaded.")

    # --- Setup Experiment ---
    L_values = [10,100,500,1000,2000,5000, 10000]
    # L_values = [10,100]
    
    # --- IMPORTANT: Updated methods dictionary ---
    # All lambdas now accept (L, device, pc1, pc2) for a consistent interface.
    # The QMC methods simply ignore pc1 and pc2.
    methods = {
        'SW':    lambda L, dev, pc1, pc2: rand_projections(dim=3, num_projections=L, device=dev),
        'EQSW':  lambda L, dev, pc1, pc2: get_eqsw_projections(L, dev),
        'GQSW':  lambda L, dev, pc1, pc2: get_gqsw_projections(L, dev),
        'SQSW':  lambda L, dev, pc1, pc2: get_sqsw_projections(L, dev),
        'DQSW':  lambda L, dev, pc1, pc2: get_dqsw_projections(L, dev),
        'CQSW':  lambda L, dev, pc1, pc2: get_cqsw_projections(L, dev),
        'BOSW': lambda L, device, pc1, pc2: get_bosw_coverage_focused(
    L, device, pc1, pc2
)

    }

    pair_indices = [(0, 1), (0, 2), (1, 3), (2, 3)] 
    # pair_indices = [(0, 1)] 

    # --- Create Subplots ---
    fig, axes = plt.subplots(1, 4, figsize=(28, 7), sharey=True)
    axes = axes.flatten()

    plot_config = {
        'SW':    {'label': 'SW', 'color': 'red', 'marker': 'o'},
        'GQSW':  {'label': 'GQSW', 'color': 'orange', 'marker': 'd'},
        'EQSW':  {'label': 'EQSW', 'color': 'green', 'marker': 'v'},
        'SQSW':  {'label': 'SQSW', 'color': 'deepskyblue', 'marker': '|'},
        'DQSW':  {'label': 'DQSW', 'color': 'olive', 'marker': 'h'},
        'CQSW':  {'label': 'CQSW', 'color': 'blue', 'marker': 's'},
        'BOSW': {'label': 'BOSW (Ours)', 'color': 'purple', 'marker': '*', 'linestyle': '--'},
    }

    # --- Run Experiments ---
    for idx, (pi, pj) in enumerate(pair_indices):
        ax = axes[idx]
        pair_label = f"{pi+1} vs {pj+1}"
        print(f"\n--- Running experiment for Pair {pair_label} ---")

        pc1, pc2 = pcs[pi], pcs[pj]
        errors = {name: [] for name in methods}

        print(f"Calculating ground truth for Pair {pair_label}...")
        gt_projections = rand_projections(dim=3, num_projections=100000, device=device)
        gt_sw_sq = one_dimensional_Wasserstein_prod(pc1, pc2, gt_projections, p=2).mean().item()
        gt_sw = np.sqrt(gt_sw_sq)
        print(f"Ground Truth SW: {gt_sw:.6f}")

        for name, proj_func in methods.items():
            for L in tqdm(L_values, desc=f"{name.upper():<6} ({pair_label})"):
                # The call is now consistent for all methods
                projections = proj_func(L, device, pc1, pc2)
                sw_sq = one_dimensional_Wasserstein_prod(pc1, pc2, projections, p=2).mean().item()
                error = np.abs(np.sqrt(sw_sq) - gt_sw)
                errors[name].append(error)

        for name, style_args in plot_config.items():
            if name in errors:
                base_style = {'linestyle': '-', 'linewidth': 2, 'markersize': 8}
                base_style.update(style_args)
                ax.plot(L_values, errors[name], **base_style)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(L_values)
        ax.set_xticklabels(L_values)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.set_xlabel("L (Number of Projections)", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Absolute Error", fontsize=12)
        ax.set_title(pair_label, fontsize=14, fontweight='bold')
        ax.grid(True, which="both", linestyle="--", alpha=0.6)

    # --- Global Legend ---
    handles = [plt.Line2D([0], [0], **plot_config[name]) for name in methods]
    labels = [plot_config[name]['label'] for name in methods]
    
    fig.legend(handles, labels, loc='lower center', ncol=len(methods), bbox_to_anchor=(0.5, -0.1), fontsize=12)
    fig.suptitle("Approximation Error of SW Estimators (including BOSW)", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("figure1_replication_with_bo.png", bbox_inches='tight', dpi=300)
    print("\n✅ Plot saved to figure1_replication_with_bo.png")
    plt.show()

if __name__ == "__main__":
    run_full_replication()
