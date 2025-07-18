import torch
import math
import gpytorch
from gpytorch.kernels import MaternKernel
from utils import one_dimensional_Wasserstein_prod

def bqsw_projections(X, Y, num_projections, device='cuda', p=2):
    """
    Hybrid Bayesian Quadrature for Sliced Wasserstein Distance.
    Combines fast diverse initialization (Method B) with exact GP refinement (Method A).
    """
    dtype = X.dtype
    dim = X.size(1)
    G = 360
    noise = 1e-6

    n_initial = min(20, num_projections // 2)
    n_refine = num_projections - n_initial

    # === Step 1: Generate unit directions (candidates) ===
    U = torch.randn((G, dim), device=device, dtype=dtype)
    U = U / torch.norm(U, dim=1, keepdim=True)

    # === Step 2: Project and compute SW distances ===
    PXs = (X @ U.T).sort(dim=0)[0]  # (N, G)
    PYs = (Y @ U.T).sort(dim=0)[0]
    diff_p = (PXs - PYs).abs().pow(p).mean(dim=0)
    y_grid = diff_p.pow(1.0 / p)  # (G,)

    # === Step 3: Diverse initialization (Method B style) ===
    selected = torch.randperm(G, device=device)[:n_initial].tolist()

    # === Step 4: GP Prior Kernel ===
    kern = MaternKernel(nu=2.5).to(dtype=dtype, device=device)
    u_grid = torch.linspace(0, 1, G, device=device, dtype=dtype).unsqueeze(-1)
    with torch.no_grad():
        gram = kern(u_grid, u_grid).evaluate()  # (G,G)

    # === Step 5: Initial Gram & Cholesky ===
    M = len(selected)
    K0 = gram[selected][:, selected].clone()
    eye = torch.eye(M, device=device, dtype=dtype)
    jitter = noise

    for _ in range(5):
        try:
            L = torch.linalg.cholesky(K0 + jitter * eye)
            break
        except RuntimeError:
            jitter *= 10
    else:
        L = torch.linalg.cholesky(K0 + jitter * eye)

    # === Step 6: Exact GP refinement (Method A style) ===
    for _ in range(n_refine):
        M = len(selected)
        k_x_all = gram[:, selected]  # (G, M)
        w_all = torch.linalg.solve_triangular(L, k_x_all.T, upper=False)  # (M, G)
        var = gram.diagonal() - w_all.pow(2).sum(dim=0)  # (G,)
        var[selected] = float('-inf')  # mask used

        new_idx = int(var.argmax().item())
        if any(abs(new_idx - old) < 1 for old in selected):
            break

        # rank-1 Cholesky update
        a = gram[selected, new_idx]  # (M,)
        w_new = torch.linalg.solve_triangular(L, a.unsqueeze(1), upper=False).squeeze(1)
        d_sq = gram[new_idx, new_idx] + noise - (w_new ** 2).sum()
        d = math.sqrt(max(d_sq.item(), 0.0))

        # Update L
        L_new = torch.zeros((M + 1, M + 1), device=device, dtype=dtype)
        L_new[:M, :M] = L
        L_new[:M, M] = w_new
        L_new[M, M] = d
        L = L_new
        selected.append(new_idx)

    # === Final output ===
    sel_tensor = torch.tensor(selected[:num_projections], device=device, dtype=torch.long)
    return U[sel_tensor]  # (num_projections, dim)

import torch
import math
from gpytorch.kernels import MaternKernel

def rbqsw_plus_projections(X, Y, num_projections, device='cuda', p=2, variance_noise_scale=0.01):
    """
    Enhanced RBQSW with dynamic random rotation and variance noise.
    - Re-selects projections every call (for dynamic gradient flow).
    - Adds noise to GP posterior variance before selection.
    - Applies fresh random rotation per call.
    """
    dtype = X.dtype
    dim = X.size(1)
    G = 360
    noise = 1e-6

    n_initial = min(20, num_projections // 2)
    n_refine = num_projections - n_initial

    # Step 1: Candidate directions
    U = torch.randn((G, dim), device=device, dtype=dtype)
    U = U / U.norm(dim=1, keepdim=True)

    # Step 2: Project and compute 1D SW distances
    PXs = (X @ U.T).sort(dim=0)[0]
    PYs = (Y @ U.T).sort(dim=0)[0]
    diff_p = (PXs - PYs).abs().pow(p).mean(dim=0)
    y_grid = diff_p.pow(1.0 / p)

    # Step 3: Initial selection
    selected = torch.randperm(G, device=device)[:n_initial].tolist()

    # Step 4: GP kernel
    kern = MaternKernel(nu=2.5).to(dtype=dtype, device=device)
    u_grid = torch.linspace(0, 1, G, device=device, dtype=dtype).unsqueeze(-1)
    gram = kern(u_grid, u_grid).evaluate()

    # Step 5: Initial Cholesky
    K0 = gram[selected][:, selected].clone()
    eye = torch.eye(len(selected), device=device, dtype=dtype)
    jitter = noise
    for _ in range(5):
        try:
            L = torch.linalg.cholesky(K0 + jitter * eye)
            break
        except RuntimeError:
            jitter *= 10
    else:
        L = torch.linalg.cholesky(K0 + jitter * eye)

    # Step 6: Greedy variance refinement (with noise)
    for _ in range(n_refine):
        k_x_all = gram[:, selected]
        w_all = torch.linalg.solve_triangular(L, k_x_all.T, upper=False)
        var = gram.diagonal() - w_all.pow(2).sum(dim=0)

        # Add Gaussian noise to encourage stochasticity
        if variance_noise_scale > 0:
            var += variance_noise_scale * torch.randn_like(var)

        var[selected] = float('-inf')
        new_idx = int(var.argmax().item())
        if any(abs(new_idx - old) < 1 for old in selected):
            break

        a = gram[selected, new_idx]
        w_new = torch.linalg.solve_triangular(L, a.unsqueeze(1), upper=False).squeeze(1)
        d_sq = gram[new_idx, new_idx] + noise - (w_new ** 2).sum()
        d = math.sqrt(max(d_sq.item(), 0.0))

        L_new = torch.zeros((len(selected)+1, len(selected)+1), device=device, dtype=dtype)
        L_new[:len(selected), :len(selected)] = L
        L_new[:len(selected), len(selected)] = w_new
        L_new[len(selected), len(selected)] = d
        L = L_new
        selected.append(new_idx)

    U_selected = U[torch.tensor(selected, device=device)]

    # Step 7: Apply fresh random rotation
    q = torch.randn(4, device=device, dtype=dtype)
    q = q / q.norm()
    R = torch.tensor([
        [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[3]*q[0], 2*q[1]*q[3] + 2*q[2]*q[0]],
        [2*q[1]*q[2] + 2*q[3]*q[0], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[1]*q[0]],
        [2*q[1]*q[3] - 2*q[2]*q[0], 2*q[2]*q[3] + 2*q[1]*q[0], 1 - 2*q[1]**2 - 2*q[2]**2]
    ], device=device, dtype=dtype)
    return U_selected @ R.T

