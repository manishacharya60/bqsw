# file: bayesian_sw.py

import os
import random
import GPy
import torch
import numpy as np
from scipy.spatial.distance import pdist
from GradientFlow.utils import one_dimensional_Wasserstein_prod

# ✅ Set global seed for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
rng = np.random.default_rng(SEED)

def get_sobol_projections(L, device='cpu'):
    soboleng = torch.quasirandom.SobolEngine(dimension=2, scramble=False)
    net = soboleng.draw(L).to(device)
    alpha, tau = net[:, [0]], net[:, [1]]
    theta = torch.cat([
        2 * torch.sqrt(tau - tau ** 2) * torch.cos(2 * np.pi * alpha),
        2 * torch.sqrt(tau - tau ** 2) * torch.sin(2 * np.pi * alpha),
        1 - 2 * tau
    ], dim=1)
    return theta

def get_coulomb_projections(L, device='cpu'):
    Z = (1 - (2 * np.arange(1, L + 1) - 1) / L).reshape(-1, 1)
    theta1 = np.arccos(Z)
    theta2 = np.mod(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
    thetas = np.concatenate([
        np.sin(theta1) * np.cos(theta2),
        np.sin(theta1) * np.sin(theta2),
        np.cos(theta1)
    ], axis=1)
    theta0 = torch.from_numpy(thetas).float()

    thetas_opt = torch.randn(L, 3, requires_grad=True, device=device)
    thetas_opt.data = theta0.to(device)

    optimizer = torch.optim.SGD([thetas_opt], lr=1)
    for _ in range(100):
        loss = (1 / (torch.cdist(thetas_opt, thetas_opt, p=1) + 1e-6)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        thetas_opt.data = thetas_opt.data / torch.sqrt(torch.sum(thetas_opt.data ** 2, dim=1, keepdim=True))

    return thetas_opt.detach()

# Corrected function for bayesian_sw.py

def vanilla_bq_sliced_wasserstein(p_A, p_B, L, device):
    """
    CORRECTED VANILLA BQ: Uses the same Sobol points as Vanilla QSW for a fair comparison.
    """
    def f(thetas_torch):
        distances = one_dimensional_Wasserstein_prod(p_A, p_B, thetas_torch, p=2)
        return distances.cpu().numpy()

    # CRITICAL FIX: Use the Sobol projections to match Vanilla QSW
    thetas_torch = get_sobol_projections(L, device=device)
    y_np = f(thetas_torch).reshape(-1, 1)
    thetas_np = thetas_torch.cpu().numpy()

    # The rest of your simple GP model is fine for a vanilla implementation
    kernel = GPy.kern.RBF(input_dim=3)
    gp_model = GPy.models.GPRegression(thetas_np, y_np, kernel)
    try:
        gp_model.optimize(messages=False)
    except:
        # Fallback if optimization fails
        pass

    # Use a simple random integration for the vanilla version
    integration_thetas = rng.standard_normal((1000, 3))
    integration_thetas /= np.linalg.norm(integration_thetas, axis=1, keepdims=True)
    posterior_mean, _ = gp_model.predict(integration_thetas)
    
    return np.sqrt(max(np.mean(posterior_mean), 0))

def bq_sliced_wasserstein(p_A, p_B, L, device):
    def f(thetas_torch):
        distances = one_dimensional_Wasserstein_prod(p_A, p_B, thetas_torch, p=2)
        return distances.cpu().numpy()

    thetas_torch = get_coulomb_projections(L, device=device)
    y_np = f(thetas_torch).reshape(-1, 1)
    thetas_np = thetas_torch.cpu().numpy()

    y_mean, y_std = np.mean(y_np), np.std(y_np) + 1e-8
    y_normalized = (y_np - y_mean) / y_std

    kernel = GPy.kern.Matern52(input_dim=3, variance=1.0, lengthscale=np.median(pdist(thetas_np)))
    gp_model = GPy.models.GPRegression(thetas_np, y_normalized, kernel, normalizer=False, noise_var=0.01)

    gp_model.kern.variance.constrain_bounded(0.1, 10)
    gp_model.kern.lengthscale.constrain_bounded(0.01, 2)
    gp_model.likelihood.variance.constrain_bounded(1e-3, 0.1)

    try:
        gp_model.likelihood.variance.fix(0.01)
        gp_model.optimize(max_iters=100, messages=False)
        gp_model.likelihood.variance.unfix()
        gp_model.optimize_restarts(num_restarts=3, robust=True, verbose=False, parallel=False)
    except Exception as e:
        print("Optimization warning:", e)

    soboleng = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
    net = soboleng.draw(5000)
    alpha, tau = net[:, [0]].numpy(), net[:, [1]].numpy()
    integration_thetas = np.column_stack([
        2 * np.sqrt(tau * (1 - tau)) * np.cos(2 * np.pi * alpha),
        2 * np.sqrt(tau * (1 - tau)) * np.sin(2 * np.pi * alpha),
        1 - 2 * tau
    ])

    posterior_mean, _ = gp_model.predict(integration_thetas)
    posterior_mean = posterior_mean * y_std + y_mean
    return np.sqrt(max(np.mean(posterior_mean), 0))
