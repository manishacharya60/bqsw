import torch
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from utils import rand_projections, one_dimensional_Wasserstein_prod

def get_bqsw_projections(
    L, device, pc1, pc2, p=2, initial_L=10, candidate_L=1000, beta=2.0, mc_integral_samples=1000, seed=None, epsilon=1e-8
):
    """
    Log-GP BQ with UCB acquisition and posterior variance.
    
    Returns:
        theta_final: [L, d] selected projection directions
        integral_mean: estimated SW
        integral_std: std deviation of the estimate
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    dim = pc1.shape[1]
    theta_list = []
    f_log_list = []

    # Step 1: Random initial directions
    theta_init = rand_projections(dim, initial_L, device)
    f_raw_init = one_dimensional_Wasserstein_prod(pc1, pc2, theta_init, p=p).squeeze()
    f_log_init = torch.log(f_raw_init + epsilon)

    theta_list.append(theta_init)
    f_log_list.append(f_log_init)

    while sum(t.shape[0] for t in theta_list) < L:
        theta_all = torch.cat(theta_list, dim=0)
        f_log_all = torch.cat(f_log_list, dim=0)

        theta_np = theta_all.cpu().numpy()
        f_log_np = f_log_all.cpu().numpy().reshape(-1, 1)

        # Scaling
        theta_scaler = StandardScaler().fit(theta_np)
        f_scaler = StandardScaler().fit(f_log_np)

        theta_scaled = theta_scaler.transform(theta_np)
        f_log_scaled = f_scaler.transform(f_log_np).ravel()

        # Fixed GP (no optimizer)
        gp = GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0),
            alpha=1e-4,
            normalize_y=False,
            optimizer=None
        )
        gp.fit(theta_scaled, f_log_scaled)

        # Candidate θ and UCB in log-space
        theta_cand = rand_projections(dim, candidate_L, device)
        theta_cand_scaled = theta_scaler.transform(theta_cand.cpu().numpy())

        mu_log_scaled, sigma_log = gp.predict(theta_cand_scaled, return_std=True)
        mu_log = f_scaler.inverse_transform(mu_log_scaled.reshape(-1, 1)).ravel()

        ucb_log = mu_log + beta * sigma_log
        best_idx = np.argmax(ucb_log)
        best_theta = theta_cand[best_idx].unsqueeze(0)

        f_raw_best = one_dimensional_Wasserstein_prod(pc1, pc2, best_theta, p=p).squeeze()
        f_log_best = torch.log(f_raw_best + epsilon)

        theta_list.append(best_theta)
        f_log_list.append(f_log_best.unsqueeze(0))

    # Final training set
    theta_final = torch.cat(theta_list, dim=0)[:L]
    f_log_final = torch.cat(f_log_list, dim=0)[:L]

    # Final GP fit on full set
    theta_np = theta_final.cpu().numpy()
    f_log_np = f_log_final.cpu().numpy().reshape(-1, 1)

    theta_scaler = StandardScaler().fit(theta_np)
    f_scaler = StandardScaler().fit(f_log_np)

    theta_scaled = theta_scaler.transform(theta_np)
    f_log_scaled = f_scaler.transform(f_log_np).ravel()

    gp = GaussianProcessRegressor(
        kernel=RBF(length_scale=1.0),
        alpha=1e-4,
        normalize_y=False,
        optimizer=None
    )
    gp.fit(theta_scaled, f_log_scaled)

    # Estimate posterior mean and variance via Monte Carlo over sphere
    theta_mc = rand_projections(dim, mc_integral_samples, device)
    theta_mc_scaled = theta_scaler.transform(theta_mc.cpu().numpy())

    mu_log_scaled, sigma_log = gp.predict(theta_mc_scaled, return_std=True)
    mu_log = f_scaler.inverse_transform(mu_log_scaled.reshape(-1, 1)).ravel()

    # Log-normal moment matching: E[f] ≈ exp(μ + σ²/2)
    f_pred_mean = np.exp(mu_log + 0.5 * sigma_log**2)
    integral_mean = float(np.mean(f_pred_mean))
    integral_var = float(np.var(f_pred_mean))
    integral_std = np.sqrt(integral_var)

    return theta_final, integral_mean, integral_std
