# Add these imports to the top of your script
import gpytorch
import torch
from qsw import *
from utils import *
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

# -------- CORRECTED Bayesian Optimized SW (BOSW) --------

def spherical_to_cartesian(coords):
    """Converts spherical coordinates (phi, theta) to 3D Cartesian vectors."""
    phi, theta = coords[..., 0], coords[..., 1]
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)

def get_bosw_projections(L, device, pc1, pc2, n_init=10, p_norm=2):
    """
    Bayesian Optimization to find optimal projections for SW.
    """
    if L <= n_init:
        # Fall back to a QMC method if L is too small for BO
        return get_sqsw_projections(L, device)

    # Define the objective function for BO
    def sw_objective(coords_2d):
        # BoTorch evaluates in batches, ensure input is at least 2D
        if coords_2d.dim() == 1:
            coords_2d = coords_2d.unsqueeze(0)
        
        projections_3d = spherical_to_cartesian(coords_2d).to(device)
        sw_sq = one_dimensional_Wasserstein_prod(pc1, pc2, projections_3d, p=p_norm)
        
        # FIX 1: Ensure the output is shape (batch_size, 1) for the GP model
        return torch.sqrt(sw_sq).unsqueeze(-1)

    # --- BO Setup ---
    search_space_bounds = torch.tensor([[0., 0.], [2 * np.pi, np.pi]], device=device, dtype=torch.float32)
    n_iter = L - n_init

    # --- Initialization ---
    initial_coords = torch.rand(n_init, 2, device=device) * search_space_bounds[1]
    
    # FIX 2: `sw_objective` now returns the correct shape, so no extra unsqueeze is needed
    train_x = initial_coords
    train_y = sw_objective(initial_coords) # Correctly shaped to (n_init, 1)

    # --- Optimization Loop ---
    for _ in range(n_iter):
        with gpytorch.settings.cholesky_jitter(1e-5):
            gp = SingleTaskGP(train_x, train_y)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

        acq_func = UpperConfidenceBound(gp, beta=0.2)
        
        new_candidate, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=search_space_bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )

        # FIX 3: Evaluate the new candidate (no .squeeze() needed)
        new_value = sw_objective(new_candidate)

        # FIX 4: Update training data with correctly shaped tensors
        train_x = torch.cat([train_x, new_candidate])
        train_y = torch.cat([train_y, new_value])

    final_projections = spherical_to_cartesian(train_x)
    return final_projections.detach()