import torch
import numpy as np
import gpytorch
from utils import *
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize

# --- NEW: Bayesian Optimization Projection Selection (BOSW) ---
from torch.quasirandom import SobolEngine

# ---- Main BO-based Projection Function ----
def get_bosw_coverage_focused(L, device, pc1, pc2, p=2):
    """
    Enhanced coverage-focused BOSW with:
    1. Better candidate generation
    2. Multi-scale coverage optimization
    3. Local refinement
    """
    
    def compute_coverage_score(candidate, existing_projs):
        """Multi-scale coverage metric"""
        if len(existing_projs) == 0:
            return 1.0
            
        # Compute distances to all existing projections
        existing_tensor = torch.stack(existing_projs)
        distances = torch.norm(existing_tensor - candidate.unsqueeze(0), dim=1)
        
        # Multi-scale coverage: consider both minimum and average distance
        min_dist = distances.min().item()
        avg_dist = distances.mean().item()
        
        # Also consider angular distances for better sphere coverage
        angular_dists = torch.acos(torch.clamp(
            torch.matmul(existing_tensor, candidate), -1, 1
        ))
        min_angular = angular_dists.min().item()
        
        # Combined score
        score = min_dist * (1 + 0.5 * avg_dist) * (1 + 0.3 * min_angular)
        return score
    
    def generate_smart_candidates(existing_projs, n_candidates, device):
        """Generate candidates biased toward gaps in coverage"""
        candidates = []
        
        # 1/3 pure random
        n_random = n_candidates // 3
        candidates.append(rand_projections(dim=3, num_projections=n_random, device=device))
        
        # 1/3 repulsive points (far from existing)
        if len(existing_projs) > 0:
            n_repulsive = n_candidates // 3
            repulsive = []
            existing_tensor = torch.stack(existing_projs)
            
            for _ in range(n_repulsive):
                # Start with random point
                point = rand_projections(dim=3, num_projections=1, device=device).squeeze()
                
                # Push away from existing points (mini gradient ascent)
                for _ in range(5):
                    dists = existing_tensor - point.unsqueeze(0)
                    forces = dists / (torch.norm(dists, dim=1, keepdim=True) ** 3 + 1e-6)
                    point = point - 0.1 * forces.sum(dim=0)
                    point = point / torch.norm(point)  # Normalize to sphere
                
                repulsive.append(point)
            
            candidates.append(torch.stack(repulsive))
        
        # 1/3 from quasi-random sequences for structured coverage
        n_structured = n_candidates - n_random - (n_candidates // 3)
        if n_structured > 0:
            # Use different QMC generators
            structured = []
            
            # Some from spiral
            if n_structured > 10:
                spiral = get_sqsw_projections(n_structured // 2, device)
                structured.append(spiral)
            
            # Some from Sobol
            sobol_n = n_structured - (n_structured // 2)
            if sobol_n > 0:
                soboleng = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
                sobol_points = soboleng.draw(sobol_n).to(device) * 2 - 1  # Map to [-1,1]
                sobol_points = sobol_points / torch.norm(sobol_points, dim=1, keepdim=True)
                structured.append(sobol_points)
            
            if structured:
                candidates.append(torch.cat(structured))
        
        return torch.cat(candidates)
    
    def local_refinement(projections, n_iters=10):
        """Locally optimize projection set for better coverage"""
        projs = projections.clone()
        projs.requires_grad_(True)
        
        optimizer = torch.optim.Adam([projs], lr=0.01)
        
        for _ in range(n_iters):
            # Repulsion loss
            dists = torch.cdist(projs, projs, p=2) + torch.eye(len(projs), device=device) * 1e6
            repulsion_loss = -(dists.min(dim=1)[0].mean())
            
            # Sphere constraint
            norm_loss = ((torch.norm(projs, dim=1) - 1) ** 2).mean()
            
            loss = repulsion_loss + 10 * norm_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Final projection to sphere
        projs = projs.detach()
        projs = projs / torch.norm(projs, dim=1, keepdim=True)
        
        return projs
    
    # Handle small L
    if L <= 30:
        # For small L, use refined quasi-random points
        projections = get_sqsw_projections(L, device)
        return local_refinement(projections, n_iters=20)
    
    # Initialize with high-quality points
    selected = []
    initial_n = min(20, L // 5)
    initial_projs = get_sqsw_projections(initial_n, device)
    selected.extend([initial_projs[i] for i in range(initial_n)])
    
    # Adaptive batch sizes for efficiency
    remaining = L - initial_n
    batch_sizes = []
    if remaining > 100:
        # Use larger batches for large L
        while remaining > 0:
            batch_size = min(50, remaining)
            batch_sizes.append(batch_size)
            remaining -= batch_size
    else:
        # One-by-one for small L
        batch_sizes = [1] * remaining
    
    # Main selection loop
    for batch_size in batch_sizes:
        # Generate smart candidates
        n_candidates = max(500, 20 * batch_size)
        candidates = generate_smart_candidates(selected, n_candidates, device)
        
        # Compute coverage scores for all candidates
        scores = torch.tensor([
            compute_coverage_score(candidates[i], selected) 
            for i in range(len(candidates))
        ], device=device)
        
        # Select top batch_size candidates with diversity
        selected_in_batch = []
        for _ in range(batch_size):
            # Get best candidate
            best_idx = torch.argmax(scores).item()
            selected_in_batch.append(candidates[best_idx])
            selected.append(candidates[best_idx])
            
            # Update scores to encourage diversity within batch
            if batch_size > 1:
                new_point = candidates[best_idx]
                for i in range(len(candidates)):
                    dist = torch.norm(candidates[i] - new_point).item()
                    scores[i] *= min(1.0, dist / 0.5)  # Penalize nearby points
        
        # Periodic refinement for better coverage
        if len(selected) % 100 == 0 and len(selected) < L:
            current_tensor = torch.stack(selected)
            refined = local_refinement(current_tensor, n_iters=5)
            selected = [refined[i] for i in range(len(refined))]
    
    # Final refinement
    final_projections = torch.stack(selected)
    if L <= 500:  # Only refine for smaller L (computational cost)
        final_projections = local_refinement(final_projections, n_iters=10)
    
    return final_projections

# -------- Equal-Area QSW (EQSW) --------
def get_eqsw_projections(L, device):
    """Equal-Area QSW (EQSW) using a Sobol sequence."""
    soboleng = torch.quasirandom.SobolEngine(dimension=2, scramble=False)
    net = soboleng.draw(L).to(device)
    alpha, tau = net[:, [0]], net[:, [1]]
    theta = torch.cat([
        2 * torch.sqrt(tau - tau ** 2) * torch.cos(2 * np.pi * alpha),
        2 * torch.sqrt(tau - tau ** 2) * torch.sin(2 * np.pi * alpha),
        1 - 2 * tau
    ], dim=1)
    return theta

# -------- Gaussian QSW (GQSW) --------
def get_gqsw_projections(L, device):
    """Gaussian QSW (GQSW) using Sobol + inverse Gaussian CDF + normalization."""
    soboleng = torch.quasirandom.SobolEngine(dimension=3, scramble=False)
    theta = soboleng.draw(L)
    theta = torch.clamp(theta, min=1e-6, max=1 - 1e-6)
    theta = torch.from_numpy(norm.ppf(theta.numpy()) + 1e-6).float()
    theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True) + 1e-8)
    return theta.to(device)

# -------- Spiral QSW (SQSW) --------
def get_sqsw_projections(L, device):
    """Spiral QSW (SQSW)."""
    indices = torch.arange(1, L + 1, device=device, dtype=torch.float32).view(-1, 1)
    Z = (1 - (2 * indices - 1) / L)
    theta1 = torch.acos(Z)
    theta2 = torch.remainder(1.8 * torch.sqrt(torch.tensor(L, dtype=torch.float32, device=device)) * theta1, 2 * np.pi)
    theta = torch.cat([
        torch.sin(theta1) * torch.cos(theta2),
        torch.sin(theta1) * torch.sin(theta2),
        torch.cos(theta1)
    ], dim=1)
    return theta

# -------- Distance-maximizing QSW (DQSW) --------
def get_dqsw_projections(L, device, iters=100):
    """Distance-maximizing QSW (DQSW) via full pairwise L1 distance maximization."""
    indices = np.arange(1, L + 1)
    Z = (1 - (2 * indices - 1) / L).reshape(-1, 1)
    theta1 = np.arccos(Z)
    theta2 = np.mod(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
    thetas = np.concatenate([
        np.sin(theta1) * np.cos(theta2),
        np.sin(theta1) * np.sin(theta2),
        np.cos(theta1)
    ], axis=1)
    thetas_opt = torch.tensor(thetas, requires_grad=True, device=device, dtype=torch.float32)
    optimizer = torch.optim.SGD([thetas_opt], lr=1.0)
    for _ in range(iters):
        optimizer.zero_grad()
        # Full pairwise L1 distances
        loss = -torch.cdist(thetas_opt, thetas_opt, p=1).mean()
        loss.backward()
        optimizer.step()
        # Normalize to stay on sphere
        thetas_opt.data = thetas_opt.data / torch.norm(thetas_opt.data, dim=1, keepdim=True)
    return thetas_opt.detach()

# -------- Coulomb-minimizing QSW (CQSW) --------
def get_cqsw_projections(L, device, iters=100):
    """Coulomb-minimizing QSW (CQSW) via inverse pairwise L1 energy minimization."""
    indices = np.arange(1, L + 1)
    Z = (1 - (2 * indices - 1) / L).reshape(-1, 1)
    theta1 = np.arccos(Z)
    theta2 = np.mod(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
    thetas = np.concatenate([
        np.sin(theta1) * np.cos(theta2),
        np.sin(theta1) * np.sin(theta2),
        np.cos(theta1)
    ], axis=1)
    thetas_opt = torch.tensor(thetas, requires_grad=True, device=device, dtype=torch.float32)
    optimizer = torch.optim.SGD([thetas_opt], lr=1.0)
    for _ in range(iters):
        optimizer.zero_grad()
        distances = torch.cdist(thetas_opt, thetas_opt, p=1) + 1e-6  # Avoid div by zero
        loss = (1 / distances).mean()
        loss.backward()
        optimizer.step()
        thetas_opt.data = thetas_opt.data / torch.norm(thetas_opt.data, dim=1, keepdim=True)
    return thetas_opt.detach()
