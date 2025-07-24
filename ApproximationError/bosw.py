# bosw.py - Fixed Fast BO for SW

import torch
import numpy as np
from scipy.spatial.distance import cdist
from utils import one_dimensional_Wasserstein_prod

def get_bosw_projections_fast(X, Y, L, device, init_method='coulomb', n_init=10):
    """
    Fast BO for SW using greedy selection from candidate pool
    """
    # For very small L, just return good projections
    if L <= 20:
        if init_method == 'coulomb':
            from qsw import get_cqsw_projections
            return get_cqsw_projections(L, device)
        else:
            from qsw import get_eqsw_projections  
            return get_eqsw_projections(L, device)
    
    # Step 1: Generate diverse candidate pool
    n_candidates = min(500, L * 5)  # Reduced for speed
    
    candidates = []
    
    # 1/3 random projections
    n_random = n_candidates // 3
    random_proj = torch.randn(n_random, 3, device=device)
    random_proj = random_proj / torch.norm(random_proj, dim=1, keepdim=True)
    candidates.append(random_proj)
    
    # 1/3 from Sobol
    from qsw import get_eqsw_projections
    n_sobol = n_candidates // 3
    sobol_proj = get_eqsw_projections(n_sobol, device)
    candidates.append(sobol_proj)
    
    # 1/3 from initialization method
    n_init_method = n_candidates - n_random - n_sobol
    if init_method == 'coulomb':
        from qsw import get_cqsw_projections
        init_proj = get_cqsw_projections(n_init_method, device)
    elif init_method == 'spiral':
        from qsw import get_sqsw_projections
        init_proj = get_sqsw_projections(n_init_method, device)
    else:
        init_proj = torch.randn(n_init_method, 3, device=device)
        init_proj = init_proj / torch.norm(init_proj, dim=1, keepdim=True)
    candidates.append(init_proj)
    
    # Combine all candidates
    all_candidates = torch.cat(candidates, dim=0)
    
    # Step 2: Compute all W values - FIX THE SHAPE ISSUE
    candidate_W_2d = one_dimensional_Wasserstein_prod(X, Y, all_candidates, p=2)
    candidate_W = candidate_W_2d.squeeze(0)  # Convert from (1, n_candidates) to (n_candidates,)
    
    # Step 3: Select diverse initial set
    selected_indices = []
    
    if n_init > 0:
        # First: projection with median W value
        W_np = candidate_W.cpu().numpy()
        median_idx = int(np.argmin(np.abs(W_np - np.median(W_np))))
        selected_indices.append(median_idx)
        
        # Rest: maximize minimum distance to selected
        candidates_np = all_candidates.cpu().numpy()
        for _ in range(1, min(n_init, L)):
            selected_np = candidates_np[selected_indices]
            min_dists = cdist(candidates_np, selected_np, metric='cosine').min(axis=1)
            min_dists[selected_indices] = -1  # Exclude already selected
            next_idx = int(np.argmax(min_dists))
            selected_indices.append(next_idx)
    
    # Step 4: Greedy selection for remaining
    for t in range(len(selected_indices), L):
        # Current SW estimate
        current_W = candidate_W[selected_indices]
        current_mean = current_W.mean().item()
        
        # Find projection that most reduces SW estimate
        best_score = float('inf')
        best_idx = -1
        
        # Evaluate all remaining candidates
        for idx in range(len(all_candidates)):
            if idx in selected_indices:
                continue
                
            # What would mean be with this projection?
            new_mean = (current_mean * len(selected_indices) + candidate_W[idx].item()) / (len(selected_indices) + 1)
            
            # Score is just the new mean (lower is better)
            if new_mean < best_score:
                best_score = new_mean
                best_idx = idx
        
        if best_idx >= 0:
            selected_indices.append(best_idx)
    
    # Return selected projections
    return all_candidates[selected_indices[:L]]


def get_bosw_projections_simple(X, Y, L, device, init_method='coulomb'):
    """
    Even simpler: Select L projections that individually give lowest W values
    """
    # Generate diverse candidates
    n_candidates = min(500, L * 5)
    
    # Mix of projection types
    candidates = []
    
    # Get various projection types
    from qsw import get_eqsw_projections, get_cqsw_projections, get_sqsw_projections
    
    if n_candidates >= 300:
        candidates.append(get_cqsw_projections(100, device))
        candidates.append(get_eqsw_projections(100, device))
        candidates.append(get_sqsw_projections(100, device))
        
        # Add some random
        n_random = n_candidates - 300
        if n_random > 0:
            random_proj = torch.randn(n_random, 3, device=device)
            random_proj = random_proj / torch.norm(random_proj, dim=1, keepdim=True)
            candidates.append(random_proj)
    else:
        # Just use one good method
        candidates.append(get_cqsw_projections(n_candidates, device))
    
    all_candidates = torch.cat(candidates, dim=0)
    
    # Compute W for each
    W_values = one_dimensional_Wasserstein_prod(X, Y, all_candidates, p=2)
    W_values = W_values.squeeze(0)  # Fix shape
    
    # Select top L with lowest W values
    _, indices = torch.topk(W_values, L, largest=False)
    
    return all_candidates[indices]