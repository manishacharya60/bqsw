import numpy as np
import torch
from torch.autograd import Variable
import ot
import random
import tqdm
from scipy.stats import norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_true_Wasserstein(X,Y,p=2):
    M = ot.dist(X.cpu().detach().numpy(), Y.cpu().detach().numpy())
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)

def rand_projections(dim, num_projections=1000,device=device):
    projections = torch.randn((num_projections, dim),device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections


def one_dimensional_Wasserstein_prod(X,Y,theta,p):
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    X_prod = X_prod.view(X_prod.shape[0], -1)
    Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    wasserstein_distance = torch.abs(
        (
                torch.sort(X_prod, dim=0)[0]
                - torch.sort(Y_prod, dim=0)[0]
        )
    )
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=0,keepdim=True)
    return wasserstein_distance


def SW(X, Y, L=10, p=2, device=device):
    dim = X.size(1)
    theta = rand_projections(dim, L,device)
    sw=one_dimensional_Wasserstein_prod(X,Y,theta,p=p).mean()
    return  torch.pow(sw,1./p)

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

def get_bosw_bayesian(L, device, pc1, pc2, p=2, beta=2.0, seed=None, ai='ucb'):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    @torch.no_grad()
    def one_d_wasserstein_mean(pc1, pc2, theta):
        return one_dimensional_Wasserstein_prod(pc1, pc2, theta, p=p).mean().sqrt()

    class LightweightGP:
        def __init__(self, kernel_lengthscale=0.5):
            self.lengthscale = kernel_lengthscale
            self.X_train, self.y_train = None, None
            self.y_mean, self.y_std = 0, 1
            self.base_noise = 1e-3

        def fit(self, X, y):
            self.X_train = X
            self.y_train = y
            self.y_mean = y.mean()
            self.y_std = y.std() + 1e-6

            if len(X) > 5:
                angles = torch.acos(torch.clamp(X @ X.T, -1, 1))
                angles_nonzero = angles[angles > 0]
                if len(angles_nonzero) > 0:
                    median_dist = angles_nonzero.median().item()
                    self.lengthscale = np.clip(median_dist * 0.5, 0.1, 2.0)

        def predict(self, X_test):
            if self.X_train is None:
                return torch.zeros(len(X_test)), torch.ones(len(X_test))
            try:
                K_train = self._kernel(self.X_train, self.X_train)
                K_test = self._kernel(X_test, self.X_train)
                K_test_test = self._kernel(X_test, X_test)

                eigenvalues = torch.linalg.eigvalsh(K_train)
                min_eig = eigenvalues.min().item()
                noise_level = max(self.base_noise, 1e-2) if min_eig < 1e-6 else self.base_noise
                noise_I = noise_level * torch.eye(len(K_train), device=K_train.device)

                for attempt in range(5):
                    try:
                        L = torch.linalg.cholesky(K_train + noise_I)
                        break
                    except:
                        noise_level *= 10
                        noise_I = noise_level * torch.eye(len(K_train), device=K_train.device)
                        if attempt == 4:
                            return self._predict_svd(K_train, K_test, K_test_test)

                alpha = torch.cholesky_solve(((self.y_train - self.y_mean) / self.y_std).unsqueeze(1), L).squeeze()
                mu = K_test @ alpha
                v = torch.linalg.solve(L, K_test.T)
                var = torch.diag(K_test_test - v.T @ v)
                return mu * self.y_std + self.y_mean, torch.sqrt(var.clamp(min=1e-6)) * self.y_std

            except:
                return torch.zeros(len(X_test)) + self.y_mean, torch.ones(len(X_test)) * self.y_std

        def _predict_svd(self, K_train, K_test, K_test_test):
            U, S, Vt = torch.linalg.svd(K_train)
            mask = S > 1e-6
            U, S, Vt = U[:, mask], S[mask], Vt[mask, :]
            K_inv_y = U @ torch.diag(1/S) @ Vt @ ((self.y_train - self.y_mean) / self.y_std)
            mu = K_test @ K_inv_y
            var = torch.diag(K_test_test).clamp(min=1e-6)
            return mu * self.y_std + self.y_mean, torch.sqrt(var) * self.y_std

        def _kernel(self, X1, X2):
            angles = torch.acos(torch.clamp(X1 @ X2.T, -0.9999, 0.9999))
            return torch.exp(-0.5 * (angles / self.lengthscale) ** 2)

    @torch.no_grad()
    def compute_enhanced_objective(proj, existing_projections, pc1, pc2, cached_current_sw=None):
        if len(existing_projections) > 0:
            existing_tensor = torch.stack(existing_projections)
            distances = torch.norm(existing_tensor - proj.unsqueeze(0), dim=1)
            min_dist = distances.min().item()
            avg_dist = distances.mean().item()
            angular_dists = torch.acos(torch.clamp(torch.matmul(existing_tensor, proj), -0.9999, 0.9999))
            min_angular = angular_dists.min().item()
            coverage_score = min_dist * (1 + 0.3 * avg_dist) * (1 + 0.2 * min_angular)
        else:
            coverage_score = 1.0

        if len(existing_projections) > 0 and cached_current_sw is not None:
            new_projs = torch.cat([existing_tensor, proj.unsqueeze(0)])
            new_sw = one_d_wasserstein_mean(pc1, pc2, new_projs)
            contribution = abs(new_sw.item() - cached_current_sw.item())
        else:
            contribution = one_d_wasserstein_mean(pc1, pc2, proj.unsqueeze(0)).item()

        raw_obj = coverage_score * (1 + 0.5 * contribution)
        return torch.sqrt(torch.tensor(raw_obj + 1e-6))  # Affine transform

    def generate_smart_candidates(existing_projs, n_candidates, device):
        candidates = []
        n_random = n_candidates // 4
        candidates.append(rand_projections(3, n_random, device))
        n_quasi = n_candidates // 4
        if n_quasi > 0:
            candidates.append(get_sqsw_projections(n_quasi, device))
        if len(existing_projs) > 0:
            n_repulsive = n_candidates // 4
            repulsive = []
            existing_tensor = torch.stack(existing_projs)
            for _ in range(n_repulsive):
                point = rand_projections(3, 1, device).squeeze()
                for _ in range(5):
                    diffs = existing_tensor - point.unsqueeze(0)
                    dists = torch.norm(diffs, dim=1, keepdim=True).clamp(min=0.1)
                    forces = (diffs / (dists ** 3)).sum(dim=0)
                    point = point + 0.1 * forces
                    point = point / torch.norm(point)
                repulsive.append(point)
            if repulsive:
                candidates.append(torch.stack(repulsive))
        n_remaining = n_candidates - sum(len(c) for c in candidates)
        if n_remaining > 0:
            candidates.append(get_sqsw_projections(n_remaining, device))
        return torch.cat(candidates)

    selected_projections = []
    gp = LightweightGP(kernel_lengthscale=0.3)

    if L <= 5:
        return get_sqsw_projections(L, device)

    n_init = min(15, L // 3)
    init_projections = get_sqsw_projections(n_init, device)
    X_train, y_train = [], []

    for i in range(n_init):
        proj = init_projections[i]
        selected_projections.append(proj)
        obj = compute_enhanced_objective(proj, selected_projections[:-1], pc1, pc2)
        X_train.append(proj)
        y_train.append(obj)

    X_train = torch.stack(X_train)
    y_train = torch.stack(y_train)

    y_train_sq = y_train ** 2

    while len(selected_projections) < L:
        if len(X_train) > 50 and len(X_train) % 50 == 0:
            unique_indices = []
            for i in range(len(X_train)):
                is_unique = True
                for j in unique_indices:
                    if torch.norm(X_train[i] - X_train[j]) < 1e-4:
                        is_unique = False
                        break
                if is_unique:
                    unique_indices.append(i)
            if len(unique_indices) < len(X_train):
                X_train = X_train[unique_indices]
                y_train = y_train[unique_indices]
                y_train_sq = y_train_sq[unique_indices]

        gp.fit(X_train, y_train)

        remaining = L - len(selected_projections)
        batch_size = min(max(1, remaining // 5), 10)
        n_candidates = min(1500, 150 * batch_size)
        candidates = generate_smart_candidates(selected_projections, n_candidates, device)

        if len(selected_projections) > 0 and len(selected_projections) < 0.9 * L:
            existing_tensor = torch.stack(selected_projections)
            dists = torch.cdist(candidates, existing_tensor, p=2)
            min_dists = dists.min(dim=1)[0]
            threshold = 0.15 if len(selected_projections) < L // 2 else 0.1
            keep_mask = min_dists > threshold
            candidates = candidates[keep_mask]
            if len(candidates) < 50:
                candidates = generate_smart_candidates(selected_projections, 200, device)

        mu_g, sigma_g = gp.predict(candidates)
        mu_f = mu_g ** 2
        sigma_f = 2 * mu_g * sigma_g
        acq_values = acquisition_function(mu_f, sigma_f, y_train_sq, kind=ai, beta=beta)

        current_sw = one_d_wasserstein_mean(pc1, pc2, torch.stack(selected_projections)) if selected_projections else None
        n_select = min(batch_size, L - len(selected_projections))
        selected_indices = []

        for _ in range(n_select):
            best_idx = torch.argmax(acq_values).item()
            selected_point = candidates[best_idx]
            selected_indices.append(best_idx)
            distances = torch.norm(candidates - selected_point.unsqueeze(0), dim=1)
            diversity_penalty = torch.exp(-15 * distances)
            acq_values *= (1 - 0.7 * diversity_penalty)

        for idx in selected_indices:
            proj = candidates[idx]
            selected_projections.append(proj)
            obj = compute_enhanced_objective(proj, selected_projections[:-1], pc1, pc2, cached_current_sw=current_sw)
            X_train = torch.cat([X_train, proj.unsqueeze(0)])
            y_train = torch.cat([y_train, obj.unsqueeze(0)])
            y_train_sq = torch.cat([y_train_sq, obj.unsqueeze(0) ** 2])

        if len(X_train) > 150:
            keep_indices = list(range(n_init)) + list(range(len(X_train) - 100, len(X_train)))
            X_train = X_train[keep_indices]
            y_train = y_train[keep_indices]
            y_train_sq = y_train_sq[keep_indices]

    if L <= 300:
        selected_projections = local_refinement_simple(torch.stack(selected_projections), device, n_iters=10)

    return torch.stack(selected_projections) if isinstance(selected_projections, list) else selected_projections

def local_refinement_simple(projections, device, n_iters=5):
    """Simple local refinement for better coverage"""
    projs = projections.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([projs], lr=0.005)
    
    for _ in range(n_iters):
        dists = torch.cdist(projs, projs, p=2) + torch.eye(len(projs), device=device) * 1e6
        repulsion_loss = -(dists.min(dim=1)[0].mean())
        norm_loss = ((torch.norm(projs, dim=1) - 1) ** 2).mean()
        
        loss = repulsion_loss + 10 * norm_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return projs.detach() / torch.norm(projs.detach(), dim=1, keepdim=True)

def acquisition_function(mu, sigma, y_train, kind, beta=2.0):
    if kind == 'ucb':
        return mu + beta * sigma
    elif kind == 'ei':
        best = y_train.max()
        z = (mu - best) / (sigma + 1e-8)
        normal = torch.distributions.Normal(0, 1)
        ei = (mu - best) * normal.cdf(z) + sigma * normal.log_prob(z).exp()
        return ei
    elif kind == 'logei':
        best = y_train.max()
        z = (mu - best) / (sigma + 1e-8)
        normal = torch.distributions.Normal(0, 1)
        ei = (mu - best) * normal.cdf(z) + sigma * normal.log_prob(z).exp()
        return torch.log(ei + 1e-8)
    elif kind == 'thompson':
        return torch.normal(mu, sigma + 1e-6)
    else:
        raise ValueError(f"Unknown acquisition kind: {kind}")