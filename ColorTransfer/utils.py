import numpy as np
import torch
from torch.autograd import Variable
import ot
import random
import tqdm
from bosw import *
from scipy.stats import norm
def compute_true_Wasserstein(X,Y,p=2):
    M = ot.dist(X.detach().numpy(), Y.detach().numpy())
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)
def compute_Wasserstein(M,device='cpu',e=0):
    if(e==0): 
        pi = ot.emd([],[],M.cpu().detach().numpy()).astype('float32')
    else:
        pi = ot.sinkhorn([], [], M.cpu().detach().numpy(),reg=e).astype('float32')
    pi = torch.from_numpy(pi).to(device)
    return torch.sum(pi*M)

def rand_projections(dim, num_projections=1000,device='cpu'):
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


def SW(X, Y, L=10, p=2, device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim, L,device)
    sw=one_dimensional_Wasserstein_prod(X,Y,theta,p=p).mean()
    return  torch.pow(sw,1./p)

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

def get_bosw_bayesian(L, device, pc1, pc2, p=2, beta=2.0, seed=None, ai='ubc'):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    @torch.no_grad()
    def one_d_wasserstein_mean(pc1, pc2, theta):
        return one_dimensional_Wasserstein_prod(pc1, pc2, theta, p=p).mean().sqrt()

    # --- Enhanced Lightweight GP with better numerical stability ---
    class LightweightGP:
        def __init__(self, kernel_lengthscale=0.5):
            self.lengthscale = kernel_lengthscale
            self.X_train, self.y_train = None, None
            self.y_mean, self.y_std = 0, 1
            self.base_noise = 1e-3  # Increased base noise
        
        def fit(self, X, y):
            self.X_train = X
            self.y_train = y
            self.y_mean = y.mean()
            self.y_std = y.std() + 1e-6
            
            # Adaptive lengthscale with bounds
            if len(X) > 5:
                angles = torch.acos(torch.clamp(X @ X.T, -1, 1))
                angles_nonzero = angles[angles > 0]
                if len(angles_nonzero) > 0:
                    median_dist = angles_nonzero.median().item()
                    # Bound the lengthscale to reasonable values
                    self.lengthscale = np.clip(median_dist * 0.5, 0.1, 2.0)
        
        def predict(self, X_test):
            if self.X_train is None:
                return torch.zeros(len(X_test)), torch.ones(len(X_test))
            
            try:
                K_train = self._kernel(self.X_train, self.X_train)
                K_test = self._kernel(X_test, self.X_train)
                K_test_test = self._kernel(X_test, X_test)
                
                # Adaptive noise based on condition number
                eigenvalues = torch.linalg.eigvalsh(K_train)
                min_eig = eigenvalues.min().item()
                
                if min_eig < 1e-6:
                    noise_level = max(self.base_noise, 1e-2)
                else:
                    noise_level = self.base_noise
                
                noise_I = noise_level * torch.eye(len(K_train), device=K_train.device)
                
                # Try Cholesky decomposition with increasing noise if needed
                max_attempts = 5
                for attempt in range(max_attempts):
                    try:
                        L = torch.linalg.cholesky(K_train + noise_I)
                        break
                    except:
                        noise_level *= 10
                        noise_I = noise_level * torch.eye(len(K_train), device=K_train.device)
                        if attempt == max_attempts - 1:
                            # Fallback: use SVD-based approach
                            return self._predict_svd(K_train, K_test, K_test_test)
                
                alpha = torch.cholesky_solve(((self.y_train - self.y_mean) / self.y_std).unsqueeze(1), L).squeeze()
                mu = K_test @ alpha
                v = torch.linalg.solve(L, K_test.T)
                var = torch.diag(K_test_test - v.T @ v)
                
                return mu * self.y_std + self.y_mean, torch.sqrt(var.clamp(min=1e-6)) * self.y_std
                
            except Exception as e:
                # Ultimate fallback: return prior
                return torch.zeros(len(X_test)) + self.y_mean, torch.ones(len(X_test)) * self.y_std
        
        def _predict_svd(self, K_train, K_test, K_test_test):
            """SVD-based prediction as fallback"""
            U, S, Vt = torch.linalg.svd(K_train)
            # Keep only significant eigenvalues
            threshold = 1e-6
            mask = S > threshold
            U = U[:, mask]
            S = S[mask]
            Vt = Vt[mask, :]
            
            # Compute predictions
            K_inv_y = U @ torch.diag(1/S) @ Vt @ ((self.y_train - self.y_mean) / self.y_std)
            mu = K_test @ K_inv_y
            
            # Approximate variance (simplified)
            var = torch.diag(K_test_test).clamp(min=1e-6)
            
            return mu * self.y_std + self.y_mean, torch.sqrt(var) * self.y_std
        
        def _kernel(self, X1, X2):
            # Add small jitter for numerical stability
            angles = torch.acos(torch.clamp(X1 @ X2.T, -0.9999, 0.9999))
            K = torch.exp(-0.5 * (angles / self.lengthscale) ** 2)
            return K

    @torch.no_grad()
    def compute_enhanced_objective(proj, existing_projections, pc1, pc2, cached_current_sw=None):
        # Multi-scale coverage score (inspired by coverage_focused)
        if len(existing_projections) > 0:
            existing_tensor = torch.stack(existing_projections)
            
            # Euclidean distances
            distances = torch.norm(existing_tensor - proj.unsqueeze(0), dim=1)
            min_dist = distances.min().item()
            avg_dist = distances.mean().item()
            
            # Angular distances for better sphere coverage
            angular_dists = torch.acos(torch.clamp(torch.matmul(existing_tensor, proj), -0.9999, 0.9999))
            min_angular = angular_dists.min().item()
            
            # Combined multi-scale coverage score
            coverage_score = min_dist * (1 + 0.3 * avg_dist) * (1 + 0.2 * min_angular)
        else:
            coverage_score = 1.0

        # Wasserstein contribution
        if len(existing_projections) > 0 and cached_current_sw is not None:
            new_projs = torch.cat([existing_tensor, proj.unsqueeze(0)])
            new_sw = one_d_wasserstein_mean(pc1, pc2, new_projs)
            contribution = abs(new_sw.item() - cached_current_sw.item())
        else:
            contribution = one_d_wasserstein_mean(pc1, pc2, proj.unsqueeze(0)).item()

        # Balanced objective with stronger coverage emphasis
        return coverage_score * (1 + 0.5 * contribution)

    def generate_smart_candidates(existing_projs, n_candidates, device):
        """Generate candidates with better coverage properties"""
        candidates = []
        
        # 1/4 pure random
        n_random = n_candidates // 4
        candidates.append(rand_projections(3, n_random, device))
        
        # 1/4 from quasi-random sequences
        n_quasi = n_candidates // 4
        if n_quasi > 0:
            quasi_points = get_sqsw_projections(n_quasi, device)
            candidates.append(quasi_points)
        
        # 1/4 repulsive points (simplified for stability)
        if len(existing_projs) > 0:
            n_repulsive = n_candidates // 4
            repulsive = []
            existing_tensor = torch.stack(existing_projs)
            
            for _ in range(n_repulsive):
                # Start with random point
                point = rand_projections(3, 1, device).squeeze()
                
                # Simple repulsion without gradient optimization
                for _ in range(5):
                    # Compute repulsive forces
                    diffs = existing_tensor - point.unsqueeze(0)
                    dists = torch.norm(diffs, dim=1, keepdim=True).clamp(min=0.1)
                    forces = (diffs / (dists ** 3)).sum(dim=0)
                    
                    # Update point
                    point = point + 0.1 * forces
                    point = point / torch.norm(point)  # Project to sphere
                
                repulsive.append(point)
            
            if repulsive:
                candidates.append(torch.stack(repulsive))
        
        # Remaining from structured sequences
        n_remaining = n_candidates - sum(len(c) for c in candidates)
        if n_remaining > 0:
            candidates.append(get_sqsw_projections(n_remaining, device))
        
        return torch.cat(candidates)

    # Remove duplicate projections
    def remove_duplicates(projections, threshold=1e-4):
        if len(projections) <= 1:
            return projections
        
        unique = [projections[0]]
        for proj in projections[1:]:
            dists = torch.norm(torch.stack(unique) - proj.unsqueeze(0), dim=1)
            if dists.min() > threshold:
                unique.append(proj)
        return unique

    selected_projections = []
    gp = LightweightGP(kernel_lengthscale=0.3)

    # Fast path for small L
    if L <= 5:
        return get_sqsw_projections(L, device)

    # --- Enhanced Initial Projections ---
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
    y_train = torch.tensor(y_train, device=device)

    # --- Enhanced BO Loop ---
    while len(selected_projections) < L:
        # Remove duplicates from training data periodically
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
        
        gp.fit(X_train, y_train)

        # Adaptive batch size
        remaining = L - len(selected_projections)
        batch_size = min(max(1, remaining // 5), 10)
        n_candidates = min(1500, 150 * batch_size)  # Slightly reduced

        # Generate smart candidates
        candidates = generate_smart_candidates(selected_projections, n_candidates, device)

        # Prune candidates that are too close
        if len(selected_projections) > 0 and len(selected_projections) < 0.9 * L:
            existing_tensor = torch.stack(selected_projections)
            dists = torch.cdist(candidates, existing_tensor, p=2)
            min_dists = dists.min(dim=1)[0]
            
            threshold = 0.15 if len(selected_projections) < L // 2 else 0.1
            keep_mask = min_dists > threshold
            candidates = candidates[keep_mask]
            
            if len(candidates) < 50:
                candidates = generate_smart_candidates(selected_projections, 200, device)

        # Compute acquisition values
        mu, sigma = gp.predict(candidates)
        acq_values = mu + beta * sigma

        # Batch selection with enhanced diversity
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

        # Update with selected points
        for idx in selected_indices:
            proj = candidates[idx]
            selected_projections.append(proj)
            obj = compute_enhanced_objective(proj, selected_projections[:-1], pc1, pc2, cached_current_sw=current_sw)
            X_train = torch.cat([X_train, proj.unsqueeze(0)])
            y_train = torch.cat([y_train, torch.tensor([obj], device=device)])

        # Memory management
        if len(X_train) > 150:
            keep_indices = list(range(n_init)) + list(range(len(X_train) - 100, len(X_train)))
            X_train = X_train[keep_indices]
            y_train = y_train[keep_indices]

    # Final refinement
    if L <= 300:
        selected_projections = local_refinement_simple(torch.stack(selected_projections), device, n_iters=10)

    if isinstance(selected_projections, list):
        return torch.stack(selected_projections)
    else:
        return selected_projections

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

# def transform_SW(src,target,src_label,origin,sw_type='sw',L=10,num_iter=1000):
#     np.random.seed(1)
#     random.seed(1)
#     torch.manual_seed(1)
#     device='cpu'
#     s = np.array(src).reshape(-1, 3)
#     s = torch.from_numpy(s).float()
#     s = torch.nn.parameter.Parameter(s)
#     t = np.array(target).reshape(-1, 3)
#     t = torch.from_numpy(t).float()
#     opt = torch.optim.SGD([s], lr=1)
#     if (sw_type == 'nqsw' or sw_type == 'rnqsw' or sw_type == 'rrnqsw'  ):
#         soboleng = torch.quasirandom.SobolEngine(dimension=3, scramble=False)
#         theta = soboleng.draw(L)
#         theta = torch.clamp(theta, min=1e-6, max=1 - 1e-6)
#         theta = torch.from_numpy(norm.ppf(theta) + 1e-6).float()
#         theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True)).to(device)
#     elif(sw_type=='qsw' or sw_type=='rqsw' or sw_type=='rrqsw'):
#         soboleng = torch.quasirandom.SobolEngine(dimension=2, scramble=False)
#         net = soboleng.draw(L)
#         alpha = net[:, [0]]
#         tau = net[:, [1]]
#         theta = torch.cat([2 * torch.sqrt(tau - tau ** 2) * torch.cos(2 * np.pi * alpha),
#                            2 * torch.sqrt(tau - tau ** 2) * torch.sin(2 * np.pi * alpha), 1 - 2 * tau], dim=1).to(
#             device)
#     elif(sw_type=='sqsw' or sw_type=='rsqsw'):
#         Z = (1 - (2 * torch.arange(1, L + 1) - 1) / L).view(-1, 1)
#         theta1 = torch.arccos(Z)
#         theta2 = torch.remainder(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
#         theta = torch.cat(
#             [torch.sin(theta1) * torch.cos(theta2), torch.sin(theta1) * torch.sin(theta2), torch.cos(theta1)],
#             dim=1)
#         theta = theta.to(device)
#     elif(sw_type=='odqsw' or sw_type=='rodqsw'):
#         Z = (1 - (2 * np.arange(1, L + 1) - 1) / L).reshape(-1, 1)
#         theta1 = np.arccos(Z)
#         theta2 = np.mod(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
#         thetas = np.concatenate([np.sin(theta1) * np.cos(theta2), np.sin(theta1) * np.sin(theta2), np.cos(theta1)],
#                                 axis=1)
#         theta0 = torch.from_numpy(thetas)
#         thetas = torch.randn(L, 3, requires_grad=True)
#         thetas.data = theta0
#         optimizer = torch.optim.SGD([thetas], lr=1)
#         for _ in range(100):
#             loss = - torch.cdist(thetas, thetas, p=1).mean()
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             thetas.data = thetas.data / torch.sqrt(torch.sum(thetas.data ** 2, dim=1, keepdim=True))
#         theta = thetas.to(device).float()
#     elif (sw_type == 'ocqsw' or sw_type=='rocqsw'):
#         Z = (1 - (2 * np.arange(1, L + 1) - 1) / L).reshape(-1, 1)
#         theta1 = np.arccos(Z)
#         theta2 = np.mod(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
#         thetas = np.concatenate([np.sin(theta1) * np.cos(theta2), np.sin(theta1) * np.sin(theta2), np.cos(theta1)],
#                                 axis=1)
#         theta0 = torch.from_numpy(thetas)
#         thetas = torch.randn(L, 3, requires_grad=True)
#         thetas.data = theta0
#         optimizer = torch.optim.SGD([thetas], lr=1)
#         for _ in range(100):
#             loss = (1 / (torch.cdist(thetas, thetas, p=1) + 1e-6)).mean()
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             thetas.data = thetas.data / torch.sqrt(torch.sum(thetas.data ** 2, dim=1, keepdim=True))
#         theta = thetas.to(device).float()
#     for _ in tqdm.tqdm(range(num_iter)):
#         opt.zero_grad()
#         if (sw_type == 'sw'):
#             g_loss = SW(s, t, L=L,p=2)
#         elif(sw_type=='nqsw' or sw_type=='qsw' or sw_type=='sqsw' or sw_type=='ocqsw' or sw_type=='odqsw'):
#             g_loss=one_dimensional_Wasserstein_prod(s,t,theta,p=2)
#         elif(sw_type=='rnqsw'):
#             soboleng = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
#             theta = soboleng.draw(L)
#             theta = torch.clamp(theta, min=1e-6, max=1 - 1e-6)
#             theta = torch.from_numpy(norm.ppf(theta) + 1e-6).float()
#             theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True)).to(device)
#             g_loss = one_dimensional_Wasserstein_prod(s, t, theta, p=2)
#         elif(sw_type=='rqsw'):
#             soboleng = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
#             net = soboleng.draw(L)
#             alpha = net[:, [0]]
#             tau = net[:, [1]]
#             theta = torch.cat([2 * torch.sqrt(tau - tau ** 2) * torch.cos(2 * np.pi * alpha),
#                                2 * torch.sqrt(tau - tau ** 2) * torch.sin(2 * np.pi * alpha), 1 - 2 * tau], dim=1).to(
#                 device)
#             g_loss = one_dimensional_Wasserstein_prod(s, t, theta, p=2)
#         elif (sw_type == 'rrnqsw' or sw_type == 'rrqsw' or sw_type == 'rsqsw' or sw_type == 'rocqsw' or sw_type == 'rodqsw'):
#             U = torch.qr(torch.randn(3, 3))[0]
#             thetaprime = torch.matmul(theta, U)
#             g_loss = one_dimensional_Wasserstein_prod(s, t, thetaprime, p=2)
#         g_loss =torch.sqrt(g_loss.mean())
#         g_loss = g_loss*s.shape[0]
#         opt.zero_grad()
#         g_loss.backward()
#         opt.step()
#         s.data = torch.clamp(s, min=0)
#     s = torch.clamp( s,min=0).cpu().detach().numpy()
#     img_ot_transf = s[src_label].reshape(origin.shape)
#     img_ot_transf = img_ot_transf / np.max(img_ot_transf) * 255
#     img_ot_transf = img_ot_transf.astype("uint8")
#     return s, img_ot_transf

def transform_SW(src, target, src_label, origin, sw_type='sw', L=100, num_iter=500):
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    device = 'cpu'

    s = np.array(src).reshape(-1, 3)
    s = torch.from_numpy(s).float()
    s = torch.nn.parameter.Parameter(s)
    t = np.array(target).reshape(-1, 3)
    t = torch.from_numpy(t).float()

    pc1, pc2 = s.detach(), t.detach()
    opt = torch.optim.SGD([s], lr=1)

    theta = None

    if sw_type == 'bosw':
        s_init = torch.from_numpy(src).float().reshape(-1, 3)
        t_init = torch.from_numpy(target).float().reshape(-1, 3)
        theta = get_bosw_bayesian(L, device, s_init, t_init, p=2, beta=2.0)

    elif sw_type in ['nqsw', 'rnqsw', 'rrnqsw']:
        soboleng = torch.quasirandom.SobolEngine(dimension=3, scramble=('r' in sw_type))
        theta = soboleng.draw(L)
        theta = torch.clamp(theta, min=1e-6, max=1 - 1e-6)
        theta = torch.from_numpy(norm.ppf(theta) + 1e-6).float()
        theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True)).to(device)
    elif sw_type in ['qsw', 'rqsw', 'rrqsw']:
        soboleng = torch.quasirandom.SobolEngine(dimension=2, scramble=('r' in sw_type))
        net = soboleng.draw(L)
        alpha, tau = net[:, [0]], net[:, [1]]
        theta = torch.cat([
            2 * torch.sqrt(tau - tau ** 2) * torch.cos(2 * np.pi * alpha),
            2 * torch.sqrt(tau - tau ** 2) * torch.sin(2 * np.pi * alpha),
            1 - 2 * tau
        ], dim=1).to(device)
    elif sw_type in ['sqsw', 'rsqsw']:
        Z = (1 - (2 * torch.arange(1, L + 1) - 1) / L).view(-1, 1)
        theta1 = torch.arccos(Z)
        theta2 = torch.remainder(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
        theta = torch.cat([
            torch.sin(theta1) * torch.cos(theta2),
            torch.sin(theta1) * torch.sin(theta2),
            torch.cos(theta1)
        ], dim=1).to(device)
    elif sw_type in ['odqsw', 'rodqsw', 'ocqsw', 'rocqsw']:
        Z = (1 - (2 * np.arange(1, L + 1) - 1) / L).reshape(-1, 1)
        theta1 = np.arccos(Z)
        theta2 = np.mod(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
        thetas = np.concatenate([
            np.sin(theta1) * np.cos(theta2),
            np.sin(theta1) * np.sin(theta2),
            np.cos(theta1)
        ], axis=1)
        theta0 = torch.from_numpy(thetas)
        thetas = torch.randn(L, 3, requires_grad=True)
        thetas.data = theta0
        optimizer = torch.optim.SGD([thetas], lr=1)
        for _ in range(100):
            loss = -torch.cdist(thetas, thetas, p=1).mean() if 'odqsw' in sw_type else \
                   (1 / (torch.cdist(thetas, thetas, p=1) + 1e-6)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            thetas.data = thetas.data / torch.sqrt(torch.sum(thetas.data ** 2, dim=1, keepdim=True))
        theta = thetas.to(device).float()
    
        # Color-aware RBOSW 
    elif sw_type == 'rbosw' or sw_type == 'color_rbosw':
        s_init = torch.from_numpy(src).float().reshape(-1, 3)
        t_init = torch.from_numpy(target).float().reshape(-1, 3)
        # Start with BOSW projections
        theta = get_bosw_bayesian(L, device, s_init, t_init, p=2, beta=2.0)
        
        # Also keep some color-specific projections
        # RGB axes are important for color transfer
        color_axes = torch.tensor([
            [1., 0., 0.],  # Red channel
            [0., 1., 0.],  # Green channel  
            [0., 0., 1.],  # Blue channel
            [1., 1., 0.],  # Yellow (R+G)
            [1., 0., 1.],  # Magenta (R+B)
            [0., 1., 1.],  # Cyan (G+B)
            [1., 1., 1.]   # Gray (R+G+B)
        ], device=device)
        color_axes = color_axes / torch.norm(color_axes, dim=1, keepdim=True)
        
        # Replace some BOSW projections with color axes
        n_color = min(7, L // 3)
        if n_color > 0:
            theta[:n_color] = color_axes[:n_color]
    
    # For tracking color evolution in RBOSW
    update_frequency = max(50, num_iter // 20)  # Update projections 20 times
    projection_update_counter = 0

    for _ in tqdm.tqdm(range(num_iter)):
        opt.zero_grad()
        if sw_type == 'sw':
            g_loss = SW(s, t, L=L, p=2)
        elif sw_type in ['nqsw', 'qsw', 'sqsw', 'ocqsw', 'odqsw', 'bosw']:
            g_loss = one_dimensional_Wasserstein_prod(s, t, theta, p=2)
        elif sw_type in ['rnqsw']:
            soboleng = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
            theta = soboleng.draw(L)
            theta = torch.clamp(theta, min=1e-6, max=1 - 1e-6)
            theta = torch.from_numpy(norm.ppf(theta) + 1e-6).float()
            theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True)).to(device)
            g_loss = one_dimensional_Wasserstein_prod(s, t, theta, p=2)

        elif sw_type == 'rqsw':
            soboleng = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
            net = soboleng.draw(L)
            alpha, tau = net[:, [0]], net[:, [1]]
            theta = torch.cat([
                2 * torch.sqrt(tau - tau ** 2) * torch.cos(2 * np.pi * alpha),
                2 * torch.sqrt(tau - tau ** 2) * torch.sin(2 * np.pi * alpha),
                1 - 2 * tau
            ], dim=1).to(device)
            g_loss = one_dimensional_Wasserstein_prod(s, t, theta, p=2)

        elif sw_type == 'rbosw':
            # Periodically update projections based on current colors
            if _ > 0 and _ % update_frequency == 0:
                # Get current color state
                s_current = s.detach()
                
                # Generate new projections considering current color distribution
                new_theta = get_color_aware_projections(L, device, s_current, t)
                
                # Blend old and new projections (momentum)
                blend_factor = 0.5 + 0.3 * (_ / num_iter)  # Increase new weight over time
                theta = (1 - blend_factor) * theta + blend_factor * new_theta
                theta = theta / torch.norm(theta, dim=1, keepdim=True)
            
            # Always apply random rotation for diversity
            U = torch.linalg.qr(torch.randn(3, 3, device=device))[0]
                        
            thetaprime = torch.matmul(theta, U)
            g_loss = one_dimensional_Wasserstein_prod(s, t, thetaprime, p=2)

        elif sw_type in ['rrnqsw', 'rrqsw', 'rsqsw', 'rocqsw', 'rodqsw']:
            U = torch.linalg.qr(torch.randn(3, 3))[0]
            thetaprime = torch.matmul(theta, U)
            g_loss = one_dimensional_Wasserstein_prod(s, t, thetaprime, p=2)

        g_loss = torch.sqrt(g_loss.mean()) * s.shape[0]
        g_loss.backward()
        opt.step()
        s.data = torch.clamp(s.data, min=0)

    s = torch.clamp(s, min=0).cpu().detach().numpy()
    img_ot_transf = s[src_label].reshape(origin.shape)
    img_ot_transf = (img_ot_transf / np.max(img_ot_transf) * 255).astype("uint8")
    return s, img_ot_transf

def get_color_aware_projections(L, device, current_colors, target_colors):
    """
    Generate projections that are aware of current color distribution
    """
    # Normalize colors
    s_norm = current_colors / 255.0
    t_norm = target_colors / 255.0
    
    projections = []
    
    # 1. Color difference direction
    mean_diff = t_norm.mean(0) - s_norm.mean(0)
    if torch.norm(mean_diff) > 1e-3:
        projections.append(mean_diff / torch.norm(mean_diff))
    
    # 2. Principal components of color difference
    color_diff = t_norm - s_norm.mean(0)
    U, S, V = torch.svd(color_diff.T @ color_diff)
    for i in range(min(3, L//4)):
        projections.append(V[:, i])
    
    # 3. Directions that maximize color separation
    n_sep = min(L//4, 10)
    for _ in range(n_sep):
        # Random linear combination of colors
        weights_s = torch.randn(s_norm.shape[0], device=device)
        weights_t = torch.randn(t_norm.shape[0], device=device)
        weights_s = weights_s / weights_s.sum()
        weights_t = weights_t / weights_t.sum()
        
        direction = weights_t @ t_norm - weights_s @ s_norm
        if torch.norm(direction) > 1e-3:
            projections.append(direction / torch.norm(direction))
    
    # 4. Fill rest with BOSW
    n_needed = L - len(projections)
    if n_needed > 0:
        bosw_projs = get_bosw_bayesian(n_needed, device, current_colors, target_colors, p=2, beta=2.0)
        projections.extend([bosw_projs[i] for i in range(n_needed)])
    
    return torch.stack(projections[:L])