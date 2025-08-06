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
    import math, random
    import numpy as np
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dtype = pc1.dtype

    @torch.no_grad()
    def one_d_wasserstein_mean(pc1, pc2, theta):
        return one_dimensional_Wasserstein_prod(pc1, pc2, theta, p=p).mean().sqrt()

    # ---------------- GP with lengthscale Bayes-averaging ----------------
    class LightweightGP:
        def __init__(self, kernel_lengthscale=0.5, base_noise=1e-3, top_k_ls=3, tau=0.25):
            self.lengthscale = float(kernel_lengthscale)
            self.base_noise  = float(base_noise)
            self.top_k_ls = int(top_k_ls)
            self.tau = float(tau)  # temperature for LS weights
            self.X_train = None; self.y_train = None
            self.y_mean = torch.tensor(0.0, device=device, dtype=dtype)
            self.y_std  = torch.tensor(1.0, device=device, dtype=dtype)
            self._ls_list = None
            self._nll_list = None

        def _kernel_ls(self, X1, X2, ls: float):
            dot = torch.clamp(X1 @ X2.T, -0.9999, 0.9999)
            ang = torch.acos(dot)
            return torch.exp(-0.5 * (ang / ls) ** 2)

        def _nll_from_K(self, K, y_std):
            n = y_std.numel()
            I = torch.eye(n, device=K.device, dtype=K.dtype)
            jitter = self.base_noise
            for _ in range(5):
                try:
                    Lc = torch.linalg.cholesky(K + jitter * I)
                    alpha = torch.cholesky_solve(y_std.view(-1,1), Lc).view(-1)
                    return 0.5 * (y_std @ alpha) + torch.log(torch.diag(Lc)).sum() + 0.5 * n * math.log(2*math.pi)
                except RuntimeError:
                    jitter *= 10.0
            # SVD fallback
            U,S,Vt = torch.linalg.svd(K)
            S = torch.clamp(S, min=1e-8)
            Kinv_y = U @ torch.diag(1/S) @ Vt @ y_std
            return 0.5 * (y_std @ Kinv_y) + 0.5 * torch.log(S).sum() + 0.5 * n * math.log(2*math.pi)

        def fit(self, X, y):
            self.X_train = X; self.y_train = y
            self.y_mean = y.mean(); self.y_std = y.std() + 1e-6
            ys = (y - self.y_mean) / self.y_std

            ls_grid = torch.tensor([0.10,0.15,0.20,0.30,0.45,0.60,0.80,1.00,1.30,1.60,2.00],
                                   device=X.device, dtype=X.dtype)
            if X.size(0) > 5:
                ang_all = torch.acos(torch.clamp(X @ X.T, -1, 1))
                valid = ang_all[ang_all > 1e-6]
                if valid.numel() > 0:
                    md = valid.median()
                    extra = torch.clamp(md * torch.tensor([0.5,0.75,1.0,1.5,2.0], device=X.device, dtype=X.dtype),
                                        0.10, 2.00)
                    ls_grid = torch.unique(torch.cat([ls_grid, extra]))
                    ls_grid, _ = torch.sort(ls_grid)

            nlls = []
            for ls in ls_grid:
                K = self._kernel_ls(X, X, float(ls))
                nlls.append(self._nll_from_K(K, ys).item())
            nlls = torch.tensor(nlls, device=X.device, dtype=torch.float64)
            order = torch.argsort(nlls)
            top_idx = order[:min(self.top_k_ls, ls_grid.numel())]
            self._ls_list = [float(ls_grid[i].item()) for i in top_idx]
            self._nll_list = [float(nlls[i].item()) for i in top_idx]
            self.lengthscale = self._ls_list[0]

        def _predict_single_ls(self, Xtest, ls):
            Xtr = self.X_train; ys = (self.y_train - self.y_mean) / self.y_std
            Ktt = self._kernel_ls(Xtr, Xtr, ls)
            Kst = self._kernel_ls(Xtest, Xtr, ls)
            Kss = self._kernel_ls(Xtest, Xtest, ls)
            n = Ktt.size(0)
            I = torch.eye(n, device=Ktt.device, dtype=Ktt.dtype)
            jitter = self.base_noise
            for _ in range(5):
                try:
                    Lc = torch.linalg.cholesky(Ktt + jitter * I)
                    v = torch.linalg.solve(Lc, Kst.T)
                    alpha = torch.cholesky_solve(ys.view(-1,1), Lc).view(-1)
                    mu_std = Kst @ alpha
                    var = torch.diag(Kss - v.T @ v).clamp_min(1e-8)
                    return mu_std, var
                except RuntimeError:
                    jitter *= 10.0
            # SVD fallback
            U,S,Vt = torch.linalg.svd(Ktt)
            S = torch.clamp(S, min=1e-8)
            Kinv = U @ torch.diag(1/S) @ Vt
            mu_std = Kst @ (Kinv @ ys)
            var = torch.diag(Kss - Kst @ (Kinv @ Kst.T)).clamp_min(1e-8)
            return mu_std, var

        def predict(self, Xtest):
            if self.X_train is None or self.X_train.numel() == 0:
                n = Xtest.size(0)
                return (torch.zeros(n, device=Xtest.device, dtype=Xtest.dtype) + self.y_mean,
                        torch.ones(n, device=Xtest.device, dtype=Xtest.dtype)  * self.y_std)
            # Bayes-average over top LS
            nll = torch.tensor(self._nll_list, device=Xtest.device, dtype=torch.float64)
            w = torch.softmax(-(nll - nll.min())/max(self.tau, 1e-6), dim=0).to(Xtest.dtype)
            mus = []; vars_ = []
            for ls in self._ls_list:
                mu_std, var_std = self._predict_single_ls(Xtest, ls)
                mu = mu_std * self.y_std + self.y_mean
                std = torch.sqrt(var_std.clamp_min(1e-12)) * self.y_std
                mus.append(mu); vars_.append(std**2)
            mu_mix = torch.zeros_like(mus[0])
            m2_mix = torch.zeros_like(vars_[0])
            for wi, mi, vi in zip(w, mus, vars_):
                mu_mix = mu_mix + wi * mi
                m2_mix = m2_mix + wi * (vi + mi**2)
            var_mix = (m2_mix - mu_mix**2).clamp_min(1e-12)
            return mu_mix, torch.sqrt(var_mix)

    # ---- sphere utilities ----
    def _normalize_rows(V: torch.Tensor):
        return V / (V.norm(dim=1, keepdim=True) + 1e-12)

    def _canonicalize_rows(V: torch.Tensor):
        s = torch.sign(torch.where(V[:,0].abs() > 1e-12, V[:,0],
                            torch.where(V[:,1].abs() > 1e-12, V[:,1], V[:,2])))
        s = torch.where(s == 0, torch.ones_like(s), s)
        return V * s.view(-1,1)

    def spherical_fibonacci(n, device, dtype):
        if n <= 0: return torch.empty(0,3, device=device, dtype=dtype)
        k = torch.arange(n, device=device, dtype=torch.float64) + 0.5
        z = 1.0 - 2.0 * k / float(n)
        phi = (2.0 * math.pi) * (k / ((1.0 + math.sqrt(5.0)) / 2.0))
        r = torch.sqrt((1.0 - z*z).clamp_min(0.0))
        x = (r * torch.cos(phi)).to(dtype)
        y = (r * torch.sin(phi)).to(dtype)
        z = z.to(dtype)
        V = torch.stack([x,y,z], dim=1)
        return _canonicalize_rows(_normalize_rows(V))

    def _tangent_basis(u: torch.Tensor):
        ref = torch.tensor((1.0, 0.0, 0.0), device=u.device, dtype=u.dtype)
        if torch.abs(u[0]) >= 0.9:
            ref = torch.tensor((0.0, 1.0, 0.0), device=u.device, dtype=u.dtype)
        b1 = torch.cross(u, ref, dim=0); n1 = b1.norm()
        if n1 < 1e-12:
            ref = torch.tensor((0.0, 0.0, 1.0), device=u.device, dtype=u.dtype)
            b1 = torch.cross(u, ref, dim=0); n1 = b1.norm()
        b1 = b1 / (n1 + 1e-12)
        b2 = torch.cross(u, b1, dim=0); b2 = b2 / (b2.norm() + 1e-12)
        return b1, b2

    def _exp_map(u: torch.Tensor, v_unit: torch.Tensor, alpha):
        a = torch.as_tensor(alpha, device=u.device, dtype=torch.float64)
        u64 = u.to(torch.float64); v64 = v_unit.to(torch.float64)
        x64 = u64 * torch.cos(a) + v64 * torch.sin(a)
        x64 = x64 / (x64.norm() + 1e-18)
        return x64.to(u.dtype)

    # ---- objectives & helpers ----
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
        return torch.sqrt(torch.tensor(raw_obj + 1e-6, device=proj.device, dtype=proj.dtype))

    @torch.no_grad()
    def current_bosw(thetas):
        if thetas is None or len(thetas) == 0: return None
        return one_d_wasserstein_mean(pc1, pc2, thetas)

    @torch.no_grad()
    def delta_improvement(proj, existing_thetas, bosw_cur):
        if bosw_cur is None:
            return one_d_wasserstein_mean(pc1, pc2, proj.unsqueeze(0))
        new_sw = one_d_wasserstein_mean(pc1, pc2, torch.cat([existing_thetas, proj.unsqueeze(0)], dim=0))
        return torch.clamp(bosw_cur - new_sw, min=0.0)

    def generate_smart_candidates(existing_projs, n_candidates, device, progress, X_train=None, y_train=None):
        chunks = []
        # random & SQSW & Sobol-on-sphere
        chunks.append(rand_projections(3, n_candidates // 4, device).to(dtype))
        k_quasi = n_candidates // 4
        if k_quasi > 0:
            chunks.append(get_sqsw_projections(k_quasi, device).to(dtype))
        k_sob = n_candidates // 4
        if k_sob > 0:
            sob = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
            u = sob.draw(k_sob).to(device)
            az = 2 * math.pi * u[:, 0]
            z  = 2 * u[:, 1] - 1
            r  = torch.sqrt((1 - z*z).clamp_min(0))
            sobol_sphere = torch.stack([r*torch.cos(az), r*torch.sin(az), z], dim=1).to(dtype)
            sobol_sphere = _normalize_rows(sobol_sphere)
            chunks.append(_canonicalize_rows(sobol_sphere))
        # spherical Fibonacci (uniform)
        k_fib = n_candidates - sum(c.size(0) for c in chunks)
        if k_fib > 0:
            chunks.append(spherical_fibonacci(k_fib, device, dtype))

        # mild repulsive around existing
        if len(existing_projs) > 0:
            n_rep = max(0, n_candidates // 8)
            rep = []
            E = torch.stack(existing_projs).to(dtype)
            for _ in range(n_rep):
                point = rand_projections(3, 1, device).squeeze(0).to(dtype)
                for _ in range(3):
                    diffs = E - point
                    d = diffs.norm(dim=1, keepdim=True).clamp(min=0.1)
                    point = (point + 0.06 * (diffs / (d**3)).sum(dim=0))
                    point = point / point.norm()
                rep.append(point)
            if rep: chunks.append(torch.stack(rep))

        V = torch.cat(chunks, dim=0)
        V = _canonicalize_rows(_normalize_rows(V))
        return V

    @torch.no_grad()
    def tangent_proxy_improve(point, existing_list, pc1, pc2, cur_sw, progress, tries=16):
        best = point
        best_val = compute_enhanced_objective(point, existing_list, pc1, pc2, cached_current_sw=cur_sw)
        r0 = 0.06 * (1.0 - progress) + 0.012
        b1, b2 = _tangent_basis(point)
        dirs = (b1, b2,
                (b1+b2)/((b1+b2).norm()+1e-12),
                (b1-b2)/((b1-b2).norm()+1e-12))
        for a in (r0, 0.5*r0, -0.5*r0, -r0):
            for vdir in dirs:
                cand = _exp_map(point, vdir, a)
                val = compute_enhanced_objective(cand, existing_list, pc1, pc2, cached_current_sw=cur_sw)
                if val > best_val:
                    best_val, best = val, cand
        for _ in range(tries):
            a = torch.randn(2, device=point.device, dtype=point.dtype); a = a/(a.norm()+1e-12)
            v = a[0]*b1 + a[1]*b2
            cand = _exp_map(point, v, r0)
            val = compute_enhanced_objective(cand, existing_list, pc1, pc2, cached_current_sw=cur_sw)
            if val > best_val:
                best_val, best = val, cand
        return best

    # ---------------- main ----------------
    selected = []
    gp = LightweightGP(kernel_lengthscale=0.3, top_k_ls=3, tau=0.25)

    if L <= 5:
        return _canonicalize_rows(get_sqsw_projections(L, device).to(dtype))

    n_init = min(15, L // 3)
    init = _canonicalize_rows(get_sqsw_projections(n_init, device).to(dtype))
    X_train, y_train = [], []
    for i in range(n_init):
        proj = init[i]
        selected.append(proj)
        obj = compute_enhanced_objective(proj, selected[:-1], pc1, pc2)
        X_train.append(proj); y_train.append(obj)
    X_train = torch.stack(X_train).to(device=device, dtype=dtype)
    y_train = torch.stack(y_train).to(device=device, dtype=dtype)
    y_train_sq = y_train ** 2

    while len(selected) < L:
        if len(X_train) > 50 and len(X_train) % 50 == 0:
            keep = []
            for i in range(len(X_train)):
                if not any(torch.norm(X_train[i] - X_train[j]) < 1e-6 for j in keep):
                    keep.append(i)
            if len(keep) < len(X_train):
                X_train = X_train[keep]; y_train = y_train[keep]; y_train_sq = y_train_sq[keep]

        gp.fit(X_train, y_train)

        remaining = L - len(selected)
        batch_size = min(max(1, remaining // 5), 10)
        progress = len(selected) / float(L)

        n_candidates = 3500 if progress <= 0.4 else (6500 if progress <= 0.7 else 9000)
        candidates = generate_smart_candidates(selected, n_candidates, device, progress, X_train, y_train)

        if len(selected) > 0 and len(selected) < int(0.95 * L):
            E = torch.stack(selected).to(dtype)
            min_d = torch.cdist(candidates, E, p=2).min(dim=1)[0]
            thr = 0.18 * (1.0 - progress) + 0.06 * progress
            keep = min_d > thr
            if keep.any(): candidates = candidates[keep]
            if candidates.size(0) < max(150, 12 * batch_size):
                candidates = torch.cat([candidates, _canonicalize_rows(get_sqsw_projections(300, device).to(dtype))], dim=0)

        mu_g, sigma_g = gp.predict(candidates)
        mu_f = mu_g ** 2
        sigma_f = (2 * mu_g * sigma_g).clamp_min(1e-12)
        kind_now = 'ei' if (ai.lower() == 'ei' or progress >= 0.6) else 'ucb'
        beta_now = max(0.25, beta * (1.0 - progress)**2) if kind_now == 'ucb' else beta
        acq = acquisition_function(mu_f, sigma_f, y_train_sq, kind=kind_now, beta=beta_now)

        # ---- pre-thin top-K for diversity BEFORE proxy scoring (SAFE VERSION)
        n_select = min(batch_size, L - len(selected))
        K = min(900, max(320, int(240 + 1500 * (progress**1.3))))
        K = min(K, acq.numel())  # never exceed available
        top_idx = torch.topk(acq, K).indices
        top_cands = candidates[top_idx]

        # greedy pre-thinning by angular distance
        pre_thr = 0.22 * (1.0 - progress) + 0.14 * progress  # ~0.22→0.14
        K_local = top_cands.size(0)
        if K_local == 0:
            # emergency fallback: at least take the global best acq
            arg = torch.argmax(acq).item()
            top_cands = candidates[arg:arg+1]
            top_idx = torch.tensor([arg], device=device)
            K_local = 1

        keep_mask = torch.zeros(K_local, dtype=torch.bool, device=device)
        kept = []
        for i in range(K_local):
            if not keep_mask[i]:
                kept.append(i)
                ang = torch.acos(torch.clamp((top_cands @ top_cands[i]), -1.0, 1.0))
                collide = (ang < pre_thr)
                keep_mask = keep_mask | collide
                keep_mask[i] = True
        kept_idx_local = torch.tensor(kept, device=device, dtype=torch.long)
        top_cands = top_cands[kept_idx_local]
        top_idx = top_idx[kept_idx_local]

        # proxy scoring
        with torch.no_grad():
            cur_sw = one_d_wasserstein_mean(pc1, pc2, torch.stack(selected)) if selected else None
            proxy_scores = []
            for j in range(top_cands.size(0)):
                s = compute_enhanced_objective(top_cands[j], selected, pc1, pc2, cached_current_sw=cur_sw)
                proxy_scores.append(s)
            proxy_scores = torch.stack(proxy_scores) if len(proxy_scores) > 0 else torch.zeros(0, device=device, dtype=dtype)

        # ---- SECOND shortlist (K2) – make sure K2 ≤ available and ≥ 1
        avail = top_cands.size(0)
        if avail == 0:
            # extreme fallback: take best global acq
            arg = torch.argmax(acq).item()
            top_cands2 = candidates[arg:arg+1]
            top_idx2 = torch.tensor([arg], device=device)
        else:
            K2_raw = max(8, avail // 10)         # scale with avail
            K2 = min(48, max(1, K2_raw))         # clamp to [1, 48]
            K2 = min(K2, avail)                   # never exceed available
            topk2_idx_local = torch.topk(proxy_scores, K2).indices
            top_cands2 = top_cands[topk2_idx_local]
            top_idx2 = top_idx[topk2_idx_local]

        # "true" delta scores for blended ranking
        true_scores = []
        with torch.no_grad():
            existing_tensor = torch.stack(selected) if selected else None
            bosw_cur = current_bosw(existing_tensor) if existing_tensor is not None else None
            for j in range(top_cands2.size(0)):
                td = delta_improvement(top_cands2[j], existing_tensor, bosw_cur)
                true_scores.append(td)
        true_scores = torch.stack(true_scores) if len(true_scores) > 0 else torch.zeros(0, device=device, dtype=dtype)

        def _z(x):
            m = x.mean(); s = x.std() + 1e-12
            return (x - m) / s

        # align indexes when avail==0 fallback happened
        if avail == 0:
            proxy_sel = torch.tensor([0], device=device)
        else:
            proxy_sel = topk2_idx_local

        gamma = float(max(0.0, min(0.4, (progress - 0.70) / 0.30)))  # ≤ 0.4 cap
        if true_scores.numel() > 0:
            blended_local = (1.0 - gamma) * _z(proxy_scores[proxy_sel]) + gamma * _z(true_scores)
        else:
            blended_local = _z(proxy_scores[proxy_sel]) if proxy_scores.numel() > 0 else torch.zeros(1, device=device, dtype=dtype)

        # selection with mild diversity
        ls = getattr(gp, 'lengthscale', 0.3)
        width = max(0.30 * ls, 0.09)
        strength = 0.60
        nms_thr = 0.55 * width

        # map blended_local back into the "top_cands" index space for greedy pick
        work_scores = torch.full((top_cands.size(0),), -1e9, device=device, dtype=dtype)
        if avail == 0:
            work_scores[0] = blended_local[0]
        else:
            work_scores[topk2_idx_local] = blended_local

        chosen_global_idx = []
        for _ in range(n_select):
            j = torch.argmax(work_scores).item()
            chosen_global_idx.append(int(top_idx[j].item()))
            if top_cands.size(0) > 1:
                ang = torch.acos(torch.clamp((top_cands @ top_cands[j]), -1.0, 1.0))
                penal = strength * torch.exp(-(ang / width) ** 2)
                work_scores = work_scores * (1.0 - penal).clamp_min(0.0)
                suppress = (ang < nms_thr)
                work_scores[suppress] = -1e9
            work_scores[j] = -1e9

        cur_sw = one_d_wasserstein_mean(pc1, pc2, torch.stack(selected)) if selected else None
        for idx in chosen_global_idx:
            proj0 = candidates[idx]
            proj0 = _normalize_rows(proj0.view(1, -1)).view(-1)
            proj0 = _canonicalize_rows(proj0.view(1, -1)).view(-1)
            proj = tangent_proxy_improve(proj0, selected, pc1, pc2, cur_sw, progress, tries=16)
            proj = _canonicalize_rows(proj.view(1, -1)).view(-1)
            selected.append(proj)
            obj = compute_enhanced_objective(proj, selected[:-1], pc1, pc2, cached_current_sw=cur_sw)
            X_train = torch.cat([X_train, proj.unsqueeze(0)], dim=0)
            y_train = torch.cat([y_train, obj.unsqueeze(0)], dim=0)
            y_train_sq = torch.cat([y_train_sq, obj.unsqueeze(0)**2], dim=0)

        if len(X_train) > 220:
            keep_idx = list(range(n_init)) + list(range(len(X_train) - 150, len(X_train)))
            X_train = X_train[keep_idx]; y_train = y_train[keep_idx]; y_train_sq = y_train_sq[keep_idx]

    if L <= 300:
        selected = local_refinement_simple(torch.stack(selected), device, n_iters=14)

    return torch.stack(selected) if isinstance(selected, list) else selected




































































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