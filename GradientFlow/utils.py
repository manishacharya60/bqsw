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
    # --- determinism ---
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dtype = pc1.dtype

    # ----------------------- BOSW proxy -----------------------
    @torch.no_grad()
    def one_d_wasserstein_mean(pc1, pc2, theta):
        return one_dimensional_Wasserstein_prod(pc1, pc2, theta, p=p).mean().sqrt()

    # ---------------- GP with geodesic RBF kernel ----------------
    class LightweightGP:
        def __init__(self, kernel_lengthscale=0.5, base_noise=1e-3):
            self.lengthscale = float(kernel_lengthscale)
            self.base_noise  = float(base_noise)
            self.X_train = None
            self.y_train = None
            self.y_mean  = torch.tensor(0.0, device=device, dtype=dtype)
            self.y_std   = torch.tensor(1.0, device=device, dtype=dtype)

        def _kernel_ls(self, X1, X2, ls: float):
            dot = torch.clamp(X1 @ X2.T, -0.9999, 0.9999)
            ang = torch.acos(dot)  # [0, pi]
            return torch.exp(-0.5 * (ang / ls) ** 2)

        def _mll(self, K, y_std):
            n = y_std.numel()
            I = torch.eye(n, device=K.device, dtype=K.dtype)
            jitter = self.base_noise
            for _ in range(5):
                try:
                    Lc = torch.linalg.cholesky(K + jitter * I)
                    alpha = torch.cholesky_solve(y_std.view(-1,1), Lc).view(-1)
                    nll = 0.5 * (y_std @ alpha) + torch.log(torch.diag(Lc)).sum() + 0.5 * n * np.log(2*np.pi)
                    return nll
                except RuntimeError:
                    jitter *= 10.0
            # SVD fallback
            U,S,Vt = torch.linalg.svd(K)
            S = torch.clamp(S, min=1e-8)
            Kinv_y = U @ torch.diag(1/S) @ Vt @ y_std
            nll = 0.5 * (y_std @ Kinv_y) + 0.5 * torch.log(S).sum() + 0.5 * n * np.log(2*np.pi)
            return nll

        def fit(self, X, y):
            self.X_train = X
            self.y_train = y
            self.y_mean  = y.mean()
            self.y_std   = y.std() + 1e-6
            ys = (y - self.y_mean) / self.y_std

            # lengthscale grid (kept from the good run)
            ls_grid = torch.tensor([0.10,0.15,0.20,0.30,0.45,0.60,0.80,1.00,1.30,1.60,2.00],
                                   device=X.device, dtype=X.dtype)
            if X.size(0) > 5:
                ang_all = torch.acos(torch.clamp(X @ X.T, -1, 1))
                valid = ang_all[ang_all > 1e-6]
                md = valid.median() if valid.numel() > 0 else torch.tensor(0.5, device=X.device, dtype=X.dtype)
                ls_med = torch.clamp(md * torch.tensor([0.5,0.75,1.0,1.5,2.0], device=X.device, dtype=X.dtype),
                                     0.10, 2.00)
                ls_grid = torch.unique(torch.cat([ls_grid, ls_med]))
                ls_grid, _ = torch.sort(ls_grid)

            best_ls, best_nll = self.lengthscale, None
            for ls in ls_grid:
                K = self._kernel_ls(X, X, float(ls))
                nll = self._mll(K, ys)
                if (best_nll is None) or (nll < best_nll):
                    best_nll, best_ls = nll, float(ls)
            self.lengthscale = best_ls

        def predict(self, Xtest):
            if self.X_train is None or self.X_train.numel() == 0:
                n = Xtest.size(0)
                return (torch.zeros(n, device=Xtest.device, dtype=Xtest.dtype) + self.y_mean,
                        torch.ones(n, device=Xtest.device, dtype=Xtest.dtype)  * self.y_std)

            Ktt = self._kernel_ls(self.X_train, self.X_train, self.lengthscale)
            Kst = self._kernel_ls(Xtest, self.X_train, self.lengthscale)
            Kss = self._kernel_ls(Xtest, Xtest, self.lengthscale)

            n = Ktt.size(0)
            I = torch.eye(n, device=Ktt.device, dtype=Ktt.dtype)
            jitter = self.base_noise
            for _ in range(5):
                try:
                    Lc = torch.linalg.cholesky(Ktt + jitter * I)
                    break
                except RuntimeError:
                    jitter *= 10.0
            else:
                U,S,Vt = torch.linalg.svd(Ktt)
                S = torch.clamp(S, min=1e-8)
                Kinv = U @ torch.diag(1/S) @ Vt
                ys = (self.y_train - self.y_mean) / self.y_std
                mu_std = Kst @ (Kinv @ ys)
                var = torch.diag(Kss - Kst @ (Kinv @ Kst.T)).clamp_min(1e-8)
                return mu_std * self.y_std + self.y_mean, torch.sqrt(var) * self.y_std

            ys = (self.y_train - self.y_mean) / self.y_std
            alpha = torch.cholesky_solve(ys.view(-1,1), Lc).view(-1)
            mu_std = Kst @ alpha
            v = torch.linalg.solve(Lc, Kst.T)
            var = torch.diag(Kss - v.T @ v).clamp_min(1e-8)

            mu = mu_std * self.y_std + self.y_mean
            std = torch.sqrt(var) * self.y_std
            return mu, std

    # ---------------- proxy objective (unchanged) ----------------
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
        return torch.sqrt(torch.tensor(raw_obj + 1e-6, device=proj.device, dtype=proj.dtype))  # maximize

    # ---------------- sphere helpers ----------------
    def _tangent_basis(u: torch.Tensor):
        ref = torch.tensor((1.0, 0.0, 0.0), device=u.device, dtype=u.dtype)
        if torch.abs(u[0]) >= 0.9:
            ref = torch.tensor((0.0, 1.0, 0.0), device=u.device, dtype=u.dtype)
        b1 = torch.cross(u, ref, dim=0)
        n1 = b1.norm()
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

    def _normalize_rows(V: torch.Tensor):
        return V / (V.norm(dim=1, keepdim=True) + 1e-12)

    # ---- antipodal folding (canonicalize ±θ) ----
    def _canonicalize_rows(V: torch.Tensor):
        # flip sign so the first nonzero component is positive
        s = torch.sign(torch.where(V[:,0].abs() > 1e-12, V[:,0],
                            torch.where(V[:,1].abs() > 1e-12, V[:,1], V[:,2])))
        s = torch.where(s == 0, torch.ones_like(s), s)
        return V * s.view(-1,1)

    def _sample_around_points(points, k_total, progress):
        if k_total <= 0 or len(points) == 0:
            return None
        pts = []
        k_per = max(1, k_total // len(points))
        radius = 0.08 * (1.0 - progress) + 0.015  # a bit tighter than before
        for c in points:
            b1, b2 = _tangent_basis(c)
            for _ in range(k_per):
                a = torch.randn(2, device=c.device, dtype=c.dtype)
                a = a / (a.norm() + 1e-12)
                v = a[0] * b1 + a[1] * b2
                pts.append(_exp_map(c, v, radius))
        return torch.stack(pts) if len(pts) > 0 else None

    # ---------------- candidates: random + SQSW + repulsive + elite(20%) ----------------
    def generate_smart_candidates(existing_projs, n_candidates, device, progress, X_train=None, y_train=None):
        chunks = []
        # random
        n_random = n_candidates // 4
        chunks.append(rand_projections(3, n_random, device).to(dtype))
        # quasi
        n_quasi = n_candidates // 4
        if n_quasi > 0:
            chunks.append(get_sqsw_projections(n_quasi, device).to(dtype))
        # repulsive around existing
        if len(existing_projs) > 0:
            n_rep = n_candidates // 6
            rep = []
            E = torch.stack(existing_projs).to(dtype)
            for _ in range(n_rep):
                point = rand_projections(3, 1, device).squeeze(0).to(dtype)
                for _ in range(4):
                    diffs = E - point
                    d = diffs.norm(dim=1, keepdim=True).clamp(min=0.1)
                    point = (point + 0.1 * (diffs / (d**3)).sum(dim=0))
                    point = point / point.norm()
                rep.append(point)
            if rep:
                chunks.append(torch.stack(rep))
        # elite (20%) with spacing
        if X_train is not None and y_train is not None and X_train.numel() > 0:
            k_el = max(0, n_candidates // 5)
            if k_el > 0:
                k_elite = max(2, min(12, X_train.size(0) // 8))
                elite_idx = torch.topk(y_train, k_elite).indices
                elite_pts = X_train[elite_idx]
                el_per = max(1, k_el // max(1, elite_pts.size(0)))
                elite_cands_list = []
                for e in elite_pts:
                    cand_e = _sample_around_points([e], el_per, progress)
                    if cand_e is not None:
                        elite_cands_list.append(cand_e)
                if elite_cands_list:
                    elite_cands = torch.cat(elite_cands_list, dim=0)
                    if elite_cands.size(0) > 1:
                        m = elite_cands.size(0)
                        keep = torch.ones(m, dtype=torch.bool, device=elite_cands.device)
                        thr = 0.06  # spacing among elite proposals
                        for i in range(m):
                            if not keep[i]: continue
                            ang = torch.acos(torch.clamp(elite_cands @ elite_cands[i], -1, 1))
                            collide = (ang < thr)
                            collide[i] = False
                            keep = keep & (~collide)
                        elite_cands = elite_cands[keep]
                    chunks.append(elite_cands)

        V = torch.cat(chunks, dim=0)
        V = _normalize_rows(V)
        V = _canonicalize_rows(V)   # << antipodal folding
        return V

    # ---------------- local micro-search on proxy ----------------
    @torch.no_grad()
    def tangent_proxy_improve(point, existing_list, pc1, pc2, cur_sw, progress, tries=6):
        best = point
        best_val = compute_enhanced_objective(point, existing_list, pc1, pc2, cached_current_sw=cur_sw)
        r0 = 0.08 * (1.0 - progress) + 0.015   # slightly smaller, safer
        b1, b2 = _tangent_basis(point)
        steps = [ r0, 0.5*r0, -0.5*r0, -r0 ]
        dirs = (b1, b2,
                (b1+b2)/((b1+b2).norm()+1e-12),
                (b1-b2)/((b1-b2).norm()+1e-12))
        for a in steps:
            for vdir in dirs:
                cand = _exp_map(point, vdir, a)
                val = compute_enhanced_objective(cand, existing_list, pc1, pc2, cached_current_sw=cur_sw)
                if val > best_val:
                    best_val, best = val, cand
        for _ in range(tries):
            a = torch.randn(2, device=point.device, dtype=point.dtype); a = a / (a.norm()+1e-12)
            v = a[0]*b1 + a[1]*b2
            cand = _exp_map(point, v, r0)
            val = compute_enhanced_objective(cand, existing_list, pc1, pc2, cached_current_sw=cur_sw)
            if val > best_val:
                best_val, best = val, cand
        return best

    # ---------------- main loop ----------------
    selected = []
    gp = LightweightGP(kernel_lengthscale=0.3)

    if L <= 5:
        return _canonicalize_rows(get_sqsw_projections(L, device).to(dtype))

    # init: coverage start
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
        # occasional de-dup
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

        # candidate pool (bigger late)
        n_candidates = 4000 if progress <= 0.4 else (9000 if progress <= 0.7 else 14000)
        candidates = generate_smart_candidates(selected, n_candidates, device, progress, X_train, y_train)

        # adaptive spacing
        if len(selected) > 0 and len(selected) < int(0.95 * L):
            E = torch.stack(selected).to(dtype)
            min_d = torch.cdist(candidates, E, p=2).min(dim=1)[0]
            thr = 0.20 * (1.0 - progress) + 0.05 * progress  # a touch milder
            keep = min_d > thr
            if keep.any():
                candidates = candidates[keep]
            if candidates.size(0) < max(150, 12 * batch_size):
                candidates = torch.cat([candidates, _canonicalize_rows(get_sqsw_projections(400, device).to(dtype))], dim=0)

        # acquisition on proxy
        mu_g, sigma_g = gp.predict(candidates)
        mu_f = mu_g ** 2
        sigma_f = (2 * mu_g * sigma_g).clamp_min(1e-12)
        kind_now = 'ei' if (ai.lower() == 'ei' or progress >= 0.6) else 'ucb'
        beta_now = max(0.25, beta * (1.0 - progress)**2) if kind_now == 'ucb' else beta
        acq = acquisition_function(mu_f, sigma_f, y_train_sq, kind=kind_now, beta=beta_now)

        # small TS only very early
        if progress < 0.4:
            ts_sample = mu_f + sigma_f.sqrt() * torch.randn_like(mu_f)
            acq = 0.9 * acq + 0.1 * ts_sample

        # shortlist by acq, then re-rank by proxy + diversity
        n_select = min(batch_size, L - len(selected))
        K = min(1200, max(300, int(200 + 3000 * (progress**1.4))))
        K = min(K, acq.numel())
        topk_idx = torch.topk(acq, K).indices
        top_cands = candidates[topk_idx]

        with torch.no_grad():
            cur_sw = one_d_wasserstein_mean(pc1, pc2, torch.stack(selected)) if selected else None
            scores = []
            for j in range(top_cands.size(0)):
                s = compute_enhanced_objective(top_cands[j], selected, pc1, pc2, cached_current_sw=cur_sw)
                scores.append(s)
            scores = torch.stack(scores)

        chosen = []
        work_scores = scores.clone()
        ls = getattr(gp, 'lengthscale', 0.3)
        width = max(0.25 * ls, 0.07)
        strength = 0.55 if progress < 0.6 else 0.75
        nms_thr = 0.7 * width

        for _ in range(n_select):
            j = torch.argmax(work_scores).item()
            chosen.append(topk_idx[j].item())

            if top_cands.size(0) > 1:
                ang = torch.acos(torch.clamp((top_cands @ top_cands[j]), -1.0, 1.0))
                penal = strength * torch.exp(-(ang / width) ** 2)
                work_scores = work_scores * (1.0 - penal).clamp_min(0.0)
                suppress = ang < nms_thr
                work_scores[suppress] = -1e9

            work_scores[j] = -1e9

        # commit picks + micro proxy refine
        cur_sw = one_d_wasserstein_mean(pc1, pc2, torch.stack(selected)) if selected else None
        for idx in chosen:
            proj0 = candidates[idx]
            proj0 = _normalize_rows(proj0.view(1, -1)).view(-1)
            proj0 = _canonicalize_rows(proj0.view(1, -1)).view(-1)
            proj = tangent_proxy_improve(proj0, selected, pc1, pc2, cur_sw, progress,
                                         tries=(6 if progress < 0.7 else 12))
            proj = _canonicalize_rows(proj.view(1, -1)).view(-1)
            selected.append(proj)
            obj = compute_enhanced_objective(proj, selected[:-1], pc1, pc2, cached_current_sw=cur_sw)
            X_train = torch.cat([X_train, proj.unsqueeze(0)], dim=0)
            y_train = torch.cat([y_train, obj.unsqueeze(0)], dim=0)
            y_train_sq = torch.cat([y_train_sq, obj.unsqueeze(0)**2], dim=0)

        # capacity control
        if len(X_train) > 240:
            keep_idx = list(range(n_init)) + list(range(len(X_train) - 160, len(X_train)))
            X_train = X_train[keep_idx]; y_train = y_train[keep_idx]; y_train_sq = y_train_sq[keep_idx]

    # gentle refinement (same)
    if L <= 300:
        S = local_refinement_simple(torch.stack(selected), device, n_iters=15)
        selected = local_refinement_simple(S, device, n_iters=12)

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