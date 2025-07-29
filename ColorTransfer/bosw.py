
from math import log
import torch
import numpy as np
import random
from sklearn.cluster import KMeans

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

def get_bosw_projections(L, device, pc1, pc2, p=2, beta=1.0, gamma=0.1):
    """
    BOSW for color transfer with cluster-aware acquisition function.
    Returns: [L x 3] projection directions on unit sphere.
    """
    class LightweightGP:
        def __init__(self, kernel_lengthscale=0.5):
            self.lengthscale = kernel_lengthscale
            self.X_train = None
            self.y_train = None
            self.y_mean = 0
            self.y_std = 1

        def fit(self, X, y):
            self.X_train = X
            self.y_train = y
            self.y_mean = y.mean()
            self.y_std = y.std() + 1e-6

        def predict(self, X_test):
            if self.X_train is None:
                return torch.zeros(len(X_test)), torch.ones(len(X_test))
            K_train = self._kernel(self.X_train, self.X_train)
            K_test = self._kernel(X_test, self.X_train)
            K_test_test = self._kernel(X_test, X_test)
            K_train_inv = torch.linalg.inv(K_train + 1e-4 * torch.eye(len(K_train)))
            y_normalized = (self.y_train - self.y_mean) / self.y_std
            mu = K_test @ K_train_inv @ y_normalized
            var = torch.diag(K_test_test - K_test @ K_train_inv @ K_test.T)
            mu = mu * self.y_std + self.y_mean
            std = torch.sqrt(var.clamp(min=1e-6)) * self.y_std
            return mu, std

        def _kernel(self, X1, X2):
            angles = torch.acos(torch.clamp(X1 @ X2.T, -1, 1))
            return torch.exp(-0.5 * (angles / self.lengthscale) ** 2)

    def cluster_contrast_score(projection, pc1, pc2, K=5):
        theta = projection.unsqueeze(0)
        X_proj = (pc1 @ theta.T).cpu().numpy()
        Y_proj = (pc2 @ theta.T).cpu().numpy()
        all_proj = np.concatenate([X_proj, Y_proj], axis=0)
        labels = KMeans(n_clusters=K, n_init=3).fit_predict(all_proj)
        centers = [all_proj[labels == k].mean() for k in range(K)]
        return np.var(centers)

    def compute_objective(projection, existing_projections, pc1, pc2):
        if len(existing_projections) > 0:
            existing_tensor = torch.stack(existing_projections)
            distances = torch.norm(existing_tensor - projection.unsqueeze(0), dim=1)
            coverage_score = distances.min().item()
        else:
            coverage_score = 1.0
        contribution = cluster_contrast_score(projection, pc1, pc2)
        return coverage_score * (1 + contribution)

    selected_projections = []
    gp = LightweightGP(kernel_lengthscale=0.3)

    if L <= 20:
        return get_sqsw_projections(L, device)

    n_init = min(10, L // 4)
    init_projections = get_sqsw_projections(n_init, device)
    X_train, y_train = [], []

    for i in range(n_init):
        proj = init_projections[i]
        selected_projections.append(proj)
        obj = compute_objective(proj, selected_projections[:-1], pc1, pc2)
        X_train.append(proj)
        y_train.append(obj)

    X_train = torch.stack(X_train)
    y_train = torch.tensor(y_train, device=device)
    remaining = L - n_init
    batch_size = min(10, max(1, remaining // 20))

    while len(selected_projections) < L:
        gp.fit(X_train, y_train)
        n_candidates = min(1000, 100 * batch_size)
        candidates = []
        candidates.append(rand_projections(dim=3, num_projections=n_candidates // 2, device=device))
        candidates.append(get_sqsw_projections(n_candidates // 2, device))
        candidates = torch.cat(candidates)
        keep_mask = torch.ones(len(candidates), dtype=torch.bool)
        for i, cand in enumerate(candidates):
            for existing in selected_projections:
                if torch.norm(cand - existing) < 0.1:
                    keep_mask[i] = False
                    break
        candidates = candidates[keep_mask]
        if len(candidates) == 0:
            candidates = rand_projections(dim=3, num_projections=100, device=device)

        with torch.no_grad():
            mu, sigma = gp.predict(candidates)
            diversity_scores = []
            for cand in candidates:
                diversity_scores.append(cluster_contrast_score(cand, pc1, pc2))
            diversity_scores = torch.tensor(diversity_scores, device=device)
            ucb = mu + beta * sigma + gamma * diversity_scores

        n_select = min(batch_size, L - len(selected_projections))
        selected_indices = []
        for _ in range(n_select):
            best_idx = torch.argmax(ucb).item()
            selected_indices.append(best_idx)
            selected_point = candidates[best_idx]
            distances = torch.norm(candidates - selected_point.unsqueeze(0), dim=1)
            penalty = torch.exp(-distances * 10)
            ucb = ucb * (1 - 0.5 * penalty)

        for idx in selected_indices:
            proj = candidates[idx]
            selected_projections.append(proj)
            obj = compute_objective(proj, selected_projections[:-1], pc1, pc2)
            X_train = torch.cat([X_train, proj.unsqueeze(0)])
            y_train = torch.cat([y_train, torch.tensor([obj], device=device)])
        if len(X_train) > 100:
            keep_indices = list(range(n_init)) + list(range(len(X_train) - 90, len(X_train)))
            X_train = X_train[keep_indices]
            y_train = y_train[keep_indices]

    return local_refinement_simple(torch.stack(selected_projections), device)

def local_refinement_simple(projections, device, n_iters=5):
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

def get_rbosw_projections(L, device, pc1, pc2, p=2, restarts=5, beta=0.8, gamma=0.2):
    best_theta = None
    best_score = -float("inf")
    for _ in range(restarts):
        theta = get_bosw_projections(L, device, pc1, pc2, p=p, beta=beta, gamma=gamma)
        angles = torch.acos(torch.clamp(theta @ theta.T, -1, 1))
        score = torch.topk(angles.flatten(), k=L, largest=False)[0][-1]  # min angular separation
        if score > best_score:
            best_score = score
            best_theta = theta
    return best_theta

