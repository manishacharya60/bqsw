import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# ------------------------ Helper Functions ------------------------ #

def normalize_safe(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

def get_projection(pc, theta):
    return pc @ theta.T

def sliced_wasserstein_1d(proj_A, proj_B):
    proj_A_sorted = np.sort(proj_A, axis=0)
    proj_B_sorted = np.sort(proj_B, axis=0)
    return np.mean((proj_A_sorted - proj_B_sorted) ** 2)

def evaluate_direction(theta, A, B):
    proj_A = get_projection(A, theta)
    proj_B = get_projection(B, theta)
    return sliced_wasserstein_1d(proj_A, proj_B)

# ------------------------ SW Estimators ------------------------ #

def sw_naive(A, B, n_proj=1000):
    directions = normalize_safe(np.random.randn(n_proj, 3))
    return np.mean([evaluate_direction(th, A, B) for th in directions])

def qsw_eq(A, B, n_proj=1024):
    sobol = qmc.Sobol(d=3, scramble=False)
    directions = normalize_safe(sobol.random(n=n_proj))
    return np.mean([evaluate_direction(th, A, B) for th in directions])

def bo_sliced_wasserstein(A, B, L=20, eval_grid_size=1000):
    dim = 3
    sobol = qmc.Sobol(d=dim, scramble=False)
    theta_init = normalize_safe(sobol.random(n=8))
    y_init = np.array([evaluate_direction(th, A, B) for th in theta_init])

    kernel = C(1.0) * RBF(length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    gp.fit(theta_init, y_init)

    for _ in range(L - len(theta_init)):
        candidate_thetas = normalize_safe(np.random.randn(eval_grid_size, dim))
        mu, sigma = gp.predict(candidate_thetas, return_std=True)
        beta = 2.0
        ucb = mu - beta * sigma
        next_theta = candidate_thetas[np.argmin(ucb)]
        y_next = evaluate_direction(next_theta, A, B)
        theta_init = np.vstack([theta_init, next_theta])
        y_init = np.append(y_init, y_next)
        gp.fit(theta_init, y_init)

    test_thetas = normalize_safe(np.random.randn(1000, dim))
    pred_mu = gp.predict(test_thetas)
    return np.mean(pred_mu)

# ------------------------ Experiment Runner ------------------------ #

def run_experiments(n_trials=20, L_values=[5, 10, 20, 30, 40]):
    results = { "Naive SW": [], "QSW (EQ)": [], "BO-SW": {L: [] for L in L_values} }

    for trial in range(n_trials):
        # Fixed synthetic point clouds (you can randomize each trial if desired)
        A = np.random.randn(100, 3)
        B = A + 0.5 * np.random.randn(100, 3)

        # Naive and QSW only once per trial
        results["Naive SW"].append(sw_naive(A, B))
        results["QSW (EQ)"].append(qsw_eq(A, B))

        # BO for each L
        for L in L_values:
            bo_val = bo_sliced_wasserstein(A, B, L=L)
            results["BO-SW"][L].append(bo_val)

    return results

def summarize_results(results):
    print("\n--- Mean ± Std of Estimated SW² over Trials ---\n")
    print("Naive SW    : {:.6f} ± {:.6f}".format(
        np.mean(results["Naive SW"]), np.std(results["Naive SW"])))
    print("QSW (EQ)    : {:.6f} ± {:.6f}".format(
        np.mean(results["QSW (EQ)"]), np.std(results["QSW (EQ)"])))
    print("\nBO-SW Results by L:")
    for L, vals in results["BO-SW"].items():
        print(f"  L={L:<3} : {np.mean(vals):.6f} ± {np.std(vals):.6f}")

# ------------------------ Run All ------------------------ #

if __name__ == "__main__":
    results = run_experiments(n_trials=20, L_values=[5, 10, 20, 30, 40])
    summarize_results(results)
