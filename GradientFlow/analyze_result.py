# File: GradientFlow/analyze_results.py

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_convergence_results(methods, L_values, ind1=0, ind2=98, seeds=[1, 2, 3]):
    """
    Loads and plots the convergence results for different SW methods.

    Args:
        methods (list): List of method names (e.g., ['QSW', 'SW', 'BQSW']).
        L_values (list): List of projection numbers to plot (e.g., [50, 100, 200]).
        ind1 (int): Index for the source shape.
        ind2 (int): Index for the target shape.
        seeds (list): List of random seeds used in the experiment.
    """
    num_steps = 6 # Based on print_steps in main_point.py
    iterations = np.arange(num_steps)

    # --- Plot 1: Distance vs. Iterations ---
    plt.figure(figsize=(12, 8))
    
    for L in L_values:
        for method in methods:
            all_distances = []
            for seed in seeds:
                filename = f"saved/{method}_L{L}_{ind1}_{ind2}_distances_seed{seed}.txt"
                if os.path.exists(filename):
                    distances = np.loadtxt(filename, delimiter=',')
                    if len(distances) == num_steps:
                        all_distances.append(distances)
                else:
                    print(f"Warning: File not found - {filename}")

            if not all_distances:
                continue

            all_distances = np.array(all_distances)
            mean_distances = np.mean(all_distances, axis=0)
            std_distances = np.std(all_distances, axis=0)

            plt.plot(iterations, mean_distances, marker='o', linestyle='-', label=f'{method} (L={L})')
            plt.fill_between(iterations, mean_distances - std_distances, mean_distances + std_distances, alpha=0.2)

    plt.xlabel("Iteration")
    plt.ylabel("True Wasserstein-2 Distance")
    plt.title("Convergence Performance: Distance vs. Iterations")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig("comparison_distance_vs_iterations.png")
    plt.show()


    # --- Plot 2: Distance vs. Time ---
    plt.figure(figsize=(12, 8))

    for L in L_values:
        for method in methods:
            all_distances = []
            all_times = []
            for seed in seeds:
                dist_file = f"saved/{method}_L{L}_{ind1}_{ind2}_distances_seed{seed}.txt"
                time_file = f"saved/{method}_L{L}_{ind1}_{ind2}_times_seed{seed}.txt"
                
                if os.path.exists(dist_file) and os.path.exists(time_file):
                    distances = np.loadtxt(dist_file, delimiter=',')
                    times = np.loadtxt(time_file, delimiter=',')
                    if len(distances) == num_steps and len(times) == num_steps:
                        all_distances.append(distances)
                        all_times.append(times)
                else:
                    print(f"Warning: File pair not found for {method}, L={L}, seed={seed}")

            if not all_distances:
                continue

            # Average time and corresponding distances
            mean_times = np.mean(np.array(all_times), axis=0)
            mean_distances = np.mean(np.array(all_distances), axis=0)
            std_distances = np.std(np.array(all_distances), axis=0)

            plt.plot(mean_times, mean_distances, marker='o', linestyle='-', label=f'{method} (L={L})')
            plt.fill_between(mean_times, mean_distances - std_distances, mean_distances + std_distances, alpha=0.2)

    plt.xlabel("Wall-Clock Time (seconds)")
    plt.ylabel("True Wasserstein-2 Distance")
    plt.title("Convergence Performance: Distance vs. Time")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig("comparison_distance_vs_time.png")
    plt.show()


if __name__ == '__main__':
    # --- Configuration for the analysis ---
    # Add the methods you want to compare to this list.
    # The names must match the file prefixes (e.g., 'QSW', 'SW', 'BQSW').
    # methods_to_compare = ['RBOSW', 'RRQSW', 'RRNQSW', 'SW', 'RQSW']
    methods_to_compare = ['SW', 'QSW', 'BOSW', 'RBOSW', 'ROCQSW']
    
    # Specify the number of projections (L) you want to see on the plots.
    L_values_to_plot = [10]
    
    # --- Run the analysis ---
    plot_convergence_results(methods_to_compare, L_values_to_plot)