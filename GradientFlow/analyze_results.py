#!/usr/bin/env python3
"""
Script to analyze all results from saved/ folder and generate comprehensive performance summary
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
import re

def parse_filename(filename):
    """Parse method, L, and seed from filename"""
    pattern = r'([A-Z]+)_L(\d+)_\d+_\d+_distances_seed(\d+)\.txt'
    match = re.match(pattern, filename)
    if match:
        method = match.group(1)
        L = int(match.group(2))
        seed = int(match.group(3))
        return method, L, seed
    return None, None, None

def read_distances(filepath):
    """Read distances from file"""
    try:
        with open(filepath, 'r') as f:
            distances = [float(line.strip()) for line in f if line.strip()]
        return distances
    except:
        return []

def analyze_saved_results():
    """Analyze all results in saved/ folder"""
    saved_dir = Path("saved")
    
    # Find all distance files
    distance_files = list(saved_dir.glob("*_distances_seed*.txt"))
    
    results = []
    
    for file in distance_files:
        method, L, seed = parse_filename(file.name)
        if method and L and seed:
            distances = read_distances(file)
            if distances:
                initial_dist = distances[0]
                final_dist = distances[-1]
                convergence_ratio = final_dist / initial_dist
                
                # Calculate convergence rate (log improvement per iteration)
                if len(distances) > 1:
                    log_improvement = np.log(initial_dist / final_dist)
                    convergence_rate = log_improvement / (len(distances) - 1)
                else:
                    convergence_rate = 0
                
                results.append({
                    'method': method,
                    'L': L,
                    'seed': seed,
                    'initial_distance': initial_dist,
                    'final_distance': final_dist,
                    'convergence_ratio': convergence_ratio,
                    'convergence_rate': convergence_rate,
                    'iterations': len(distances),
                    'distances': distances
                })
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Summary statistics by method and L
    summary_stats = df.groupby(['method', 'L']).agg({
        'final_distance': ['mean', 'std', 'min', 'max'],
        'convergence_rate': ['mean', 'std'],
        'convergence_ratio': ['mean', 'std'],
        'iterations': 'mean'
    }).round(6)
    
    # Flatten column names
    summary_stats.columns = [f"{col[0]}_{col[1]}" for col in summary_stats.columns]
    summary_stats = summary_stats.reset_index()
    
    print("=== PERFORMANCE ANALYSIS SUMMARY ===")
    print("\nKey Findings:")
    print("-" * 50)
    
    # Analyze BOSW performance at different L values
    BOSW_results = df[df['method'] == 'BOSW']
    if not BOSW_results.empty:
        BOSW_l10 = BOSW_results[BOSW_results['L'] == 10]
        BOSW_l100 = BOSW_results[BOSW_results['L'] == 100]
        
        if not BOSW_l10.empty and not BOSW_l100.empty:
            print(f"BOSW L=10:  Final distance = {BOSW_l10['final_distance'].mean():.6f} ± {BOSW_l10['final_distance'].std():.6f}")
            print(f"BOSW L=100: Final distance = {BOSW_l100['final_distance'].mean():.6f} ± {BOSW_l100['final_distance'].std():.6f}")
            
            if BOSW_l100['final_distance'].mean() > BOSW_l10['final_distance'].mean():
                print("⚠️  BOSW performs WORSE at L=100 than L=10!")
            else:
                print("✓ BOSW performance improves with higher L")
    
    # Compare methods at L=100
    l100_results = df[df['L'] == 100]
    if not l100_results.empty:
        print(f"\nL=100 Performance Comparison:")
        for method in l100_results['method'].unique():
            method_data = l100_results[l100_results['method'] == method]
            mean_final = method_data['final_distance'].mean()
            std_final = method_data['final_distance'].std()
            print(f"{method:8s}: {mean_final:.6f} ± {std_final:.6f}")
    
    # Find best performing method overall
    best_method = summary_stats.loc[summary_stats['final_distance_mean'].idxmin()]
    print(f"\nBest Overall Performance: {best_method['method']} at L={best_method['L']}")
    print(f"Final distance: {best_method['final_distance_mean']:.6f}")
    
    # Analysis of convergence patterns
    print(f"\nConvergence Analysis:")
    print("-" * 30)
    
    # Check which methods fail to converge properly
    poor_convergers = summary_stats[summary_stats['convergence_rate_mean'] < 0.1]
    if not poor_convergers.empty:
        print("Methods with poor convergence:")
        for _, row in poor_convergers.iterrows():
            print(f"  {row['method']} L={row['L']}: rate={row['convergence_rate_mean']:.4f}")
    
    print(f"\nFull Summary Table:")
    print("=" * 80)
    print(summary_stats.to_string(index=False))
    
    # Save detailed results
    summary_stats.to_csv('results_summary.csv', index=False)
    print(f"\nDetailed results saved to: results_summary.csv")
    
    return df, summary_stats

def identify_BOSW_issues(df):
    """Identify specific issues with BOSW implementation"""
    print("\n=== BOSW ISSUE ANALYSIS ===")
    
    BOSW_data = df[df['method'] == 'BOSW']
    
    if BOSW_data.empty:
        print("No BOSW data found!")
        return
    
    print("\nBOSW Performance Issues:")
    print("-" * 40)
    
    # Check convergence stagnation
    for _, row in BOSW_data.iterrows():
        distances = row['distances']
        if len(distances) > 3:
            # Check if distances plateau after iteration 1
            late_distances = distances[2:]
            if len(late_distances) > 1:
                late_variation = np.std(late_distances) / np.mean(late_distances)
                if late_variation < 0.1:  # Less than 10% variation
                    print(f"  L={row['L']}, seed={row['seed']}: Convergence stagnates after iteration 1")
                    print(f"    Distances: {distances}")
    
    # Compare with other methods
    print(f"\nComparison with other methods at same L values:")
    for L in BOSW_data['L'].unique():
        print(f"\nL={L}:")
        l_data = df[df['L'] == L]
        for method in l_data['method'].unique():
            method_data = l_data[l_data['method'] == method]
            mean_final = method_data['final_distance'].mean()
            print(f"  {method:8s}: {mean_final:.6f}")

if __name__ == "__main__":
    df, summary = analyze_saved_results()
    identify_BOSW_issues(df)