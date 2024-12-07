import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
code_path = project_root/"code"
results_path = project_root / "results" / "plots"
results_path.mkdir(parents=True, exist_ok=True)
sys.path.append(str(code_path))
from utility_functions_bandit import*
from utility_functions_optim import*

def parse_params_from_file(file_path):
    """Parse parameters from a given file."""
    params = {}
    with open(file_path, "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            params[key.strip()] = float(value.strip()) if '.' in value else int(value.strip())
    return params

def main(refinement_type, func, M, SEARCH_SPACE_L, SEARCH_SPACE_H, NUM_REFINEMENT, THOMPSON_LOOP_LEN):
    search_space = np.linspace(SEARCH_SPACE_L, SEARCH_SPACE_H, M + 1)
    intervals = [[search_space[i], search_space[i + 1]] for i in range(len(search_space) - 1)]
    P_0 = np.ones(M) / M
    H_0 = get_entropy(P_0)
    empirical_dist = {key: [] for key in range(M)}
    interval_params = {key: [0, 1] for key in range(M)}

    for refinement in range(1, NUM_REFINEMENT + 1):
        print(f"=========== Refinement {refinement} ===========")

        for t in range(THOMPSON_LOOP_LEN):
            interval_samples = [np.random.normal(interval_params[i][0], np.sqrt(interval_params[i][1])) for i in range(M)]
            chosen_sample_idx = np.argmax(interval_samples)
            chosen_interval = intervals[chosen_sample_idx]
            chosen_point = np.random.uniform(*chosen_interval)

            if func == 1:
                reward = phi_1_noisy(chosen_point, seed=42)
            else:
                reward = phi_2_noisy(chosen_point, seed=42)

            empirical_dist[chosen_sample_idx].append(reward)
            interval_params[chosen_sample_idx][0] = np.mean(empirical_dist[chosen_sample_idx])
            interval_params[chosen_sample_idx][1] = max(np.var(empirical_dist[chosen_sample_idx]), 1e-3)

            mu_list = [interval_params[i][0] for i in range(M)]
            var_list = [interval_params[i][1] for i in range(M)]
            P_t = build_proposal_probabilities(mu_list, var_list)

            H_t = get_entropy(P_t)
            print(f"Iteration {t}: H(t) = {H_t:.4f}, H(0) = {H_0:.4f}")

            if H_t <= 0.5 * H_0:
                print(f"Stopping Thompson Sampling at iteration {t}.")
                break

        highest_prob_interval_idx = np.argmax(P_t)
        if refinement_type == 1:
            intervals = refine_interval_and_neighbors(intervals, highest_prob_interval_idx, M)
        elif refinement_type == 2:
            intervals = refine_interval_with_cdf(intervals, P_t, highest_prob_interval_idx, M)

        M = len(intervals)
        print(f"Refined Intervals: {intervals}; HPIdx: {highest_prob_interval_idx}\n")

    # visualization:
    x = np.linspace(0, 1, 1000)
    if func == 1:
        y = phi_1(x)
        y_noisy = phi_1_noisy(x, seed=42)
        title = "$\\phi(x) = x\\cdot(1-x)$"
    else:
        y = phi_2(x)
        y_noisy = phi_2_noisy(x, seed=42)
        title = "$\\phi(x) = 2 + 2x(1-2x) + 1/50 \\cdot \\sin{(52 \\pi x)}$"

    plt.plot(x, y_noisy, label="Y Noisy")
    plt.plot(x, y, label="Y True")
    plt.axvline(x=intervals[highest_prob_interval_idx][0], color='red', linestyle='--', linewidth=1, label="Lower Limit")
    plt.axvline(x=intervals[highest_prob_interval_idx][1], color='red', linestyle='--', linewidth=1, label="Upper Limit")
    plt.legend()
    plt.title(title)
    plot_path = results_path / f"function_maximization{func}.png"
    print(f"Saving plot to: {plot_path}")
    plt.savefig(results_path/"function_maximization", dpi=300)
    try:
        plt.savefig(plot_path, dpi=300)
        print("Plot saved successfully.")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refinement_type", type=int, choices=[1, 2], help="Type of refinement: 1 for neighbors, 2 for CDF.")
    parser.add_argument("--func", type=int, choices=[1, 2], help="Function to maximize: 1 for phi_1, 2 for phi_2.")
    parser.add_argument("--param_file", type=str, help="Path to parameter file.")

    args = parser.parse_args()

    if args.param_file:
        file_params = parse_params_from_file(args.param_file)
        refinement_type = file_params.get("refinement_type", 1)
        func = file_params.get("func", 1)
        M = file_params.get("M", 200)
        SEARCH_SPACE_L = file_params.get("SEARCH_SPACE_L", 0)
        SEARCH_SPACE_H = file_params.get("SEARCH_SPACE_H", 1)
        NUM_REFINEMENT = file_params.get("NUM_REFINEMENT", 5)
        THOMPSON_LOOP_LEN = file_params.get("THOMPSON_LOOP_LEN", 1000)
    else:
        refinement_type = args.refinement_type
        func = args.func
        M = 200
        SEARCH_SPACE_L = 0
        SEARCH_SPACE_H = 1
        NUM_REFINEMENT = 5
        THOMPSON_LOOP_LEN = 1000

    main(refinement_type, func, M, SEARCH_SPACE_L, SEARCH_SPACE_H, NUM_REFINEMENT, THOMPSON_LOOP_LEN)
