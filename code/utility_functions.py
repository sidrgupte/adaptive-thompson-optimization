import numpy as np
import random, math

def constant_q_simulation(n_rounds, epsilon, q):
    N_H, N_T = 0, 0
    Q_H, Q_T = 0.0, 0.0
    total_rewards = []
    Q_H_progress, Q_T_progress = [], []
    action_counts = []

    for t in range(1, n_rounds + 1):
        if random.random() < epsilon:
            action = random.choice(["H", "T"])
        else:
            action = "H" if Q_H > Q_T else "T"

        outcome = "H" if random.random() < q else "T"
        reward = 1 if action == outcome else 0

        if action == "H":
            N_H += 1
            Q_H = Q_H + (reward - Q_H) / N_H
        else:
            N_T += 1
            Q_T = Q_T + (reward - Q_T) / N_T

        total_rewards.append(reward if len(total_rewards) == 0 else total_rewards[-1] + reward)
        Q_H_progress.append(Q_H)
        Q_T_progress.append(Q_T)
        action_counts.append((N_H, N_T))

    return Q_H_progress, Q_T_progress, action_counts, total_rewards


# Simulate dynamic q experiment
def dynamic_q_simulation(n_rounds, epsilon, q_t_func):
    N_H, N_T = 0, 0
    Q_H, Q_T = 0.0, 0.0
    total_rewards = []
    Q_H_progress, Q_T_progress = [], []
    q_t_values = []
    action_counts = []

    for t in range(1, n_rounds + 1):
        q_t = q_t_func(t)
        q_t_values.append(q_t)

        if random.random() < epsilon:
            action = random.choice(["H", "T"])
        else:
            action = "H" if Q_H > Q_T else "T"

        outcome = "H" if random.random() < q_t else "T"
        reward = 1 if action == outcome else 0

        if action == "H":
            N_H += 1
            Q_H = Q_H + (reward - Q_H) / N_H
        else:
            N_T += 1
            Q_T = Q_T + (reward - Q_T) / N_T

        total_rewards.append(reward if len(total_rewards) == 0 else total_rewards[-1] + reward)
        Q_H_progress.append(Q_H)
        Q_T_progress.append(Q_T)
        action_counts.append((N_H, N_T))

    return Q_H_progress, Q_T_progress, action_counts, total_rewards, q_t_values

def simulate_thompson_sampling(n_rounds, p_h_func):
    # Initialize Beta distribution parameters
    alpha_H, beta_H = 1, 1  # For Heads
    alpha_T, beta_T = 1, 1  # For Tails

    rewards = []  # To track cumulative rewards
    equity_curve = []  # Cumulative reward over time
    posterior_H_means = []  # Posterior mean for Heads
    posterior_T_means = []  # Posterior mean for Tails

    for t in range(1, n_rounds + 1):
        # Current probability of Heads
        p_h_t = p_h_func(t)
        
        # Sample from Beta distributions
        sample_H = np.random.beta(alpha_H, beta_H)
        sample_T = np.random.beta(alpha_T, beta_T)
        
        # Resolve ties by random choice
        if sample_H == sample_T:
            action = random.choice(["H", "T"])
        else:
            action = "H" if sample_H > sample_T else "T"

        # Simulate the coin flip
        outcome = "H" if random.random() < p_h_t else "T"
        reward = 1 if action == outcome else 0
        
        # Update Beta distribution for the chosen action
        if action == "H":
            alpha_H += reward
            beta_H += 1 - reward
        else:
            alpha_T += reward
            beta_T += 1 - reward
        
        # Track cumulative rewards and posterior means
        rewards.append(reward)
        equity_curve.append(sum(rewards))
        posterior_H_means.append(alpha_H / (alpha_H + beta_H))
        posterior_T_means.append(alpha_T / (alpha_T + beta_T))

    return equity_curve, posterior_H_means, posterior_T_means