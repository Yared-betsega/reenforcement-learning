import numpy as np
import math

def simulate_arm_pull(arm_mean):
    # Simulate pulling an arm by returning a reward from a normal distribution centered at 'arm_mean'
    return np.random.normal(arm_mean, scale=1.0)

def ucb(num_arms, num_iterations):
    Q = np.zeros(num_arms)  # Initialize Q-values for each arm
    counts = np.zeros(num_arms)  # Count of pulls for each arm
    total_pulls = 0

    # Initialize by pulling each arm once
    for arm in range(num_arms):
        reward = simulate_arm_pull(arm_means[arm])
        Q[arm] = reward  # Initial reward becomes the initial estimate
        counts[arm] += 1
        total_pulls += 1

    for _ in range(num_arms, num_iterations):
        # Calculate UCB for each arm
        ucb_values = Q + np.sqrt(2 * np.log(total_pulls) / counts)
        arm = np.argmax(ucb_values)  # Choose the arm with the highest UCB

        # Pull the chosen arm and update
        reward = simulate_arm_pull(arm_means[arm])
        counts[arm] += 1
        total_pulls += 1
        # Update the Q-value for the chosen arm using incremental average
        Q[arm] += (reward - Q[arm]) / counts[arm]

    return Q

num_arms = 10
arm_means = np.linspace(0, 9, num_arms)  # Linearly spaced means for the arms
num_iterations = 1000

# Perform UCB
Q_values = ucb(num_arms, num_iterations)
optimal_arm = np.argmax(Q_values)  # Find the arm with the highest Q-value

print("Q-values for each arm:", Q_values)
print("Optimal arm to pull according to UCB Algorithm:", optimal_arm)
