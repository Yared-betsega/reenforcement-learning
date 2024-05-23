import numpy as np
import random

def simulate_arm_pull(arm_mean):
    # Simulate pulling an arm by returning a reward from a normal distribution centered at 'arm_mean'
    return np.random.normal(arm_mean, scale=1.0)

def epsilon_greedy(num_arms, num_iterations, epsilon=0.1):
    Q = np.zeros(num_arms)  # Initialize Q-values for each arm
    counts = np.zeros(num_arms)  # Count of pulls for each arm to calculate average correctly

    for _ in range(num_iterations):
        if random.random() < epsilon:
            # Exploration: choose a random arm
            arm = random.randint(0, num_arms - 1)
        else:
            # Exploitation: choose the arm with the highest Q-value
            arm = np.argmax(Q)
        
        # Pull the arm and get the reward
        reward = simulate_arm_pull(arm_means[arm])
        
        # Update the counts and Q-value for the chosen arm
        counts[arm] += 1
        Q[arm] += (reward - Q[arm]) / counts[arm]  # Incremental average update

    return Q

num_arms = 10
arm_means = np.linspace(0, 9, num_arms)  # Linearly spaced means for the arms
num_iterations = 1000
epsilon = 0.1  # Exploration probability

# Perform Epsilon-Greedy Policy
Q_values = epsilon_greedy(num_arms, num_iterations, epsilon)
optimal_arm = np.argmax(Q_values)  # Find the arm with the highest Q-value

print("Q-values for each arm:", Q_values)
print("Optimal arm to pull according to Epsilon-Greedy Policy:", optimal_arm)
