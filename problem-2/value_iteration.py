import numpy as np
import random

def simulate_arm_pull(arm):
    # Assume each arm returns a reward from a normal distribution (mean varies, std = 1)
    return np.random.normal(loc=arm_means[arm], scale=1.0)

def value_iteration_bandit(arms, iterations, discount_factor=0.95):
    values = np.zeros(arms)  # Start with zero value for each arm
    for _ in range(iterations):
        new_values = np.copy(values)
        for arm in range(arms):
            # Simulate pulling each arm
            reward = simulate_arm_pull(arm)
            # Update the value estimate for the arm
            new_values[arm] = reward + discount_factor * values[arm]
        values = new_values
    return values

arms = 10  # Number of arms
arm_means = np.linspace(0, 9, arms)  # Varying means for each arm to simulate varying rewards
iterations = 1000  # Number of iterations for value iteration

# Perform value iteration
arm_values = value_iteration_bandit(arms, iterations)
optimal_arm = np.argmax(arm_values)  # Find the arm with the highest estimated value

print("Estimated values for each arm:", arm_values)
print("Optimal arm to pull:", optimal_arm)
