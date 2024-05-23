import numpy as np
import random

def simulate_arm_pull(arm):
    # Returns a reward from a normal distribution centered at the mean of the arm.
    return np.random.normal(loc=arm_means[arm], scale=1.0)

def policy_evaluation(arm, arm_means, iterations, discount_factor=0.95):
    total_reward = 0
    for _ in range(iterations):
        reward = simulate_arm_pull(arm)
        total_reward += reward
    average_reward = total_reward / iterations
    # Considering discount factor though typically not used in bandit problems
    return average_reward * discount_factor

def policy_iteration(arm_means, iterations, evaluation_iterations):
    num_arms = len(arm_means)
    policy = random.randint(0, num_arms - 1)  # Start with a random arm as the initial policy
    values = np.zeros(num_arms)  # Initialize the values for each arm

    for _ in range(iterations):
        # Evaluate and update the value of the current policy
        values[policy] = policy_evaluation(policy, arm_means, evaluation_iterations, discount_factor=0.95)

        # Policy Improvement: Find a better arm, if any
        for arm in range(num_arms):
            arm_value = policy_evaluation(arm, arm_means, evaluation_iterations, discount_factor=0.95)
            values[arm] = arm_value  # Update the estimated value for each arm
            if arm_value > values[policy]:
                policy = arm

    return policy, values

arms = 10  # Number of arms
arm_means = np.linspace(0, 9, arms)  # Varying means for each arm
iterations = 10  # Number of iterations for policy iteration
evaluation_iterations = 1000  # Number of pulls used to evaluate each arm

# Perform Policy Iteration
optimal_arm, arm_values = policy_iteration(arm_means, iterations, evaluation_iterations)

print("Estimated values for each arm:", arm_values)
print("Optimal arm to pull according to policy iteration:", optimal_arm)
