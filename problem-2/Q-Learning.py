import numpy as np

def simulate_arm_pull(arm_mean):
    # Simulate pulling an arm by returning a reward from a normal distribution centered at 'arm_mean'
    return np.random.normal(arm_mean, scale=1.0)

def q_learning(num_arms, num_iterations, alpha=0.1, gamma=0.9):
    Q = np.zeros(num_arms)  # Initialize Q-values for each arm

    for _ in range(num_iterations):
        arm = np.random.choice(num_arms)  # Choose an arm at random (exploration)
        reward = simulate_arm_pull(arm_means[arm])
        
        # Q-Learning update rule
        Q[arm] += alpha * (reward - Q[arm])

    return Q

num_arms = 10
arm_means = np.linspace(0, 9, num_arms)  # Linearly spaced means for the arms
num_iterations = 1000

# Perform Q-Learning
Q_values = q_learning(num_arms, num_iterations)
optimal_arm = np.argmax(Q_values)  # Find the arm with the highest Q-value

print("Q-values for each arm:", Q_values)
print("Optimal arm to pull according to Q-Learning:", optimal_arm)
