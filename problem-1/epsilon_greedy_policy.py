import numpy as np
import random

def initialize_grid(n, m, start, goal, obstacles):
    grid = np.zeros((n, m))
    for obs in obstacles:
        grid[obs] = -10  # Marking obstacles
    grid[start] = -1  # Starting point
    grid[goal] = 10  # Goal point
    return grid

def q_learning_with_epsilon_greedy(grid, episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    n, m = grid.shape
    Q = np.zeros((n, m, 4))  # Q-values for each state and action
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    for episode in range(episodes):
        state = (0, 0)  # Start from the beginning of the grid
        while state != (n-1, m-1):  # Loop until the goal state is reached
            if random.uniform(0, 1) < epsilon:
                action = random.choice(range(4))  # Explore: choose a random action
            else:
                action = np.argmax(Q[state[0], state[1], :])  # Exploit: choose the best known action

            next_state = (state[0] + actions[action][0], state[1] + actions[action][1])

            # Ensure next_state is within the grid boundaries
            if next_state[0] < 0 or next_state[0] >= n or next_state[1] < 0 or next_state[1] >= m:
                next_state = state  # If outside grid, stay in place

            # Calculate reward for the next state
            reward = grid[next_state]

            # Q-learning update formula
            old_value = Q[state[0], state[1], action]
            next_max = np.max(Q[next_state[0], next_state[1], :])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            Q[state[0], state[1], action] = new_value

            state = next_state  # Move to the next state

    # Derive the optimal policy from the Q-values
    policy = np.zeros((n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            policy[i, j] = np.argmax(Q[i, j, :])  # Choose the action with the highest Q-value at each state
    return Q, policy

# Define the grid world environment
n, m = 5, 5
start = (0, 0)
goal = (4, 4)
obstacles = [(1, 2), (2, 2), (3, 2)]
grid = initialize_grid(n, m, start, goal, obstacles)

# Perform Q-Learning with Epsilon-Greedy Policy
Q_values, optimal_policy = q_learning_with_epsilon_greedy(grid, 1000)  # 1000 episodes for learning
print("Q-values:\n", Q_values)
print("Optimal Policy:\n", optimal_policy)
