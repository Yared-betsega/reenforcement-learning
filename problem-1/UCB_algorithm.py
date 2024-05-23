import numpy as np
import math

def initialize_grid(n, m, start, goal, obstacles):
    grid = np.zeros((n, m))
    for obs in obstacles:
        grid[obs] = -10  # Marking obstacles
    grid[start] = -1  # Starting point
    grid[goal] = 10  # Goal point
    return grid

def ucb_grid_world(grid, episodes, exploration_factor=2):
    n, m = grid.shape
    Q = np.zeros((n, m, 4))  # Q-values for each state and action
    N = np.zeros((n, m, 4))  # Count of each state-action pair
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    
    for episode in range(episodes):
        state = (0, 0)  # Start from the beginning of the grid
        while state != (n-1, m-1):
            current_counts = N[state[0], state[1], :]
            total = np.sum(current_counts)
            ucb_values = Q[state[0], state[1], :] + np.sqrt((2 * math.log(total + 1)) / (current_counts + 1)) * exploration_factor
            action = np.argmax(ucb_values)  # Choose the action with the highest UCB

            next_state = (state[0] + actions[action][0], state[1] + actions[action][1])

            # Ensure next_state is within the grid boundaries
            if next_state[0] < 0 or next_state[0] >= n or next_state[1] < 0 or next_state[1] >= m:
                next_state = state  # Stay in place if move is out of bounds

            # Calculate reward for the next state
            reward = grid[next_state]

            # Update Q-value and counts
            Q[state[0], state[1], action] = (N[state[0], state[1], action] * Q[state[0], state[1], action] + reward) / (N[state[0], state[1], action] + 1)
            N[state[0], state[1], action] += 1

            state = next_state  # Move to the next state

    # Derive the optimal policy from the Q-values
    policy = np.zeros((n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            policy[i, j] = np.argmax(Q[i, j, :])  # Choose the best action at each state based on Q-values
    return Q, policy

# Define the grid world environment
n, m = 5, 5
start = (0, 0)
goal = (4, 4)
obstacles = [(1, 2), (2, 2), (3, 2)]
grid = initialize_grid(n, m, start, goal, obstacles)

# Perform UCB algorithm
Q_values, optimal_policy = ucb_grid_world(grid, 1000)  # 1000 episodes for learning
print("Q-values:\n", Q_values)
print("Optimal Policy:\n", optimal_policy)
