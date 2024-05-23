import numpy as np

def initialize_grid(n, m, start, goal, obstacles):
    grid = np.zeros((n, m))
    for obs in obstacles:
        grid[obs] = -10  # Indicating obstacles
    grid[start] = -1  # Starting point
    grid[goal] = 10  # Goal point
    return grid

def policy_evaluation(policy, grid, discount_factor=0.9, theta=0.1):
    value_map = np.zeros(grid.shape)
    while True:
        delta = 0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if (i, j) in [(0, 0), (4, 4), *obstacles]:
                    continue
                v = value_map[i, j]
                action = policy[i, j]
                ni, nj = i + action[0], j + action[1]
                reward = grid[ni, nj]
                value_map[i, j] = reward + discount_factor * value_map[ni, nj]
                delta = max(delta, abs(v - value_map[i, j]))
        if delta < theta:
            break
    return value_map

def policy_improvement(value_map, grid, discount_factor=0.9):
    policy = np.zeros(grid.shape, dtype=(int, 2))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if (i, j) in [(0, 0), (4, 4), *obstacles]:
                continue
            actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
            action_values = []
            for action in actions:
                ni, nj = i + action[0], j + action[1]
                if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                    reward = grid[ni, nj]
                    action_value = reward + discount_factor * value_map[ni, nj]
                    action_values.append(action_value)
                else:
                    action_values.append(float('-inf'))  # Cannot move in this direction
            best_action = actions[np.argmax(action_values)]
            policy[i, j] = best_action
    return policy

def policy_iteration(grid, discount_factor=0.9, theta=0.1):
    policy = np.zeros(grid.shape, dtype=(int, 2))  # Initial random policy
    is_policy_stable = False
    while not is_policy_stable:
        value_map = policy_evaluation(policy, grid, discount_factor, theta)
        new_policy = policy_improvement(value_map, grid, discount_factor)
        if np.array_equal(new_policy, policy):
            is_policy_stable = True
        policy = new_policy
    return policy, value_map

# Define the grid world environment
n, m = 5, 5
start = (0, 0)
goal = (4, 4)
obstacles = [(1, 2), (2, 2), (3, 2)]
grid = initialize_grid(n, m, start, goal, obstacles)

# Perform policy iteration
optimal_policy, optimal_value = policy_iteration(grid)
print(optimal_policy)
print(optimal_value)
