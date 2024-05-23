import numpy as np

def initialize_grid(n, m, start, goal, obstacles):
    grid = np.zeros((n, m))
    for obs in obstacles:
        grid[obs] = -10  # Indicating obstacles
    grid[start] = -1  # Starting point
    grid[goal] = 10  # Goal point
    return grid

def value_iteration(grid, discount_factor=0.9, theta=0.1):
    value_map = np.zeros(grid.shape)
    policy = np.zeros(grid.shape, dtype=int)
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    converge = False
    
    while not converge:
        delta = 0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if (i, j) in [(0, 0), (4, 4), *obstacles]:
                    continue  # Skip update for start, goal and obstacles
                v = value_map[i, j]
                max_value = float('-inf')
                for action in actions:
                    ni, nj = i + action[0], j + action[1]
                    if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                        reward = grid[ni, nj]
                        value = reward + discount_factor * value_map[ni, nj]
                        if value > max_value:
                            max_value = value
                            policy[i, j] = actions.index(action)
                value_map[i, j] = max_value
                delta = max(delta, abs(v - value_map[i, j]))
        converge = delta < theta
    return value_map, policy

# Define the grid world environment
n, m = 5, 5
start = (0, 0)
goal = (4, 4)
obstacles = [(1, 2), (2, 2), (3, 2)]
grid = initialize_grid(n, m, start, goal, obstacles)

# Perform value iteration
value_map, policy = value_iteration(grid)
print(value_map, policy)