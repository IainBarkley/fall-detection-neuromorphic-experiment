# Code modified from: https://github.com/zuoxingdong/mazelab/blob/master/mazelab/generators/
import numpy as np
from itertools import product as cartesian_product


def random_maze(width=81, height=51, complexity=.75, density=.75):
    r"""Generate a random maze array. 
    
    It only contains two kind of objects, obstacle and free space. The numerical value for obstacle
    is ``1`` and for free space is ``0``. 
    
    Code from https://en.wikipedia.org/wiki/Maze_generation_algorithm
    """
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))
    # Build actual maze
    Z = np.zeros(shape, dtype=bool)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1
    # Make aisles
    for i in range(density):
        x, y = np.random.randint(0, shape[1]//2 + 1) * 2, np.random.randint(0, shape[0]//2 + 1) * 2
        Z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:             neighbours.append((y, x - 2))
            if x < shape[1] - 2:  neighbours.append((y, x + 2))
            if y > 1:             neighbours.append((y - 2, x))
            if y < shape[0] - 2:  neighbours.append((y + 2, x))
            if len(neighbours):
                y_,x_ = neighbours[np.random.randint(0, len(neighbours))]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = 1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
                    
    return Z.astype(int)


class MazeGenerator(object):
    def __init__(self):
        self.maze = None
    
    def sample_state(self):
        raise NotImplementedError
        
    def get_maze(self):
        return self.maze


class RandomBlockMazeGenerator(MazeGenerator):
    def __init__(self, maze_size, obstacle_ratio):
        super(MazeGenerator, self).__init__()
        
        self.maze_size = maze_size
        self.obstacle_ratio = obstacle_ratio
        
        self.maze = self._generate_maze()
        
    def _generate_maze(self):
        maze_size = self.maze_size# - 2  # Without the wall
        
        maze = np.zeros([maze_size, maze_size]) 
        
        # List of all possible locations
        all_idx = np.array(list(cartesian_product(range(maze_size), range(maze_size))))

        # Randomly sample locations according to obstacle_ratio
        random_idx_idx = np.random.choice(maze_size**2, size=int(self.obstacle_ratio*maze_size**2), replace=False)
        random_obs_idx = all_idx[random_idx_idx]

        # Fill obstacles
        for idx in random_obs_idx:
            maze[idx[0], idx[1]] = 1

        # Padding with walls, i.e. ones
        maze = np.pad(maze, 1, 'constant', constant_values=1)
        
        return maze
    
    def sample_state(self):
        """Randomly sample an initial state and a goal state"""
        # Get indices for all free spaces, i.e. zero
        free_space = np.where(self.maze == 0)
        free_space = list(zip(free_space[0], free_space[1]))

        # Sample indices for initial state and goal state
        init_idx, goal_idx = np.random.choice(len(free_space), size=2, replace=False)
        
        # Convert initial state to a list, goal states to list of list
        init_state = list(free_space[init_idx])
        goal_states = [list(free_space[goal_idx])]  # TODO: multiple goals
        
        return init_state, goal_states  
