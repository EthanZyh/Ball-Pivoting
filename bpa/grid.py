import numpy as np

class Grid:
    '''Grid for fast point searching'''

    def __init__(self, points, radius):
        # points coordiantes are in [0, 10]
        self.delta = 2 * radius
        self.delta2 = self.delta * self.delta
        self.points = points
        self.grid = {}
        for id, p in enumerate(points):
            grid_coord = tuple(np.floor(p / self.delta).astype(int))
            if grid_coord not in self.grid:
                self.grid[grid_coord] = []
            self.grid[grid_coord].append(id)
        self.neighbors_coord_delta = np.array(
            [[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1], [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
             [0, -1, -1], [0, -1, 0], [0, -1, 1], [0, 0, -1], [0, 0, 0], [0, 0, 1], [0, 1, -1], [0, 1, 0], [0, 1, 1],
             [1, -1, -1], [1, -1, 0], [1, -1, 1], [1, 0, -1], [1, 0, 0], [1, 0, 1], [1, 1, -1], [1, 1, 0], [1, 1, 1]],
            dtype=int
        )
    
    def get_neighbors(self, point):
        '''get points that are within self.delta from a point'''
        grid_coord = np.floor(point / self.delta).astype(int)
        neighbors = [self.grid.get(tuple(grid_coord+coord_delta), []) for coord_delta in self.neighbors_coord_delta]
        neighbors = [id for ids in neighbors for id in ids if np.dot(point-self.points[id], point-self.points[id]) < self.delta2]
        return neighbors
        