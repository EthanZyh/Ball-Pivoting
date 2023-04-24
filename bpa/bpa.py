import numpy as np
from utils import math3d
from utils import io

from grid import Grid

class BPA_solver:
    "Ball Pivoting Algorithm Solver"

    def __init__(self, points, normals, radius=0.4):
        self.points = points # N x 3
        self.normals = normals # N x 3
        self.faces = []
        self.radius = radius
        self.preprocess()
        self.grid = Grid(self.points, self.radius)
        self.point_index_to_try = 3 # np.random.randint(len(self.points))
        self.is_used_point = np.zeros(len(self.points), dtype=bool)
        self.third_node_of_edge = {} # (p1, p2) -> third node
        self.edge_fringe = [] # (= active edges)
    
    def preprocess(self):
        self.min_pos = np.min(self.points, axis=0) # 3
        self.max_pos = np.max(self.points, axis=0) # 3
        self.box_size = np.max(self.max_pos - self.min_pos) # 1
        self.points = (self.points - self.min_pos) * 10 / self.box_size # N x 3

    def postprocess(self):
        self.points = self.points * self.box_size / 10 + self.min_pos

    def add_triangle(self, p1, p2, p3):
        self.third_node_of_edge[(p1,p2)] = p3
        self.third_node_of_edge[(p2,p3)] = p1
        self.third_node_of_edge[(p3,p1)] = p2
        self.is_used_point[p1] = True
        self.is_used_point[p2] = True
        self.is_used_point[p3] = True
        self.faces.append([p1,p3,p2]) # reverse order to make normal pointing inner

    def find_seed_triangle(self):
        for i in range(len(self.points)):
            p1 = (self.point_index_to_try + i) % len(self.points)
            p1_neighbors = self.grid.get_neighbors(self.points[p1])
            p1_neighbors = sorted(p1_neighbors, key=lambda p2: np.sum((self.points[p1] - self.points[p2]) ** 2))
            for p2 in p1_neighbors:
                if p2 == p1:
                    continue
                # potential_p3s = self.grid.get_neighbors(self.points[p2]) # = p1_neighbors?
                potential_p3s = [p3 for p3 in p1_neighbors \
                                if np.sum((self.points[p2] - self.points[p3]) ** 2) < 4 * self.radius ** 2]
                for p3 in potential_p3s:
                    if p3 == p1 or p3 == p2:
                        continue
                    if (p1, p2) in self.third_node_of_edge or (p2, p1) in self.third_node_of_edge or \
                        (p1, p3) in self.third_node_of_edge or (p3, p1) in self.third_node_of_edge or \
                        (p2, p3) in self.third_node_of_edge or (p3, p2) in self.third_node_of_edge:
                        continue  # already connected
                    if np.dot(np.cross(self.points[p2] - self.points[p1], self.points[p3] - self.points[p1]), self.normals[p1]) > 0:
                        p2, p3 = p3, p2
                    center = math3d.get_center_from_triangle_and_radius(self.points[p1], self.points[p2], self.points[p3], self.radius)
                    if center is None:
                        continue
                    if any([np.dot(center-self.points[p], center-self.points[p]) < self.radius ** 2 for p in p1_neighbors if p not in [p1, p2, p3]]):
                        continue
                    print(f"add tri: {p1},{p2},{p3}")
                    # found one!
                    self.add_triangle(p1, p2, p3)
                    self.point_index_to_try = (p1 + 1) % len(self.points)
                    return [(p1,p2), (p2,p3), (p3,p1)]
        # not found
        return None
    
    def expand_triangle(self, edge_index):
        ''' 
        Expand a triangle along a given edge.
        Return whether successful.
        '''
        p1, p2 = self.edge_fringe[edge_index]
        p0 = self.third_node_of_edge[(p1,p2)] # exising triangle (p0,p1,p2)
        if (p2,p1) in self.third_node_of_edge:
            return False
        p1_neighbors = self.grid.get_neighbors(self.points[p1])
        potential_p3s = [p3 for p3 in p1_neighbors if p3 != p1 and p3 != p2 and \
                        (p1,p3) not in self.third_node_of_edge and \
                        (p3,p2) not in self.third_node_of_edge and \
                        np.sum((self.points[p2] - self.points[p3]) ** 2) < 4 * self.radius ** 2]
        best_p3 = None
        min_angle = np.inf
        for p3 in potential_p3s: # potential new triangle: (p1,p3,p2)
            if p3 == p0:
                continue
            if np.dot(np.cross(self.points[p3] - self.points[p1], self.points[p2] - self.points[p1]), self.normals[p1]) >= 0:
                continue
            center = math3d.get_center_from_triangle_and_radius(self.points[p1], self.points[p3], self.points[p2], self.radius)
            if center is None:
                continue
            angle = math3d.dihedral_angle(self.points[p1], self.points[p2], self.points[p0], center)
            if angle < min_angle:
                min_angle = angle
                best_p3 = p3
        if best_p3 is None:
            return False
        p3 = best_p3
        # check no points inside the sphere
        center = math3d.get_center_from_triangle_and_radius(self.points[p1], self.points[p3], self.points[p2], self.radius)
        angle = math3d.dihedral_angle(self.points[p1], self.points[p2], self.points[p0], center)
        for p in p1_neighbors:
            if p not in [p1, p2, p3]:
                if np.dot(center-self.points[p], center-self.points[p]) < self.radius ** 2:
                    # print("!!!!!", p0 == p, angle)
                    return False
        # found one!
        self.add_triangle(p1, p3, p2)
        self.edge_fringe.pop(edge_index)
        if (p3, p1) in self.edge_fringe:
            self.edge_fringe.remove((p3, p1))
        else:
            self.edge_fringe.append((p1, p3))
        if (p2, p3) in self.edge_fringe:
            self.edge_fringe.remove((p2, p3))
        else:
            self.edge_fringe.append((p3, p2))
        return True
            
    def solve(self):
        expand_try_count = 0
        import os 
        os.makedirs("output", exist_ok=True)
        io.write_obj_file(f"output/{expand_try_count:04d}.obj", {"v": self.points, "f": self.faces})
        while True:
            self.edge_fringe = self.find_seed_triangle() # linked list of edges in the fringe
            if self.edge_fringe is None:
                break
            edge_index = 0
            while edge_index < len(self.edge_fringe):
                expand_try_count += 1
                if not self.expand_triangle(edge_index):
                    edge_index += 1
                if expand_try_count % 200 == 0:
                    print(expand_try_count)
                    io.write_obj_file(f"output/{expand_try_count:04d}.obj", {"v": self.points, "f": self.faces})
        self.postprocess()
        return {"v": self.points, "f": self.faces}