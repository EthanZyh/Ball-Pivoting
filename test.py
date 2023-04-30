import trimesh 
import numpy as np
import bpa.utils.math3d as math3d

# mesh = trimesh.load("output/2800.obj")
# mesh.export("output/test.obj")
# vertices = np.array(mesh.vertices)
# faces = mesh.faces
# r = 0.8

# num = 0
# for face in faces:
#     p1, p2, p3 = face[0], face[1], face[2]
#     center = math3d.get_center_from_triangle_and_radius(vertices[p1], vertices[p3], vertices[p2], r)
#     for p in range(len(vertices)):
#         if np.dot(vertices[p]-center, vertices[p]-center) < r**2:
#             if p in [p1, p2, p3]:
#                 continue
#             print("Error", num)
#     num += 1


mesh = trimesh.load("data/bunny.obj")
vertices = np.array(mesh.vertices)
faces = mesh.faces
vn = np.array(mesh.vertex_normals)

vertices += 0.1 * vn
mesh = trimesh.Trimesh(vertices=vertices, process=False)
mesh.export("output/!test.obj")
