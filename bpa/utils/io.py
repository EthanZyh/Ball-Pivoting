import trimesh
import numpy as np

def read_obj_file(file_name) -> dict:
    mesh = trimesh.load(file_name)
    vertices = np.array(mesh.vertices)
    normals = np.array(mesh.vertex_normals)
    return {"v": vertices, "vn": normals}

def write_obj_file(file_name, mesh:dict):
    mesh = trimesh.Trimesh(vertices=mesh["v"], faces=mesh.get("f", None), process=False)
    mesh.export(file_name)
