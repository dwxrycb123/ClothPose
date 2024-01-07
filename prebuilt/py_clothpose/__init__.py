import numpy as np
import trimesh
from .py_clothpose import L2Energy, Mesh, Points, show_mesh, show_points, \
    Optimizer, Mesh2PointEnergy, Point2MeshEnergy, BarrierEnergy, ARAPEnergy

__all__ = ['ARAPEnergy', 'BarrierEnergy', 'L2Energy', 'Mesh',
           'Mesh2PointEnergy', 'Optimizer', 'Point2MeshEnergy', 'Points',
           'show_mesh', 'show_points']


def read_mesh(mesh_path: str) -> Mesh:
    mesh = trimesh.load_mesh(mesh_path)
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.faces)
    return Mesh(verts, tris)


def read_points(pcd_path: str) -> Points:
    pcd = trimesh.load(pcd_path)
    return Points(np.asarray(pcd.vertices))
