import sys
sys.path.append('.')
from build.py_clothpose import L2Energy, Mesh, Points, show_mesh, show_points, \
    Optimizer, Mesh2PointEnergy, Point2MeshEnergy, BarrierEnergy, ARAPEnergy
import json
import numpy as np
import trimesh
from icecream import ic


def read_mesh(mesh_path):
    mesh = trimesh.load_mesh(mesh_path)
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.faces)
    return Mesh(verts, tris)


def read_points(pcd_path):
    pcd = trimesh.load(pcd_path)
    return Points(np.asarray(pcd.vertices))


def show_geoms(geoms):
    scene = trimesh.Scene()
    for g in geoms:
        scene.add_geometry(g)
    scene.show()


rest_mesh_path = './resources/init/rest_repaired.obj'
pc_path = './resources/pcd/156.ply'
rest_mesh = read_mesh(rest_mesh_path)
coarse_mesh = read_mesh('./resources/init/ClothMesh_156.obj')
init_pc = read_points('./resources/init/init.ply')
points = read_points(pc_path)
current_mesh = rest_mesh.clone()
show_geoms([trimesh.load_mesh(rest_mesh_path)])

with open('./resources/init/rest_verts_mapping.json', 'r') as f:
    anchor_inds = json.load(f)
energies = [L2Energy(anchor_inds, 0.1)]
optimizer = Optimizer(rest_mesh, current_mesh, 3e-3, 1e4,
                      1.0, 1., energies, 2e-2)
optimizer.set_coarse_mesh(coarse_mesh)
optimizer.solve(128, 3)

post_energies = [Mesh2PointEnergy(2.), Point2MeshEnergy(2.)]
post_optimizer = Optimizer(rest_mesh, current_mesh,
                           3e-3, 1e5, 10., 1., post_energies, 2e-2)
post_optimizer.set_pcd(init_pc)
post_optimizer.solve(128, 3)
res_mesh = trimesh.Trimesh(
    np.asarray(current_mesh.vertices),
    np.asarray(current_mesh.triangles))
pc = trimesh.load('./resources/pcd/156.ply')
show_geoms([res_mesh, pc])
