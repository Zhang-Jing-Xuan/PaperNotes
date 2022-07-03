# coding:utf-8
from cv2 import line
import open3d as o3d
import numpy as np
import os
from plyfile import PlyData, PlyElement

chair_list=[7934,6845,7832,7633,7419,7569,7614,7953,7034,4218,6121,4512,5051]
statue_list=[113953,113287,109335,110761,111827,115150,111344,112827,114749,108844,110997,110680,113267,114490,111887,109765,110892,113068,109491,111720,112235,111272,114251,114336,105386,112265,115100,112916,109711,112372,113922,114287]

def read_mesh(filename,list_):
    """ read XYZ for each vertex.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = len(chair_list)
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x'][list_]
        vertices[:, 1] = plydata['vertex'].data['y'][list_]
        vertices[:, 2] = plydata['vertex'].data['z'][list_]
        vertices[:, 3] = plydata['vertex'].data['red'][list_]
        vertices[:, 4] = plydata['vertex'].data['green'][list_]
        vertices[:, 5] = plydata['vertex'].data['blue'][list_]
    return vertices[:,0:3]

def pbl(vertics):
    opoints=o3d.utility.Vector3dVector(vertics)
    obox=o3d.geometry.OrientedBoundingBox.create_from_points(opoints)
    olineset=o3d.geometry.LineSet.create_from_oriented_bounding_box(obox)
    return opoints,obox,olineset

def Visualize():
    v=o3d.visualization
    vis=v.Visualizer()
    # vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd=o3d.io.read_point_cloud(filename)
    vis.add_geometry(pcd)
    vis.add_geometry(olineset)
    v.RenderOption.line_width=5.0
    render_option = vis.get_render_option()
    render_option.point_size = 3
    vis.run()
    vis.destroy_window()

ROOT = "C:\\Users\\zjx61\\Desktop\\scene0565"
filename = os.path.join(ROOT, "scene0565_00_vh_clean_2.ply")
vertics = read_mesh(filename,statue_list)
opoints,obox,olineset=pbl(vertics)
color = [[1, 0, 0] for i in range(12)]
olineset.colors = o3d.utility.Vector3dVector(color) 
Visualize()