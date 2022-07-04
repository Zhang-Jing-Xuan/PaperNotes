# coding:utf-8
from cv2 import line
import open3d as o3d
import numpy as np
import os
from plyfile import PlyData, PlyElement

chair_list=[7934,6845,7832,7633,7419,7569,7614,7953,7034,4218,6121,4512,5051]

def custom_draw_geometry_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(0, 10.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd,olineset],
                                                              rotate_view)

def read_mesh(filename):
    """ read XYZ for each vertex.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = len(chair_list)
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x'][chair_list]
        vertices[:, 1] = plydata['vertex'].data['y'][chair_list]
        vertices[:, 2] = plydata['vertex'].data['z'][chair_list]
        vertices[:, 3] = plydata['vertex'].data['red'][chair_list]
        vertices[:, 4] = plydata['vertex'].data['green'][chair_list]
        vertices[:, 5] = plydata['vertex'].data['blue'][chair_list]
    return vertices[:,0:3]

def pbl(vertics):
    opoints=o3d.utility.Vector3dVector(vertics)
    box=o3d.geometry.OrientedBoundingBox.create_from_points(opoints)
    lineset=o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
    return opoints,box,lineset

def Visualize():
    v=o3d.visualization
    vis=v.Visualizer()
    # vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd=o3d.io.read_point_cloud(filename)
    vis.add_geometry(pcd)
    vis.add_geometry(olineset)
    # v.RenderOption.line_width=5.0
    # render_option = vis.get_render_option()
    # render_option.point_size = 3

    # ctr = vis.get_view_control()
    # print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    # ctr.change_field_of_view(step=-40)
    # print("Field of view (after changing) %.2f" % ctr.get_field_of_view())

    # custom_draw_geometry_with_rotation(pcd)
    vis.run()
    vis.destroy_window()

ROOT = "C:\\Users\\zjx61\\Desktop\\scene0565"
filename = os.path.join(ROOT, "scene0565_00_vh_clean_2.ply")
vertics = read_mesh(filename)
opoints,obox,olineset=pbl(vertics)
color = [[1, 0, 0] for i in range(12)]
olineset.colors = o3d.utility.Vector3dVector(color) 
Visualize()