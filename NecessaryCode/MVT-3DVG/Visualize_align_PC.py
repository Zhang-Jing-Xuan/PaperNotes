# coding:utf-8
import open3d as o3d
import numpy as np
import os
from plyfile import PlyData, PlyElement


# pcd = o3d.io.read_point_cloud("C:\\Users\\zjx61\\Desktop\\scene0565\\scene0565_00_vh_clean_2.labels.ply")
# pcd = o3d.io.read_point_cloud("C:\\Users\\zjx61\\Desktop\\scene0565\\aligned.ply")
# print(pcd)
# print(np.asarray(pcd.points))
# o3d.visualization.draw_geometries([pcd],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])
def read_mesh(filename):
    """ read XYZ for each vertex.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        vertices[:, 3] = plydata['vertex'].data['red']
        vertices[:, 4] = plydata['vertex'].data['green']
        vertices[:, 5] = plydata['vertex'].data['blue']
    return vertices, plydata['face']


def write_mesh(vertices, faces):
    new_vertices = []
    for i in range(vertices.shape[0]):
        new_vertices.append((
            vertices[i][0],
            vertices[i][1],
            vertices[i][2],
            vertices[i][3],
            vertices[i][4],
            vertices[i][5],
        ))
    vertices = np.array(new_vertices,
                        dtype=[("x", np.dtype("float32")),
                               ("y", np.dtype("float32")),
                               ("z", np.dtype("float32")),
                               ("red", np.dtype("uint8")),
                               ("green", np.dtype("uint8")),
                               ("blue", np.dtype("uint8"))])
    vertices = PlyElement.describe(vertices, "vertex")
    mesh = PlyData([vertices, faces])
    mesh.write("C:\\Users\\zjx61\\Desktop\\aligned.ply")

if __name__ == '__main__':
    # 1. 得到 ply ⽂件，分别得到 x, y, z, r, g, b
    ROOT = "C:\\Users\\zjx61\\Desktop\\scene0565"
    filename = os.path.join(ROOT, "scene0565_00_vh_clean_2.ply")
    vertices, faces = read_mesh(filename)
    # 2. 得到偏移
    meta_file = os.path.join(ROOT, "scene0565_00.txt")
    lines = open(meta_file).readlines()
    axis_align_matrix = None
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
    # 3. 将点云 ply ⽂件偏移
    if axis_align_matrix != None:
        axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
        pts = np.ones((vertices.shape[0], 4))
        pts[:, 0:3] = vertices[:, :3]
        pts = np.dot(pts, axis_align_matrix.transpose())
        aligned_vertices = np.copy(vertices)
        aligned_vertices[:, 0:3] = pts[:, 0:3]
    write_mesh(aligned_vertices, faces)


# from six.moves import cPickle
# import numpy as np
# def unpickle_data(file_name, python2_to_3=False):
#     """
#     Restore data previously saved with pickle_data().
#     :param file_name: file holding the pickled data.
#     :param python2_to_3: (boolean), if True, pickle happened under python2x, unpickling under python3x.
#     :return: an generator over the un-pickled items.
#     Note, about implementing the python2_to_3 see
#         https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
#     """
#     in_file = open(file_name, 'rb')
#     if python2_to_3:
#         size = cPickle.load(in_file, encoding='latin1')
#     else:
#         size = cPickle.load(in_file)

#     for _ in range(size):
#         if python2_to_3:
#             yield cPickle.load(in_file, encoding='latin1')
#         else:
#             yield cPickle.load(in_file)
#     in_file.close()

# all_scans = unpickle_data("C:\\Users\\zjx61\\Desktop\\test_resultall_vis.pkl")
# instance_labels = set()
# '''
# dict_keys(['guessed_correctly', 'confidences_probs', 'contrasted_objects', 'target_pos', 'context_size', 'guessed_correctly_among_true_class', 'utterance', 'stimulus_id', 'object_ids', 'target_object_id', 'distrators_pos'])
# '''
# # 48*156
# for scan in all_scans:
#     print(scan[0].keys())
# with open("record.txt","a+") as f:
#     print("distrators_pos",scan[0]['distrators_pos'][0][0],file=f)
#     print("target_pos",scan[0]['target_pos'][0],file=f)
#     print("context_size",scan[0]['context_size'][0],file=f)
#     print("target_object_id",scan[0]['target_object_id'][0],file=f)
#     print("object_ids",scan[0]['object_ids'][0],file=f)
#     print("stimulus_id",scan[0]['stimulus_id'][0][0],file=f)
#     print("utterance",scan[0]['utterance'][0][0],file=f)
