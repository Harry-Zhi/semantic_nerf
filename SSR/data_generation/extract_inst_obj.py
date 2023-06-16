# reference https://github.com/facebookresearch/Replica-Dataset/issues/17#issuecomment-538757418

from plyfile import *
import numpy as np
import trimesh


# path_in = 'path/to/mesh_semantic.ply'
path_in = '/home/xin/data/vmap/room_0_debug/habitat/mesh_semantic.ply'

print("Reading input...")
mesh = trimesh.load(path_in)
# mesh.show()
file_in = PlyData.read(path_in)
vertices_in = file_in.elements[0]
faces_in = file_in.elements[1]

print("Filtering data...")
objects = {}
sub_mesh_indices = {}
for i, f in enumerate(faces_in):
     object_id = f[1]
     if not object_id in objects:
         objects[object_id] = []
         sub_mesh_indices[object_id] = []
     objects[object_id].append((f[0],))
     sub_mesh_indices[object_id].append(i)
     sub_mesh_indices[object_id].append(i+faces_in.data.shape[0])


print("Writing data...")
for object_id, faces in objects.items():
    path_out = path_in + f"_{object_id}.ply"
    # print("sub_mesh_indices[object_id] ", sub_mesh_indices[object_id])
    obj_mesh = mesh.submesh([sub_mesh_indices[object_id]], append=True)
    in_n = len(sub_mesh_indices[object_id])
    out_n = obj_mesh.faces.shape[0]
    # print("obj id ", object_id)
    # print("in_n ", in_n)
    # print("out_n ", out_n)
    # print("faces ", len(faces))
    # assert in_n == out_n
    obj_mesh.export(path_out)
    # faces_out = PlyElement.describe(np.array(faces, dtype=[('vertex_indices', 'O')]), 'face')
    # print("faces out ", len(PlyData([vertices_in, faces_out]).elements[1].data))
    # PlyData([vertices_in, faces_out]).write(path_out+"_cmp.ply")

