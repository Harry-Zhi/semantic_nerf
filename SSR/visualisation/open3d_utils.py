import open3d as o3d
import numpy as np


def draw_segment(t1, t2, color=(1., 1., 0.)):
    points = [t1, t2]

    lines = [[0, 1]]

    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set # line-segment


def draw_trajectory(scene, transform_wc, color=(1., 1., 0.), name="trajectory"):
    for i in range(trajectory.shape[0] - 1):
        t1 = transform_wc[i, :3, 3]
        t2 = transform_wc[i+1, :3, 3]
        segment = draw_segment(t1, t2, color)
        scene.scene.add_geometry("{}_{}".format(name, i), segment, material)
        scene.force_redraw()

def draw_camera_frustrums(scene, material, intrinsics, transform_wc, scale=1.0, color=(1, 0, 0), name="camera"):
    for i in range(len(transform_wc)):
        camera_frustum = gen_camera_frustrum(intrinsics, transform_wc[i])
        scene.scene.add_geometry("{}_{}".format(name, i), camera_frustum, material)
        scene.force_redraw()


def gen_camera_frustrum(intrinsics, transform_wc, scale=1.0, color=(1, 0, 0)):
    """
    intrinsics: camera intrinsic matrix
    scale: the depth of the frustum front plane
    color: frustum line colours
    """
    print("Draw camera frustum using o3d.geometry.LineSet.")
    w = intrinsics['cx'] * 2
    h = intrinsics['cy'] * 2
    xl = scale * -intrinsics['cx'] / intrinsics['fx'] # 3D coordinate of minimum x
    xh = scale * (w - intrinsics['cx']) / intrinsics['fx'] # 3D coordinate of maximum x
    yl = scale * -intrinsics['cy'] / intrinsics['fy'] # 3D coordinate of minimum y
    yh = scale * (h - intrinsics['cy']) / intrinsics['fy'] # 3D coordinate of maximum y
    verts = [
            0, 0, 0, # 0 - camera center
            xl, yl, scale, # 1 - upper left
            xh, yl, scale,  # 2 - upper right
            xh, yh, scale,  # 3 - bottom right
            xl, yh, scale,  # 4 - bottom leff
            ]
    
    lines = [
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [1, 2],
    [1, 4],
    [3, 2],
    [3, 4],
    ]

    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    line_set = line_set.transform(transform_wc)
    return line_set # camera frustun



def integrate_rgbd_tsdf(tsdf_volume, rgb, dep, depth_trunc, T_wc, intrinsic):
    for i in range(0, len(T_wc)):
        print("Integrate {:d}-th image into the volume.".format(i))
        color = o3d.geometry.Image(rgb[i])
        depth = o3d.geometry.Image(dep[i])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_trunc=depth_trunc,
            depth_scale=1,
            convert_rgb_to_intensity=False,
        )

        T_cw = np.linalg.inv(T_wc[i])

        tsdf_volume.integrate(
            image=rgbd,
            intrinsic=intrinsic,
            extrinsic=T_cw,
        )
    return tsdf_volume

def tsdf2mesh(tsdf):
        mesh = tsdf.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh



def integrate_dep_pcd(dep, T_wc, intrinsic):
    # http://www.open3d.org/docs/latest/tutorial/Advanced/multiway_registration.html#Make-a-combined-point-cloud
    
    pcd_list = []
    pcd_combined = o3d.geometry.PointCloud()
    for i in range(0, len(T_wc)):
        depth = o3d.geometry.Image(dep[i])
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_map,
            intrinsic,
            depth_scale=1,
            stride=1,
            project_valid_depth_only=True)
        pcd.transform(T_WC[i])
        pcd_combined+= pcd

    # pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.02)
    print("Merge point clouds from multiple views.")
    # Flip it, otherwise the pointcloud will be upside down
    pcd_combined.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd_combined],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
    return pcd_combined


def draw_pc(batch_size,
            pcs_cam,
            T_WC_batch_np,
            im_batch=None,
            scene=None):

    pcs_w = []
    for batch_i in range(batch_size):
        T_WC = T_WC_batch_np[batch_i]
        pc_cam = pcs_cam[batch_i]

        col = None
        if im_batch is not None:
            img = im_batch[batch_i]
            col = img.reshape(-1, 3)

        pc_tri = trimesh.PointCloud(vertices=pc_cam, colors=col)
        pc_tri.apply_transform(T_WC)
        pcs_w.append(pc_tri.vertices)

        if scene is not None:
            scene.add_geometry(pc_tri)

    pcs_w = np.concatenate(pcs_w, axis=0)
    return pcs_w



def trimesh_to_open3d(src):
    dst = o3d.geometry.TriangleMesh()
    dst.vertices = o3d.utility.Vector3dVector(src.vertices)
    dst.triangles = o3d.utility.Vector3iVector(src.faces)
    vertex_colors = src.visual.vertex_colors[:, :3].astype(np.float) / 255.0
    dst.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    dst.compute_vertex_normals()

    return dst


def clean_mesh(o3d_mesh, keep_single_cluster=False, min_num_cluster=200):
    import copy

    o3d_mesh_clean = copy.deepcopy(o3d_mesh)
    # http://www.open3d.org/docs/release/tutorial/geometry/mesh.html?highlight=cluster_connected_triangles
    triangle_clusters, cluster_n_triangles, cluster_area = o3d_mesh_clean.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    if keep_single_cluster:
        # keep the largest cluster.!
        largest_cluster_idx = np.argmax(cluster_n_triangles)
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        o3d_mesh_clean.remove_triangles_by_mask(triangles_to_remove)
        o3d_mesh_clean.remove_unreferenced_vertices()
        print("Show mesh with largest cluster kept")
    else:
        # remove small clusters
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_num_cluster
        o3d_mesh_clean.remove_triangles_by_mask(triangles_to_remove)
        o3d_mesh_clean.remove_unreferenced_vertices()
        print("Show mesh with small clusters removed")


    return o3d_mesh_clean