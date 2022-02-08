import os
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from SSR.datasets.replica import replica_datasets
from SSR.datasets.scannet import scannet_datasets
from SSR.datasets.replica_nyu import replica_nyu_cnn_datasets
from SSR.datasets.scannet import scannet_datasets
import open3d as o3d

from SSR.training import trainer
from SSR.models.model_utils import run_network
from SSR.geometry.occupancy import grid_within_bound
from SSR.visualisation import open3d_utils
import numpy as np
import yaml
import json

import skimage.measure as ski_measure
import time
from imgviz import label_colormap
import trimesh


@torch.no_grad()
def render_fn(trainer, rays, chunk):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            trainer.render_rays(rays[i:i+chunk])

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="/home/shuaifeng/Documents/PhD_Research/CodeRelease/SemanticSceneRepresentations/SSR/configs/SSR_room0_config_test.yaml", help='config file name.')

    parser.add_argument('--mesh_dir', type=str, required=True, help='Path to scene file, e.g., ROOT_PATH/Replica/mesh/room_0/')
    parser.add_argument('--training_data_dir', type=str, required=True, help='Path to rendered data.')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to the directory saving training logs and ckpts.')

    parser.add_argument('--use_vertex_normal', action="store_true", help='use vertex normals to compute color')
    parser.add_argument('--near_t', type=float, default=2.0, help='the near bound factor to start the ray')
    parser.add_argument('--sem', action="store_true")
    parser.add_argument('--grid_dim', type=int, default=256)
    parser.add_argument('--gpu', type=str, default="", help='GPU IDs.')



    args = parser.parse_args()

    config_file_path = args.config_file

    # Read YAML file
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    if len(args.gpu)>0:
        config["experiment"]["gpu"] = args.gpu
    print("Experiment GPU is {}.".format(config["experiment"]["gpu"]))
    trainer.select_gpus(config["experiment"]["gpu"])
    

    to8b_np = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)
        
    # Cast intrinsics to right types
    ssr_trainer = trainer.SSRTrainer(config)


    near_t = args.near_t
    mesh_dir = args.mesh_dir
    training_data_dir = args.training_data_dir
    save_dir = args.save_dir
    mesh_recon_save_dir = os.path.join(save_dir, "mesh_reconstruction")
    os.makedirs(mesh_recon_save_dir, exist_ok=True)


    info_mesh_file = os.path.join(mesh_dir, "habitat", "info_semantic.json")
    with open(info_mesh_file, "r") as f:
        annotations = json.load(f)
        
    instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])
    instance_id_to_semantic_label_id[instance_id_to_semantic_label_id<=0] = 0
    semantic_classes = np.unique(instance_id_to_semantic_label_id)
    num_classes = len(semantic_classes) # including void class--0
    label_colour_map = label_colormap()[semantic_classes]
    valid_colour_map = label_colour_map[1:]

    total_num = 900
    step = 5
    ids = list(range(total_num))
    train_ids = list(range(0, total_num, step))
    test_ids = [x+2 for x in train_ids]    

    replica_data_loader = replica_datasets.ReplicaDatasetCache(data_dir=training_data_dir,
                                                                train_ids=train_ids, test_ids=test_ids,
                                                                img_h=config["experiment"]["height"],
                                                                img_w=config["experiment"]["width"])

    ssr_trainer.set_params_replica()
    ssr_trainer.prepare_data_replica(replica_data_loader)

    ##########################

    # Create nerf model, init optimizer
    ssr_trainer.create_ssr()
    # Create rays in world coordinates
    ssr_trainer.init_rays()

    # load_ckpt into NeRF
    ckpt_path = os.path.join(save_dir, "checkpoints", "200000.ckpt")
    print('Reloading from', ckpt_path)
    ckpt = torch.load(ckpt_path)

    start = ckpt['global_step']
    ssr_trainer.ssr_net_coarse.load_state_dict(ckpt['network_coarse_state_dict'])
    ssr_trainer.ssr_net_fine.load_state_dict(ckpt['network_fine_state_dict'])
    ssr_trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    ssr_trainer.training = False  # enable testing mode before rendering results, need to set back during training!
    ssr_trainer.ssr_net_coarse.eval()
    ssr_trainer.ssr_net_fine.eval()


    level = 0.45 # level = 0
    threshold = 0.2
    draw_cameras = True
    grid_dim = args.grid_dim
            
    train_Ts_np =  replica_data_loader.train_samples["T_wc"]
    mesh_file = os.path.join(mesh_dir,"mesh.ply")
    assert os.path.exists(mesh_file)

    trimesh_scene = trimesh.load(mesh_file, process=False)

    to_origin_transform, extents = trimesh.bounds.oriented_bounds(trimesh_scene)
    T_extent_to_scene = np.linalg.inv(to_origin_transform)
    scene_transform = T_extent_to_scene
    scene_extents = extents
    grid_query_pts, scene_scale = grid_within_bound([-1.0, 1.0], scene_extents, scene_transform, grid_dim=grid_dim)

    grid_query_pts = grid_query_pts.cuda().reshape(-1,1,3) # Num_rays, 1, 3-xyz
    viewdirs = torch.zeros_like(grid_query_pts).reshape(-1, 3) 
    st = time.time()
    print("Initialise Trimesh Scenes")

    with torch.no_grad():
        chunk = 1024
        run_MLP_fn  =  lambda pts: run_network(inputs=pts, viewdirs=torch.zeros_like(pts).squeeze(1), 
            fn=ssr_trainer.ssr_net_fine, embed_fn=ssr_trainer.embed_fn,
            embeddirs_fn=ssr_trainer.embeddirs_fn, netchunk=int(2048*128))

        raw = torch.cat([run_MLP_fn(grid_query_pts[i: i+chunk]).cpu() for i in range(0, grid_query_pts.shape[0], chunk)], dim=0)
        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        alpha = raw[..., 3] # [N]
        sem_logits = raw[..., 4:]  # [N_rays, N_samples, num_class]
        label_fine = logits_2_label(sem_logits).view(-1).cpu().numpy()
        vis_label_colour = label_colour_map[label_fine+1]

    print("Finish Computing Semantics!")
    print()

    def occupancy_activation(alpha, distances):
        occ = 1.0 - torch.exp(-F.relu(alpha) * distances)
        # notice we apply RELU to raw sigma before computing alpha
        return occ

    # voxel_size = (ssr_trainer.far - ssr_trainer.near) / grid_dim # or self.N_importance
    voxel_size = (ssr_trainer.far - ssr_trainer.near) / ssr_trainer.N_importance # or self.N_importance
    occ = occupancy_activation(alpha, voxel_size)
    print("Compute Occupancy Grids")
    occ = occ.reshape(grid_dim, grid_dim, grid_dim)
    occupancy_grid = occ.detach().cpu().numpy()

    print('fraction occupied:', (occupancy_grid > threshold).mean())
    print('Max Occ: {}, Min Occ: {}, Mean Occ: {}'.format(occupancy_grid.max(), occupancy_grid.min(), occupancy_grid.mean()))
    vertices, faces, vertex_normals, _ = ski_measure.marching_cubes(occupancy_grid, level=level, gradient_direction='ascent')
    print()

    dim = occupancy_grid.shape[0]
    vertices = vertices / (dim - 1)
    mesh = trimesh.Trimesh(vertices=vertices, vertex_normals=vertex_normals, faces=faces)

    # Transform to [-1, 1] range
    mesh_canonical = mesh.copy()
    mesh_canonical.apply_translation([-0.5, -0.5, -0.5])
    mesh_canonical.apply_scale(2)

    scene_scale = scene_extents/2.0
    # Transform to scene coordinates
    mesh_canonical.apply_scale(scene_scale)
    mesh_canonical.apply_transform(scene_transform)
    # mesh.show()
    exported = trimesh.exchange.export.export_mesh(mesh_canonical, os.path.join(mesh_recon_save_dir, 'mesh_canonical.ply'))
    print("Saving Marching Cubes mesh to mesh_canonical.ply !")
    exported = trimesh.exchange.export.export_mesh(mesh_canonical, os.path.join(mesh_recon_save_dir, 'mesh.ply'))
    print("Saving Marching Cubes mesh to mesh.ply !")


    o3d_mesh = open3d_utils.trimesh_to_open3d(mesh)
    o3d_mesh_canonical  = open3d_utils.trimesh_to_open3d(mesh_canonical)

    print('Removing noise ...')
    print(f'Original Mesh has {len(o3d_mesh_canonical.vertices)/1e6:.2f} M vertices and {len(o3d_mesh_canonical.triangles)/1e6:.2f} M faces.')
    o3d_mesh_canonical_clean = open3d_utils.clean_mesh(o3d_mesh_canonical, keep_single_cluster=False, min_num_cluster=400)

    vertices_ = np.array(o3d_mesh_canonical_clean.vertices).reshape([-1, 3]).astype(np.float32)
    triangles = np.asarray(o3d_mesh_canonical_clean.triangles) # (n, 3) int
    N_vertices = vertices_.shape[0]
    print(f'Denoised Mesh has {len(o3d_mesh_canonical_clean.vertices)/1e6:.2f} M vertices and {len(o3d_mesh_canonical_clean.triangles)/1e6:.2f} M faces.')

    print("###########################################")
    print()
    print("Using Normals for colour predictions!")
    print()
    print("###########################################")
    
    ## use normal vector method as suggested by the author, see https://github.com/bmild/nerf/issues/44
    mesh_recon_save_dir = os.path.join(mesh_recon_save_dir,"use_vertex_normal")
    os.makedirs(mesh_recon_save_dir, exist_ok=True)

    selected_mesh = o3d_mesh_canonical_clean
    rays_d = - torch.FloatTensor(np.asarray(selected_mesh.vertex_normals)) # use negative normal directions as ray marching directions
    near = 0.1  * torch.ones_like(rays_d[:, :1])
    far = 10.0 * torch.ones_like(rays_d[:, :1])
    rays_o = torch.FloatTensor(vertices_) - rays_d * near * args.near_t
    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True).float()
    rays = torch.cat([rays_o, rays_d, near, far, viewdirs], -1)

    # provide ray directions as input
    rays = rays.cuda()
    with torch.no_grad():
        chunk=4096
        # chunk=80*1024
        results = render_fn(ssr_trainer, rays, chunk)

        # combine the output and write to file
        if args.sem:
            labels = logits_2_label(results["sem_logits_fine"]).numpy()
            vis_labels =  valid_colour_map[labels]
            v_colors  =  vis_labels
        else:
            rgbs = results["rgb_fine"].numpy()
            rgbs = to8b_np(rgbs)
            v_colors  =  rgbs

    v_colors = v_colors.astype(np.uint8)


    o3d_mesh_canonical_clean.vertex_colors = o3d.utility.Vector3dVector(v_colors/255.0)

    if args.sem:
        o3d.io.write_triangle_mesh(os.path.join(mesh_recon_save_dir, 'semantic_mesh_canonical_dim{}neart_{}.ply'.format(grid_dim, near_t)), o3d_mesh_canonical_clean)
        print("Saving Marching Cubes mesh to semantic_mesh_canonical_dim{}neart_{}.ply".format(grid_dim, near_t))
    else:
        o3d.io.write_triangle_mesh(os.path.join(mesh_recon_save_dir, 'colour_mesh_canonical_dim{}neart_{}.ply'.format(grid_dim, near_t)), o3d_mesh_canonical_clean)
        print("Saving Marching Cubes mesh to colour_mesh_canonical_dim{}neart_{}.ply".format(grid_dim, near_t))

    print('Done!')


if __name__=='__main__':
    train()