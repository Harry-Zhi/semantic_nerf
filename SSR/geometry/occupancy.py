import torch
import numpy as np


def grid_within_bound(occ_range, extents, transform, grid_dim):
        range_dist = occ_range[1] - occ_range[0]
        bounds_tranform_np = transform

        bounds_tranform = torch.from_numpy(bounds_tranform_np).float()
        scene_scale_np = extents / (range_dist * 0.9)
        scene_scale = torch.from_numpy(scene_scale_np).float()

        # todo: only make grid once, then only transform!
        grid_pc = make_3D_grid(
            occ_range,
            grid_dim,
            transform=bounds_tranform,
            scale=scene_scale,
        )
        grid_pc = grid_pc.view(-1, 1, 3)

        return grid_pc, scene_scale

def make_3D_grid(occ_range, dim, transform=None, scale=None):
    t = torch.linspace(occ_range[0], occ_range[1], steps=dim)
    grid = torch.meshgrid(t, t, t)
    grid_3d_norm = torch.cat(
        (grid[0][..., None],
         grid[1][..., None],
         grid[2][..., None]), dim=3
    )

    if scale is not None:
        grid_3d = grid_3d_norm * scale
    if transform is not None:
        R1 = transform[None, None, None, 0, :3]
        R2 = transform[None, None, None, 1, :3]
        R3 = transform[None, None, None, 2, :3]

        grid1 = (R1 * grid_3d).sum(-1, keepdim=True)
        grid2 = (R2 * grid_3d).sum(-1, keepdim=True)
        grid3 = (R3 * grid_3d).sum(-1, keepdim=True)
        grid_3d = torch.cat([grid1, grid2, grid3], dim=-1)

        trans = transform[None, None, None, :3, 3]
        grid_3d = grid_3d + trans

    return grid_3d

def make_3D_grid_np(occ_range, dim, device, transform=None, scale=None):
    t = torch.linspace(occ_range[0], occ_range[1], steps=dim, device=device)
    t = np.linspace(occ_range[0], occ_range[1], num=dim)
    grid = np.meshgrid(t, t, t) # list of 3 elements of shape [dim, dim, dim]

    grid_3d_norm = np.concatenate(
        (grid[0][..., None],
         grid[1][..., None],
         grid[2][..., None]), axis=3
    ) # shape of [dim, dim, dim, 3]

    if scale is not None:
        grid_3d = grid_3d_norm * scale
    if transform is not None:
        R1 = transform[None, None, None, 0, :3]
        R2 = transform[None, None, None, 1, :3]
        R3 = transform[None, None, None, 2, :3]

        grid1 = (R1 * grid_3d).sum(-1, keepdim=True)
        grid2 = (R2 * grid_3d).sum(-1, keepdim=True)
        grid3 = (R3 * grid_3d).sum(-1, keepdim=True)
        grid_3d = np.concatenate([grid1, grid2, grid3], dim=-1)

        trans = transform[None, None, None, :3, 3]
        grid_3d = grid_3d + trans

    return grid_3d



def chunk_alphas(pc, chunk_size, fc_occ_map, n_embed_funcs, B_layer,):
    n_pts = pc.shape[0]
    n_chunks = int(np.ceil(n_pts / chunk_size))
    alphas = []
    for n in range(n_chunks):
        start = n * chunk_size
        end = start + chunk_size
        chunk = pc[start:end, :]
        points_embedding = embedding.positional_encoding(
            chunk, B_layer, num_encoding_functions=n_embed_funcs
        )
        alpha = fc_occ_map(points_embedding, full=True).squeeze(dim=-1)
        alphas.append(alpha)
    alphas = torch.cat(alphas, dim=-1)

    return alphas
