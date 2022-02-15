import torch
import numpy as np

# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


# Ray helpers
def get_rays_camera(B, H, W, fx, fy,  cx, cy, depth_type, convention="opencv"):

    assert depth_type is "z" or depth_type is "euclidean"
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H))  # pytorch's meshgrid has indexing='ij', we transpose to "xy" moode

    i = i.t().float()
    j = j.t().float()

    size = [B, H, W]

    i_batch = torch.empty(size)
    j_batch = torch.empty(size)
    i_batch[:, :, :] = i[None, :, :]
    j_batch[:, :, :] = j[None, :, :]

    if convention == "opencv":
        x = (i_batch - cx) / fx
        y = (j_batch - cy) / fy
        z = torch.ones(size)
    elif convention == "opengl":
        x = (i_batch - cx) / fx
        y = -(j_batch - cy) / fy
        z = -torch.ones(size)
    else:
        assert False

    dirs = torch.stack((x, y, z), dim=3)  # shape of [B, H, W, 3]

    if depth_type == 'euclidean':
        norm = torch.norm(dirs, dim=3, keepdim=True)
        dirs = dirs * (1. / norm)

    return dirs


def get_rays_world(T_WC, dirs_C):
    R_WC = T_WC[:, :3, :3]  # Bx3x3
    dirs_W = torch.matmul(R_WC[:, None, ...], dirs_C[..., None]).squeeze(-1)
    origins = T_WC[:, :3, -1]  # Bx3
    origins = torch.broadcast_tensors(origins[:, None, :], dirs_W)[0]
    return origins, dirs_W


def get_rays_camera_np(B, H, W, fx, fy,  cx, cy, depth_type, convention="opencv"):
    assert depth_type is "z" or depth_type is "euclidean"
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')  # pytorch's meshgrid has default indexing='ij'

    size = [B, H, W]

    i_batch = np.empty(size, dtype=np.float32)
    j_batch = np.empty(size, dtype=np.float32)
    i_batch[:, :, :] = i[None, :, :]
    j_batch[:, :, :] = j[None, :, :]

    if convention == "opencv":
        x = (i_batch - cx) / fx
        y = (j_batch - cy) / fy
        z = np.ones(size, dtype=np.float32)
    elif convention == "opengl":
        x = (i_batch - cx) / fx
        y = -(j_batch - cy) / fy
        z = -np.ones(size, dtype=np.float32)
    else:
        assert False

    dirs = np.stack((x, y, z), axis=3)  # shape of [B, H, W, 3]

    if depth_type == 'euclidean':
        norm = np.norm(dirs, axis=3, keepdim=True)
        dirs = dirs * (1. / norm)

    return dirs


def get_rays_world_np(T_WC, dirs_C):
    R_WC = T_WC[:, :3, :3]  # Bx3x3
    dirs_W = (R_WC * dirs_C[..., np.newaxis, :]).sum(axis=-1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # sum([B,3,3] * [B, H, W, 1, 3], axis=-1)  -->  [B, H, W, 3]
    origins = T_WC[:, :3, -1]  # Bx3

    return origins, dirs_W


def ndc_rays(H, W, focal, near, rays_o, rays_d):

    # Shift ray origins to near plane
    # solves for the t value such that o + t * d = -near
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def stratified_bins(min_depth,
                    max_depth,
                    n_bins,
                    n_rays,
                    device):

    bin_limits = torch.linspace(
        min_depth,
        max_depth,
        n_bins + 1,
        device=device,
    )
    lower_limits = bin_limits[:-1]
    bin_length = (max_depth - min_depth) / (n_bins)
    increments = torch.rand(n_rays, n_bins, device=device) * bin_length
    z_vals = lower_limits[None, :] + increments

    return z_vals


def sampling_index(n_rays, batch_size, h, w):

    index_b = np.random.choice(np.arange(batch_size)).reshape((1, 1))  # sample one image from the full trainiing set
    index_hw = torch.randint(0, h * w, (1, n_rays))

    return index_b, index_hw


# Hierarchical sampling using inverse CDF transformations
def sample_pdf(bins, weights, N_samples, det=False):
    """ Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: N_rays x (N_samples_coarse - 1)
        weights: N_rays x (N_samples_coarse - 2)
        N_samples: N_samples_fine
        det: deterministic or not
    """
    # Get pdf
    weights = weights + 1e-5  # prevent nans, prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)  # N_rays x (N_samples - 2)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # N_rays x (N_samples_coarse - 1)
    # padded to 0~1 inclusive, (N_rays, N_samples-1)

    # Take uniform samples
    if det:  # generate deterministic samples
        u = torch.linspace(0., 1., steps=N_samples,  device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples],  device=bins.device)
        # (N_rays, N_samples_fine)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf.detach(), u, right=True)  # N_rays x N_samples_fine
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (N_rays, N_samples_fine, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]  # (N_rays, N_samples_fine, N_samples_coarse - 1)

    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)  # N_rays, N_samples_fine, 2
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)  # N_rays, N_samples_fine, 2

    denom = (cdf_g[..., 1]-cdf_g[..., 0])  # # N_rays, N_samples_fine
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


def create_rays(num_rays, Ts_c2w, height, width, fx, fy, cx, cy, near, far, c2w_staticcam=None, depth_type="z",
              use_viewdirs=True, convention="opencv"):
    """
    convention: 
    "opencv" or "opengl". It defines the coordinates convention of rays from cameras.
    OpenCv defines x,y,z as right, down, forward while OpenGl defines x,y,z as right, up, backward (camera looking towards forward direction still, -z!)
    Note: Use either convention is fine, but the corresponding pose should follow the same convention.

    """
    print('prepare rays')

    rays_cam = get_rays_camera(num_rays, height, width, fx, fy, cx, cy, depth_type=depth_type, convention=convention) # [N, H, W, 3]

    dirs_C = rays_cam.view(num_rays, -1, 3)  # [N, HW, 3]
    rays_o, rays_d = get_rays_world(Ts_c2w, dirs_C)  # origins: [B, HW, 3], dirs_W: [B, HW, 3]

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # c2w_staticcam: If not None, use this transformation matrix for camera,
            # while using other c2w argument for viewing directions.
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays_world(c2w_staticcam, dirs_C)  # origins: [B, HW, 3], dirs_W: [B, HW, 3]

        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    return rays

