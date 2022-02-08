import torch
import torch.nn.functional as F
from SSR.training.training_utils import batchify


def run_network_compund(inputs, fn, embed_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.

    Input: [N_rays, N_samples, 3]
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    compund_fn = lambda x: fn(embed_fn(x))

    outputs_flat = batchify(compund_fn, netchunk)(inputs_flat)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.

    Input: [N_rays, N_samples, 3]
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs



def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, enable_semantic=True, 
                num_sem_class=0, endpoint_feat=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
        raw_noise_std: random perturbations added to ray samples
        
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]  # # (N_rays, N_samples_-1)
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).cuda()], -1)  # [N_rays, N_samples]

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std
        noise = noise.cuda()
    else:
        noise = 0.

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]


    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    # [1, 1-a1, 1-a2, ...]
    # [N_rays, N_samples+1] sliced by [:, :-1] to [N_rays, N_samples]

    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    # [N_rays, 3], the accumulated opacity along the rays, equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

    if enable_semantic:
        assert num_sem_class>0
        # https://discuss.pytorch.org/t/multi-class-cross-entropy-loss-and-softmax-in-pytorch/24920/2
        sem_logits = raw[..., 4:4+num_sem_class]  # [N_rays, N_samples, num_class]
        sem_map = torch.sum(weights[..., None] * sem_logits, -2)  # [N_rays, num_class]
    else:
        sem_map = torch.tensor(0)


    if endpoint_feat:
        feat = raw[..., -128:] # [N_rays, N_samples, feat_dim] take the last 128 dim from predictions
        feat_map = torch.sum(weights[..., None] * feat, -2)  # [N_rays, feat_dim]
    else:
        feat_map = torch.tensor(0)

    depth_map = torch.sum(weights * z_vals, -1)  # (N_rays,)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[..., None])
        if enable_semantic:
            sem_map = sem_map + (1.-acc_map[..., None])
    
    return rgb_map, disp_map, acc_map, weights, depth_map, sem_map, feat_map

