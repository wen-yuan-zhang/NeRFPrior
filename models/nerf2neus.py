import trimesh
import mcubes
import torch
from models.tensoRF import TensorVMSplit
import numpy as np
import skimage
import plyfile
import os
from torch import nn
import torch.nn.functional as F


class PriorNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """
        """
        super(PriorNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim



def query_alpha_color_nerf(nerf: PriorNeRF, _xyz_sampled, viewdirs=None, z_vals=None):
    # xyz_sampled: [N, 3]
    # viewdirs: [N_rays, 3]
    xyz_sampled = _xyz_sampled * 2
    batch_size, n_samples, _ = xyz_sampled.shape
    inputs_flat = torch.reshape(xyz_sampled, [-1, 3])
    embedded = nerf.embed_fn(inputs_flat)

    if viewdirs is None:
        viewdirs = torch.zeros(batch_size,3,dtype=torch.float32).cuda()
    input_dirs = viewdirs[:, None].expand(xyz_sampled.shape)
    input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
    embedded_dirs = nerf.embeddirs_fn(input_dirs_flat)
    embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = nerf(embedded)

    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, dists[0][0].expand(dists[..., :1].shape)], -1)

    # dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    outputs_flat = outputs_flat.reshape(batch_size, n_samples, -1)
    alpha = raw2alpha(outputs_flat[..., 3], dists)  # [N_rays, N_samples]

    return alpha, None

def query_density_nerf(nerf, _xyz_sampled):
    # xyz_sampled: [N, 3]
    # viewdirs: [N_rays, 3]
    xyz_sampled = _xyz_sampled * 2
    batch_size, n_samples, _ = xyz_sampled.shape
    inputs_flat = torch.reshape(xyz_sampled, [-1, 3])
    embedded = nerf.embed_fn(inputs_flat)

    viewdirs = torch.zeros(batch_size, 3, dtype=torch.float32).cuda()
    input_dirs = viewdirs[:, None].expand(xyz_sampled.shape)
    input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
    embedded_dirs = nerf.embeddirs_fn(input_dirs_flat)
    embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = nerf(embedded)

    return outputs_flat[..., 3]


def load_nerf(ckpt_path):
    embed_fn, input_ch = get_embedder(10, 0)
    embeddirs_fn, input_ch_views = get_embedder(4, 0)
    model_fine = PriorNeRF(D=8, W=256, input_ch=input_ch, output_ch=5, skips=[4],
                      input_ch_views=input_ch_views, use_viewdirs=True).cuda()
    print('Reloading from', ckpt_path)
    ckpt = torch.load(ckpt_path)
    model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    model_fine.embed_fn = embed_fn
    model_fine.embeddirs_fn = embeddirs_fn
    return model_fine


def validate_mesh(nerf):
    object_bbox_min = np.array([-1.01, -1.01, -1.01])  # only used in extract
    object_bbox_max = np.array([1.01, 1.01, 1.01])
    bound_min = torch.tensor(object_bbox_min, dtype=torch.float32)
    bound_max = torch.tensor(object_bbox_max, dtype=torch.float32)

    resolution = 256
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_density_nerf(nerf,pts.reshape(1,-1,3)).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val

    vertices, triangles = mcubes.marching_cubes(u, 30)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]

    os.makedirs(os.path.join('debug'), exist_ok=True)
    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.export(os.path.join('debug/nerf.ply'))