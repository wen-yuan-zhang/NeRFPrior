import torch
from models.tensoRF import TensorVMSplit
import numpy as np
import skimage
import plyfile
import os
# from test_ours import *


def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso)/step_ratio)


def query_alpha_color(tensorf: TensorVMSplit, xyz_sampled, viewdirs=None, prior_mode='cat', requires_grad=False):
    # xyz_sampled: [N, 3]
    # viewdirs: [N_rays, 3]

    dim = 3
    if len(xyz_sampled.shape) == 3:
        N_rays, N_samples, _ = xyz_sampled.shape
        xyz_sampled = xyz_sampled.reshape(-1, 3)
    # if len(xyz_sampled.shape) == 2:
    #     dim = 2
    #     xyz_sampled = xyz_sampled.unsqueeze(0)

    # neighboring 8 vertices
    # after: xyz_samples: [N, 9, 3] or [N, 1, 3]
    if prior_mode.startswith('local'):
        xyz_sampled = query_grid_vertices(tensorf, xyz_sampled)
    else:
        xyz_sampled = xyz_sampled.unsqueeze(1)

    mask_outbbox = ((tensorf.aabb[0] > xyz_sampled) | (xyz_sampled > tensorf.aabb[1])).any(dim=-1)
    ray_valid = ~mask_outbbox

    if tensorf.alphaMask is not None:
        alphas = tensorf.alphaMask.sample_alpha(xyz_sampled[ray_valid])
        alpha_mask = alphas > 0
        ray_invalid = ~ray_valid
        ray_invalid[ray_valid] |= (~alpha_mask)
        ray_valid = ~ray_invalid

    sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)

    if ray_valid.any():
        xyz_sampled = tensorf.normalize_coord(xyz_sampled)
        sigma_feature = tensorf.compute_densityfeature(xyz_sampled[ray_valid], requires_grad=requires_grad)

        validsigma = tensorf.feature2density(sigma_feature)
        sigma[ray_valid] = validsigma

    # alpha, weight, bg_weight = raw2alpha(sigma, tensorf.stepSize * tensorf.distance_scale)
    # app_mask = weight > tensorf.rayMarch_weight_thres
    alpha = 1. - torch.exp(-sigma * tensorf.stepSize * tensorf.distance_scale)
    # alpha = 1-alpha

    if prior_mode == 'local_mean':
        alpha = torch.mean(alpha, dim=1, keepdim=True)

    if viewdirs is None:
        # alpha: [N, 1] or [N, 9]
        return alpha, None


    rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
    viewdirs = viewdirs.reshape(-1, 1, 3).expand(xyz_sampled.shape)
    app_mask = ray_valid
    if app_mask.any():
        app_features = tensorf.compute_appfeature(xyz_sampled[app_mask])
        valid_rgbs = tensorf.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features)
        rgb[app_mask] = valid_rgbs

    # alpha: [N, 1] or [N, 9]
    # rgb: [N, 3] or [N, 9*3]
    if prior_mode == 'local_mean':
        rgb = torch.mean(rgb, dim=1, keepdim=False)
    elif prior_mode == 'local_cat':
        rgb = rgb.reshape(rgb.shape[0], -1)
    else:
        rgb = rgb.reshape(-1, 3)
    return alpha, rgb


def load_tensorf(ckpt, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    ckpt = torch.load(ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = TensorVMSplit(**kwargs)
    tensorf.load(ckpt)

    # 把密集体素网格的坐标点预先加载好
    gridSize = tensorf.gridSize
    samples = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, gridSize[0]),
        torch.linspace(0, 1, gridSize[1]),
        torch.linspace(0, 1, gridSize[2]),
    ), -1).to(tensorf.device)
    dense_xyz = tensorf.aabb[0] * (1 - samples) + tensorf.aabb[1] * samples
    tensorf.dense_xyz = dense_xyz       # [X, Y, Z, 3]

    return tensorf


def construct_alpha_color_grid(tensorf: TensorVMSplit):
    alpha, dense_xyz = tensorf.getDenseAlpha()
    tensorf.alpha_grid = alpha

    rgb = alpha = torch.zeros(dense_xyz.shape[:3]+[3])


def query_grid_vertices(tensorf: TensorVMSplit, pts):
    # pts: [N, 3]
    # return: [N, 9, 3]
    bbox = tensorf.aabb
    pts_normalized = (pts-bbox[0]) / (bbox[1] - bbox[0])        # \in [0, 1]
    pts_gridded = pts_normalized * (tensorf.gridSize-1)        # \in [0, 256]
    pts_lower = torch.floor(pts_gridded)        # [N, 3]
    verts = pts_lower.unsqueeze(1).repeat([1,8,1])
    # x-, y-, z-

    # x+, y-, z-
    verts[:, 1, 0] += 1
    # x+, y+, z-
    verts[:, 2, :2] += 1
    # x-, y+, z-
    verts[:, 3, 1] += 1
    # x-, y-, z+
    verts[:, 4, 2] += 1
    # x+, y-, z+
    verts[:, 5, 0] += 1
    verts[:, 5, 2] += 1
    # x+, y+, z+
    verts[:, 6, :] += 1
    # x-, y+, z+
    verts[:, 7, 1:] += 1

    mask_outbbox = ((pts_gridded <= 0) | (pts_gridded >= (tensorf.gridSize-1))).any(dim=-1)  # [N,]
    mask_outbbox = mask_outbbox.unsqueeze(-1).repeat([1, 8])    # [N, 9]
    mask_inbbox = ~mask_outbbox

    # [N,8,3]
    grid_vertices = pts.unsqueeze(1).repeat([1,8,1])
    verts = verts.long()[mask_inbbox]
    grid_vertices[mask_inbbox] = tensorf.dense_xyz[verts[:,0], verts[:,1], verts[:,2]]
    # grid_vertices = tensorf.dense_xyz[verts[:,:,0], verts[:,:,1], verts[:,:,2]]
    all_vertices = torch.cat([pts.unsqueeze(1), grid_vertices], 1)


    return all_vertices


def query_occupancy_confidence(model, pts):
    # model: rendering.network
    # pts: [N, 3]
    N, _ = pts.shape
    grid_points = query_grid_vertices(model.tensorf, pts)     # [N, 9, 3]
    grid_points = grid_points.reshape(N*9, 3)

    g = []
    occs = []
    chunk = N
    for i in range(0, N*9, chunk):
        with torch.enable_grad():
            p = grid_points[i:i+N]
            p.requires_grad_(True)
            queried_occ, _ = query_alpha_color(model.tensorf, p, None, prior_mode='cat', requires_grad=True)
            y = queried_occ
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=p,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True, allow_unused=True)[0]
            _g = gradients.unsqueeze(1)
        grid_points.requires_grad_(False)
        g.append(_g.detach())
        occs.append(queried_occ.squeeze())
        del _g

    g = torch.cat(g, 0).squeeze().reshape(N, 9, 3)   # [N, 9, 3]
    occs = torch.cat(occs, 0).reshape(N, 9)     # only for test
    normals_ = g[:, :, :] / (g[:, :, :].norm(2, dim=2).unsqueeze(-1) + 10 ** (-5))
    normals_ = normals_.reshape(N, 9, 3)


    confidence = torch.var(normals_, 1).sum(1)
    length = g[:, :, :].norm(2, dim=2)
    confidence = torch.var(length, 1)
    confidence = torch.var(occs, 1)
    ref_pts = normals_[:, 0:1, :].repeat([1,8,1])
    grid_pts = normals_[:, 1:, :]
    cosine = torch.mul(ref_pts, grid_pts).sum(-1)
    confidence = torch.var(cosine, 1)


    return confidence, normals_[:, 0, :]


def query_entropy(occ, entropy_num=5):
    # pts: [N_rays, N_samples, 3], occ: [N_rays, N_samples]
    # pts_4_ = pts[:, :-4, :]
    # pts_3_ = pts[:, 1:-3, :]
    expand = (entropy_num-1)//2
    occs = []
    for i in range(expand):
        occ_ = occ[:, i:(-expand-(expand-i))]
        occs.append(occ_)
    occs.append(occ[:, expand:-expand])
    for i in range(expand):
        if i != expand -1:
            occ_ = occ[:, expand+i+1:(-expand+i+1)]
        else:
            occ_ = occ[:, expand+i+1:]
        occs.append(occ_)
    # occ_2_ = occ[:, 0:-4]
    # occ_1_ = occ[:, 1:-3]
    # occ_ = occ[:, 2:-2]
    # occ_1 = occ[:, 3:-1]
    # occ_2 = occ[:, 4:]
    # occ_5around = torch.stack([occ_2_, occ_1_, occ_, occ_1, occ_2], 2)   # [N_rays, N_samples, 5]
    occ_around = torch.stack(occs, 2)
    occ_around = torch.softmax(occ_around, 2)
    log_occ_around = torch.log(occ_around+1e-8)
    entropy = (-occ_around * log_occ_around).sum(2)   # [N_rays, N_samples]

    return entropy


def tensorf_volume_rendering(xyz_sampled, z_vals, camera_world, viewdirs, tensorf):
    # tensorf volume rendering
    # xyz_sampled: [N_rays, N_samples, 3]
    # z_vals: [N_rays, N_samples]
    # viewdirs: [N_rays, 3]

    # from test_ours import visual_ray_points
    # visual_ray_points(xyz_sampled)

    # xyz_sampled, z_vals, ray_valid = tensorf.sample_ray(camera_world, viewdirs, is_train=False, N_samples=tensorf.nSamples)

    mask_outbbox = ((tensorf.aabb[0] > xyz_sampled) | (xyz_sampled > tensorf.aabb[1])).any(dim=-1)
    ray_valid = ~mask_outbbox
    dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)


    viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

    if tensorf.alphaMask is not None:
        alphas = tensorf.alphaMask.sample_alpha(xyz_sampled[ray_valid])
        alpha_mask = alphas > 0
        ray_invalid = ~ray_valid
        ray_invalid[ray_valid] |= (~alpha_mask)
        ray_valid = ~ray_invalid

    sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
    rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

    if ray_valid.any():
        xyz_sampled = tensorf.normalize_coord(xyz_sampled)
        sigma_feature = tensorf.compute_densityfeature(xyz_sampled[ray_valid])

        validsigma = tensorf.feature2density(sigma_feature)
        sigma[ray_valid] = validsigma

    alpha, weight, bg_weight = raw2alpha(sigma, dists * tensorf.distance_scale)

    app_mask = weight > tensorf.rayMarch_weight_thres

    if app_mask.any():
        app_features = tensorf.compute_appfeature(xyz_sampled[app_mask])
        valid_rgbs = tensorf.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features)
        rgb[app_mask] = valid_rgbs

    acc_map = torch.sum(weight, -1)
    rgb_map = torch.sum(weight[..., None] * rgb, -2)

    return rgb_map


### deprecated
def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    ply_filename_out,
    bbox,
    level=0.5,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1]-bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    faces = faces[...,::-1] # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0,0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0,1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0,2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)