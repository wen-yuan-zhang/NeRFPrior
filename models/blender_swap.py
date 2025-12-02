import torch, cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *


class BlendSwapDataset(Dataset):
    def __init__(self, conf, split='train', N_vis=-1):
        self.device = torch.device('cuda')
        self.N_vis = N_vis
        self.root_dir = conf.get_string('data_dir')
        scene = conf.get_string('scene')
        self.root_dir = os.path.join(self.root_dir, scene)
        
        self.split = split
        self.is_stack = False
        self.downsample = 1.0
        self.define_transforms()

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        # self.define_proj_mat()

        self.white_bg = True

    def read_meta(self):

        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = int(self.meta['w'] / self.downsample), int(self.meta['h'] / self.downsample)
        self.img_wh = [w, h]
        self.focal_x = 0.5 * w / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.focal_y = 0.5 * h / np.tan(0.5 * self.meta['camera_angle_y'])  # original focal length
        self.cx, self.cy = self.meta['cx'], self.meta['cy']

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal_x, self.focal_y], center=[self.cx, self.cy])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal_x, 0, self.cx], [0, self.focal_y, self.cy], [0, 0, 1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_pretrained_rgbs = []
        self.all_depth = []
        self.all_rgb_std = []
        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):  # img_list:#
            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix'])
            pose = pose @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses.append(c2w)

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}")
            self.image_paths += [image_path]
            img = Image.open(image_path)

            img = self.transform(img)  # (4, h, w)

            rgb_std = self.cal_rgb_std(img.permute(1, 2, 0))
            rgb_std = np.where(rgb_std > 10 / 255.0, 1.0, 0.0)
            self.all_rgb_std.append(torch.Tensor(rgb_std).cpu())

            img = img.view(-1, w * h).permute(1, 0)  # (h*w, 4) RGBA
            if img.shape[-1] == 4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs.append(img)
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays.append(torch.cat([rays_o, rays_d], 1))  # (h*w, 6)

        self.poses = torch.stack(self.poses)  #(N, 4, 4)
        self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 6)
        self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
 
        self.h, self.w = h, w
        # [w*h, 9]
        self.all_neighbor_idx = self.query_neighbor_idx(torch.arange(w*h))

        # for Neus exp_runner
        self.n_images = len(self.image_paths)
        self.pose_all = self.poses
        self.object_bbox_min = np.array([-1.01, -1.01, -1.01])  # only used in extract
        self.object_bbox_max = np.array([ 1.01,  1.01,  1.01])
        self.scale_mats_np = [np.eye(4)]

    def define_transforms(self):
        self.transform = T.ToTensor()

    # def define_proj_mat(self):
    #     self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:, :3]

    def __len__(self):
        return len(self.all_rgbs)

    def cal_rgb_std(self, img):
        """
        :param img: [h, w, 3], rgb\in [0, 255]
        :return: [h, w] \in [0, 1]
        """
        img = np.array(img, np.float64)
        kernel = (3, 3)
        kernel = (9, 9)
        E_square = cv2.blur(img ** 2, kernel)
        square_E = cv2.blur(img, kernel) ** 2
        sharp_img = np.sqrt(np.abs(E_square - square_E))

        gray_img = sharp_img.max(2)
        gray_min = np.min(gray_img)
        gray_max = np.max(gray_img)
        gray_img = np.clip(gray_img, gray_min, gray_max)
        gray_img = (gray_img - gray_min) / (gray_max - gray_min + 1e-6)
        return gray_img

    def query_neighbor_idx(self, idx):
        # query idx's neighbors. idx: [N,]
        # 0 1 2
        # 3 * 4
        # 5 6 7
        # return: [N, 9]
        pad = 4
        row, col = idx // self.w, idx % self.w
        r0 = r1 = r2 = torch.maximum(row-pad, torch.zeros_like(row))
        r3 = r4 = row
        r5 = r6 = r7 = torch.minimum(row+pad, torch.ones_like(row)*(self.h-1))
        c0 = c3 = c5 = torch.maximum(col-pad, torch.zeros_like(col))
        c1 = c6 = col
        c2 = c4 = c7 = torch.minimum(col+pad, torch.ones_like(col)*(self.w-1))

        idx0 = r0 * self.w + c0
        idx1 = r1 * self.w + c1
        idx2 = r2 * self.w + c2
        idx3 = r3 * self.w + c3
        idx4 = r4 * self.w + c4
        idx5 = r5 * self.w + c5
        idx6 = r6 * self.w + c6
        idx7 = r7 * self.w + c7
        neighbor_idx = torch.stack([idx0, idx1, idx2, idx3, idx, idx4, idx5, idx6, idx7], 1)  # [N, 9]
        border_mask = (idx % self.w == 0) | (idx % self.w == (self.w - 1)) | (idx // self.w == 0) | (
                    idx // self.w == (self.h - 1))
        neighbor_idx[border_mask] = idx[border_mask].unsqueeze(1).repeat([1, 9])
        return neighbor_idx



    # def gen_random_rays_at(self, img_idx, batch_size):
    #     pixels_x = torch.randint(low=0, high=self.w, size=[batch_size])
    #     pixels_y = torch.randint(low=0, high=self.h, size=[batch_size])
    #     color = self.all_rgbs[img_idx] # [h, w, 3]
    #     color = color[(pixels_y, pixels_x)]  # [batch_size, 3]
    #     mask = torch.ones_like(color, dtype=torch.float)
    #     all_rays = self.all_rays[img_idx].reshape(self.h, self.w, 6) # [h, w, 6]
    #     rand_rays = all_rays[(pixels_y, pixels_x)] # [batch_size, 6]
    #     return torch.cat([rand_rays, color, mask[:, :1]], dim=-1).to(self.device)

    def gen_random_rays_at(self, img_idx, batch_size, mode):
        if mode == 'batch':
            pixels_x = torch.randint(low=0, high=self.w, size=[batch_size])
            pixels_y = torch.randint(low=0, high=self.h, size=[batch_size])

            color = self.all_rgbs[img_idx] # [h, w, 3]
            color = color[(pixels_y, pixels_x)]  # [batch_size, 3]
            mask = torch.ones_like(color, dtype=torch.float)
            all_rays = self.all_rays[img_idx].reshape(self.h, self.w, 6) # [h, w, 6]
            rand_rays = all_rays[(pixels_y, pixels_x)] # [batch_size, 6]
            return torch.cat([rand_rays, color, mask[:, :1]], dim=-1).to(self.device)
        elif mode == 'patch':
            pixels_x = torch.randint(low=0, high=self.w, size=[batch_size // 9])
            pixels_y = torch.randint(low=0, high=self.h, size=[batch_size // 9])
            pixel_idx = pixels_y * self.h + pixels_x
            rand_idx = self.all_neighbor_idx[pixel_idx].reshape(-1)
            pixels_x = rand_idx % self.w
            pixels_y = rand_idx // self.w
            patch_rgb_std = self.all_rgb_std[img_idx].reshape(-1, 1)[rand_idx]
            color = self.all_rgbs[img_idx]  # [h, w, 3]
            color = color[(pixels_y, pixels_x)]  # [batch_size, 3]
            mask = torch.ones_like(color, dtype=torch.float)
            all_rays = self.all_rays[img_idx].reshape(self.h, self.w, 6)  # [h, w, 6]
            rand_rays = all_rays[(pixels_y, pixels_x)]  # [batch_size, 6]
            return torch.cat([rand_rays, color, mask[:, :1], patch_rgb_std], dim=-1).to(self.device)


    def near_far_from_sphere(self, rays_o, rays_d):
        # copied from dataset.py
        # a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        # b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        # mid = 0.5 * (-b) / a
        # near = mid - 1.0
        # far = mid + 1.0

        near = torch.zeros(rays_o.shape[0], 1).cuda()
        far = torch.ones(rays_o.shape[0], 1).cuda() * 3
        return near, far

    def gen_rays_at(self, img_idx, resolution_level=1):
        all_rays = self.all_rays[img_idx].reshape(self.h, self.w, 6) # [h, w, 6]
        rays_o = all_rays[:, :, :3].to(self.device)
        rays_d = all_rays[:, :, 3:].to(self.device)
        return rays_o, rays_d

    def image_at(self, idx, resolution_level):
        img = cv2.imread(self.image_paths[idx])
        return (cv2.resize(img, (self.w // resolution_level, self.h // resolution_level))).clip(0, 255)

    
    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        # only used in novel view synthesis
        raise NotImplementedError()

    def __getitem__(self, idx):
        sample = {}
        img_raynum = self.h*self.w
        if self.split == 'train':
            imgid = self.__len__() - 1 - idx
            # imgid = 100
            all_rays = self.all_rays[imgid]
            rand_idx = np.random.choice(img_raynum, self.batch_size, replace=False)
            # rand_idx = np.arange(10740, 10840)

            # batch_rand_idx = np.random.choice(img_raynum, self.batch_size // 9, replace=False)
            # # batch_rand_idx = np.array([64500, 64510, 64520, 64530, 64540, 64550, 64560, 64570, 64580, 64590, 64560])    # debug
            # rand_idx = self.all_neighbor_idx[batch_rand_idx].reshape(-1)  # [N*9]
            # sample.update({'patch_rgb_std': self.all_rgb_std[imgid].reshape(-1)[batch_rand_idx]})
            sample.update({'patch_rgb_std': self.all_rgb_std[imgid].reshape(-1)[rand_idx]})

            sample.update({'rays_o': all_rays[rand_idx, :3],
                      'rays_d': all_rays[rand_idx, 3:6],
                      'rgb': self.all_rgbs[imgid].reshape(-1, 3)[rand_idx],
                    #   'all_trg_rays_o': all_rays[:, :3],
                    #   'all_trg_rays_d': all_rays[:, 3:6],
                    #   'trg_rgb': self.all_rgbs[imgid],
                    #   'trg_extrinsics': self.poses[imgid],
                      'idx': idx,
                      'imgid': imgid,
                      'h': self.h,
                      'w': self.w,
                      'intrinsics': self.intrinsics,
                      'pixel_idx': rand_idx,
                      })

            # debug
            # rays_d = sample['rays_d'].unsqueeze(0)
            # rays_o = sample['rays_o'].unsqueeze(0)
            # c2w = self.poses[imgid]
            # world_pos = rays_d * 0.5 + rays_o     # [W, H, 3]
            # world_pos = torch.cat([world_pos, torch.ones([rays_d.shape[0], rays_d.shape[1], 1])], 2)    # [h*w, 4]
            # # K * R^-1 * xyz
            # camera_pos = torch.einsum('ij,abj->abi', c2w.inverse(), world_pos)    # [W, H, 4]
            # camera_pos = torch.einsum('abj,ij->abi', world_pos, c2w.inverse())
            # # camera_pos[..., 1] *= -1
            # # camera_pos[..., 2] *= -1
            # uv1 = torch.einsum('ij, abj->abi', self.intrinsics, camera_pos[..., :3])     # [W, H, 3]
            # uv1[..., :] /= uv1[..., 2:]
            # uv1 = uv1[..., :2] - 0.5
            # uv1 = uv1[:,:,[1,0]]


            # newview
            src_view_num = 3
            src_offsets = np.random.choice(np.concatenate((np.arange(-60, -9), np.arange(10, 61))), src_view_num, replace=False)
            src_idx = []
            for offset in src_offsets:
                _idx = imgid + offset
                if _idx >= len(self.all_rgbs) or _idx < 0:
                    _idx = imgid - offset
                src_idx.append(_idx)
            src_imgid = src_idx
            src_all_rays = self.all_rays[src_imgid]
            # sample.update({'src_rgb': self.all_rgbs[src_imgid],
            #                'src_extrinsics': self.poses[src_imgid],
            #                'src_rays_o': src_all_rays[..., :3].reshape(src_view_num, self.h, self.w, 3),
            #                'src_rays_d': src_all_rays[..., 3:6].reshape(src_view_num, self.h, self.w, 3),
            #                'offsets': src_offsets,})

            return sample
        elif self.split == 'test':
            idx = np.random.randint(self.__len__())
            # idx = 901
            all_rays = self.all_rays[idx]
            sample = {'rays_o': all_rays[:, :3],
                      'rays_d': all_rays[:, 3:6],
                      'rgb': self.all_rgbs[idx].reshape(-1,3),
                      'idx': idx}
            return sample
