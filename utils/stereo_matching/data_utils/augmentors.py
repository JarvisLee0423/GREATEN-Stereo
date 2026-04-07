import os
import cv2
import glob
import math
import random
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import ColorJitter, Compose, functional
from PIL import Image
from skimage import color
from typing import Any, List, Tuple, Union

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def get_kitti_images() -> list:
    return sorted(glob.glob("datasets/KITTI/training/image_2/*_10.png"))


def get_eth3d_images() -> list:
    return sorted(glob.glob("datasets/ETH3D/two_view_training/*/im0.png"))


def get_middlebury_images() -> list:
    root = "datasets/Middlebury/MiddEval3"
    with open(os.path.join(root, "official_train.txt"), "r") as file:
        lines = file.read().splitlines()
    return sorted([os.path.join(root, "trainingQ", f"{name}/im0.png") for name in lines])


def transfer_color(image: np.ndarray, style_mean: float, style_stddev: float) -> np.ndarray:
    reference_image_lab = color.rgb2lab(image)
    reference_stddev = np.std(reference_image_lab, axis=(0, 1), keepdims=True)
    reference_mean = np.mean(reference_image_lab, axis=(0, 1), keepdims=True)

    reference_image_lab = reference_image_lab - reference_mean
    lamb = style_stddev / reference_stddev
    style_image_lab = lamb * reference_image_lab
    output_image_lab = style_image_lab + style_mean
    l, a, b = np.split(output_image_lab, 3, axis=2)
    l = l.clip(0, 100)
    output_image_lab = np.concatenate((l, a, b), axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        output_image_rgb = color.lab2rgb(output_image_lab) * 255
        return output_image_rgb


class AdjustGamma(object):
    def __init__(self, gamma_min: Union[int, float], gamma_max: Union[int, float], gain_min: Union[int, float]=1.0, gain_max: Union[int, float]=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = gamma_min, gamma_max, gain_min, gain_max
    
    def __call__(self, sample: Image.Image) -> Union[torch.Tensor, Image.Image]:
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return functional.adjust_gamma(sample, gamma, gain)
    
    def __repr__(self) -> str:
        return f"Adjust Gamma ({self.gamma_min}, {self.gamma_max}) and Gain ({self.gain_min}, {self.gain_max})."


class DispAugmentor:
    def __init__(
        self,
        crop_size: Union[List[Union[int, float]], Tuple[Union[int, float]]],
        min_scale: float=-0.2,
        max_scale: float=0.5,
        do_flip: bool=True,
        yjitter: bool=False,
        saturation_range: List[float]=[0.6, 1.4],
        gamma: List[Union[int, float]]=[1, 1, 1, 1],
    ):
        # Spatial augmentation params.
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 1.0
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # Flip augmentation params.
        self.yjitter = yjitter
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # Photometric augmentation params.
        self.photo_aug = Compose([
            ColorJitter(brightness=0.4, contrast=0.4, saturation=saturation_range, hue=0.5 / 3.14),
            AdjustGamma(*gamma),
        ])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
    
    def color_transform(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray]:
        """ Photometric augmentation. """

        # Asymmetric.
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
        # Symmetric.
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)
        
        return img1, img2
    
    def eraser_transform(self, img1: np.ndarray, img2: np.ndarray, bounds: List[int]=[50, 100]) -> Tuple[np.ndarray]:
        """ Occlusion augmentation. """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
        
        return img1, img2
    
    def spatial_transform(self, img1: np.ndarray, img2: np.ndarray, disp: np.ndarray) -> Tuple[np.ndarray]:
        # Randomly sample scale.

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd),
        )

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if (np.random.rand() < self.spatial_aug_prob) or (ht < self.crop_size[0]) or (wd < self.crop_size[1]):
            # Rescale the images.
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            disp = cv2.resize(disp, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

            disp = disp * [scale_x, scale_y]
        
        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == "hf": # h-flip.
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                disp = disp[:, ::-1] * [-1.0, 1.0]
            
            if np.random.rand() < self.h_flip_prob and self.do_flip == "h": # h-flip for stereo.
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp
            
            if np.random.rand() < self.v_flip_prob and self.do_flip == "v": # v-flip.
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                disp = disp[::-1, :] * [1.0, -1.0]
        
        if self.yjitter:
            y0 = np.random.randint(2, img1.shape[0] - self.crop_size[0] - 2)
            x0 = np.random.randint(2, img1.shape[1] - self.crop_size[1] - 2)

            y1 = y0 + np.random.randint(-2, 2 + 1)
            img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            img2 = img2[y1:y1 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            disp = disp[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

            img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            disp = disp[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        
        return img1, img2, disp
    
    def __call__(self, img1: np.ndarray, img2: np.ndarray, disp: np.ndarray) -> Tuple[np.ndarray]:
        img1, img2 = self.color_transform(img1, img2)
        # img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, disp = self.spatial_transform(img1, img2, disp)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        disp = np.ascontiguousarray(disp)

        return img1, img2, disp


class SparseDispAugmentor:
    def __init__(
        self,
        crop_size: Union[List[Union[int, float]], Tuple[Union[int, float]]],
        min_scale: float=-0.2,
        max_scale: float=0.5,
        do_flip: bool=False,
        yjitter: bool=False,
        saturation_range: List[float]=[0.7, 1.3],
        gamma: List[Union[int, float]]=[1, 1, 1, 1],
    ):
        # Spatial augmentation params.
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # Flip augmentation params.
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # Photometric augmentation params.
        self.photo_aug = Compose([
            ColorJitter(brightness=0.3, contrast=0.3, saturation=saturation_range, hue=0.3 / 3.14),
            AdjustGamma(*gamma),
        ])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
    
    def color_transform(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray]:
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2
    
    def eraser_transform(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray]:
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
        
        return img1, img2
    
    def resize_sparse_disp_map(self, disp: np.ndarray, valid: np.ndarray, fx: float=1.0, fy: float=1.0) -> Tuple[np.ndarray]:
        ht, wd = disp.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)
        coords = coords.reshape(-1, 2).astype(np.float32)
        disp = disp.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        disp0 = disp[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        disp1 = disp0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        disp1 = disp1[v]

        disp_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        disp_img[yy, xx] = disp1
        valid_img[yy, xx] = 1

        return disp_img, valid_img
    
    def spatial_transform(self, img1: np.ndarray, img2: np.ndarray, disp: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray]:
        # Randomly sample scale.
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd),
        )

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if (np.random.rand() < self.spatial_aug_prob) or (ht < self.crop_size[0]) or (wd < self.crop_size[1]):
            # Rescale the images.
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            disp, valid = self.resize_sparse_disp_map(disp, valid, fx=scale_x, fy=scale_y)
        
        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == "hf": # h-flip.
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                disp = disp[:, ::-1] * [-1.0, 1.0]
            
            if np.random.rand() < self.h_flip_prob and self.do_flip == "h": # h-flip for stereo.
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp
            
            if np.random.rand() < self.v_flip_prob and self.do_flip == "v": # v-flip.
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                disp = disp[::-1, :] * [1.0, -1.0]
        
        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        disp = disp[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, disp, valid

    def __call__(self, img1: np.ndarray, img2: np.ndarray, disp: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray]:
        img1, img2 = self.color_transform(img1, img2)
        # img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, disp, valid = self.spatial_transform(img1, img2, disp, valid)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        disp = np.ascontiguousarray(disp)
        valid = np.ascontiguousarray(valid)

        return img1, img2, disp, valid


class PhysicalSpecularTransparentAugmentor:
    def __init__(
        self,
        aug_prob: float=0.7,
        use_noise_content_prob: float=0.2,
        add_spotlight_prob: float=0.6,
        global_spotlight_prob: float=0.5,
        irregular_shape_prob: float=0.6,
        max_global_spotlights: int=3,
        min_patch_ratio: float=0.3,
        max_patch_ratio: float=0.7,
        min_fake_disp: int=5,
        max_fake_disp: int=35,
        spotlight_intensity_min: float=120.0,
        spotlight_intensity_max: float=255.0,
        feather_sigma: float=15.0,
    ):
        self.aug_prob = aug_prob
        self.use_noise_content_prob = use_noise_content_prob
        self.add_spotlight_prob = add_spotlight_prob
        self.global_spotlight_prob = global_spotlight_prob
        self.irregular_shape_prob = irregular_shape_prob
        
        self.max_global_spotlights = max_global_spotlights
        self.min_patch_ratio = min_patch_ratio
        self.max_patch_ratio = max_patch_ratio
        self.min_fake_disp = min_fake_disp
        self.max_fake_disp = max_fake_disp

        self.spotlight_min = spotlight_intensity_min
        self.spotlight_max = spotlight_intensity_max
        self.feather_sigma = feather_sigma
    
    def get_random_texture_source(self, img: torch.Tensor) -> torch.Tensor:
        B = img.shape[0]
        if B > 1:
            perm = torch.randperm(B, device=img.device)
            offset = (perm == torch.arange(B, device=img.device)).long()
            perm = (perm + offset) % B

            return img[perm]
        else:
            return torch.flip(img, dims=[2, 3])

    def generate_irregular_blob(self, h: int, w: int, blob_radius_min: float, blob_radius_max: float, device: torch.device, softness: float=10.0) -> torch.Tensor:
        y_range = torch.arange(h, device=device).float()
        x_range = torch.arange(w, device=device).float()
        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

        grid_y = 2.0 * (grid_y / h) - 1.0
        grid_x = 2.0 * (grid_x / w) - 1.0

        center_x = (random.random() - 0.5) * 0.6
        center_y = (random.random() - 0.5) * 0.6

        axis_x = random.uniform(blob_radius_min, blob_radius_max)
        axis_y = random.uniform(blob_radius_min, blob_radius_max)

        angle = random.uniform(0, math.pi)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        shifted_x = grid_x - center_x
        shifted_y = grid_y - center_y

        rot_x = shifted_x * cos_a + shifted_y * sin_a
        rot_y = -shifted_x * sin_a + shifted_y * cos_a

        dist_sq = (rot_x / axis_x) ** 2 + (rot_y / axis_y) ** 2

        mask = torch.sigmoid(softness * (1.0 - dist_sq))

        return mask.view(1, 1, h, w)
    
    def generate_feathered_rect_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        y = torch.arange(h, device=device).float()
        x = torch.arange(w, device=device).float()

        dist_y = torch.min(y, h - 1 - y)
        dist_x = torch.min(x, w - 1 - x)

        dist_y = dist_y.view(-1, 1).expand(h, w)
        dist_x = dist_x.view(1, -1).expand(h, w)

        min_dist = torch.min(dist_y, dist_x)

        fade_size = min(h, w) * 0.15

        mask = torch.clamp(min_dist / (fade_size + 1e-6), 0.0, 1.0)
        mask = torch.pow(mask, 0.5)

        return mask.view(1, 1, h, w)

    def generate_content_strip(self, source_img: torch.Tensor, h: int, w: int, total_width: int) -> torch.Tensor:
        B, C, H, W = source_img.shape
        device = source_img.device

        if random.random() < self.use_noise_content_prob:
            content = torch.ones(B, C, h, total_width, device=device) * random.random() * 255.0
        else:
            req_h, req_w = h, total_width
            if H < req_h or W < req_w:
                source_img = F.interpolate(source_img, size=(max(H, req_h), max(W, req_w)), mode="bilinear", align_corners=False)
                H, W = source_img.shape[2], source_img.shape[3]
            
            sy = random.randint(0, H - req_h)
            sx = random.randint(0, W - req_w)
            content = source_img[:, :, sy:sy + req_h, sx:sx + req_w]

        return content
    
    def apply_global_spotlights(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        device = img.device
        out_img = img.clone()

        for b in range(B):
            if random.random() > self.global_spotlight_prob:
                continue

            num_spots = random.randint(1, self.max_global_spotlights)
            for _ in range(num_spots):
                spot_h = random.randint(int(H * 0.05), int(H * 0.2))
                spot_w = random.randint(int(W * 0.05), int(W * 0.2))

                top = random.randint(0, max(0, H - spot_h))
                left = random.randint(0, max(0, W - spot_w))

                spot_mask = self.generate_irregular_blob(spot_h, spot_w, 0.5, 0.8, device, softness=5.0)

                intensity = random.uniform(self.spotlight_min, self.spotlight_max)

                roi = out_img[b:b+1, :, top:top+spot_h, left:left+spot_w]
                highlight = roi + (spot_mask * intensity)

                out_img[b:b+1, :, top:top+spot_h, left:left+spot_w] = torch.clamp(highlight, 0, 255)
        
        return out_img
    
    def __call__(self, left: torch.Tensor, right: torch.Tensor, disp_gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.aug_prob:
            return left, right
        
        B, C, H, W = left.shape
        device = left.device

        aug_left = left.clone()
        aug_right = right.clone()

        texture_source = self.get_random_texture_source(left if random.random() > 0.5 else right)
        for b in range(B):
            target_h = random.randint(int(H * self.min_patch_ratio), int(H * self.max_patch_ratio))
            target_w = random.randint(int(W * self.min_patch_ratio), int(W * self.max_patch_ratio))

            fake_disp = random.randint(self.min_fake_disp, self.max_fake_disp)
            # if random.random() < 0.5: fake_disp = -fake_disp

            # total_w = target_w + abs(fake_disp)
            total_w = target_w + fake_disp
            strip = self.generate_content_strip(texture_source[b:b + 1], target_h, target_w, total_w)

            if random.random() < 0.5:
                tex_l = strip[:, :, :, :target_w]
                tex_r = strip[:, :, :, fake_disp:]
            else:
                tex_l = strip[:, :, :, fake_disp:]
                tex_r = strip[:, :, :, :target_w]

            # if fake_disp > 0:
            #     tex_l = strip[:, :, :, fake_disp:]
            #     tex_r = strip[:, :, :, :target_w]
            # else:
            #     tex_l = strip[:, :, :, :target_w]
            #     tex_r = strip[:, :, :, abs(fake_disp):]
            
            if random.random() < self.irregular_shape_prob:
                patch_mask = self.generate_irregular_blob(target_h, target_w, 0.5, 0.8, device, softness=10.0)
            else:
                patch_mask = self.generate_feathered_rect_mask(target_h, target_w, device)

            lx, ly = 0, 0
            valid = False
            median_disp = 0.0
            for _ in range(5):
                lx = random.randint(0, max(0, W - target_w))
                ly = random.randint(0, max(0, H - target_h))

                patch_disp = disp_gt[b:b+1, :, ly:ly + target_h, lx:lx + target_w]
                valid_mask = (patch_disp > 0)
                if valid_mask.float().mean() < 0.2: continue

                median_disp = patch_disp[valid_mask].median().item()
                valid = True
                break

            if not valid: continue

            rx_start = int(lx - median_disp)
            rx_end = rx_start + target_w
            r_img_start = max(0, rx_start)
            r_img_end = min(W, rx_end)

            roi_l = aug_left[b:b + 1, :, ly:ly + target_h, lx:lx + target_w]
            aug_left[b:b + 1, :, ly:ly + target_h, lx:lx + target_w] = roi_l * (1 - patch_mask) + tex_l * patch_mask

            if r_img_end > r_img_start:
                patch_x_start = r_img_start - rx_start
                patch_x_end = patch_x_start + (r_img_end - r_img_start)

                tex_r_valid = tex_r[:, :, :, patch_x_start:patch_x_end]
                mask_r_valid = patch_mask[:, :, :, patch_x_start:patch_x_end]

                roi_r = aug_right[b:b + 1, :, ly:ly + target_h, r_img_start:r_img_end]
                aug_right[b:b + 1, :, ly:ly + target_h, r_img_start:r_img_end] = roi_r * (1 - mask_r_valid) + tex_r_valid * mask_r_valid

            if random.random() < self.add_spotlight_prob:
                spot_mask = self.generate_irregular_blob(target_h, target_w, 0.5, 0.8, device, softness=5.0)
                intensity = random.uniform(self.spotlight_min, self.spotlight_max)
                
                dice = random.random()
                if dice < 0.4:
                    roi_l_final = aug_left[b:b + 1, :, ly:ly + target_h, lx:lx + target_w]
                    highlight = roi_l_final + (spot_mask * intensity * patch_mask)
                    aug_left[b:b + 1, :, ly:ly + target_h, lx:lx + target_w] = torch.clamp(highlight, 0, 255)
                elif dice < 0.8:
                    if r_img_end > r_img_start:
                        spot_r_valid = spot_mask[:, :, :, patch_x_start:patch_x_end]
                        mask_r_valid = patch_mask[:, :, :, patch_x_start:patch_x_end]
                        roi_r_final = aug_right[b:b+1, :, ly:ly+target_h, r_img_start:r_img_end]
                        highlight = roi_r_final + (spot_r_valid * intensity * mask_r_valid)
                        aug_right[b:b+1, :, ly:ly+target_h, r_img_start:r_img_end] = torch.clamp(highlight, 0, 255)
                else:
                    roi_l_final = aug_left[b:b + 1, :, ly:ly + target_h, lx:lx + target_w]
                    highlight_l = roi_l_final + (spot_mask * intensity * patch_mask)
                    aug_left[b:b + 1, :, ly:ly + target_h, lx:lx + target_w] = torch.clamp(highlight_l, 0, 255)
                    if r_img_end > r_img_start:
                        spot_r_valid = spot_mask[:, :, :, patch_x_start:patch_x_end]
                        mask_r_valid = patch_mask[:, :, :, patch_x_start:patch_x_end]
                        roi_r_final = aug_right[b:b+1, :, ly:ly+target_h, r_img_start:r_img_end]
                        highlight_r = roi_r_final + (spot_r_valid * intensity * mask_r_valid)
                        aug_right[b:b+1, :, ly:ly+target_h, r_img_start:r_img_end] = torch.clamp(highlight_r, 0, 255)
        
        aug_left = self.apply_global_spotlights(aug_left)
        aug_right = self.apply_global_spotlights(aug_right)
        
        return torch.clamp(aug_left, 0, 255), torch.clamp(aug_right, 0, 255)


class SpecularTransparentAugmentor:
    def __init__(
        self,
        aug_prob: float=0.8,
        use_noise_content_prob: float=0.3,
        add_spotlight_prob: float=0.5,
        min_patch_ratio: float=0.2,
        max_patch_ratio: float=0.5,
        feather_sigma_range: Tuple[float, float]=(5.0, 15.0),
        min_fake_disp: int=50,
        max_fake_disp: int=200,
        spotlight_intensity_min: float=100.0,
        spotlight_intensity_max: float=200.0,
    ):
        self.aug_prob = aug_prob
        self.use_noise_content_prob = use_noise_content_prob
        self.add_spotlight_prob = add_spotlight_prob
        
        self.min_patch_ratio = min_patch_ratio
        self.max_patch_ratio = max_patch_ratio
        self.feather_sigma_range = feather_sigma_range
        self.min_fake_disp = min_fake_disp
        self.max_fake_disp = max_fake_disp

        self.spotlight_min = spotlight_intensity_min
        self.spotlight_max = spotlight_intensity_max
    
    def get_random_texture_source(self, img: torch.Tensor) -> torch.Tensor:
        B = img.shape[0]
        if B > 1:
            perm = torch.randperm(B, device=img.device)
            offset = (perm == torch.arange(B, device=img.device)).long()
            perm = (perm + offset) % B

            return img[perm]
        else:
            return torch.flip(img, dims=[2, 3])
    
    def generate_feathered_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros((1, 1, h, w), device=device)
        border_h = int(h * 0.15)
        border_w = int(w * 0.15)

        if h > 2 * border_h and w > 2 * border_w:
            mask[:, :, border_h:h - border_h, border_w:w - border_w] = 1.0
        else:
            mask[:, :, h // 2:h // 2 + 1, w // 2:w // 2 + 1] = 1.0
        
        sigma = random.uniform(self.feather_sigma_range[0], self.feather_sigma_range[1])
        kernel_size = int(sigma * 4) | 1
        blurrer = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        mask = blurrer(mask)
        mask = mask / (mask.max() + 1e-8)

        return mask
    
    def generate_base_content(self, source_img: torch.Tensor, h: int, w: int) -> torch.Tensor:
        B, C, H, W = source_img.shape
        device = source_img.device

        if random.random() < self.use_noise_content_prob:
            content = torch.rand(B, C, h, w, device=device) * 255.0
        else:
            sy = random.randint(0, max(0, H - h))
            sx = random.randint(0, max(0, W - w))
            content = source_img[:, :, sy:sy + h, sx:sx + w]
            if content.shape[2] != h or content.shape[3] != w:
                content = F.interpolate(source_img, size=(h, w), mode="bilinear", align_corners=False)
        
        return content
    
    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.aug_prob:
            return left, right
        
        B, C, H, W = left.shape
        device = left.device

        aug_left = left.clone()
        aug_right = right.clone()

        if random.random() > 0.5:
            texture_source_img = self.get_random_texture_source(left)
        else:
            texture_source_img = self.get_random_texture_source(right)
        
        for b in range(B):
            target_h = random.randint(int(H * self.min_patch_ratio), int(H * self.max_patch_ratio))
            target_w = random.randint(int(W * self.min_patch_ratio), int(W * self.max_patch_ratio))

            patch_content = self.generate_base_content(texture_source_img[b:b+1], target_h, target_w)
            patch_mask = self.generate_feathered_mask(target_h, target_w, device)
            global_alpha = random.uniform(0.7, 1.0)
            patch_mask = patch_mask * global_alpha

            lx = random.randint(0, max(0, W - target_w))
            ly = random.randint(0, max(0, H - target_h))
            fake_disp = random.randint(self.min_fake_disp, self.max_fake_disp)
            if random.random() < 0.5:
                fake_disp = -fake_disp
            rx = lx - fake_disp

            roi_l = aug_left[b:b + 1, :, ly:ly + target_h, lx:lx + target_w]
            aug_left[b:b + 1, :, ly:ly + target_h, lx:lx + target_w] = roi_l * (1 - patch_mask) + patch_content * patch_mask

            r_start = max(0, rx)
            r_end = min(W, rx + target_w)
            if r_end > r_start:
                patch_x_start = max(0, -rx)
                patch_x_end = patch_x_start + (r_end - r_start)
                valid_content = patch_content[:, :, :, patch_x_start:patch_x_end]
                valid_mask = patch_mask[:, :, :, patch_x_start:patch_x_end]

                roi_r = aug_right[b:b + 1, :, ly:ly + target_h, r_start:r_end]
                aug_right[b:b + 1, :, ly:ly + target_h, r_start:r_end] = roi_r * (1 - valid_mask) + valid_content * valid_mask

                if random.random() < self.add_spotlight_prob:
                    intensity = random.uniform(self.spotlight_min, self.spotlight_max)

                    target_view = random.choice(["left", "right"])
                    if target_view == "left":
                        current_roi = aug_left[b:b + 1, :, ly:ly + target_h, lx:lx + target_w]
                        highlighted_roi = current_roi + (intensity * patch_mask)
                        aug_left[b:b + 1, :, ly:ly + target_h, lx:lx + target_w] = highlighted_roi
                    else:
                        current_roi_r = aug_right[b:b + 1, :, ly:ly + target_h, r_start:r_end]
                        highlighted_roi_r = current_roi_r + (intensity * valid_mask)
                        aug_right[b:b + 1, :, ly:ly + target_h, r_start:r_end] = highlighted_roi_r
        
        return torch.clamp(aug_left, 0, 255), torch.clamp(aug_right, 0, 255)


class PhotometricConsistencyDestroyer:
    def __init__(
        self,
        aug_prob: float=0.7,
        specular_prob: float=0.3,
        transparent_prob: float=0.3,
        spotlight_prob: float=0.2,
        eraser_prob: float=0.2,
        feather_sigma: int=15,
        kernel_size: int=61,
        hard_edge_prob: float=0.3,
    ):
        self.aug_prob = aug_prob
        
        # Normalize the augmentation probabilities.
        total_type_prob = specular_prob + transparent_prob + spotlight_prob + eraser_prob
        self.probs = {
            "specular": specular_prob / total_type_prob,
            "transparent": transparent_prob / total_type_prob,
            "spotlight": spotlight_prob / total_type_prob,
            "eraser": eraser_prob / total_type_prob,
        }

        self.hard_edge_prob = hard_edge_prob
        self.kernel_size = kernel_size
        self.feather_sigma = feather_sigma
        self.gaussian_kernel = self.create_gaussian_kernel(kernel_size, feather_sigma)
    
    def create_gaussian_kernel(self, kernel_size: int, sigma: int) -> torch.Tensor:
        coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        return kernel.view(1, 1, kernel_size, kernel_size)
    
    def generate_mask(self, img_shape: List[int], device: Any) -> torch.Tensor:
        B, C, H, W = img_shape

        mask = torch.zeros((B, 1, H, W), device=device)

        for b in range(B):
            w_size = random.randint(int(W * 0.2), int(W * 0.5))
            h_size = random.randint(int(H * 0.2), int(H * 0.5))

            x = random.randint(0, max(1, W - w_size))
            y = random.randint(0, max(1, H - h_size))

            mask[b, :, y:y + h_size, x:x + w_size] = 1.0
        
        if random.random() < self.hard_edge_prob:
            return mask
        else:
            if self.gaussian_kernel.device != device:
                self.gaussian_kernel = self.gaussian_kernel.to(device)
            
            padding = self.kernel_size // 2
            soft_mask = F.conv2d(mask, self.gaussian_kernel, padding=padding)

            max_vals = soft_mask.amax(dim=(2, 3), keepdim=True) + 1e-6
            soft_mask = soft_mask / max_vals

            return soft_mask
    
    def get_texture_source(self, img: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        B = img.shape[0]

        if B > 1:
            perm = torch.randperm(B, device=img.device)

            offset = (perm == torch.arange(B, device=img.device)).long()
            perm = (perm + offset) % B

            return img[perm], True
        return img, False
    
    def apply_specular_aug(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        device = img.device

        source_img, is_swapped = self.get_texture_source(img)
        reflection_texture = torch.flip(source_img, dims=[3])
        alpha = 0.6 + torch.rand(B, 1, 1, 1, device=device) * 0.3
        content = alpha * reflection_texture + (1 - alpha) * img

        img_aug = img * (1 - mask) + content * mask

        return torch.clamp(img_aug, 0, 255)
    
    def apply_transparent_aug(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        device = img.device

        source_img, is_swapped = self.get_texture_source(img)

        shift_x = random.randint(30, 150)
        shift_y = random.randint(-40, 40)

        if random.random() < 0.5:
            shift_x = -shift_x

        refracted_texture = torch.roll(source_img, shifts=(shift_x, shift_y), dims=(3, 2))
        alpha = torch.rand(B, 1, 1, 1, device=device) * 0.3 + 0.4

        content_in_glass = img * (1 - alpha) + refracted_texture * alpha
        img_aug = img * (1 - mask) + content_in_glass * mask

        return torch.clamp(img_aug, 0, 255)
    
    def apply_spotlight_aug(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        device = img.device

        intensity = torch.rand(B, 1, 1, 1, device=device) * 100 + 150

        content = img + intensity
        content = torch.clamp(content, 0, 255)

        img_aug = img * (1 - mask) + content * mask

        return torch.clamp(img_aug, 0, 255)
    
    def apply_eraser_aug(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        noise = torch.rand_like(img) * 255

        img_aug = img * (1 - mask) + noise * mask

        return torch.clamp(img_aug, 0, 255)
    
    def __call__(self, left_img: torch.Tensor, right_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > self.aug_prob:
            return left_img, right_img
        
        aug_left = left_img.clone()
        aug_right = right_img.clone()
        
        device = left_img.device
        mask = self.generate_mask(left_img.shape, device)

        target_is_left = random.random() < 0.5
        if target_is_left:
            target_img = aug_left
        else:
            target_img = aug_right
        
        r = random.random()
        cumulative_prob = 0.0

        if r < (cumulative_prob := cumulative_prob + self.probs["specular"]):
            target_img = self.apply_specular_aug(target_img, mask)
        elif r < (cumulative_prob := cumulative_prob + self.probs["transparent"]):
            target_img = self.apply_transparent_aug(target_img, mask)
        elif r < (cumulative_prob := cumulative_prob + self.probs["spotlight"]):
            target_img = self.apply_spotlight_aug(target_img, mask)
        else:
            target_img = self.apply_eraser_aug(target_img, mask)
        
        if target_is_left:
            aug_left = target_img
        else:
            aug_right = target_img
        
        return aug_left, aug_right
