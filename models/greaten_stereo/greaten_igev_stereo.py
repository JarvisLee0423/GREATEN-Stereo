import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import List, Tuple, Union
from utils.utils import estimate_normals, autocast
from utils.stereo_matching.data_utils.augmentors import PhysicalSpecularTransparentAugmentor
from models.greaten_stereo.updaters import BasicMultiUpdateBlock
from models.greaten_stereo.cost_volumes import ChannelExtensionSimpleExcitiveAttentionCombinedVolume, NormalGuidedChannelExtensionSimpleExcitiveAttentionCombinedVolume, CombinedVolumeSampler
from models.greaten_stereo.basic_modules import context_upsample, Conv2x, Conv2xIN, BasicConvIN
from models.greaten_stereo.feature_extractors import Mobilenetv2Encoder, MultiBasicEncoder, GateMaskGenerator, GatedContextGeometricFusion, SurfaceNormalSimpleEncoder
from modules.backbones.depth_anything.depth_anything import DepthAnythingV2


st_augmentor = PhysicalSpecularTransparentAugmentor()


class GREATENStereo(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(GREATENStereo, self).__init__()

        self.args = args
        self.freezing_module_list = []

        feat_channels = [96, 64, 192, 160]
        context_channels = args.channels

        self.cnet = MultiBasicEncoder(
            out_channels=[args.channels, context_channels],
            norm_fn="instance",
            downsample=args.n_downsample,
        )

        self.update_block = BasicMultiUpdateBlock(self.args, channels=args.channels)

        self.context_zqr_convs = nn.ModuleList([
            nn.Conv2d(context_channels[i], args.channels[i] * 3, 3, padding=3 // 2)
            for i in range(self.args.n_gru_layers)
        ])

        self.feature = Mobilenetv2Encoder()

        # Create feature extractor stems.
        self.stem_2 = nn.Sequential(
            BasicConvIN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        )
        self.stem_4 = nn.Sequential(
            BasicConvIN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48),
            nn.ReLU(),
        )
        self.spx = nn.Sequential(
            nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1),
        )
        self.spx_2 = Conv2xIN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConvIN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24),
            nn.ReLU(),
        )
        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(
            nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1),
        )

        if self.args.infer_normal:
            print("Requiering DepthAnything backbone for normal estimation...")
            self.conv = BasicConvIN(feat_channels[0], feat_channels[0], kernel_size=3, padding=1, stride=1)
            self.desc = nn.Conv2d(feat_channels[0], feat_channels[0], kernel_size=1, padding=0, stride=1)

            # Create normal feature extractor stems.
            self.surface_normal_encoder = SurfaceNormalSimpleEncoder(in_channels=3, feat_channels=feat_channels)
            self.surface_normal_stem_2 = nn.Sequential(
                BasicConvIN(3, 32, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(32),
                nn.ReLU(),
            )
            self.upsample_stem_2x = nn.Sequential(
                BasicConvIN(64, 32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(32),
                nn.ReLU(),
            )

            self.excitive_attention_volume = NormalGuidedChannelExtensionSimpleExcitiveAttentionCombinedVolume(96, feat_channels, self.args.max_disp, 8)

            self.gate_mask_generator = GateMaskGenerator(feat_channels[0], reduction=16)
            self.gated_context_geometric_fusion_4x = GatedContextGeometricFusion(feat_channels[0])
            self.gated_context_geometric_fusion_8x = GatedContextGeometricFusion(feat_channels[1])
            self.gated_context_geometric_fusion_16x = GatedContextGeometricFusion(feat_channels[2])
            self.gated_context_geometric_fusion_32x = GatedContextGeometricFusion(feat_channels[3])

            self.intermediate_layer_idx = {
                "vits": [2, 5, 8, 11],
                "vitb": [2, 5, 8, 11],
                "vitl": [4, 11, 17, 23],
                "vitg": [9, 19, 29, 39],
            }
            mono_model_configs = {
                "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
                "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
                "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
                "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
            }
            dim_list_config = mono_model_configs[self.args.backbone_type]["features"]
            dim_list = []
            dim_list.append(dim_list_config)
            depth_anything = DepthAnythingV2(**mono_model_configs[args.backbone_type])
            state_dict_dpt = torch.load(args.backbone_ckpt, map_location="cpu")
            print(f"Loading ckpt for the backbone {os.path.basename(args.backbone_ckpt)}...")
            depth_anything.load_state_dict(state_dict_dpt, strict=True)
            print("Done Loading!")
            self.mono_encoder = depth_anything.pretrained
            self.mono_decoder = depth_anything.depth_head
            print(f"Freezing the gradient in mono encoder...")
            self.mono_encoder.requires_grad_(False)
            self.mono_decoder.requires_grad_(False)
            print("Done Freezing!")

            del depth_anything, state_dict_dpt

            self.net_stem = nn.Sequential(
                nn.Conv2d(args.channels[2] + feat_channels[0] * 2, args.channels[2], kernel_size=1, padding=0, stride=1),
                nn.Conv2d(args.channels[1] + feat_channels[1] * 2, args.channels[1], kernel_size=1, padding=0, stride=1),
                nn.Conv2d(args.channels[0] + feat_channels[2] * 2, args.channels[0], kernel_size=1, padding=0, stride=1),
            )
            self.inp_stem = nn.Sequential(
                nn.Conv2d(context_channels[2] + feat_channels[0] * 2, context_channels[2], kernel_size=1, padding=0, stride=1),
                nn.Conv2d(context_channels[1] + feat_channels[1] * 2, context_channels[1], kernel_size=1, padding=0, stride=1),
                nn.Conv2d(context_channels[0] + feat_channels[2] * 2, context_channels[0], kernel_size=1, padding=0, stride=1),
            )

            print("Done setting up normal-guided GREATEN-Stereo model.")
        else:
            self.excitive_attention_volume = ChannelExtensionSimpleExcitiveAttentionCombinedVolume(96, feat_channels, self.args.max_disp, 8)

            self.net_stem = nn.Sequential(
                nn.Conv2d(args.channels[2] + feat_channels[0], args.channels[2], kernel_size=1, padding=0, stride=1),
                nn.Conv2d(args.channels[1] + feat_channels[1], args.channels[1], kernel_size=1, padding=0, stride=1),
                nn.Conv2d(args.channels[0] + feat_channels[2], args.channels[0], kernel_size=1, padding=0, stride=1),
            )
            self.inp_stem = nn.Sequential(
                nn.Conv2d(context_channels[2] + feat_channels[0], context_channels[2], kernel_size=1, padding=0, stride=1),
                nn.Conv2d(context_channels[1] + feat_channels[1], context_channels[1], kernel_size=1, padding=0, stride=1),
                nn.Conv2d(context_channels[0] + feat_channels[2], context_channels[0], kernel_size=1, padding=0, stride=1),
            )

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def freeze_bn(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            if isinstance(module, nn.SyncBatchNorm) and name in self.freezing_module_list:
                module.eval()
    
    def mark_module_for_freezing(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self.freezing_module_list.append(name)
    
    def normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        normalizer = torchvision.transforms.Normalize(
            mean=self.mean,
            std=self.std,
            inplace=False,
        )

        return normalizer(img).contiguous()
    
    def upsample_disp(self, disp: torch.Tensor, mask_feat_4: torch.Tensor, stem_2x: torch.Tensor) -> torch.Tensor:
        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp * 4, spx_pred).unsqueeze(1)
        
        return up_disp

    def infer_mono(self, left_img: torch.Tensor, right_img: torch.Tensor) -> Tuple[torch.Tensor, List[Union[list, torch.Tensor]]]:
        height_ori, width_ori = left_img.shape[2:]
        resize_left_img = F.interpolate(left_img, scale_factor=14 / 16, mode="bilinear", align_corners=False)
        resize_right_img = F.interpolate(right_img, scale_factor=14 / 16, mode="bilinear", align_corners=False)

        patch_h, patch_w = resize_left_img.shape[-2] // 14, resize_left_img.shape[-1] // 14
        feat_left_encoder = self.mono_encoder.get_intermediate_layers(resize_left_img, self.intermediate_layer_idx[self.args.backbone_type], return_class_token=True)
        feat_right_encoder = self.mono_encoder.get_intermediate_layers(resize_right_img, self.intermediate_layer_idx[self.args.backbone_type], return_class_token=True)

        depth_mono_left = self.mono_decoder(feat_left_encoder, patch_h, patch_w)
        depth_mono_left = F.relu(depth_mono_left)
        depth_mono_left = F.interpolate(depth_mono_left, size=(height_ori, width_ori), mode="bilinear", align_corners=False)
        depth_mono_left = (depth_mono_left - torch.amin(depth_mono_left, dim=(1, 2, 3), keepdim=True)) / (torch.amax(depth_mono_left, dim=(1, 2, 3), keepdim=True) - torch.amin(depth_mono_left, dim=(1, 2, 3), keepdim=True)).contiguous()
        
        depth_mono_right = self.mono_decoder(feat_right_encoder, patch_h, patch_w)
        depth_mono_right = F.relu(depth_mono_right)
        depth_mono_right = F.interpolate(depth_mono_right, size=(height_ori, width_ori), mode="bilinear", align_corners=False)
        depth_mono_right = (depth_mono_right - torch.amin(depth_mono_right, dim=(1, 2, 3), keepdim=True)) / (torch.amax(depth_mono_right, dim=(1, 2, 3), keepdim=True) - torch.amin(depth_mono_right, dim=(1, 2, 3), keepdim=True)).contiguous()

        return depth_mono_left, depth_mono_right
    
    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor, iters: int=12, disp_init: torch.Tensor=None, disp_gt: torch.Tensor=None, test_mode: bool=False) -> Tuple[Union[torch.Tensor, Union[List[torch.Tensor], dict, int, float]]]:
        """ Estimate disparity between pair of frames. """

        # Take the specular-transparent augmentation during the training.
        if self.training and not test_mode and self.args.apply_st_augmentation and disp_gt is not None:
            st_aug_left_img, st_aug_right_img = st_augmentor(left_img, right_img, disp_gt)
        else:
            st_aug_left_img = left_img.clone()
            st_aug_right_img = right_img.clone()
        st_aug_left_img_vis = st_aug_left_img.clone()
        st_aug_right_img_vis = st_aug_right_img.clone()
        st_aug_left_img = self.normalize_image(st_aug_left_img / 255.0)
        st_aug_right_img = self.normalize_image(st_aug_right_img / 255.0)
        left_img = self.normalize_image(left_img / 255.0)
        right_img = self.normalize_image(right_img / 255.0)
        # left_img = (2 * (left_img / 255.0) - 1.0).contiguous()
        # right_img = (2 * (right_img / 255.0) - 1.0).contiguous()
        if self.args.infer_normal:
            depth_mono_left, depth_mono_right = self.infer_mono(left_img, right_img)
            # Get the 1 / 4 scale mono depth.
            depth_mono_left_4x = F.interpolate(depth_mono_left, scale_factor=1 / 4, mode="bilinear", align_corners=False)
            depth_mono_right_4x = F.interpolate(depth_mono_right, scale_factor=1 / 4, mode="bilinear", align_corners=False)
            # Convert the mono depth to surface normal.
            normals_left = estimate_normals(depth_mono_left, normal_gain=depth_mono_left.shape[-1] / 10)
            normals_right = estimate_normals(depth_mono_right, normal_gain=depth_mono_right.shape[-1] / 10)
            # Get the 1 / 4 scale surface normal.
            normals_left_4x = estimate_normals(depth_mono_left_4x, normal_gain=depth_mono_left_4x.shape[-1] / 10)
            normals_right_4x = estimate_normals(depth_mono_right_4x, normal_gain=depth_mono_right_4x.shape[-1] / 10)
            # Get the 1 / 4 specular-transparent augmented image.
            st_aug_left_img_4x = F.interpolate(st_aug_left_img, scale_factor=1 / 4, mode="bilinear", align_corners=False)
            st_aug_right_img_4x = F.interpolate(st_aug_right_img, scale_factor=1 / 4, mode="bilinear", align_corners=False)

        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            feat_left = self.feature(st_aug_left_img)
            feat_right = self.feature(st_aug_right_img)
            stem_2x = self.stem_2(st_aug_left_img)
            stem_4x = self.stem_4(stem_2x)
            stem_2y = self.stem_2(st_aug_right_img)
            stem_4y = self.stem_4(stem_2y)
            feat_left[0] = torch.cat((feat_left[0], stem_4x), 1)
            feat_right[0] = torch.cat((feat_right[0], stem_4y), 1)
            if self.args.infer_normal:
                feat_left[0] = self.desc(self.conv(feat_left[0]))
                feat_right[0] = self.desc(self.conv(feat_right[0]))
                
                normals_feat_left = self.surface_normal_encoder(normals_left.detach())
                normals_feat_right = self.surface_normal_encoder(normals_right.detach())

                # Generate the left gate mask from the highest resolution feature.
                gate_mask_left_4x = self.gate_mask_generator(feat_left[0], normals_feat_left[0], st_aug_left_img_4x, normals_left_4x)
                # Generate the left gate mask resolution pyramid.
                gate_mask_left_8x = F.interpolate(gate_mask_left_4x, scale_factor=1 / 2, mode="bilinear", align_corners=False)
                gate_mask_left_16x = F.interpolate(gate_mask_left_4x, scale_factor=1 / 4, mode="bilinear", align_corners=False)
                gate_mask_left_32x = F.interpolate(gate_mask_left_4x, scale_factor=1 / 8, mode="bilinear", align_corners=False)
                gate_mask_left = [
                    gate_mask_left_4x, gate_mask_left_8x, gate_mask_left_16x, gate_mask_left_32x,
                ]

                # Generate the right gate mask from the highest resolution feature.
                gate_mask_right_4x = self.gate_mask_generator(feat_right[0], normals_feat_right[0], st_aug_right_img_4x, normals_right_4x)
                # Generate the right gate mask resolution pyramid.
                gate_mask_right_8x = F.interpolate(gate_mask_right_4x, scale_factor=1 / 2, mode="bilinear", align_corners=False)
                gate_mask_right_16x = F.interpolate(gate_mask_right_4x, scale_factor=1 / 4, mode="bilinear", align_corners=False)
                gate_mask_right_32x = F.interpolate(gate_mask_right_4x, scale_factor=1 / 8, mode="bilinear", align_corners=False)
                gate_mask_right = [
                    gate_mask_right_4x, gate_mask_right_8x, gate_mask_right_16x, gate_mask_right_32x,
                ]

                # Apply the gated left context-geometric feature fusion.
                context_geometric_left_4x, gated_context_left_4x = self.gated_context_geometric_fusion_4x(feat_left[0], normals_feat_left[0], gate_mask_left[0])
                context_geometric_left_8x, gated_context_left_8x = self.gated_context_geometric_fusion_8x(feat_left[1], normals_feat_left[1], gate_mask_left[1])
                context_geometric_left_16x, gated_context_left_16x = self.gated_context_geometric_fusion_16x(feat_left[2], normals_feat_left[2], gate_mask_left[2])
                context_geometric_left_32x, gated_context_left_32x = self.gated_context_geometric_fusion_32x(feat_left[3], normals_feat_left[3], gate_mask_left[3])
                context_geometric_left = [
                    context_geometric_left_4x, context_geometric_left_8x, context_geometric_left_16x, context_geometric_left_32x,
                ]
                gated_context_left = [
                    gated_context_left_4x, gated_context_left_8x, gated_context_left_16x, gated_context_left_32x,
                ]
                
                # Apply the gated right context-geometric feature fusion.
                context_geometric_right_4x, gated_context_right_4x = self.gated_context_geometric_fusion_4x(feat_right[0], normals_feat_right[0], gate_mask_right[0])
                context_geometric_right_8x, gated_context_right_8x = self.gated_context_geometric_fusion_8x(feat_right[1], normals_feat_right[1], gate_mask_right[1])
                context_geometric_right_16x, gated_context_right_16x = self.gated_context_geometric_fusion_16x(feat_right[2], normals_feat_right[2], gate_mask_right[2])
                context_geometric_right_32x, gated_context_right_32x = self.gated_context_geometric_fusion_32x(feat_right[3], normals_feat_right[3], gate_mask_right[3])
                context_geometric_right = [
                    context_geometric_right_4x, context_geometric_right_8x, context_geometric_right_16x, context_geometric_right_32x,
                ]
                gated_context_right = [
                    gated_context_right_4x, gated_context_right_8x, gated_context_right_16x, gated_context_right_32x,
                ]

                # Apply the global excitive attention volume construction.
                context_geometric_left, match_left, match_right, global_volume, init_disp = self.excitive_attention_volume(
                    context_geometric_left, context_geometric_right,
                    gated_context_left, gated_context_right,
                    normals_feat_left, normals_feat_right,
                    gate_mask_left, gate_mask_right,
                )

                # Generate the upsampling features.
                gate_mask_left_2x = F.interpolate(gate_mask_left_4x, scale_factor=2.0, mode="bilinear", align_corners=False)
                normals_stem_2x = self.surface_normal_stem_2(normals_left.detach())
                upsample_stem_2x = self.upsample_stem_2x(
                    torch.cat([stem_2x * gate_mask_left_2x, normals_stem_2x], dim=1)
                )
            else:
                context_left, match_left, match_right, global_volume, init_disp = self.excitive_attention_volume(feat_left, feat_right)
            
            if not test_mode:
                if self.args.infer_normal:
                    xspx = self.spx_4(context_geometric_left_4x)
                    xspx = self.spx_2(xspx, upsample_stem_2x)
                else:
                    xspx = self.spx_4(feat_left[0])
                    xspx = self.spx_2(xspx, stem_2x)
                spx_pred = self.spx(xspx)
                spx_pred = F.softmax(spx_pred, 1)
            
            if self.args.infer_normal:
                cnet_list = self.cnet(st_aug_left_img, num_layers=self.args.n_gru_layers)
                net_list = [torch.tanh(self.net_stem[i](torch.cat([x[0] * gate_mask_left[i], normals_feat_left[i], context_geometric_left[i]], dim=1))) for i, x in enumerate(cnet_list)]
                inp_list = [torch.relu(self.inp_stem[i](torch.cat([x[1] * gate_mask_left[i], normals_feat_left[i], context_geometric_left[i]], dim=1))) for i, x in enumerate(cnet_list)]
            else:
                cnet_list = self.cnet(left_img, num_layers=self.args.n_gru_layers)
                net_list = [torch.tanh(self.net_stem[i](torch.cat([x[0], context_left[i]], dim=1))) for i, x in enumerate(cnet_list)]
                inp_list = [torch.relu(self.inp_stem[i](torch.cat([x[1], context_left[i]], dim=1))) for i, x in enumerate(cnet_list)]
            inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]
        
        cv_block = CombinedVolumeSampler
        feat_batch, _, feat_h, feat_w = match_left.shape
        cv_fn = cv_block(match_left.float(), match_right.float(), global_volume.float(), radius=self.args.cv_radius, num_levels=self.args.cv_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
        disp = init_disp
        disp_preds = []

        # GRUs iterations to update disparity.
        for iter in range(iters):
            disp = disp.detach()
            geo_feat = cv_fn(disp, coords)
            with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res ConvGRU.
                    net_list = self.update_block(net_list, inp_list, iter16=True, iter08=False, iter04=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru: # Update low-res ConvGRU and mid-res ConvGRU.
                    net_list = self.update_block(net_list, inp_list, iter16=self.args.n_gru_layers==3, iter08=True, iter04=False, update=False)
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers >= 2)
            
            disp = disp + delta_disp
            if test_mode and iter < iters - 1:
                continue

            # Upsample predictions.
            if self.args.infer_normal:
                disp_up = self.upsample_disp(disp, mask_feat_4, upsample_stem_2x)
            else:
                disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)
        
        if test_mode:
            if self.args.infer_normal:
                return init_disp, disp_up, ({"apc": cv_fn.local_volume_pyramid, "ccv": cv_fn.global_volume_pyramid}, feat_batch, feat_h, feat_w, [st_aug_left_img_vis, normals_left, normals_left_4x, gate_mask_left_4x], [st_aug_right_img_vis, normals_right, normals_right_4x, gate_mask_right_4x])
            else:
                return init_disp, disp_up, ({"apc": cv_fn.local_volume_pyramid, "ccv": cv_fn.global_volume_pyramid}, feat_batch, feat_h, feat_w)
        
        init_disp = context_upsample(init_disp * 4.0, spx_pred.float()).unsqueeze(1)

        if self.args.infer_normal:
            return (init_disp, [st_aug_left_img_vis, normals_left, normals_left_4x, gate_mask_left_4x], [st_aug_right_img_vis, normals_right, normals_right_4x, gate_mask_right_4x]), disp_preds
        else:
            return init_disp, disp_preds
