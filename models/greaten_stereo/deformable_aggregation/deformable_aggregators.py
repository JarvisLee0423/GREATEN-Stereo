import torch
import torch.nn as nn
from typing import Tuple, Union
from einops import rearrange
from models.greaten_stereo.deformable_aggregation.key_points_generators import Sparse1DKeyPointsGenerator, Sparse2DKeyPointsGenerator

try:
    from utils.stereo_matching.cuda_utils.deformable_aggregation import DeformableAggregationFunction as DAF
except:
    DAF = None


def to_3d(inputs: torch.Tensor, dim: int=1) -> torch.Tensor:
    if dim == 1:
        return rearrange(inputs, "b c h w -> (b h) w c")
    else:
        return rearrange(inputs, "b c h w -> b (h w) c")


def to_4d(inputs: torch.Tensor, b: Union[int, float], h: Union[int, float], w: Union[int, float]=None, dim: int=1) -> torch.Tensor:
    if dim == 1:
        return rearrange(inputs, "(b h) w c -> b c h w", b=b, h=h)
    else:
        return rearrange(inputs, "b (h w) c -> b c h w", h=h, w=w)


class Deformable1DFeatureAggregator(nn.Module):
    def __init__(
        self,
        feat_channels: int,
        num_pts: int=9,
        num_groups: int=8,
        proj_drop: float=0.0,
        use_deformable_func: bool=True,
    ):
        super(Deformable1DFeatureAggregator, self).__init__()

        if feat_channels % num_groups != 0:
            raise ValueError(
                f"The feat_channels must be divisible by num_groups, but got {feat_channels} and {num_groups}."
            )
        self.feat_channels = feat_channels
        self.group_channels = int(feat_channels / num_groups)
        self.num_pts = num_pts
        self.num_groups = num_groups
        self.use_deformable_func = use_deformable_func and DAF is not None
        assert self.use_deformable_func, "Current implementation only supports to use deformable aggregation cuda kernel."
        self.norm = nn.LayerNorm(feat_channels)
        self.norm_context = nn.LayerNorm(feat_channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.key_points_generator = Sparse1DKeyPointsGenerator(num_pts, feat_channels)
        self.weight_proj = nn.Conv2d(feat_channels, num_groups * self.num_pts, kernel_size=1)
        self.value_proj = nn.Conv2d(feat_channels, feat_channels, kernel_size=1)
        self.output_proj = nn.Conv2d(feat_channels, feat_channels, kernel_size=1)
    
    def get_weights(self, feats: torch.Tensor) -> torch.Tensor:
        bs, _, H, W = feats.shape
        weights = self.weight_proj(feats).permute(0, 2, 3, 1).reshape(
            bs, H * W, -1, self.num_groups
        ).reshape(
            bs, H * W, 1, 1, self.num_pts, self.num_groups
        )

        return weights
    
    def forward(self, feats1: torch.Tensor, feats2: torch.Tensor, anchor_points: torch.Tensor) -> Tuple[torch.Tensor]:
        bs, _, H, W = feats1.shape
        feats1 = to_4d(self.norm(to_3d(feats1, dim=2)), bs, H, W, dim=2)
        feats2 = to_4d(self.norm_context(to_3d(feats2, dim=2)), bs, H, W, dim=2)
        # Get value.
        feats_value = self.value_proj(feats2)
        feature_maps = DAF.feature_maps_format([feats_value.unsqueeze(1)])
        # Get attention weights (query x key).
        weights = self.get_weights(feats1)
        weights = weights.permute(0, 1, 4, 2, 3, 5).contiguous().reshape(
            bs,
            H * W,
            self.num_pts,
            1, 1,
            self.num_groups,
        )
        weights = weights.flatten(2, 4).softmax(dim=-2).reshape(
            bs,
            H * W * self.num_pts,
            1, 1,
            self.num_groups,
        )
        # Get the sampling points.
        key_points = self.key_points_generator(feats1, anchor_points)
        key_points = key_points.reshape(bs, H * W * self.num_pts, 2).unsqueeze(-2)

        output = DAF.apply(
            *feature_maps, key_points, weights
        ).reshape(bs, H * W, self.num_pts, self.feat_channels)
        output = output.sum(dim=2)
        output = output.reshape(bs, H, W, self.feat_channels).permute(0, 3, 1, 2)
        output = self.proj_drop(self.output_proj(output))

        return output, key_points.squeeze(-2).reshape(bs, H, W, self.num_pts, 2)


class NormalGuidedDeformable1DFeatureAggregator(nn.Module):
    def __init__(
        self,
        feat_channels: int,
        num_pts: int=9,
        num_groups: int=8,
        proj_drop: float=0.0,
        use_deformable_func: bool=True,
    ):
        super(NormalGuidedDeformable1DFeatureAggregator, self).__init__()

        if feat_channels % num_groups != 0:
            raise ValueError(
                f"The feat_channels must be divisible by num_groups, but got {feat_channels} and {num_groups}."
            )
        self.feat_channels = feat_channels
        self.group_channels = int(feat_channels / num_groups)
        self.num_pts = num_pts
        self.num_groups = num_groups
        self.use_deformable_func = use_deformable_func and DAF is not None
        assert self.use_deformable_func, "Current implementation only supports to use deformable aggregation cuda kernel."
        self.norm = nn.LayerNorm(feat_channels)
        self.norm_context = nn.LayerNorm(feat_channels)
        self.norm_geometric = nn.LayerNorm(feat_channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.context_key_points_generator = Sparse1DKeyPointsGenerator(num_pts, feat_channels)
        self.geometric_key_points_generator = Sparse1DKeyPointsGenerator(num_pts, feat_channels)
        self.weight_context_proj = nn.Conv2d(feat_channels, num_groups * self.num_pts, kernel_size=1)
        self.weight_geometric_proj = nn.Conv2d(feat_channels, num_groups * self.num_pts, kernel_size=1)
        self.value_context_proj = nn.Conv2d(feat_channels, feat_channels, kernel_size=1)
        self.value_geometric_proj = nn.Conv2d(feat_channels, feat_channels, kernel_size=1)
        self.output_proj = nn.Conv2d(feat_channels * 2, feat_channels, kernel_size=1)
    
    def get_weights(self, match: torch.Tensor) -> torch.Tensor:
        bs, _, H, W = match.shape
        context_weights = self.weight_context_proj(match).permute(0, 2, 3, 1).reshape(
            bs, H * W, -1, self.num_groups
        ).reshape(
            bs, H * W, 1, 1, self.num_pts, self.num_groups
        )
        geometric_weights = self.weight_geometric_proj(match).permute(0, 2, 3, 1).reshape(
            bs, H * W, -1, self.num_groups
        ).reshape(
            bs, H * W, 1, 1, self.num_pts, self.num_groups
        )

        return context_weights, geometric_weights
    
    def forward(self, mask: torch.Tensor, match: torch.Tensor, context: torch.Tensor, geometric: torch.Tensor, context_anchor_points: torch.Tensor, geometric_anchor_points: torch.Tensor) -> Tuple[torch.Tensor]:
        bs, _, H, W = match.shape
        match = to_4d(self.norm(to_3d(match, dim=2)), bs, H, W, dim=2)
        context = to_4d(self.norm_context(to_3d(context, dim=2)), bs, H, W, dim=2)
        geometric = to_4d(self.norm_geometric(to_3d(geometric, dim=2)), bs, H, W, dim=2)
        # Get value.
        context_feats_value = self.value_context_proj(context)
        geometric_feats_value = self.value_geometric_proj(geometric)
        context_feature_maps = DAF.feature_maps_format([context_feats_value.unsqueeze(1)])
        geometric_feature_maps = DAF.feature_maps_format([geometric_feats_value.unsqueeze(1)])
        # Get attention weights (query x key).
        context_weights, geometric_weights = self.get_weights(match)
        context_weights = context_weights.permute(0, 1, 4, 2, 3, 5).contiguous().reshape(
            bs,
            H * W,
            self.num_pts,
            1, 1,
            self.num_groups,
        )
        context_weights = context_weights.flatten(2, 4).softmax(dim=-2).reshape(
            bs,
            H * W * self.num_pts,
            1, 1,
            self.num_groups,
        )
        geometric_weights = geometric_weights.permute(0, 1, 4, 2, 3, 5).contiguous().reshape(
            bs,
            H * W,
            self.num_pts,
            1, 1,
            self.num_groups,
        )
        geometric_weights = geometric_weights.flatten(2, 4).softmax(dim=-2).reshape(
            bs,
            H * W * self.num_pts,
            1, 1,
            self.num_groups,
        )
        # Get the sampling points.
        context_key_points = self.context_key_points_generator(match, context_anchor_points)
        context_key_points = context_key_points.reshape(bs, H * W * self.num_pts, 2).unsqueeze(-2)
        geometric_key_points = self.geometric_key_points_generator(match, geometric_anchor_points)
        geometric_key_points = geometric_key_points.reshape(bs, H * W * self.num_pts, 2).unsqueeze(-2)

        context_output = DAF.apply(
            *context_feature_maps, context_key_points, context_weights
        ).reshape(bs, H * W, self.num_pts, self.feat_channels)
        context_output = context_output.sum(dim=2)
        context_output = context_output.reshape(bs, H, W, self.feat_channels).permute(0, 3, 1, 2)
        
        geometric_output = DAF.apply(
            *geometric_feature_maps, geometric_key_points, geometric_weights
        ).reshape(bs, H * W, self.num_pts, self.feat_channels)
        geometric_output = geometric_output.sum(dim=2)
        geometric_output = geometric_output.reshape(bs, H, W, self.feat_channels).permute(0, 3, 1, 2)
        
        output = torch.cat([context_output * mask, geometric_output], dim=1)
        output = self.proj_drop(self.output_proj(output))

        return output, context_key_points.squeeze(-2).reshape(bs, H, W, self.num_pts, 2), geometric_key_points.squeeze(-2).reshape(bs, H, W, self.num_pts, 2)


class Deformable2DFeatureAggregator(nn.Module):
    def __init__(
        self,
        feat_channels: int,
        num_pts: int=9,
        num_groups: int=8,
        proj_drop: float=0.0,
        use_deformable_func: bool=True,
    ):
        super(Deformable2DFeatureAggregator, self).__init__()

        if feat_channels % num_groups != 0:
            raise ValueError(
                f"The feat_channels must be divisible by num_groups, but got {feat_channels} and {num_groups}."
            )
        self.feat_channels = feat_channels
        self.group_channels = int(feat_channels / num_groups)
        self.num_pts = num_pts
        self.num_groups = num_groups
        self.use_deformable_func = use_deformable_func and DAF is not None
        assert self.use_deformable_func, "Current implementation only supports to use deformable aggregation cuda kernel."
        self.proj_drop = nn.Dropout(proj_drop)
        self.key_points_generator = Sparse2DKeyPointsGenerator(num_pts, feat_channels)
        self.weights_proj = nn.Conv2d(feat_channels, num_groups * self.num_pts, kernel_size=1)
        self.value_proj = nn.Conv2d(feat_channels, feat_channels, kernel_size=1)
        self.output_proj = nn.Conv2d(feat_channels, feat_channels, kernel_size=1)
    
    def get_weights(self, feats: torch.Tensor) -> torch.Tensor:
        bs, _, H, W = feats.shape
        weights = self.weights_proj(feats).permute(0, 2, 3, 1).reshape(
            bs, H * W, -1, self.num_groups
        ).reshape(
            bs, H * W, 1, 1, self.num_pts, self.num_groups
        )
        
        return weights
    
    def forward(self, feats: torch.Tensor, anchor_points: torch.Tensor) -> torch.Tensor:
        bs, _, H, W = feats.shape
        # Get value.
        feats_value = self.value_proj(feats)
        feature_maps = DAF.feature_maps_format([feats_value.unsqueeze(1)])
        # Get attention weights (query x key).
        weights = self.get_weights(feats)
        weights = weights.permute(0, 1, 4, 2, 3, 5).contiguous().reshape(
            bs,
            H * W,
            self.num_pts,
            1, 1,
            self.num_groups,
        )
        weights = weights.flatten(2, 4).softmax(dim=-2).reshape(
            bs,
            H * W * self.num_pts,
            1, 1,
            self.num_groups,
        )
        # Get the sampling points.
        key_points = self.key_points_generator(feats, anchor_points)
        key_points = key_points.reshape(bs, H * W * self.num_pts, 2).unsqueeze(-2)

        output = DAF.apply(
            *feature_maps, key_points, weights
        ).reshape(bs, H * W, self.num_pts, self.feat_channels)
        output = output.sum(dim=2)
        output = output.reshape(bs, H, W, self.feat_channels).permute(0, 3, 1, 2)
        output = self.proj_drop(self.output_proj(output))

        return output
