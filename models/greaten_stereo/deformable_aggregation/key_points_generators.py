import torch
import torch.nn as nn


class Sparse1DKeyPointsGenerator(nn.Module):
    def __init__(self, num_pts: int, feat_channels: int):
        super(Sparse1DKeyPointsGenerator, self).__init__()

        self.num_pts = num_pts
        self.feat_channels = feat_channels

        self.sampling_offsets = nn.Conv2d(feat_channels, num_pts, kernel_size=1)
    
    def forward(self, query: torch.Tensor, anchor_points: torch.Tensor) -> torch.Tensor:
        bs, _, H, W = query.shape
        H_tensor = torch.tensor(H).to(query.device)
        W_tensor = torch.tensor(W).to(query.device)
        spatial_shape = torch.stack([W_tensor, H_tensor], dim=-1)
        sampling_offsets = self.sampling_offsets(query).permute(0, 2, 3, 1).view(bs, H, W, self.num_pts, 1)
        # TODO [Need extra verification]: Use the same linear layer to generate the offsets for both left and right image.
        # TODO [Need extra verification]: Ideally, the matched points from left and right get the similar feature vector.
        # TODO [Need extra verification]: Then the same linear layer may generate similar offsets values for both left and right image.
        # TODO [Need extra verification]: However, the matched pixels from left and right has totally different anchor positions.
        # TODO [Need extra verification]: Specifically, in the same coordinate, the left pixel position usually lies on the right side of the matched right pixel position.
        # TODO [Need extra verification]: Therefore, the similar offsets values for the left and right can not be used in the same way.
        # TODO [Need extra verification]: For left -> left + offsets (Assume the linear layer generate the nagetive values).
        # TODO [Need extra verification]: For right -> right - offsets (Assume the linear layer generate the nagetive values).
        sampling_offsets_left, sampling_offsets_right = sampling_offsets.chunk(2, dim=0)
        sampling_offsets = torch.cat([-sampling_offsets_left, sampling_offsets_right], dim=0)
        zeros_offsets = torch.zeros_like(sampling_offsets)
        sampling_offsets = torch.cat([sampling_offsets, zeros_offsets], dim=-1)
        sampling_offsets = sampling_offsets / spatial_shape[None, None, None, None, :]
        if len(anchor_points.shape) == 4:
            sampling_points = anchor_points[:, :, :, None, :] + sampling_offsets
        else:
            sampling_points = anchor_points + sampling_offsets

        return sampling_points


class Sparse2DKeyPointsGenerator(nn.Module):
    def __init__(self, num_pts: int, feat_channels: int):
        super(Sparse2DKeyPointsGenerator, self).__init__()
        
        self.num_pts = num_pts
        self.feat_channels = feat_channels

        self.sampling_offsets = nn.Conv2d(feat_channels, num_pts * 2, kernel_size=1)
    
    def forward(self, query: torch.Tensor, anchor_points: torch.Tensor) -> torch.Tensor:
        bs, _, H, W = query.shape
        H_tensor = torch.tensor(H).to(query.device)
        W_tensor = torch.tensor(W).to(query.device)
        spatial_shape = torch.stack([W_tensor, H_tensor], dim=-1)
        sampling_offsets = self.sampling_offsets(query).permute(0, 2, 3, 1).view(bs, H, W, self.num_pts, 2)
        sampling_offsets = sampling_offsets / spatial_shape[None, None, None, None, :]
        sampling_points = anchor_points[:, :, :, None, :] + sampling_offsets

        return sampling_points
