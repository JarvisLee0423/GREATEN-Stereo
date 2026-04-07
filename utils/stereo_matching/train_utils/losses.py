import torch
import torch.nn.functional as F
from typing import List


def get_mask(disp_gt: torch.Tensor, valid: torch.Tensor, max_disp: int=700) -> torch.Tensor:
    # Exclude invalid pixels and extremely large displacements.
    mag = torch.sum(disp_gt ** 2, dim=1).sqrt()

    # Exclude extremely large displacements.
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    return valid


def get_metrics(disp_pred: torch.Tensor, disp_gt: torch.Tensor, valid: torch.Tensor, max_disp: int=700) -> dict:
    valid = get_mask(disp_gt, valid, max_disp)

    epe = torch.sum((disp_pred - disp_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        "train/epe": epe.mean().item(),
        "train/1px": (epe < 1).float().mean().item(),
        "train/3px": (epe < 3).float().mean().item(),
        "train/5px": (epe < 5).float().mean().item(),
        "train/bad1": (epe > 1).float().mean().item(),
        "train/bad2": (epe > 2).float().mean().item(),
        "train/bad5": (epe > 5).float().mean().item(),
    }

    return metrics


def smooth_l1_loss(disp_pred: torch.Tensor, disp_gt: torch.Tensor, valid: torch.Tensor, max_disp: int=700) -> torch.Tensor:
    disp_loss = 0.0

    valid = get_mask(disp_gt, valid, max_disp)
    valid = valid.bool() & ~torch.isnan(disp_pred)

    disp_loss += F.smooth_l1_loss(disp_pred[valid], disp_gt[valid], size_average=True)

    return disp_loss


def masked_smooth_l1_loss(disp_pred: torch.Tensor, disp_gt: torch.Tensor, occ_mask: torch.Tensor, valid: torch.Tensor, max_disp: int=700) -> torch.Tensor:
    disp_loss = 0.0

    valid = get_mask(disp_gt, valid, max_disp)
    mask = valid * occ_mask

    disp_loss += F.smooth_l1_loss(disp_pred[mask.bool()], disp_gt[mask.bool()], size_average=True)

    return disp_loss


def sequence_loss(disp_preds: List[torch.Tensor], disp_gt: torch.Tensor, valid: torch.Tensor, loss_gamma: float=0.9, max_disp: int=700) -> torch.Tensor:
    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0

    valid = get_mask(disp_gt, valid, max_disp)

    for i in range(n_predictions):
        # assert not torch.isnan(disp_preds[i]).any() and not torch.isinf(disp_preds[i]).any()
        # We adjust the loss gamma so it is consistent for any number of iterations.
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool() & ~torch.isnan(i_loss)].mean()
    
    return disp_loss


def sequence_error_loss(error_map_feats: List[torch.Tensor], disp_init: torch.Tensor, disp_preds: List[torch.Tensor], disp_gt: torch.Tensor, valid: torch.Tensor, loss_gamma: float=0.9, max_disp: int=700) -> torch.Tensor:
    n_predictions = len(error_map_feats)
    assert n_predictions >= 1
    error_loss = 0.0

    disp_preds.insert(0, disp_init)
    valid = F.interpolate(valid.unsqueeze(1), scale_factor=0.25, mode="nearest").squeeze()
    disp_gt = F.interpolate(disp_gt, scale_factor=0.25, mode="nearest") / 4
    valid = get_mask(disp_gt, valid, max_disp // 4)

    for i in range(n_predictions):
        assert not torch.isnan(disp_preds[i]).any() and not torch.isinf(disp_preds[i]).any()
        # We adjust the loss gamma so it is consistent for any number of iterations.
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        error_map_feat = error_map_feats[i]
        error_map = torch.exp(-error_map_feat)
        disp_pred = disp_preds[i]
        disp_pred = F.interpolate(disp_pred, scale_factor=0.25, mode="nearest") / 4
        error = torch.abs(disp_pred.detach() - disp_gt) / disp_gt
        i_loss = error_map * error + 0.03 * error_map_feat
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        error_loss += i_weight * i_loss[valid.bool()].mean()

    return error_loss


def sequence_gauss_loss(disp_refine: torch.Tensor, final_disp_preds: List[torch.Tensor], mu_preds: List[torch.Tensor], w_preds: List[torch.Tensor], sigma_preds: List[torch.Tensor], disp_gt: torch.Tensor, valid: torch.Tensor, gauss_num: int, max_disp: int=512) -> torch.Tensor:
    """ Loss function defined over sequence of disparity predictions. """

    n_predictions = len(mu_preds)
    assert n_predictions >= 1
    disp_loss = 0.0

    # Exclude extremely large displacements.
    valid = (disp_gt < max_disp) & (valid[:, None].bool()) & (disp_gt >= 0)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()
    N, C, H, W = mu_preds[0].shape
    i_weights = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    
    if (valid == False).all():
        disp_loss = 0
    else:
        for i in range(n_predictions):
            assert not torch.isnan(mu_preds[i]).any() and not torch.isinf(mu_preds[i]).any()
            assert not torch.isnan(w_preds[i]).any() and not torch.isinf(w_preds[i]).any()
            assert not torch.isnan(sigma_preds[i]).any() and not torch.isinf(sigma_preds[i]).any()
            assert not torch.isnan(final_disp_preds[i]).any() and not torch.isinf(final_disp_preds[i]).any()
            # Split the segment.
            mu_preds[i] = mu_preds[i].view(N // gauss_num, gauss_num, 1, H, W)
            w_preds[i] = w_preds[i].view(N // gauss_num, gauss_num, 1, H, W)
            sigma_preds[i] = sigma_preds[i].view(N // gauss_num, gauss_num, 1, H, W)
            w = w_preds[i]

            i_loss1 = (final_disp_preds[i] - disp_gt).abs()
            i_loss2 = torch.mean((mu_preds[i] - disp_gt[:, None]).abs(), dim=1)
            disp_loss += i_weights[i] * (
                i_loss1.view(-1)[valid.view(-1)].mean() + i_loss2.view(-1)[valid.view(-1)].mean()
            )
        disp_loss += 1.4 * F.smooth_l1_loss(disp_refine[valid], disp_gt[valid], size_average=True)

    return disp_loss
