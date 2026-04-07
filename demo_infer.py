import torch
import torch.nn.functional as F
import os
import cv2
import argparse
import importlib
import numpy as np
from utils.utils import InputPadder, vis_normals


DEVICE = "cuda"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def demo(args):
    # Get model.
    exp_name = args.name
    assert "-stereo" in exp_name, f"Wrong experiment name {exp_name}. Formate of name argument must be 'xxx-stereo'."
    exp_name = exp_name.split(".")[0]
    exp_name = exp_name.replace("-", "_")
    if len(exp_name.split("_")) == 3:
        model_name = exp_name.split("_")[0].upper() + "Stereo"
        module_name = exp_name.split("_")[0] + "_" + exp_name.split("_")[-1]
    else:
        model_name = exp_name.split("_stereo")[0].upper() + "Stereo"
        module_name = exp_name
    module = getattr(importlib.import_module(f"models.{module_name}.{exp_name}"), model_name)
    model = torch.nn.DataParallel(module(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    os.makedirs(args.output_directory, exist_ok=True)
    file_stem = f"{args.output_directory}/{args.stereo_image_path.split('/')[-1].split('.')[0]}"
    os.makedirs(file_stem, exist_ok=True)

    with torch.no_grad():
        stereo_image_path = args.stereo_image_path
        stereo_image = cv2.cvtColor(cv2.imread(stereo_image_path), cv2.COLOR_BGR2RGB)
        stereo_image = np.array(stereo_image).astype(np.uint8)[..., :3]
        H, W, _ = stereo_image.shape
        image1 = stereo_image[:, :W // 2, :]
        image2 = stereo_image[:, W // 2:, :]
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        image1 = image1[None].to(DEVICE)
        image2 = image2[None].to(DEVICE)
        image1 = F.interpolate(image1, scale_factor=1.0 / 2, mode="bilinear", align_corners=False)
        image2 = F.interpolate(image2, scale_factor=1.0 / 2, mode="bilinear", align_corners=False)
    
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        output = model(image1, image2, iters=args.valid_iters, test_mode=True)
        if args.infer_normal:
            normal_left = padder.unpad(output[2][-2][1]).squeeze()
            normal_right = padder.unpad(output[2][-1][1]).squeeze()
            gate_mask_left = output[2][-2][-1].squeeze().cpu().numpy()
            gate_mask_right = output[2][-1][-1].squeeze().cpu().numpy()
        disp = padder.unpad(output[1])
        image1 = padder.unpad(image1)
        image2 = padder.unpad(image2)
        disp = disp.cpu().numpy().squeeze()
        if args.infer_normal:
                normal_left = vis_normals(normal_left, None)
                normal_right = vis_normals(normal_right, None)
                cv2.imwrite(f"{file_stem}/normal_left.png", cv2.cvtColor(np.uint8(normal_left), cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"{file_stem}/normal_right.png", cv2.cvtColor(np.uint8(normal_right), cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"{file_stem}/gate_mask_left.png", np.uint8(gate_mask_left * 255))
                cv2.imwrite(f"{file_stem}/gate_mask_right.png", np.uint8(gate_mask_right * 255))
        image1 = image1.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image2 = image2.squeeze(0).permute(1, 2, 0).cpu().numpy()
        disp = cv2.applyColorMap(np.uint8(disp / disp.max() * 255), cv2.COLORMAP_JET)
        cv2.imwrite(f"{file_stem}/image_left.png", cv2.cvtColor(np.uint8(image1), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{file_stem}/image_right.png", cv2.cvtColor(np.uint8(image2), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{file_stem}/disp_pred.png", disp)
        if args.infer_normal:
            images = np.hstack([image1, image2])
            normals = np.hstack([normal_left, normal_right])
            disps = np.hstack([cv2.cvtColor(np.zeros_like(disp), cv2.COLOR_BGR2RGB), cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)])
            all = np.vstack([images, normals, disps])
        else:
            images = np.hstack([image1, image2])
            disps = np.hstack([cv2.cvtColor(np.zeros_like(disp), cv2.COLOR_BGR2RGB), cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)])
            all = np.vstack([images, disps])
        cv2.imwrite(f"{file_stem}/all.png", cv2.cvtColor(np.uint8(all), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="great-igev-stereo", help="name your experiment.")
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/kitti/kitti15.pth')
    parser.add_argument("--backbone_type", required=False, help="the type of the backbone used in the network ('MobileNetV2' or 'ResidualNet' for default if not set this argument explicitly).")
    parser.add_argument("--backbone_ckpt", required=False, help="the path of the backbone checkpoint.")
    parser.add_argument("--stereo_image_path", help="path to the stereo frames", default="./demo_images.jpg")
    parser.add_argument('--output_directory', help="directory to save output", default="./experiments/great_stereo/igev-based/demo")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument("--infer_normal", default=False, action="store_true", help="infer the normal map.")
    parser.add_argument("--precision_dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="choose mixed precision type: float16, bfloat16 or float32.")
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument("--shared_backbone", action="store_true", help="use a single backbone for the context and feature encoders.")
    parser.add_argument('--cv_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--cv_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument("--slow_fast_gru", action="store_true", help="iterate the low-res GRUs more frequently.")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument("--channels", nargs="+", type=int, default=[128] * 3, help="hidden state and context channels.")
    parser.add_argument("--context_norm", type=str, default="batch", choices=["group", "batch", "instance", "none"], help="normalization of context encoder")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

    args = parser.parse_args()

    demo(args)
