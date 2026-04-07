python3 demo_infer.py --name greaten-igevdepthany-stereo \
--restore_ckpt ./greaten-igevdepthany-stereo-submit.pth \
--backbone_type vitl \
--backbone_ckpt ./experiments/modules/backbones/depth_anything/depth_anything_v2_vitl.pth \
--stereo_image_path ./demo_test.jpg \
--output_directory ./experiments/greaten_stereo/igev-based/depthany/vis/demo \
--infer_normal
