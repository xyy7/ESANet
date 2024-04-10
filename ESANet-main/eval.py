# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse
import os
import sys

DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(DIR)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src.confusion_matrix import ConfusionMatrixTensorflow
from src.prepare_data import prepare_data


def set_free_cpu(rate=0.1, need_cpu=20):
    import os

    import psutil

    cpuinfo = psutil.cpu_percent(interval=0.5, percpu=True)
    freecpu = []
    for i, cinfo in enumerate(cpuinfo):
        if cinfo > rate:
            continue
        freecpu.append(i)
    os.sched_setaffinity(os.getpid(), freecpu[-need_cpu:])


if __name__ == "__main__":
    # set_free_cpu()
    # arguments
    parser = ArgumentParserRGBDSegmentation(
        description="Efficient RGBD Indoor Sematic Segmentation (Evaluation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_common_args()
    args = parser.parse_args()

    # dataset
    args.pretrained_on_imagenet = False  # we are loading other weights anyway

    _, data_loader, *add_data_loader = prepare_data(args, with_input_orig=True)  # 有可能跳转是错误的,debug才正常
    if args.valid_full_res:
        # cityscapes only -> use dataloader that returns full resolution images
        data_loader = add_data_loader[0]

    n_classes = data_loader.dataset.n_classes_without_void

    # model and checkpoint loading
    model, device = build_model(args, n_classes=n_classes)
    checkpoint = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print("Loaded checkpoint from {}".format(args.ckpt_path))

    model.eval()
    model.to(device)

    n_samples = 0

    confusion_matrices = dict()

    cameras = data_loader.dataset.cameras
    for camera in cameras:
        confusion_matrices[camera] = dict()
        confusion_matrices[camera] = ConfusionMatrixTensorflow(n_classes)
        n_samples_total = len(data_loader.dataset)
        with data_loader.dataset.filter_camera(camera):
            for i, sample in enumerate(data_loader):
                n_samples += sample["image"].shape[0]
                print(f"\r{n_samples}/{n_samples_total}", end="")

                image = sample["image"].to(device)  # [B, 3, 480, 640]
                depth = sample["depth"].to(device)  # [B, 1, 480, 640]
                label_orig = sample["label_orig"]  # [B, 480, 640]  # 不一定是480,640
                _, image_h, image_w = label_orig.shape

                with torch.no_grad():
                    if args.modality == "rgbd":
                        inputs = (image, depth)
                    elif args.modality == "rgb":
                        inputs = (image,)
                    elif args.modality == "depth":
                        inputs = (depth,)

                    pred, depth = model(*inputs)

                    pred = F.interpolate(pred, (image_h, image_w), mode="bilinear", align_corners=False)  # 插值回原来的大小
                    pred = torch.argmax(pred, dim=1)

                    if args.draw:
                        prediction = np.array(pred.cpu())
                        prediction = prediction.squeeze().astype(np.uint8)
                        if args.rec_data_dir is None:
                            print("with_void")
                            pred_colored = data_loader.dataset.color_label(label_orig, with_void=True)
                            pred_colored = pred_colored[0]
                            save_path = os.path.join("0928seg.png")
                        else:
                            print("without_void")
                            save_path = os.path.join(args.rec_data_dir, "0928seg.png")
                            pred_colored = data_loader.dataset.color_label(prediction, with_void=False)
                        img = Image.fromarray(pred_colored).convert("RGB")
                        img.save(save_path)

                    # ignore void pixels
                    mask = label_orig > 0  # 不计算有孔洞的
                    label = torch.masked_select(label_orig, mask)
                    pred = torch.masked_select(pred, mask.to(device))  # [1,480,640] [1,531,681]

                    # In the label 0 is void but in the prediction 0 is wall.
                    # In order for the label and prediction indices to match
                    # we need to subtract 1 of the label.
                    label -= 1
                    # copy the prediction to cpu as tensorflow's confusion
                    # matrix is faster on cpu
                    pred = pred.cpu()

                    label = label.numpy()
                    pred = pred.numpy()

                    confusion_matrices[camera].update_conf_matrix(label, pred)

                print(f"\r{i + 1}/{len(data_loader)}", end="")
                miou, _ = confusion_matrices[camera].compute_miou()
                print(f"\rCamera: {camera} mIoU: {100*miou:0.2f}")
        miou, _ = confusion_matrices[camera].compute_miou()
        print(f"\rCamera: {camera} mIoU: {100*miou:0.2f}")

    confusion_matrices["all"] = ConfusionMatrixTensorflow(n_classes)

    # sum confusion matrices of all cameras
    for camera in cameras:
        confusion_matrices["all"].overall_confusion_matrix += confusion_matrices[camera].overall_confusion_matrix
    miou, _ = confusion_matrices["all"].compute_miou()

    print(f"All Cameras, mIoU: {100*miou:0.2f}")

    with open(f"{os.path.abspath(os.path.dirname(os.path.dirname(__file__)))}/result.txt", "a") as file:
        file.write(f"{args.rec_data_dir} All Cameras, mIoU: {100*miou:0.2f}" + "\n")
