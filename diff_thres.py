import sys
import os
import numpy as np
from pathlib import Path
import cv2 as cv
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from unet_transfer import UNet16, input_size
import matplotlib.pyplot as plt
import argparse
from os.path import join
from PIL import Image
from torch.utils.data import DataLoader
import gc
from utils import load_unet_vgg16, load_unet_resnet_101, load_unet_resnet_34
from tqdm import tqdm
from data_loader import ImgDataSet


def evaluate_img(model, img):
    input_width, input_height = input_size[0], input_size[1]

    img_1 = cv.resize(img, (input_width, input_height), cv.INTER_AREA)
    X = train_tfms(Image.fromarray(img_1))
    X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]

    mask = model(X)

    mask = torch.sigmoid(mask[0, 0]).data.cpu().numpy()
    mask = cv.resize(mask, (img_width, img_height), cv.INTER_AREA)
    return mask


def evaluate_img_patch(model, img):
    input_width, input_height = input_size[0], input_size[1]

    img_height, img_width, img_channels = img.shape

    if img_width < input_width or img_height < input_height:
        return evaluate_img(model, img)

    stride_ratio = 0.1
    stride = int(input_width * stride_ratio)

    normalization_map = np.zeros((img_height, img_width), dtype=np.int16)

    patches = []
    patch_locs = []
    for y in range(0, img_height - input_height + 1, stride):
        for x in range(0, img_width - input_width + 1, stride):
            segment = img[y:y + input_height, x:x + input_width]
            normalization_map[y:y + input_height, x:x + input_width] += 1
            patches.append(segment)
            patch_locs.append((x, y))

    patches = np.array(patches)
    if len(patch_locs) <= 0:
        return None

    preds = []
    for i, patch in enumerate(patches):
        patch_n = train_tfms(Image.fromarray(patch))
        X = Variable(patch_n.unsqueeze(0)).cuda()  # [N, 1, H, W]
        masks_pred = model(X)
        mask = torch.sigmoid(masks_pred[0, 0]).data.cpu().numpy()
        preds.append(mask)

    probability_map = np.zeros((img_height, img_width), dtype=float)
    for i, response in enumerate(preds):
        coords = patch_locs[i]
        probability_map[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += response

    return probability_map


def disable_axis():
    plt.axis('off')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_ticklabels([])
    plt.gca().axes.get_yaxis().set_ticklabels([])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_json_path', type=str,
                        help='input dataset directory')
    parser.add_argument('-model_path', type=str, help='trained model path')
    parser.add_argument('-model_type', type=str, choices=['vgg16', 'resnet101', 'resnet34'])
    parser.add_argument('-threshold', type=float, default=0.2, help='threshold to cut off crack response')
    parser.add_argument('-target_dir', type=str, required=False, help=None)
    args = parser.parse_args()

    if args.target_dir != '':
        os.makedirs(args.target_dir, exist_ok=True)
        for img_path in Path(args.target_dir).glob('*.*'):
            os.remove(str(img_path))

    if args.model_type == 'vgg16':
        model = load_unet_vgg16(args.model_path)
    elif args.model_type == 'resnet101':
        model = load_unet_resnet_101(args.model_path)
    elif args.model_type == 'resnet34':
        model = load_unet_resnet_34(args.model_path)
        print(model)
    else:
        print('undefind model name pattern')
        exit()

    channel_means = [0.485, 0.456, 0.406]
    channel_stds = [0.229, 0.224, 0.225]

    test_dataset = ImgDataSet(args.dataset_json_path, isTrain=False, img_transform=None, mask_transform=None)
    img_paths = test_dataset.imList
    mask_paths = test_dataset.maskList
    # paths = [path for path in Path(args.img_dir).glob('*.jpg')]



    for img_path, mask_path in tqdm(zip(img_paths, mask_paths)):
        # print(str(path))
        img_path = Path(img_path)
        train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])

        img_0 = Image.open(str(img_path))
        mask_0 = Image.open(str(mask_path))

        img_0 = np.asarray(img_0)
        if len(img_0.shape) != 3:
            print(f'incorrect image shape: {img_path.name}{img_0.shape}')
            continue

        img_0 = img_0[:, :, :3]

        img_height, img_width, img_channels = img_0.shape

        prob_map_full = evaluate_img(model, img_0)

        nrows, ncols = 3, 4

        fig = plt.figure()
        ax = fig.add_subplot(nrows, ncols, 1)
        ax.set_axis_off()
        ax.title.set_text('Original\nImage')
        ax.imshow(img_0)
        ax = fig.add_subplot(nrows, ncols, 2)
        ax.set_axis_off()
        ax.title.set_text('Original\nMask')
        ax.imshow(mask_0, cmap='gray')
        for i, threshold in enumerate(np.arange(0, 1, 0.1), start=3):
            threshold = np.around(threshold, decimals=1)
            prob_map_viz_full = prob_map_full.copy()
            prob_map_viz_full[prob_map_viz_full < threshold] = 0.0

            ax = fig.add_subplot(nrows, ncols, i)
            ax.title.set_text('Threshold\n{}'.format(str(threshold)))
            ax.set_axis_off()
            ax.imshow(prob_map_viz_full, cmap='gray')

        fig.tight_layout()
        plt.savefig(join(args.target_dir, f'{img_path.stem}.jpg'), dpi=500)
        plt.close('all')