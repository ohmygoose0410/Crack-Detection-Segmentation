from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_loader import ImgDataSet
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix,jaccard_score
import pandas as pd
import json
import torchvision.transforms as transforms
from PIL import Image
from evaluate_unet_each_epoch import evaluate_img
from utils import load_unet_vgg16, load_unet_resnet_101, load_unet_resnet_34
from unet_transfer import input_size
import cv2 as cv

def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def general_dice(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return dice(y_true, y_pred)

def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def general_jaccard(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return jaccard(y_true, y_pred)

def mpa(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    FP = len(np.where(y_pred - y_true == 1)[0])
    FN = len(np.where(y_true - y_pred == 1)[0])
    TP = len(np.where(y_pred + y_true == 2)[0])
    TN = len(np.where(y_pred + y_true == 0)[0])

    pos = (TP + 1e-15) / (TP + FN + 1e-15)
    neg = (TN + 1e-15) / (TN + FP + 1e-15)

    return (pos+neg)/2

def miou(y_true, y_pred):
    return jaccard_score(y_true.flatten(), y_pred.flatten(), average='macro')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-dataset_jsonfile', type=str, required=False, help='path where ground truth images are located')
    arg('-model_type', type=str, required=True, choices=['vgg16', 'resnet101', 'resnet34'])
    arg('-model_path', type=str, help='trained model path')
    arg('-threshold', type=float, default=0.2, required=False,  help='crack threshold detection')
    arg('-target_dir', type=str, required=False, help=None)
    args = parser.parse_args()

    result_dice = []
    result_jaccard = []
    result_mpa = []
    result_miou = []

    preds = np.zeros((0))
    truths = np.zeros((0))

    valid_dataset = ImgDataSet(args.dataset_jsonfile, isTrain=False, img_transform=None, mask_transform=None)
    mskPaths = valid_dataset.maskList
    imgPaths = valid_dataset.imList
    dataloader = [list(i) for i in zip(imgPaths, mskPaths)]

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]

    if args.model_type == 'vgg16':
        model = load_unet_vgg16(args.model_path)
    elif args.model_type  == 'resnet101':
        model = load_unet_resnet_101(args.model_path)
    elif args.model_type  == 'resnet34':
        model = load_unet_resnet_34(args.model_path)
    else:
        print('undefind model name pattern')
        exit()

    # paths = [path for path in  Path(args.ground_truth_dir).glob('*')]
    if not os.path.exists(os.path.join(args.target_dir, 'confusion_matrix.json')):
        for data in tqdm(dataloader):
            img_pth, mask_pth = data[0], data[1]
            img_pth = Path(img_pth)
            mask_pth = Path(mask_pth)

            train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])

            img_0 = Image.open(str(img_pth))
            img_0 = np.asarray(img_0)
            if len(img_0.shape) != 3:
                print(f'incorrect image shape: {img_pth.name}{img_0.shape}')
                continue

            img_0 = img_0[:,:,:3]
            img_0_shape = img_0.shape
            prob_map_full = evaluate_img(model, img_0, img_0_shape, train_tfms)
            y_pred = (prob_map_full > args.threshold).astype(np.uint8)
            y_true = (cv.imread(str(mask_pth), 0) > 0).astype(np.uint8)

            preds = np.concatenate([preds, np.array(y_pred).flatten()], axis=0)
            truths = np.concatenate([truths, np.array(y_true).flatten()], axis=0)

            result_dice += [dice(y_true, y_pred)]
            result_jaccard += [jaccard(y_true, y_pred)]
            result_mpa += [mpa(y_true, y_pred)]
            result_miou += [miou(y_true, y_pred)]

        cf_matrix = confusion_matrix(truths, preds)
        
        data = {}
        data['cf_matrix']=cf_matrix.tolist()
        data['dice']=result_dice
        data['jaccard']=result_jaccard
        data['mpa']=result_mpa
        data['miou']=result_miou
        with open(os.path.join(args.target_dir, 'confusion_matrix.json'), 'w', newline='') as jsonfile:
            json.dump(data, jsonfile)
    else:
        with open(os.path.join(args.target_dir, 'confusion_matrix.json')) as jsf:
            data_load = json.load(jsf)
        cf_matrix = np.asarray(data_load['cf_matrix'])
        result_dice = data_load['dice']
        result_jaccard = data_load['jaccard']
        result_mpa = data_load['mpa']
        result_miou = data_load['miou']
    
    percentage_1d = np.array([cf_matrix[0][0]/np.sum(cf_matrix[0]),
                              cf_matrix[0][1]/np.sum(cf_matrix[0]),
                              cf_matrix[1][0]/np.sum(cf_matrix[1]),
                              cf_matrix[1][1]/np.sum(cf_matrix[1])])

    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:,d}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                        percentage_1d]

    percentage_2d = np.around(percentage_1d.reshape(2,2)*100).astype(np.uint8)
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    class_names = ["Non-Crack", "Crack"]
    df_cm = pd.DataFrame(percentage_2d, class_names, class_names)
    conMatrix = sns.heatmap(df_cm, annot=labels, fmt='', cmap='Blues', vmin=0, vmax=100)
    conMatrix.set(xlabel="Predicted Pixel", ylabel="True Pixel")
    figure = conMatrix.get_figure()
    figure.savefig(os.path.join(args.target_dir,'confusion_matrix.jpg'), dpi=500)

    print('Dice = ', np.mean(result_dice), np.std(result_dice))
    print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard))
    print('MPA = ', np.mean(result_mpa), np.std(result_mpa))
    print('MIoU = ', np.mean(result_miou), np.std(result_miou))