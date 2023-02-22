from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from data_loader import ImgDataSet
from sklearn.metrics import jaccard_score
import torch
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
    arg('-dataset_jsonfile', type=str, default='D:/master_course materials/Module Course/crack_detection/public_crack_image/IP/crack_dataset_split.json', required=False, help='path where ground truth images are located')
    arg('-pred_dir', type=str, default='./vgg16/img/out_pred_dir', required=False,  help='path with predictions')
    arg('-model_path', type=str, help='trained model path')
    arg('-threshold', type=float, default=0.2, required=False,  help='crack threshold detection')
    args = parser.parse_args()

    result_dice = []
    result_jaccard = []
    result_mpa = []
    result_miou = []

    best_state = torch.load(args.model_path)
    min_val_los = best_state['valid_loss']

    valid_dataset = ImgDataSet(args.dataset_jsonfile, isTrain=False, img_transform=None, mask_transform=None)
    mskPaths = valid_dataset.maskList

    # paths = [path for path in  Path(args.ground_truth_dir).glob('*')]
    for path in tqdm(mskPaths):
        y_true = (cv.imread(str(path), 0) > 0).astype(np.uint8)

        pred_file_name = Path(args.pred_dir) / Path(path).name
        if not pred_file_name.exists():
            print(f'missing prediction for file {path.name}')
            continue

        pred_image = (cv.imread(str(pred_file_name), 0) > 255 * args.threshold).astype(np.uint8)
        y_pred = pred_image

        result_dice += [dice(y_true, y_pred)]
        result_jaccard += [jaccard(y_true, y_pred)]
        result_mpa += [mpa(y_true, y_pred)]
        result_miou += [miou(y_true, y_pred)]

    print('BCELoss = ', min_val_los)
    print('Dice = ', np.mean(result_dice), np.std(result_dice))
    print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard))
    print('MPA = ', np.mean(result_mpa), np.std(result_mpa))
    print('MIoU = ', np.mean(result_miou), np.std(result_miou))