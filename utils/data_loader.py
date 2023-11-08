import os
from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import time
import argparse
from tqdm import tqdm

img_size = (448, 448)
debug = False

def cv_show_img(img):
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def matplot_show_img(img, mask, sec):
    ax1 = plt.subplot(131)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    plt.imshow(img)
    ax2 = plt.subplot(132)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    plt.imshow(mask)
    ax3 = plt.subplot(133)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    plt.imshow(img)
    plt.imshow(mask, alpha=0.5)
    plt.show(block=False)
    plt.pause(sec)
    plt.close()

def copy_dataset(in_dir, out_dir, id):
    print(f'target path: {in_dir}')
    print(f'destination path: {out_dir}')
    print(f'start copying {id}')

    Path(os.path.join(*[out_dir, "images"])).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(*[out_dir, "masks"])).mkdir(parents=True, exist_ok=True)

    dest_mask_dir = os.path.join(*[in_dir, 'masks'])
    dest_img_dir  = os.path.join(*[in_dir, 'images'])

    cnt = 0
    imgpaths, maskpaths = [], []
    paths = [list(p) for p in zip(Path(dest_img_dir).glob('*.jpg'),Path(dest_mask_dir).glob('*.jpg'))]
    
    for path in tqdm(paths):
        assert path[0].name == path[1].name, f"mask path:\r{path[0]}\n\nimg path:\r{path[1]}\nMismatch!!!!"
        
        img = cv.imread(str(path[0]))
        mask = cv.imread(str(path[1]))
        
        thresh = 128

        img = cv.resize(img, dsize=img_size)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        mask = cv.resize(mask, dsize=img_size, interpolation=cv.INTER_NEAREST)
        mask = cv.threshold(mask, thresh, 255, cv.THRESH_BINARY)[1]
        mask = (mask > 0).astype(np.uint8) * 255

        # if debug:
        #     plt.subplot(131)
        #     plt.imshow(img)
        #     plt.subplot(132)
        #     plt.imshow(mask)
        #     palette =  list(set(tuple(x) for x in mask.reshape(-1, 1)))
        #     print(palette)
        #     # 如果不指定顏色空間的話，matplotlib默認的cmap為十色環，也就是說不是我們直觀的
        #     # 灰度空間或是熱度(hot)空間，只是單純的顏色循環。舉例來說，第一個出現的值會被映
        #     # 射成"紫色"，不管0、1或255，當出現新的值會被映射為黃色，接著是翠綠色。總結來說
        #     # ，與值無關，指與其出現的先後順序有關。
        #     plt.subplot(133)
        #     plt.imshow(img)
        #     plt.imshow(mask, alpha=0.5)
        #     plt.show()

        img_path = os.path.join(*[out_dir, "images",  f'{path[0].stem}.jpg'])
        mask_path = os.path.join(*[out_dir, "masks", f'{path[1].stem}.jpg'])
        cv.imwrite(filename=img_path,  img = img)
        cv.imwrite(filename=mask_path, img = mask)
        imgpaths.append(img_path)
        maskpaths.append(mask_path)

        cnt +=1

    data = {"imgpaths": imgpaths,
            "maskpaths": maskpaths}
    with open(os.path.join(*[out_dir, f"{id}.json"]), 'w', newline='') as jsonfile:
        json.dump(data,jsonfile)

    print(f'copied {cnt} image-mask pairs from {id}')

def split_dataset(dataset_json, test_size, save_jsonfile: str):
    with open(dataset_json) as jsf:
        data_load = json.load(jsf)
        imPaths = data_load['imgpaths']
        maskPaths = data_load['maskpaths']

    train_image_paths, test_image_paths, train_mask_paths, test_mask_paths = train_test_split(imPaths,
                                                                                              maskPaths,
                                                                                              test_size=test_size,
                                                                                              random_state=42)
    data = {}  
    data['train_image_list']=train_image_paths
    data['train_mask_list']=train_mask_paths
    data['test_image_list']=test_image_paths
    data['test_mask_list']=test_mask_paths

    with open(save_jsonfile, 'w', newline='') as jsonfile:
        json.dump(data, jsonfile)

class ImgDataSet(Dataset):
    def __init__(self, dataset_json, isTrain=True, img_transform=None, mask_transform=None):
        self.dataset_json = dataset_json
        
        with open(self.dataset_json) as jsf:
            data_load = json.load(jsf)
            if isTrain:
                self.imList = data_load['train_image_list']
                self.maskList = data_load['train_mask_list']
            else:
                self.imList = data_load['test_image_list']
                self.maskList = data_load['test_mask_list']

        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __getitem__(self, i):
        img = Image.open(self.imList[i])
        mask = Image.open(self.maskList[i])
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return img, mask 

    def __len__(self):
        return len(self.imList)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DataLoader & Image Preprocessing')
    parser.add_argument('-data_dir', type=str, help='input dataset directory')
    parser.add_argument('-save_data_dir', type=str, help='output dataset directory')
    parser.add_argument('-dataset_id', default='crack_dataset', type=str, help='dataset name')
    args = parser.parse_args()

    copy_dataset(args.data_dir, args.save_data_dir, args.dataset_id)

    debug = False
    
    if debug:
        dataset = ImgDataSet(os.path.join(*[args.save_data_dir, f"{args.dataset_id}.json"]))

        for i in range(dataset.__len__()): 
            img,mask = dataset.__getitem__(i)
            matplot_show_img(img, mask, 1)