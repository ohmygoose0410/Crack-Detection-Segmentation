import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    ap = argparse.ArgumentParser()
    ap.add_argument("-model_dir", default='./model', type=str, required=True)
    ap.add_argument("-target_dir", default='.', type=str, required=True)
    ap.add_argument("-title", default='Training/Validation Loss', type=str, required=False)
    args = ap.parse_args()

    paths = [path for path in Path(args.model_dir).glob('*.pt')]
    paths = sorted(paths)
    epochs = []
    tr_losses = []
    vl_losses = []
    for path in tqdm(paths):
        if 'epoch' not in path.stem:
            continue
        #load the min loss so far
        parts = path.stem.split('_')
        epoch = int(parts[-1])
        epochs.append(epoch)
        state = torch.load(path)
        val_los = state['valid_loss']
        train_loss = float(state['train_loss'])
        tr_losses.append(train_loss)
        vl_losses.append(val_los)

    sorted_idxs = np.argsort(epochs)
    tr_losses = [tr_losses[idx] for idx in sorted_idxs]
    vl_losses = [vl_losses[idx] for idx in sorted_idxs]

    # print(tr_losses)
    # print(vl_losses)
    # print("last loss: ", tr_losses[-1])
    data = {}
    data['tr_losses']=tr_losses
    data['vl_losses']=vl_losses
    with open(os.path.join(args.target_dir, 'training_loss.json'), 'w', newline='') as jsonfile:
        json.dump(data, jsonfile)

    plt.plot(tr_losses[1:], label='train_loss')
    plt.plot(vl_losses[1:], label='valid_loss')
    plt.title(args.title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.target_dir, 'training_loss.jpg'), dpi=500)
    plt.show()





