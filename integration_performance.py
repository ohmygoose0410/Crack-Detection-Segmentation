import json 
import numpy as np
import os
import matplotlib.pyplot as plt

def json2dict(model_type, filename):
    with open(os.path.join(root_dir, model_type, filename)) as jsf:
        output = json.load(jsf)
    return output

if __name__ == '__main__':
    root_dir = './performance/new_dataset'
    model_type = ['vggunet', 'effunet', 'resunet']
    classes = {'vggunet':'VGGU-Net',
               'effunet':'EfficientU-Net',
               'resunet':'ResU-Net'}
    eval_each_model = {}
    train_loss_model = {}
    for m in model_type:
        output_eval = json2dict(m, 'evaluate.json')
        output_loss = json2dict(m, 'training_loss.json')
        eval_each_model[m] = output_eval
        train_loss_model[m] = output_loss

    fig1, axis1 = plt.subplots()
    fig2, axis2 = plt.subplots()
    for key,val in eval_each_model.items():
        axis1.plot(val['miou'][:50], label=classes[key])
        axis2.plot(val['mpa'][:50], label=classes[key])
    axis1.set_title('Average MIoU')
    axis1.set_xlabel('Epoch')
    axis1.set_ylabel('mean intersection over union')
    axis1.legend(loc=(0.7,0.25))
    fig1.savefig(os.path.join(root_dir, 'avg_miou.jpg'), dpi=500)

    axis2.set_title('Average MPA')
    axis2.set_xlabel('Epoch')
    axis2.set_ylabel('mean pixel accuracy')
    axis2.legend()
    fig2.savefig(os.path.join(root_dir, 'avg_mpa.jpg'), dpi=500)

    fig3, axis3 = plt.subplots()
    for key,val in train_loss_model.items():
        axis3.plot(val['tr_losses'][:50], label=classes[key], alpha=0.5)
    axis3.set_title('Training Loss')
    axis3.set_xlabel('Epoch')
    axis3.set_ylabel('Loss')
    # axis3.set_ylim(ymin=0, ymax=1)
    axis3.legend(loc=(0.7,0.75))
    fig3.savefig(os.path.join(root_dir, 'train_loss.jpg'), dpi=500)
    
