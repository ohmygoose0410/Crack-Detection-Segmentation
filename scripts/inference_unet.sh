#!/bin/bash

if command -v python3 &> /dev/null
then    
    python3 ./inference_unet.py \
        -dataset_jsonfile ./public_crack_image/after_image_preprocess/crack_dataset_split.json \
        -model_path ./model/model_best.pt \
        -model_type vgg16 \
        -out_viz_dir ./vgg16/img/out_viz_dir \
        -out_pred_dir ./vgg16/img/out_pred_dir/ \
        -threshold 0.2
else
    echo "python3 could not be found~~"
fi

exit