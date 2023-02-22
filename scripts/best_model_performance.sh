#!/bin/bash

if command -v python3 &> /dev/null
then    
    python3 ./best_model_performance.py \
        -dataset_jsonfile ./public_crack_image/after_image_preprocess/crack_dataset_split.json \
        -pred_dir ./vgg16/img/out_pred_dir \
        -model_path ./model/model_best.pt \
        -threshold 0.2
else
    echo "python3 could not be found~~"
fi

exit