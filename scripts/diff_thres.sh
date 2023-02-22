#!/bin/bash

if command -v python3 &> /dev/null
then    
    python3 ./diff_thres.py \
        -dataset_json_path ./public_crack_image/after_image_preprocess/crack_dataset_split.json \
        -model_path ./model/model_best.pt \
        -model_type vgg16 \
        -threshold 0.2 \
        -target_dir ./vgg16/img/diff_thres
else
    echo "python3 could not be found~~"
fi

exit