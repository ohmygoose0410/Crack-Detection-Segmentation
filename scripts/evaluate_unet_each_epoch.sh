#!/bin/bash

if command -v python3 &> /dev/null
then    
    python3 ./evaluate_unet_each_epoch.py \
        -model_dir ./model \
        -model_type vgg16 \
        -dataset_jsonfile ./public_crack_image/after_image_preprocess/crack_dataset_split.json \
        -threshold 0.2 \
        -target_dir ./vgg16
else
    echo "python3 could not be found~~"
fi

exit