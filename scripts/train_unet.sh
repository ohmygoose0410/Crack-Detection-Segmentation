#!/bin/bash

if command -v python3 &> /dev/null
then
	python3 ./train_unet.py \
        -n_epoch 2 \
        -lr 0.001 \
        -momentum 0.9 \
        -print_freq 20 \
        -weight_decay 1e-4 \
        -batch_size 1 \
        -num_workers 4 \
        -orig_json_path ./crack_dataset/image_preprocessing/crack_dataset.json \
        -split_json_dir ./crack_dataset/image_preprocessing \
        -model_dir ./model \
        -model_type vgg16
else
    echo "python3 could not be found~~"
fi

exit 0