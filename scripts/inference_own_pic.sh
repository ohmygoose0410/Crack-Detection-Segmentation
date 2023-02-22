#!/bin/bash

if command -v python3 &> /dev/null
then    
    python3 ./inference_own_pic.py \
        -img_path ./real_crack_img/origin/1.png ./real_crack_img/origin/2.png \
        -mask_path ./real_crack_img/mask/1.jpg ./real_crack_img/mask/2.jpg \
        -model_path ./model/model_best.pt \
        -model_type vgg16 \
        -out_viz_dir ./vgg16/own_img/out_viz_dir \
        -out_pred_dir ./vgg16/own_img/out_pred_dir/ \
        -threshold 0
else
    echo "python3 could not be found~~"
fi

exit