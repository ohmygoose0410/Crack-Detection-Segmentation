#!/bin/bash

if command -v python3 &> /dev/null
then
	python3 ./data_loader.py \
    -data_dir ./crack_dataset/mixed_crack_dataset \
	-save_data_dir ./crack_dataset/image_preprocessing \
	-dataset_id crack_dataset
else
    echo "python3 could not be found."
fi

exit 0