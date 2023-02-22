#!/bin/bash

if command -v python3 &> /dev/null
then    
    python3 ./plot_training_graph.py \
        -model_dir ./model \
        -target_dir ./vgg16 \
        -title "Training/Validation Loss"
        
else
    echo "python3 could not be found~~"
fi

exit