#!/bin/bash

python active_training_DFW.py --mixture_ratio 1; python generateMatrix.py; python ROC.py; mv 1000_point_ROC_case1.png curves/1x.png
python active_training_DFW.py --mixture_ratio 2; python generateMatrix.py; python ROC.py; mv 1000_point_ROC_case1.png curves/2x.png
python active_training_DFW.py --mixture_ratio 4; python generateMatrix.py; python ROC.py; mv 1000_point_ROC_case1.png curves/4x.png
python active_training_DFW.py --mixture_ratio 8; python generateMatrix.py; python ROC.py; mv 1000_point_ROC_case1.png curves/8x.png
