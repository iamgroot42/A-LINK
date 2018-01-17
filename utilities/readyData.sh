#!/bin/bash

unzip -d ./ $1
rm ./MultiPie51/Thumbs.db
rm ./MultiPie51/convert.exe.stackdump

cp -r MultiPie51 data
mv MultiPie51 data_temp

mkdir fileLists
mkdir data_segragated

python utilities/process.py data_temp/ data_segragated/
python utilities/bisect_into_paths.py data_segragated/ fileLists/

mkdir data_final
mkdir data_final/highres
mkdir data_final/lowres

python utilities/generate_image_dirs.py data_final/highres data/ fileLists/highResData.txt
python utilities/generate_image_dirs.py data_final/lowres data/ fileLists/lowResData.txt

mkdir data_final/highres/TRAIN data_final/highres/VAL
mkdir data_final/lowres/TRAIN data_final/lowres/VAL

python utilities/process.py data_final/highres/train/ data_final/highres/TRAIN/
python utilities/process.py data_final/highres/val/ data_final/highres/VAL/
python utilities/process.py data_final/lowres/train/ data_final/lowres/TRAIN/
python utilities/process.py data_final/lowres/val/ data_final/lowres/VAL/

rm -r data_final/highres/train data_final/highres/val
rm -r data_final/lowres/train data_final/lowres/val

mv data_final/highres/TRAIN/ data_final/highres/train/
mv data_final/highres/VAL/ data_final/highres/val/
mv data_final/lowres/TRAIN/ data_final/lowres/train/
mv data_final/lowres/VAL/ data_final/lowres/val/

rm -r data_temp
rm -r data_segragated
