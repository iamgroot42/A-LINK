#!/bin/bash

# Make dir structure
mkdir ARCFace_models
echo "Starting first phase of downloads"
cd ARCFace_models
wget -O disguisedModel.h5 https://www.dropbox.com/s/rd0lkodp45bh0y8/disguisedModel.h5?dl=1
wget -O ensemble1.h5 https://www.dropbox.com/s/d8fj8v4je6tddra/ensemble1.h5?dl=1
cd ..

mkdir arcface_model
cd arcface_model
mkdir model-r100-ii
cd model-r100-ii

# Download files
echo "Starting second phase of downloads"
wget -O model-symbol.json https://www.dropbox.com/s/q0ksraci5p4apbg/model-symbol.json?dl=1
wget -O model-0000.params https://www.dropbox.com/s/hubikzhg0il8qgx/model-0000.params?dl=1
wget -O log https://www.dropbox.com/s/pitbddphd8jdojs/log?dl=1

echo "Done"
