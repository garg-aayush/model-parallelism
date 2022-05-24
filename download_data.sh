#!/bin/bash
kaggle competitions download -c carvana-image-masking-challenge -f train_hq.zip
unzip train_hq.zip
mv train_hq/* data/carvana/data/imgs/
rm -d train_hq
rm train_hq.zip

kaggle competitions download -c carvana-image-masking-challenge -f train_masks.zip
unzip train_masks.zip
mv train_masks/* data/carvana/data/masks/
rm -d train_masks
rm train_masks.zip