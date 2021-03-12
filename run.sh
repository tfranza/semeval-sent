#!/bin/bash

if [ "$1" = "train_tiny" ]; then
  	python run.py train --embedding-type='w2v' --max-epoch=5 --single-fold
elif [ "$1" = "train" ]; then
  	python run.py train --embedding-type='glv' --max-epoch=50 --single-fold
elif [ "$1" = "train_noprep" ]; then
  	python run.py train --embedding-type='glv'  
#elif [ "$1" = "train_local" ]; then
#  	python run.py train --train-src=../data/train_features.csv --train-tgt=../data/train_targets_scored.csv --name='test' --batch-size=32 --hidden-size=300 --lr=0.0001 --patience=3 --num-folds=10 --valid-niter=650 --max-epoch=1200 --uniform-init=0.0001 --dropout=0.3  --cuda --save-to=./model/model.bin --enable-metrics --resampling='(remedial,1)'
elif [ "$1" = "predict" ]; then
    python run.py predict --single-fold
else
	echo "Invalid Option Selected"
fi
