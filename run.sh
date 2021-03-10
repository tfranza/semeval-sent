#!/bin/bash

if [ "$1" = "train_tiny" ]; then
  	python run.py train --embedding-type='w2v' --max-epoch=5 --single-fold
elif [ "$1" = "train" ]; then
  	python run.py train --embedding-type='w2v' --max-epoch=50 --single-fold
elif [ "$1" = "train_noprep" ]; then
  	python run.py train --embedding-type='w2v' --max-epoch=50 --single-fold --prep-verbs=-1 --prep-emoticon-emojis=-1 --prep-abbreviations=-1 --prep-elongations=-1 --prep-lowering=0 --dont-save-prep --dont-load-prep
#elif [ "$1" = "train_local" ]; then
#  	python run.py train --train-src=../data/train_features.csv --train-tgt=../data/train_targets_scored.csv --name='test' --batch-size=32 --hidden-size=300 --lr=0.0001 --patience=3 --num-folds=10 --valid-niter=650 --max-epoch=1200 --uniform-init=0.0001 --dropout=0.3  --cuda --save-to=./model/model.bin --enable-metrics --resampling='(remedial,1)'
#elif [ "$1" = "predict" ]; then
#    echo "Predicting "
#    python run.py  predict --num-folds=10 model/model.bin ../data/test_features.csv prediction.csv
else
	echo "Invalid Option Selected"
fi
