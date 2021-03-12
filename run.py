#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
	run.py train [options]
	run.py predict [options]

Options:
    -h --help                          show this screen.
    --train-src=<file>                 train set source file                        [default: datasets/semeval2019-task3/train.txt]
    --dev-src=<file>                   dev set source file                          [default: datasets/semeval2019-task3/dev.txt]
    --prep-text-save-to=<file>         preprocessed text save file                  [default: datasets/semeval2019-task3/prep-text.txt]
    --model-save-to=<file>             model save path                              [default: models/model]

    --seed=<int>                       seed                                         [default: 0]

    --dont-load-prep                   not loading previously prep text
    --dont-save-prep                   not saving the current prep text
    --prep-verbs=<int>                 raw text verbs regularization                [default: 0]
    --prep-lowering=<int>              raw text lowering                            [default: 1]
    --prep-emoticon-emojis=<int>       raw text emoticons and emoji substitution    [default: 2]
    --prep-abbreviations=<int>         raw text abbreviations substitution          [default: 3]
    --prep-elongations=<int>           raw text elongations removal                 [default: 4]
    --prep-misspellings=<int>          raw text misspellings correction             [default: -1]

    --embedding-type=<string>          embedding type among {w2v,glv,brt,rbt}       [default: glv]
    --batch-size=<int>                 batch size                                   [default: 256]
    --hidden-size=<int>                hidden size                                  [default: 128]
    --num-classes=<int>                number of classes                            [default: 4]
    --num-layers=<int>                 number of BiLSTMs layers                     [default: 2]

    --num-folds=<int>                  number of folds                              [default: 4]
    --single-fold                      compute only the first fold

    --lr=<float>                       learning rate                                [default: 0.001]
    --lr-decay=<float>                 learning rate decay                          [default: 1]
    --max-decays=<int>                 max number of decays                         [default: 20]
    --max-epoch=<int>                  max number of epochs                         [default: 70]
    --patience-limit=<int>             value for early stopping                     [default: 5]
"""

from docopt import docopt
import numpy as np
import torch

from train import train
from eval import predict

def main():
	args = docopt(__doc__)
	print(args,'\n')
	# Check pytorch version
	assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

	# seed the random number generators
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	seed = int(args['--seed'])
	torch.manual_seed(seed)
	if device=='cuda':
		torch.cuda.manual_seed(seed)
	np.random.seed(seed * 13 // 7)

	if args['train']:
		train(args, device)
	elif args['predict']:
		predict(args)
	else:
		raise RuntimeError('invalid run mode')

if __name__ == '__main__':
	main()

