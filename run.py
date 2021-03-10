#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
	train.py train [options]

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

    --num-folds=<int>                  number of folds                              [default: 4]
    --single-fold                      compute only the first fold

    --lr=<float>                       learning rate                                [default: 0.1]
    --lr-decay=<float>                 learning rate decay                          [default: 1]
    --max-decays=<int>                 max number of decays                         [default: 20]
    --max-epoch=<int>                  max number of epochs                         [default: 70]
    --patience-limit=<int>             value for early stopping                     [default: 5]
"""

from docopt import docopt
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import LambdaLR

from models import FirstNeuralNetwork
from model_preprocessing import preprocess, generate_kfolds

def create_model(args, pretrained):
	model = FirstNeuralNetwork(
		pretrained, 
		pretrained.shape[1], 
		int(args['--hidden-size']), 
		int(args['--num-classes'])
	)
	return model

def train_val_fold(args, model, dataloaders, begin_time, fold_id):
	optimizer = optim.Adam(model.parameters(), lr=float(args['--lr']))
	lr_decaying_fn = lambda epoch: (0.7 ** epoch) * 0.1
	lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_decaying_fn) 
	loss_fn = model.loss_fn
	
	print('begin training on fold %d ...' % (fold_id+1), file=sys.stderr)
	patience = 0
	num_decays = 0
	epoch_train_losses = []
	epoch_valid_losses = [1e10,1e10]
	train_dl, valid_dl = dataloaders
	epoch = 0
	while True:
		################# epoch training #################
		model.train()
		epoch_time = time.time()
		cum_batch_samples = 0
		batch_losses = []
		for (X_batch, y_batch) in train_dl:
			# zeroing gradients
			optimizer.zero_grad()
			# calling the forward
			y_pred = model.forward(X_batch)
			cum_batch_samples += len(y_batch)
			# computing loss function
			batch_loss = loss_fn(y_pred, y_batch)
			batch_losses.append(batch_loss.item())
			# compute the gradient
			batch_loss.backward()
			# pass gradients back into the model
			optimizer.step()
		epoch_train_losses.append(np.mean(batch_losses))
		valid_loss = evaluate_fold(model, valid_dl)
		epoch_valid_losses.append(valid_loss)
		print('   fold %d > epoch %d, avg.train.batchloss %.4f, avg.train.loss %.4f, avg.valid.batchloss %.4f, speed %.2f samples/sec, time elapsed %.2f secs' 
			% (fold_id+1, epoch+1, np.mean(batch_losses), np.mean(batch_losses)/int(args['--batch-size']), valid_loss, cum_batch_samples/(time.time()-epoch_time), time.time()-begin_time), file=sys.stderr)
		##################################################

		########### check for early stopping #############
		if epoch_valid_losses[-1] > epoch_valid_losses[-2-patience]:
			print('     + increasing patience from %d to %d' % (patience, patience+1))
			patience += 1
			if patience == int(args['--patience-limit']):
				print('     + patience limit hit!')
				break
			if num_decays < int(args['--max-decays']):
				prev_lr = optimizer.param_groups[0]['lr']
				lr_scheduler.step()
				num_decays += 1
				print('     + decaying learning rate from %.6f to %.6f' % (prev_lr, optimizer.param_groups[0]['lr']),file=sys.stderr)
			else:
				print('     + max amount of decays hit!')
				break
		else:
			if patience > 0:
				print('    + resetting patience from %d to 0' % (patience))
				patience = 0
			model.save(args['--model-save-to'] +'_'+str(fold_id+1))
			torch.save(optimizer.state_dict(), args['--model-save-to'] +'_'+str(fold_id+1) + '.optim')
		##################################################

		############## check for max epochs ##############
		if epoch == int(args['--max-epoch'])-1:
			print('     + max epoch hit!')
			break
		else:
			epoch += 1
		##################################################

	best_val_avg_batchloss = epoch_valid_losses[-1-patience]
	print('   validation > avg.batchloss %.4f, avg.loss %.4f' % (best_val_avg_batchloss, best_val_avg_batchloss/int(args['--batch-size'])), file=sys.stderr)
	return best_val_avg_batchloss

def evaluate_fold(model, valid_dataloader):
	loss_fn = model.loss_fn
	model.eval()
	batch_losses = []
	with torch.no_grad():
		for (X_batch, y_batch) in valid_dataloader:
			y_pred = model.forward(X_batch)
			batch_loss = loss_fn(y_pred, y_batch)
			batch_losses.append(batch_loss)
	cum_loss = np.sum(batch_losses)
	avg_batchloss = cum_loss/len(batch_losses)
	return avg_batchloss

def run(args, device):
	training_set = pd.read_csv(args['--train-src'], delimiter='\t')
	pretrained, embeddings, labels = preprocess(args, training_set)

	# converting pretrained dict vectors, inputs and outputs to tensors
	pretrained = torch.FloatTensor(pretrained)
	embeddings = torch.stack(list(map(lambda sent: torch.LongTensor(sent), embeddings)))
	labels = torch.LongTensor(labels)

	# dividing training set in batches and train/valid sets
	kfold_dataloaders = generate_kfolds(args, (embeddings,labels), device)
	
	begin_time = time.time()
	val_losses = []
	model = create_model(args, pretrained).to(device)
	print(model,'\n')
	for fold_id, (train_batches, valid_batches) in enumerate(kfold_dataloaders):
		val_avg_batchloss = train_val_fold(args, model, (train_batches, valid_batches), begin_time, fold_id)
		val_losses.append(val_avg_batchloss)
		if args['--single-fold']:
			break
		else:
			model = create_model(args, pretrained).to(device)
	
	idx_best_model = val_losses.index(max(val_losses))+1
	best_model = create_model(args, pretrained).to(device).load(args['--model-save-to']+'_'+str(idx_best_model))
	print('')

	predict(args, best_model, pretrained)

#	prev_loss = (min(hist_losses) if len(hist_losses)>0 else epoch_loss)

	# if increase in loss is greater  than 0.01% break
#	dff = prev_loss - epoch_loss
#	if dff < 0 and prev_loss < (0.997*epoch_loss):
#		print("Breaking with loss ", epoch_loss)
#		break

#	if len(hist_losses) > int(args['--max-epoch']):
#		print("Max epochs reached with loss ", epoch_loss)
#		break

	
def predict(args, model, pretrained):
	dev_set = pd.read_csv(args['--dev-src'], delimiter='\t')
	_, dev_embeddings, dev_labels = preprocess(args, dev_set, pretrained, train_flag=False)

	# converting inputs to tensors
	dev_embeddings = list(map(lambda sent: torch.LongTensor(sent), dev_embeddings))
	dev_embeddings = torch.stack(dev_embeddings)
	dev_labels = torch.LongTensor(dev_labels)

	with torch.no_grad():
		prob_pred = model.forward(dev_embeddings)
	y_pred = torch.argmax(prob_pred, dim=1).numpy()
	y_true = dev_labels.numpy()

	from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

	macro_precision = 0
	macro_recall = 0 
	macro_f1 = 0
	for id, c in enumerate(['others', 'happy', 'sad', 'angry']):
	    y_true_c = (y_true==id)+[0]*len(y_true)
	    y_pred_c = (y_pred==id)+[0]*len(y_pred)
	    prec_c = precision_score(y_true_c, y_pred_c, zero_division=0)
	    rec_c = recall_score(y_true_c, y_pred_c, zero_division=0)
	    f1_c = f1_score(y_true_c, y_pred_c, zero_division=0)
	    if id!=0:
	    	macro_precision += prec_c
	    	macro_recall += rec_c 
	    print('P: ', np.round(prec_c, decimals=4))
	    print('R: ', np.round(rec_c, decimals=4))
	    print('F1: ', np.round(f1_c, decimals=4))
	    print(confusion_matrix(y_true_c, y_pred_c))
	    print()
	macro_precision /= 3
	macro_recall /= 3
	macro_f1 = (2*macro_precision*macro_recall)/(macro_precision+macro_recall)
	print('Macro-Avg:', np.round(macro_f1,decimals=4))

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
        run(args, device)
    #elif args['predict']:
    #    predict(args)
    else:
        raise RuntimeError('invalid run mode')

if __name__ == '__main__':
    main()

