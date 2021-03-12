
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import LambdaLR

from models import create_model
from model_preprocessing import preprocess, generate_kfolds
from eval import evaluate_fold

def train(args, device):
	training_set = pd.read_csv(args['--train-src'], delimiter='\t')
	pretrained, embeddings, labels = preprocess(args, training_set)

	# converting pretrained dict vectors, inputs and outputs to tensors
	pretrained = torch.FloatTensor(pretrained)
	embeddings = torch.stack(list(map(lambda sent: torch.LongTensor(sent), embeddings)))
	labels = torch.LongTensor(labels)

	# dividing training set in batches and train/valid sets
	kfold_dataloaders = generate_kfolds(args, (embeddings,labels), device)
	
	begin_time = time.time()
	model = create_model(args, pretrained).to(device)
	print(model,'\n')
	for fold_id, train_valid_batches in enumerate(kfold_dataloaders):
		train_val_fold(
			args, 
			create_model(args, pretrained).to(device), 
			train_valid_batches, 
			begin_time, 
			fold_id
		)

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
		print('   fold %d > epoch %d, avg.train.b_loss %.4f, avg.train.loss %.4f, avg.valid.b_loss %.4f, speed %.2f samples/sec, time elapsed %.2f secs' 
			% (fold_id+1, epoch+1, np.mean(batch_losses), np.mean(batch_losses)/int(args['--batch-size']), valid_loss, cum_batch_samples/(time.time()-epoch_time), time.time()-begin_time), file=sys.stderr)
		##################################################

		########### check for early stopping #############
		if epoch_valid_losses[-1] > epoch_valid_losses[-2-patience]:
			print('     + increasing patience from %d to %d' % (patience, patience+1))
			patience += 1
			if patience == int(args['--patience-limit']):
				print('     + patience limit hit!')
				break
			####### decaying learning rate whenever patience is increased 
			if num_decays < int(args['--max-decays']):
				prev_lr = optimizer.param_groups[0]['lr']
				lr_scheduler.step()
				num_decays += 1
				print('     + decaying learning rate from %.6f to %.6f' % (prev_lr, optimizer.param_groups[0]['lr']),file=sys.stderr)
			else:
				print('     + max amount of decays hit!')
				break
			#############################################################
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

