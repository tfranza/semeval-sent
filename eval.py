
import pandas as pd
import numpy as np
import torch

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score

from model_preprocessing import preprocess
from models import FirstNeuralNetwork

def predict(args):
	dev_set = pd.read_csv(args['--dev-src'], delimiter='\t')

	_, dev_embeddings, dev_labels = preprocess(args, dev_set, train_flag=False)

	# converting inputs to tensors
	dev_embeddings = list(map(lambda sent: torch.LongTensor(sent), dev_embeddings))
	dev_embeddings = torch.stack(dev_embeddings)
	dev_labels = torch.LongTensor(dev_labels).numpy()
	
	# computing metrics for each fold
	fold_metrics = []
	for i in range(int(args['--num-folds'])):
		fold_model, train_losses, val_losses = FirstNeuralNetwork.load(args['--model-save-to']+'_'+str(i+1))
		fold_metrics += [compute_single_fold_metrics( fold_model, dev_embeddings, dev_labels, int(args['--num-classes']) )]
		if args['--single-fold']:
			break
	
	# averaging metrics for each fold (macro averaging)
	macro_metrics = fold_metrics[0]
	for fold_metric in fold_metrics[1:]:
		for metric_key, metric_values in fold_metric.items():
			macro_metrics[metric_key] = np.add(macro_metrics[metric_key], np.array(metric_values)/len(fold_metrics), casting="unsafe")
	
	print_evaluation_metric(macro_metrics['confusion'], 'Confusion Matrices')
	print_evaluation_metric(macro_metrics['precision'], 'Precision')
	print_evaluation_metric(macro_metrics['recall'], 'Recall')
	print_evaluation_metric(macro_metrics['f1'], 'F1')
	print_evaluation_metric(macro_metrics['mcc'], 'Matthews Correlation Coefficient')
	print('   '+'Receiver Operating Characteristic - AUC: %.4f' % macro_metrics['roc-auc'])
	print('   '+'F1-others: %.4f' % macro_metrics[1:]/3)

def compute_single_fold_metrics(model, X_val, y_val, num_classes):
	fold_metrics = dict()
	with torch.no_grad():
		prob_pred = model.forward(X_val)
		y_pred = torch.argmax(prob_pred, dim=1).numpy()
	fold_metrics['confusion'] = [confusion_matrix(y_val[:,idx_class], y_pred[:,idx_class]).flatten() for idx_class in range(num_classes)]   # [TN, FP, FN, TP]
	fold_metrics['precision'] = [precision_score(y_val[:,idx_class], y_pred[:,idx_class], zero_division=0) for idx_class in range(num_classes)]
	fold_metrics['recall'] = [recall_score(y_val[:,idx_class], y_pred[:,idx_class], zero_division=0) for idx_class in range(num_classes)]
	fold_metrics['f1'] = [f1_score(y_val[:,idx_class], y_pred[:,idx_class], zero_division=0) for idx_class in range(num_classes)]
	fold_metrics['mcc'] = [matthews_corrcoef(y_val[:,idx_class], y_pred[:,idx_class]) for idx_class in range(num_classes)]
	fold_metrics['roc-auc'] = roc_auc_score(y_val[:,1:], y_pred[:,1:], average='macro')
	return fold_metrics

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

def print_evaluation_metric(metrics, name_metric):
	print('   '+name_metric+':')
	for metric in metrics:
		print('     + others: %.4f' % metrics[0])
		print('     + happy: %.4f' % metrics[1])
		print('     + sad: %.4f' % metrics[2])
		print('     + angry: %.4f' % metrics[3])
