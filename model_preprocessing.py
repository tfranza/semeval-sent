import gensim
import gensim.downloader as api
import numpy as np
import re
import torch
import torch.utils.data as data

from utils.raw_text_preprocessing import RawTextPreprocessor

#############################################################################################
def preprocess(args, data, train_flag=True):
	if 'semeval2019-task3' in args['--train-src']:
		tokenized_sentences = extract_features(args, data, 'semeval2019-task3', train_flag)['sentences']
		labels = extract_labels(data, 'semeval2019-task3')
		pretrained, input_embeddings = generate_embeddings(args, tokenized_sentences)
		return pretrained, input_embeddings, labels

#############################################################################################
def extract_features(args, data, dataset_name, train_flag=True):
	features = dict()
	preprocessor = None 
	if dataset_name=='semeval2019-task3':
		sentences = data.turn1 + ' <eos> ' + data.turn2 + ' <eos> ' + data.turn3
		sentences = [simple_preprocessing(sentence) for sentence in sentences]
		#preprocessor = RawTextPreprocessor(sentences)
#		if train_flag:
#			if (args['--dont-load-prep']):
#				steps_dict = dict(map(reversed, 					 # dict containing (key: seq order, value: prep step)
#					dict( 							# dict containing (key: prep step, value: seq order)
#						(k,int(args[k])) 
#						for k in ['--prep-verbs','--prep-lowering','--prep-emoticon-emojis','--prep-abbreviations','--prep-elongations','--prep-misspellings'] 
#						if args[k] != '-1'
#					).items()
#				))
#				steps = [steps_dict[i] for i in range(len(steps_dict))]		# list containing ordered prep steps
#				preprocessor.apply_steps(steps)
#				if not args['--dont-save-prep']:
#					preprocessor.save_to(args['--prep-text-save-to'])
#			else:
#				preprocessor.load_from(args['--prep-text-save-to'])
		#features['sentences'] = preprocessor.get_sentences(tokenized=True)
		features['sentences'] = [sentence.split(" ") for sentence in sentences]

	return features

def extract_labels(data, dataset_name):
	labels = None
	if dataset_name=='semeval2019-task3':
		labels = data.label.map({'others':[1,0,0,0], 'happy':[0,1,0,0], 'sad':[0,0,1,0], 'angry':[0,0,0,1]})
	return labels

def generate_embeddings(args, sentencelist):
	if args['--embedding-type']=='w2v':
		print('Loading word2vec model... \n')
		model = gensim.models.Word2Vec(sentences = sentencelist)
	elif args['--embedding-type']=='glv100':
		print('Loading GloVe-wiki-gigaword-100 model... \n')
		model = api.load('glove-wiki-gigaword-100')
	elif args['--embedding-type']=='glv300':
		print('Loading GloVe-wiki-gigaword-300 model... \n')
		model = api.load('glove-wiki-gigaword-300')
	elif args['--embedding-type']=='fasttext':
		print('Loading fasttext-wiki-news-subwords-300 model... \n')
		model = api.load('fasttext-wiki-news-subwords-300')
	else:
		print('Unable to find the given model name for embeddings:', args['--embedding-type'])
	#vectors = model.wv.vectors
	#vocab = model.wv.vocab
	model.wv['<eos>'] = np.zeros((100,))
	model.wv['<unk>'] = np.zeros((100,))

	word_to_freq = dict()
	for wordlist in sentencelist:
		for word in wordlist:
			if word in word_to_freq.keys():
				word_to_freq[word] += 1
			else: 
				word_to_freq[word] = 1
	vocab_words = sorted(list(word_to_freq.keys()), key=lambda k: word_to_freq[k], reverse=True)
	word_to_idx = { word:idx for idx, word in enumerate(vocab_words, start=3) }
	#word_to_idx['<pad>'] = 0
	word_to_idx['<unk>'] = 1
	word_to_idx['<eos>'] = 2
	idx_to_word = { idx:word for word, idx in word_to_idx.items() }
		
	embedding_size = model.wv.vectors.shape[1]
	
#	np.random.seed(embedding_size)
#	pretrained = np.insert(vectors, 0, np.random.uniform(vectors.min(),vectors.max(),embedding_size), axis=0)
#	pretrained = np.insert(pretrained, 0, np.zeros(embedding_size), axis=0)

	#word_to_idx = {word:idx for idx,word in enumerate(vocab, start=2)}
	
	# substituing words with ids
	input_embeddings = []
	for wordlist in sentencelist:
		input_embeddings.append( list(map(lambda word: word_to_idx[word] if word in word_to_idx else 1, wordlist)) )

	pretrained = np.zeros(shape=(len(word_to_idx)+3,embedding_size))
	for idx, word in idx_to_word.items():
		if word in model.wv.vocab:
			pretrained[idx] = model.wv[word]
		else:
			pretrained[idx] = model.wv['<unk>']

	# padding zeros into (shorter) embeddings
	sent_embed_size = max([len(wordlist) for wordlist in sentencelist])
	input_embeddings = list(map(lambda sent: np.pad(sent,(0,sent_embed_size-len(sent))), input_embeddings))


	print(pretrained.shape)
	return pretrained, input_embeddings

def generate_kfolds(args, dataset, device):
	training_sentence_data, training_label_data = dataset

	k_folds = int(args['--num-folds'])
	
	# features and labels grouped by category 
	indices = [torch.nonzero(training_label_data==c).reshape((-1,)) for c in range(int(args['--num-classes']))]
	sent_features = [training_sentence_data[idxs] for idxs in indices]
	sent_labels = [training_label_data[idxs] for idxs in indices]

	# contains num of examples each part has to contain according to the label
	ex_per_part = [int(len(idxs)/k_folds) for idxs in indices]
	
	# cycle needed to ensure that others, happy, sad and angry categories are balanced in folds
	fold_features = []
	fold_labels = []
	for i in range(k_folds):
		fold_features_per_category = [
			cat_sent_features[ (i)*ex_per_part[idx_cat] : (i+1)*ex_per_part[idx_cat] ] 
			for idx_cat, cat_sent_features in enumerate(sent_features)
		]
		fold_features.append(torch.row_stack(fold_features_per_category))

		fold_labels_per_category = [
			cat_sent_labels[ (i)*ex_per_part[idx_cat] : (i+1)*ex_per_part[idx_cat] ]
			for idx_cat, cat_sent_labels in enumerate(sent_labels)
		]
		fold_labels.append(torch.row_stack(fold_labels_per_category))
	
	# generating training and validation set for each fold
	fold_trainval_dataloaders = []
	for i in range(k_folds):
		train_features = torch.row_stack([fs for j,fs in enumerate(fold_features) if j!=i])
		train_labels = torch.row_stack([ls for j,ls in enumerate(fold_labels) if j!=i])
		train_dataset = data.TensorDataset(train_features.to(device), train_labels.to(device))
		train_dataloader = data.DataLoader(train_dataset, batch_size=int(args['--batch-size']), shuffle=True)
		
		valid_features = fold_features[i]
		valid_labels = fold_labels[i]
		valid_dataset = data.TensorDataset(valid_features.to(device), valid_labels.to(device))
		valid_dataloader = data.DataLoader(valid_dataset, batch_size=int(args['--batch-size']), shuffle=True)

		fold_trainval_dataloaders.append( (train_dataloader, valid_dataloader) )
		if args['--single-fold']:
			break

	return fold_trainval_dataloaders


def simple_preprocessing(sentence):
	sentence = sentence.lower()
	sentence = re.sub(r'[!]+', ' ! ', sentence)
	sentence = re.sub(r'[?]+', ' ? ', sentence)
	sentence = re.sub(r'[,]+', ' , ', sentence)
	sentence = re.sub(r'[.]+', ' . ', sentence)
	sentence = re.sub(r'[ ]+', ' ', sentence)
	return sentence.strip()















