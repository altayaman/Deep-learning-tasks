import fasttext
from gensim.models import word2vec
import pandas as pd
import logging
import sys
import os.path
import time
from datetime import datetime
from IPython.display import display, HTML


def print_list(ls):
	for i in ls:
		print(i)
	print()
	

## main function
if __name__ == "__main__":
	print('='*100)
	print('Start date and time:\n',datetime.now().strftime('%Y-%m-%d %H:%M'),'\n')                # '%Y-%m-%d %H:%M:%S'

	# Skipgram model
	start = time.time()
	
	# Set parameters
	embedding_dim = 100   # default 100
	min_word_count_ = 1   # ignore words that are less than count of min_word_count in corpus. (default=5)
	context = 4           # Context window size, default 5
	epochs = 60           # default 5
	threads_ = 4          # Number of threads to run in parallel


	input_file_ = 'sampled_descriptionary_sample_size_5000.txt'
	out_model_name = 'fasttext_classifier__dict_sampled_5000'

	input_file_ = 'scorecards_for_fasttext_processed.txt'
	out_model_name = 'fasttext_classifier__scorecards'

	# Print parameters
	print('Model parameters')
	print('%-20s' % '   Word vector size',':',embedding_dim)
	print('%-20s' % '   min word count',':',min_word_count_)
	print('%-20s' % '   Window size',':',context)
	print('%-20s' % '   Epochs',':',epochs)
	print('%-20s' % '   Threads',':',threads_)
	print('\nTraining Fasttext model ...\n')	
	classifier = fasttext.supervised(input_file_, 
									out_model_name, 
									label_prefix='__label__', 
									epoch=epochs, 
									dim=embedding_dim, 
									min_count=min_word_count_, 
									ws=context,
									thread=threads_)
	print("Model training took %g s\n" % (time.time() - start))


	print("Testing model ...")
	result = classifier.test(input_file_)
	print('Precision:', result.precision)
	print('Recall:', result.recall)
	print('Number of examples:', result.nexamples)

	# CBOW model
	#model = fasttext.cbow('data.txt', 'model')
	#print(model.words) # list of words in dictionary