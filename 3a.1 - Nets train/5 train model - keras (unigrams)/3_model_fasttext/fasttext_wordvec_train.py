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
	print('Start date and time:\n',datetime.now().strftime('%Y-%m-%d %H:%M'),'\n')                # '%Y-%m-%d %H:%M:%S'

	# Skipgram model
	start = time.time()
	
	# Set parameters
	num_features = 400     # Word vector dimensionality. (default=100)
	min_word_count_ = 1   # ignore words that are less than count of min_word_count in corpus. (default=5)
	threads_ = 4          # Number of threads to run in parallel
	context = 20           # Context window size
	iter_ = 100

	input_file_ = 'sampled_descriptionary_sample_size_5000.txt'
	out_model_name = 'fasttext__vec'+str(num_features)+'_win'+str(context)+'__dict_sampled_5000'

	input_file_ = 'scorecards_for_fasttext_processed.txt'
	out_model_name = 'fasttext__vec'+str(num_features)+'_win'+str(context)+'__scorecards'

	# Print parameters
	print('\nModel parameters')
	print('%-20s' % '   Word vector size',':',num_features)
	print('%-20s' % '   min word count',':',min_word_count_)
	print('%-20s' % '   Window size',':',context)
	print('%-20s' % '   Iterations',':',iter_)
	print('%-20s' % '   Input file name',':',input_file_)
	print('%-20s' % '   Output file name',':',out_model_name)
	print('\nTraining Fasttext model ...\n')
	
	model = fasttext.skipgram(input_file=input_file_, 
							output=out_model_name, 
							min_count=min_word_count_, 
							epoch=iter_, 
							dim=num_features, 
							thread = threads_, 
							ws = context)

	print("Model training took %g s\n" % (time.time() - start))

	model = word2vec.Word2Vec.load_word2vec_format(out_model_name+'.vec')

	#6 Test Word2Vec model
	test_words = ['screwdriver','pliers','smartphone','baby','bowtie']
	print('='*100)
	print('Test for '+str(len(test_words))+' words:')
	for word in test_words:
		print('   Similar to ' + word)
		print_list(model.most_similar(word, topn=5))


	# CBOW model
	#model = fasttext.cbow('data.txt', 'model')
	#print(model.words) # list of words in dictionary