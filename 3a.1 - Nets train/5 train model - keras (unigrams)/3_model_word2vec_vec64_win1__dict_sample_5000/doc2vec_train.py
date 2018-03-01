import pandas as pd
import logging
import sys
import os
import time
from datetime import datetime
from IPython.display import display, HTML
from gensim.models import word2vec, Phrases, Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim.utils import to_unicode
from random import shuffle
#from nltk.corpus import brown, movie_reviews, treebank




def NGramGenerator_wordwise_interval(phrase, min_ngram, max_ngram):
	all_ngram_lists = []

    #printable_ = 'abcdefghijklmnopqrstuvwxyz0123456789 '
    #s_split = "".join((char if char in printable_ else "") for char in phrase).split()
	phrase_processed = process_string(phrase)
	s_split = phrase_processed.split()
    
	for n in range(max_ngram, min_ngram - 1, -1):
		n_gram = [s_split[i:i+n] for i in range(len(s_split)-n+1)]
		all_ngram_lists.extend(n_gram)
        
	all_ngrams = []
	for n_gram in all_ngram_lists:
		all_ngrams.extend([' '.join(n_gram)])
    
	return all_ngrams

def print_list(ls):
	for i in ls:
		print(i)
	print()



## main function
if __name__ == "__main__":
	print('Start date and time:\n',datetime.now().strftime('%Y-%m-%d %H:%M'),'\n')                # '%Y-%m-%d %H:%M:%S'
	#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	#0 Import additional script
	pardir = os.path.abspath(os.path.join(os.path.realpath(__file__), '../../'))
	print('Importing process_string.py \nfrom '+pardir+"/2_common_aux_script ...\n")
	sys.path.append(pardir+"/2_common_aux_script")
	from process_string import process_string
	sys.path.remove(pardir+"/2_common_aux_script")

	#1 Read data
	print('Reading data ...')
	path = pardir+"/1_data/"
	file_name = "sampled_descriptionary_sample_size_30.csv"
	sentences_df = pd.read_csv(path + file_name, header = 0, nrows = None)
	print('Data shape:',sentences_df.shape)


	#2 Prepare data for word2vec model
	sentences_ls = []
	for index, row in sentences_df.iterrows():
	    sentence = row['description']
	    sentences_ls.append(LabeledSentence(to_unicode(process_string(sentence)).split(),[str(index)]))
	sentences_ls[0]
	#LabeledSentence(words=['it', 'jeans', 'maternity', 'skinny', 'jeans', 'dark', 'wash', 'm'], tags=[0])


    #3 Set parameters
	min_word_count = 1    # ignore words that are less than count of min_word_count in corpus. (default=5)
	context = 3           # Context window size
	num_features = 64     # Word vector dimensionality
	downsampling = 1e-3   # Downsample setting for frequent words
	num_workers = 4       # Number of threads to run in parallel
	epochs_ = 1000

	# Print parameters
	print('\nModel parameters')
	print('%-20s' % '   Word vector size',':',num_features)
	print('%-20s' % '   min word count',':',min_word_count)
	print('%-20s' % '   Window size',':',context)
	print('%-20s' % '   Iterations',':',epochs_)
	#print('%-20s' % '   Techinque used',':',('Skip-gram' if sg == 1 else 'CBoW'))	
	

	#4 Build vocab
	print('\nCreating vocabulary for Doc2Vec model ...')
	model = Doc2Vec(min_count=min_word_count, window=context, size=num_features, sample=downsampling, negative=5, workers=num_workers)
	model.build_vocab(sentences_ls)


	#5 Start training Doc2Vec model
	print('\nTraining Doc2Vec model ...\n')
	start = time.time()
	print('Completed epochs:')
	for epoch in range(epochs_):
		if((epoch+1) % 10 == 0):
			print('%g/%g, ' % ((epoch+1), epochs_), sep=' ', end='', flush=True)
		shuffle(sentences_ls)
		model.train(sentences_ls)	
	print("Model training took %g s\n" % (time.time() - start))

	## If Word2Vec model training finished and no more updates, only querying
	model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)


	#6 Save model
	model_name = 'doc2vect_vec'+str(num_features)+'_win'+str(context)+'__dict_sample_5000'
	model.save(model_name)
	print('Model saved as %s \n' % model_name)


	#7 Test Word2Vec model
	test_words = ['screwdriver','pliers','smartphone']
	print('='*100)
	print('Test for '+str(len(test_words))+' words:')
	for word in test_words:
		print('   Similar to ' + word)
		print_list(model.most_similar(word, topn=5))









