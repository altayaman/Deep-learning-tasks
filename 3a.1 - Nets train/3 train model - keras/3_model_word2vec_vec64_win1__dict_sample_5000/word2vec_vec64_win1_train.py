import pandas as pd
import logging
import sys
import os.path
import time
from datetime import datetime
from IPython.display import display, HTML
from gensim.models import word2vec, Phrases
from nltk.corpus import brown, movie_reviews, treebank




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
	path = "/Users/altay.amanbay/Desktop/new node booster/experiments/train data from descriptionary nodes by sampling/3 - Picking samples from each node/sampled descriptionary/"
	file_name = "sampled_descriptionary_sample_size_5000.csv"
	data_file = ""
	sentences_df = pd.read_csv(path + file_name, header = 0, nrows = None)
	print('Data shape:',sentences_df.shape)


	#2 Prepare data for word2vec model
	sentences_ls = []
	for index, row in sentences_df.iterrows():
		sentence = row['description']
		tokens_ls = NGramGenerator_wordwise_interval(sentence,1,1)
		sentences_ls.append(tokens_ls)
	print('First sentence tokens:',sentences_ls[0])


	#sentences_ls = [['I','like','deep learning'],['I', 'like','NLP'],['I','enjoy','flying']]


    #3 Set parameters
	num_features = 64     # Word vector dimensionality. (default=100)
	min_word_count = 1    # ignore words that are less than count of min_word_count in corpus. (default=5)
	num_workers = 2       # Number of threads to run in parallel
	context = 1           # Context window size
	downsampling = 1e-3   # Downsample setting for frequent words
	iter_ = 1000
	sg = 1  # if 1 skip-gram technique is used, else CBoW. (default=0)

	# Print parameters
	print('\nModel parameters')
	print('%-20s' % '   Word vector size',':',num_features)
	print('%-20s' % '   min word count',':',min_word_count)
	print('%-20s' % '   Window size',':',context)
	print('%-20s' % '   Iterations',':',iter_)
	print('%-20s' % '   Techinque used',':',('Skip-gram' if sg == 1 else 'CBoW'))	
	print('\nTraining Word2Vec model ...\n')

	

	#4 Start training Word2Vec model
	start = time.time()
	model = word2vec.Word2Vec(sentences_ls, sg = 1, workers = num_workers, size = num_features, iter = iter_, min_count = min_word_count, window = context, sample = downsampling, seed = 1)
	#print(model.vocab['1'])
	print("Model training took %g s\n" % (time.time() - start))

	## If Word2Vec model training finished and no more updates, only querying
	model.init_sims(replace=True)

	#5 Save model
	model_name = 'word2vect_vec'+str(num_features)+'_win'+str(context)+'__dict_sample_5000'
	model.save(model_name)
	print('Model saved as %s \n' % model_name)

	#6 Test Word2Vec model
	test_words = ['screwdriver','pliers','smartphone']
	print('='*100)
	print('Test for '+str(len(test_words))+' words:')
	for word in test_words:
		print('   Similar to ' + word)
		print_list(model.most_similar(word, topn=5))




	## Train collocation model
	#bigram_transformer = Phrases(sentences_ls)

	# Test

	# Example
	#sent = [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there']
	#print(bigram_transformer[sent])
	# Expected output: [u'the', u'mayor', u'of', u'new_york', u'was', u'there'

	#model = Word2Vec(bigram_transformer[sentences_ls], size=100, ...)


	# For testing
	#print(model.doesnt_match(['sheets','pods','pacs','packs']))
	#print(model.similarity('oz','and'))
	#print_list(model.most_similar(positive = ['deep learning', 'NLP'], negative = ['I'], topn=5))
	#print_list(model.most_similar(negative = ['deep learning', 'NLP'], positive = ['I'], topn=5))
	#print_list(model.most_similar(negative = ['1', 'seventh'], positive = ['2'], topn=5))





