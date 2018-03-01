import pandas as pd
import logging
import sys
import time
from IPython.display import display, HTML
from gensim.models import word2vec, Phrases
from nltk.corpus import brown, movie_reviews, treebank

def NGramGenerator_wordwise_interval(phrase, min_ngram, max_ngram):
    all_ngram_lists = []
    #s_split = phrase.split()
    s_split = "".join((char if char.isalnum() else " ") for char in phrase).split()
    
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
	
	#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	#1 Read data
	print('Reading data ...')
	#sentences_df = pd.read_csv('/Users/altay.amanbay/Desktop/3 - Clustered data/items_lc_ano_hClustered_complete.csv', header = 0, nrows = None)
	#sentences_df = pd.read_csv('/Users/altay.amanbay/Desktop/Caulking guns.tsv', sep = '\t', header = 0, nrows = None)
	sentences_df = pd.read_csv('/Users/altay.amanbay/Desktop/word2vec/data/sampled_descriptionary_sample_size_5000.csv', header = 0, nrows = None)
	print('Data shape:',sentences_df.shape)


	#2 Prepare data for word2vec model
	sentences_ls = []
	for index, row in sentences_df.iterrows():
		#tokens_ls = NGramGenerator_wordwise_interval(row[0] + ' ' + str(row[1]),1,1)
		sentence = row['description']
		sentence = sentence.lower()
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
	model_name = 'word2vect_vec'+str(num_features)+'_win'+str(context)
	model.save(model_name)
	print('Model saved as %s \n' % model_name)

	#6 Test Word2Vec model
	words = ['screwdriver','pliers','smartphone']
	for word in words:
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





