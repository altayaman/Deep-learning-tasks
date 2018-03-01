import itertools
from gensim.models.word2vec import Text8Corpus
from glove import Glove
import glove
import pandas as pd
import logging
import sys
import time
from IPython.display import display, HTML
from gensim.models import word2vec, Phrases
#from nltk.corpus import brown, movie_reviews, treebank

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


#sentences = list(itertools.islice(Text8Corpus('text8.zip'),None))
#print(type(sentences))

## main function
if __name__ == "__main__":
	
	#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	#1 Read data
	print('Reading data ...')
	#sentences_df = pd.read_csv('/Users/altay.amanbay/Desktop/3 - Clustered data/items_lc_ano_hClustered_complete.csv', header = 0, nrows = None)
	#sentences_df = pd.read_csv('/Users/altay.amanbay/Desktop/Caulking guns.tsv', sep = '\t', header = 0, nrows = None)
	sentences_df = pd.read_csv('sampled_descriptionary_sample_size_5000.csv', header = 0, nrows = 10)
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

	glove = glove(no_components=100, learning_rate=0.05)



