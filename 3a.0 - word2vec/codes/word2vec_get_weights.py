import pandas as pd
import logging
import sys
import time
from IPython.display import display, HTML
from gensim.models import word2vec, Phrases
from nltk.corpus import brown, movie_reviews, treebank
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering


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
    file_path = '/Users/altay.amanbay/Desktop/word2vec/word2vect_64__dict_sample_1000'
    model = word2vec.Word2Vec.load(file_path)

    print(model.vocab.keys())
    sys.exit()


    #word vector embeddings from model into dictionary
    word2vec_dict={}
    for word in model.vocab.keys():
        try:
            word2vec_dict[word]=model[word]
        except:    
            pass