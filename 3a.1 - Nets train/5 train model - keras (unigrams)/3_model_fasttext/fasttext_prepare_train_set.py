import pandas as pd
import logging
import sys
import os.path
import time
from datetime import datetime
from IPython.display import display, HTML
from gensim.models import word2vec, Phrases
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
	#file_name = "sampled_descriptionary_sample_size_5000.csv"
	file_name = "scorecards_for_fasttext.csv"
	sentences_df = pd.read_csv(path + file_name, header = 0, nrows = None)
	sentences_df.rename(columns={'\ufeff"description"': 'description'}, inplace=True)
	print('Data shape:',sentences_df.shape)
	print('Data shape:',sentences_df.head(2))


	#1.2 Preprocess

	# a: Drop rows with NaN in any column
	sentences_df.dropna()
	
	# b:
	sentences_df['description'] = sentences_df['description'].apply(lambda x: process_string(x))

	#sentences_df = sentences_df[sentences_df['description'].str.len() > 4]
	# c: Drop rows where token count less than 1 in description_mod1 column
	selected_indices = sentences_df['description'].apply(lambda x: len(str(x).split()) > 1)
	sentences_df = sentences_df[selected_indices]

	sentences_df.drop_duplicates(subset=['description','category_path'], inplace = True, keep='first')
	sentences_df.drop_duplicates(subset=['description'], inplace = True, keep=False)
	print('Deduplicated data shape:',sentences_df.shape,'\n')


	#2 Prepare data for word2vec model
	outfile_name = 'sampled_descriptionary_sample_size_5000.txt'
	outfile_name = 'scorecards_for_fasttext_processed.txt'
	print('Outputting text file ...')
	print('File name: ',outfile_name)
	with open(outfile_name, 'w') as outfile:
		sentences_ls = []
		for index, row in sentences_df.iterrows():
			sentence = row['description']
			#print(sentence)
			label = str(row['category_id'])
			label_prefix = '__label__'
			#print('label: ',label,' type:',type(label))
			outfile.write(label_prefix+label+' ,'+sentence+'\n')








