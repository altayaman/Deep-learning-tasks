import numpy as np
import pandas as pd
import time
from threading import Lock
import os
import sys

#from __future__ import print_function
import logging
from dateutil.parser import parse as parse_date
from elasticsearch import Elasticsearch
import Multinomial_NB__functions as m


# experimental start
def main_1():
    ## get trace logger and set level
    tracer = logging.getLogger('elasticsearch.trace')
    tracer.setLevel(logging.INFO)
    tracer.addHandler(logging.FileHandler('/tmp/es_trace.log'))

    ## instantiate local ES client, connects to localhost:9200 by default
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

    # Read data
    df = pd.read_csv('/Users/altay.amanbay/Desktop/Power tools csv.csv', encoding = "ISO-8859-1", nrows=None)
    print('Data size:',df.shape)
    #df.head(5)

    
    # Insert items into ES
    start = time.time()
    lock = Lock()
    rgx_pattern = r'[^a-zA-Z0-9.-/"\']'
    print('Indexing is in process ...')
    for idx, row in df.iterrows():
        sentence_text = row['description']
        class_ = row['annotated_category_name']
        
        #lock.acquire()
        counter = counter + 1
        #lock.release()

        m.insert_doc_into_index_ngrams(sentence_text, rgx_pattern, class_, "naive_bayes", "type_1", es, counter)

    print('Indexing process took %g s' % (time.time()-start))    

# inserting csv files row by row. takes much longer.
def main_2():
    ## get trace logger and set level
    tracer = logging.getLogger('elasticsearch.trace')
    tracer.setLevel(logging.INFO)
    tracer.addHandler(logging.FileHandler('/tmp/es_trace.log'))

    ## instantiate local ES client, connects to localhost:9200 by default
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

    # Read data
    path = '/Users/altay.amanbay/Desktop/new node booster/experiments/train data from descriptionary nodes by sampling/1 - Descriptionary nodes into separate csv files/Descriptionary nodes in separate csv files/'
    print('\nReading files from following path:')
    print(path)


    print('\nGlobal descriptionary nodes indexing is starting ...')
    start = time.time()

    # iterate through csv files and index them in ES
    rgx_pattern = r'[^a-zA-Z0-9.-/"\']'
    counter = 0
    for file in os.listdir(path):
        # Pick only csv files
        if file.endswith(".csv"):
            if(file == '1-332.csv'):
                continue
            
            # Read data from csv file
            print('\n'+'='*100)
            print('Reading file %s ...' % file)
            df = pd.read_csv(os.path.join(path, file))
            print('Data shape ', df.shape[0])
            #print(df.head(2))
            #sys.exit()
            #print(os.path.join(path, file))

    
            # Insert items into ES
            local_start = time.time()
            #lock = Lock()
            print('\nLocal indexing is in process ...')
            for idx, row in df.iterrows():
                sentence_text = row['description']
                class_ = row['category_path']
                
                #lock.acquire()
                counter = counter + 1
                #lock.release()

                m.insert_doc_into_index_ngrams(sentence_text, rgx_pattern, class_, "naive_bayes", "type_1", es, counter)

            print('Local indexing is complete for file: ',file)
            print('Local indexing process took %g s' % (time.time()-local_start))

    print('\nGlobal descriptionary nodes indexing took %g s' % (time.time()-start))

# inserting csv files as a bulk. works faster.
def main_3():
    ## get trace logger and set level
    tracer = logging.getLogger('elasticsearch.trace')
    tracer.setLevel(logging.INFO)
    tracer.addHandler(logging.FileHandler('/tmp/es_trace.log'))

    ## instantiate local ES client, connects to localhost:9200 by default
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

    # Read data
    path = '/Users/altay.amanbay/Desktop/new node booster/experiments/train data from descriptionary nodes by sampling/1 - Descriptionary nodes into separate csv files/Descriptionary nodes in separate csv files 2/'
    print('\nReading files from following path:')
    print(path)


    print('\nGlobal descriptionary nodes indexing is starting ...')
    start = time.time()

    
    # Get max doc id as starting
    max_val_response = m.get_max_value("naive_bayes", "type_1", 'doc_id', es)
    doc_id_counter = max_val_response['aggregations']['max_value']['value']
    if(doc_id_counter==None):
        doc_id_counter = 1
        print('Starting doc_id counter',doc_id_counter)
    else:
        doc_id_counter = int(doc_id_counter) + 1
        print('Starting doc_id counter',doc_id_counter)

    # iterate through csv files (descriptionary nodes) in folder and index them in ES
    rgx_pattern = r'[^a-zA-Z0-9.-/"\']'
    for file in os.listdir(path):
        # Pick only csv files
        if file.endswith(".csv"):
            if(file == '1-332.csv'):
                continue
            
            # Read data from csv file
            print('\n'+'='*100)
            print('Reading file %s ...' % file)
            df = pd.read_csv(os.path.join(path, file))
            print('Data shape ', df.shape[0])
            #print(df.head(2))
            #sys.exit()
            #print(os.path.join(path, file))

    
            # Insert items into ES
            local_start = time.time()
            #lock = Lock()
            print('\nLocal indexing is in process ...')
            doc_id_counter = m.insert_bulk_doc_into_index_ngrams(df, rgx_pattern, "naive_bayes", "type_1", es, doc_id_counter)

            print('Local indexing is complete for file: ',file)
            print('Local indexing process took %g s' % (time.time()-local_start))

    print('\nGlobal descriptionary nodes indexing took %g s' % (time.time()-start))



## main function
if __name__ == "__main__":

    main_3()


