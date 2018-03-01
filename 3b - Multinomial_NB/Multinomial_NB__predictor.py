import numpy as np
import pandas as pd
import math
import time
from threading import Lock
from sqlalchemy import create_engine
from sqlalchemy import text
import psycopg2  ## for RedShift
import sys
import getpass
from optparse import OptionParser
from configparser import SafeConfigParser
from multiprocessing import Pool
from functools import partial


#from __future__ import print_function
import logging
from dateutil.parser import parse as parse_date
from elasticsearch import Elasticsearch
import Multinomial_NB__functions as m

## Read arguments
def get_args():
    help_text = """CME Node Recall Booster"""
    parser = OptionParser(usage=help_text)

    # RedShift credentials
    #parser.add_option("-H", "--host",           dest="host",            default = "rops.dwh.prod.slicetest.com", help="the url for the DB", metavar="HOST_ADDRESS")
    #parser.add_option("-d", "--db",             dest="database",        default = "sliceds",             help="the logical db name")
    #parser.add_option("-u", "--username",       dest="username",        default = "infoprod_ops_admin",  help="the username for the DB", metavar="NAME")

    # MsSQL credentials
    parser.add_option("-H",    "--host",           dest="host",            default = "RAD1.dwh.prod.slicetest.com", help="the url for the DB", metavar="HOST_ADDRESS")
    parser.add_option("-d",    "--db",             dest="database",        default = "altay",                       help="the logical db name")
    parser.add_option("-u",    "--username",       dest="username",        default = "u_altay.amanbay",             help="the username for the DB", metavar="NAME")
    
    parser.add_option("-p",    "--password",       dest="password")
    parser.add_option("-m",    "--model",          dest="model",           help="file that contains classifier, countvectorizer and their meta data")
    parser.add_option("-i",    "--input_table",    dest="input_table")
    parser.add_option("-o",    "--output_table",   dest="output_table")
    parser.add_option("--pkl", "--pickle",         dest="pickle",          help="the pickle file if data is cached", metavar="PICKLE")
    parser.add_option("--cf",  "--config_file",    dest="config_file",     help="config_file that has all settings")

    (options, args) = parser.parse_args()
    # if options.output_table is not None and options.model is not None and (options.pickle is not None or options.input_table is not None):
    #     password = getpass.getpass("\nPlease enter the password for the DB: ")
    #     options.password = password

    #     print("\nEstablished connection with the database")
    #     global global_db_engine
    #     global_db_engine = get_db_engine(options)
    #     return (options, args)
    # else:
    #     print("Need to specify Output table and model file and one of the followings:")
    #     print('Input table')
    #     print('OR')
    #     print('Pickled input data')
    #     sys.exit()

    return (options, args)

def check_if_file_exists(file_path):
    try:
        with open(file_path) as infile:
            pass
    except IOError:
        print('  ERROR: '+file_path+' file not found\n')
        sys.exit()

def get_db_engine_2(config_set_, username_, password_, host_, database_, port_):
    # for RedShift and Post
    if(config_set_ == 'redshift'):
        url = ''.join(['postgresql://', username_, ":", password_, "@", host_, ':',port_, '/', database_])

    # for MsSQL
    elif(config_set_ == 'mssql'):
        url = ''.join(['mssql+pymssql://', username_, ":", password_, "@", host_, ':',port_, '/', database_])

    print()
    print('='*100)
    print('%-22s' % 'Connecting to url',':',url)
    engine = create_engine(url)
    return engine

def get_db_engine(options):
    # for RedShift
    #url = "".join(["postgresql://", options.username, ":", options.password, "@", options.host, ":5439/", options.database])

    # for MsSQL
    url = "".join(["mssql+pyodbc://", options.username, ":", options.password, "@", options.host, ":1143/", options.database])

    engine = create_engine(url)
    return engine

def check_n_get_config_options(options_, args):
    parser = SafeConfigParser()

    if(options_.config_file):
        parser.read(options_.config_file)
        prediction_option_ls = parser.options('prediction')

        options_ls_1 = ['test_data', 'model_file', 'model_file_aux', 'prediction_dest', 'pickle']
        options_ls_2 = ['test_data', 'model_file', 'model_file_aux','prediction_dest']
        options_ls_3 = ['pickle','model_file','model_file_aux','prediction_dest']
        
        # retuns following values
        # parser, test_data_arg_section, test_data_arg, model_file_name, prediction_arg_section, prediction_arg, pickle_file_name
        if(set(prediction_option_ls).issuperset(set(options_ls_1))):
            return parser,  \
                   (parser.get('prediction', 'test_data')).split('+')[0],        \
                   (parser.get('prediction', 'test_data')).split('+')[1],        \
                   parser.get('prediction', 'model_file'),                       \
                   parser.get('prediction', 'model_file_aux'),                   \
                   (parser.get('prediction', 'prediction_dest')).split('+')[0],  \
                   (parser.get('prediction', 'prediction_dest')).split('+')[1],  \
                   parser.get('prediction', 'pickle')
        elif(set(prediction_option_ls).issuperset(set(options_ls_2))):
            return parser,  \
                   (parser.get('prediction', 'test_data')).split('+')[0],  \
                   (parser.get('prediction', 'test_data')).split('+')[1],   \
                   parser.get('prediction', 'model_file'),                 \
                   parser.get('prediction', 'model_file_aux'),                   \
                   (parser.get('prediction', 'prediction_dest')).split('+')[0],  \
                   (parser.get('prediction', 'prediction_dest')).split('+')[1],  \
                   None
        elif(set(prediction_option_ls).issuperset(set(options_ls_3))):
            return parser,   \
                   None,     \
                   None,     \
                   parser.get('prediction', 'model_file'),      \
                   parser.get('prediction', 'model_file_aux'),                   \
                   (parser.get('prediction', 'prediction_dest')).split('+')[0],  \
                   (parser.get('prediction', 'prediction_dest')).split('+')[1],  \
                   parser.get('prediction', 'pickle')
        else:
            print('Some prediction argumnets are missing')
            sys.exit()
    else:
        print(' --cf option is not passed')
        sys.exit()

def get_DBapi_result_proxy(parser, test_data_arg_section, test_data_arg, pickle_file):

    if(test_data_arg_section == 'config_file'):
        test_file_name = parser.get(test_data_arg_section, test_data_arg)
        check_if_file_exists(test_file_name)
        df = pd.read_csv(test_file_name)
        print('%-22s' % ' Test data source' ,':',test_file_name)
        print('%-22s' % ' Size of dataframe',':',df.shape)

        if parser.get('prediction', 'pickle'):
            df.to_pickle(parser.get('prediction', 'pickle'))
            print('%-22s' % '\n Test data pickled as',':',parser.get('prediction', 'pickle'))

    #elif(test_data_arg_section == 'config_redshift'):
    elif(parser.get(test_data_arg_section, 'db_type') == 'redshift'):
        # Get database engine
        db_type  = parser.get(test_data_arg_section, 'db_type')
        username = parser.get(test_data_arg_section, 'username')
        password = parser.get(test_data_arg_section, 'password')
        host     = parser.get(test_data_arg_section, 'host')
        database = parser.get(test_data_arg_section, 'database')
        port     = parser.get(test_data_arg_section, 'port')
        engine   = get_db_engine_2(db_type, username, password, host, database, port)
        print(" Established connection with the database")

        # Fetch data from db
        start = time.time()
        print("\n Getting ResultProxy ...")
        test_data_table_name = parser.get(test_data_arg_section, test_data_arg)
        #df = pd.read_sql_query('SELECT * FROM ' + test_data_table_name + ' limit 3', engine)
        qry = 'SELECT * FROM ' + test_data_table_name #+ ' limit 3'
        conn = engine.connect()
        resultProxy = conn.execute(qry)

        ## Get elapsed time
        end = time.time()
        print(" Getting ResultProxy took %g s" % (end - start))
        print('%-22s' % ' Test data source' ,':',test_data_table_name,' from RedShift')
        #print(" Found " + str(len(df)) + " entries")

    #elif(test_data_arg_section == 'config_mssql'):
    elif(parser.get(test_data_arg_section, 'db_type') == 'mssql'):
        # Get database engine
        db_type  = parser.get(test_data_arg_section, 'db_type')
        username = parser.get(test_data_arg_section, 'username')
        password = parser.get(test_data_arg_section, 'password')
        host     = parser.get(test_data_arg_section, 'host')
        database = parser.get(test_data_arg_section, 'database')
        port     = parser.get(test_data_arg_section, 'port')
        engine   = get_db_engine_2(db_type, username, password, host, database, port)
        print(" Established connection with the database")

        # Fetch data from db
        start = time.time()
        print("\n Getting ResultProxy ...")
        test_data_table_name = parser.get(test_data_arg_section, test_data_arg)
        #df = pd.read_sql_query('SELECT top 3 * FROM ' + test_data_table_name, engine)
        qry = 'SELECT top 3 * FROM ' + test_data_table_name
        conn = engine.connect()
        df = conn.execute(qry)
        resultProxy = conn.execute(qry)

        ## Get elapsed time
        end = time.time()
        print(" Getting ResultProxy took %g s" % (end - start))
        print('%-22s' % ' Test data source' ,':',test_data_table_name,' from MsSQL')
        #print(" Found " + str(len(df)) + " entries")

    #elif parser.get('prediction', 'pickle'):
    else:
        #pkl_file = parser.get('prediction', 'pickle')
        #check_if_file_exists(pkl_file)
        check_if_file_exists(test_data_arg_section)
        #df = pd.read_pickle(pkl_file)
        df = pd.read_pickle(test_data_arg_section)
        #print('%-22s' % ' Test data source',':',pkl_file)
        print('%-22s' % ' Test data source',':',test_data_arg_section)
        print('%-22s' % ' Size of dataframe',':',df.shape)

    return resultProxy

def get_data_df_2(parser, test_data_arg_section, test_data_arg, pickle_file):

    if(test_data_arg_section == None):
        #pkl_file = parser.get('prediction', 'pickle')
        #check_if_file_exists(pkl_file)
        check_if_file_exists(pickle_file)
        #df = pd.read_pickle(pkl_file)
        df = pd.read_pickle(pickle_file)
        #print('%-22s' % ' Test data source',':',pkl_file)
        print('%-22s' % 'Test data source',':',pickle_file)
        print('%-22s' % 'Size of dataframe',':',df.shape)

    elif(test_data_arg_section == 'config_file'):
        test_file_name = parser.get(test_data_arg_section, test_data_arg)
        check_if_file_exists(test_file_name)
        df = pd.read_csv(test_file_name)
        print('%-22s' % 'Test data source' ,':',test_file_name)
        print('%-22s' % 'Size of dataframe',':',df.shape)

        if parser.get('prediction', 'pickle'):
            df.to_pickle(parser.get('prediction', 'pickle'))
            print('%-22s' % '\n Test data pickled as',':',parser.get('prediction', 'pickle'))

    #elif(test_data_arg_section == 'config_redshift'):
    elif(parser.get(test_data_arg_section, 'db_type') == 'redshift'):
        # Get database engine
        db_type  = parser.get(test_data_arg_section, 'db_type')
        username = parser.get(test_data_arg_section, 'username')
        password = parser.get(test_data_arg_section, 'password')
        host     = parser.get(test_data_arg_section, 'host')
        database = parser.get(test_data_arg_section, 'database')
        port     = parser.get(test_data_arg_section, 'port')
        engine   = get_db_engine_2(db_type, username, password, host, database, port)
        print("Established connection with the RedShift")

        # Fetch data from db
        start = time.time()
        print("\nReading input test data from RedShift ...")
        test_data_table_name = parser.get(test_data_arg_section, test_data_arg)
        df = pd.read_sql_query('SELECT * FROM ' + test_data_table_name, engine)
        #print('SELECT * FROM ' + test_data_table_name)

        ## Get elapsed time
        end = time.time()
        print("Reading data from RedShift took %g s" % (end - start))
        print("Found " + str(len(df)) + " entries")

        #if parser.get('prediction', 'pickle'):
        if pickle_file:
            #print(parser.get('prediction', 'pickle'))
            #df.to_pickle(parser.get('prediction', 'pickle'))
            df.to_pickle(pickle_file)
            #print('%-22s' % 'Test data pickled as',':',parser.get('prediction', 'pickle'))
            print('%-22s' % '\n Test data pickled as',':',pickle_file)

    elif(parser.get(test_data_arg_section, 'db_type') == 'mssql'):
        # Get database engine
        db_type  = parser.get(test_data_arg_section, 'db_type')
        username = parser.get(test_data_arg_section, 'username')
        password = parser.get(test_data_arg_section, 'password')
        host     = parser.get(test_data_arg_section, 'host')
        database = parser.get(test_data_arg_section, 'database')
        port     = parser.get(test_data_arg_section, 'port')
        engine   = get_db_engine_2(db_type, username, password, host, database, port)
        print("Established connection with the MsSQL")

        # Fetch data from db
        start = time.time()
        print("\nReading input test data from MsSQL ...")
        test_data_table_name = parser.get(test_data_arg_section, test_data_arg)
        df = pd.read_sql_query('SELECT * FROM ' + test_data_table_name, engine)
        #print('SELECT * FROM ' + test_data_table_name)

        ## Get elapsed time
        end = time.time()
        print("Reading data from MsSQL took %g s" % (end - start))
        print("Found " + str(len(df)) + " entries")

        #if parser.get('prediction', 'pickle'):
        if pickle_file:
            #print(parser.get('prediction', 'pickle'))
            #df.to_pickle(parser.get('prediction', 'pickle'))
            df.to_pickle(pickle_file)
            #print('%-22s' % 'Test data pickled as',':',parser.get('prediction', 'pickle'))
            print('%-22s' % '\nTest data pickled as',':',pickle_file)

    #elif parser.get('prediction', 'pickle'):
    # else:
    #     #pkl_file = parser.get('prediction', 'pickle')
    #     #check_if_file_exists(pkl_file)
    #     check_if_file_exists(test_data_arg_section)
    #     #df = pd.read_pickle(pkl_file)
    #     df = pd.read_pickle(test_data_arg_section)
    #     #print('%-22s' % ' Test data source',':',pkl_file)
    #     print('%-22s' % ' Test data source',':',test_data_arg_section)
    #     print('%-22s' % ' Size of dataframe',':',df.shape)

    #else:
    #    print(" Need to specify either input db table or pickle file or input file")
    #    sys.exit()

    return df

def get_ranges_for_df(input_df_size_, insertion_chunk_size_):
    ranges_ = []

    c = 0
    while(True):
        if(input_df_size_ >= insertion_chunk_size_):
            input_df_size_ = input_df_size_ - insertion_chunk_size_
            range_ = (c * insertion_chunk_size_ , (c+1) * insertion_chunk_size_ - 1)
            ranges_.extend([range_])
            c = c + 1
            if(input_df_size_ == 0):
                break
        else:
            if(input_df_size_-1 < c*insertion_chunk_size_):
                range_ = (c * insertion_chunk_size_, (c*insertion_chunk_size_) + input_df_size_ - 1)
            else:
                range_ = (c * insertion_chunk_size_, input_df_size_ - 1)
            ranges_.extend([range_])
            break

    return ranges_

def get_prediction_results_laplace(item_string, es):
    regex_pattern = r'[^a-zA-Z0-9.-/"\']'
    regex_pattern = r'^([a-z]|[A-Z]|[0-9]|[\.]|[-]|[/]|["]|[\'])'
    ngrams_ls = m.NGramGenerator_wordwise_interval(item_string, regex_pattern, 1, 1)

    # Classes list and probs for each ngram token for each class
    classes_ls, ngram_probs_dict = m.get_ngram_probs_laplace(item_string, regex_pattern, es)
    if(len(classes_ls) == 0):
        return 'NA', 0

    docs_share_by_class_dict = m.docs_share_by_class(classes_ls, 'naive_bayes', 'type_1', 'category_path.keyword', 'doc_id', es)

    # Calculate original predictions
    predictions_dict = {}
    for class_ in classes_ls:
        prob = 1000000

        for ngram in ngrams_ls:
            if(ngram_probs_dict[ngram][class_]['result'] == 0):
                continue
            prob = prob * ngram_probs_dict[ngram][class_]['result']
        predictions_dict[class_] = prob * docs_share_by_class_dict[class_]
            
    
    # Normalize predictions by converting them into Liklihood
    probs = {}
    if(len(predictions_dict) == 1):
        #print('\nProba: 1')
        for k,v in predictions_dict.items():
            probs[k] = 1
    else:
        sum_probs = sum(predictions_dict.values())
        #print('\nsum of probs:',sum_probs)
        
        #print('\nLiklihood predictions:')
        for k,v in predictions_dict.items():
            #print(k,':',v)
            probs[k] = v/sum_probs
            
    predicted_class = max(probs, key=probs.get)  # pick key by max value
    predicted_prob = probs[predicted_class]
    #print('\nFinal prediction:')
    #print(predicted_class, predicted_prob)

    return predicted_class, predicted_prob
    #return pd.Series({'Prediction':predicted_class, 'Prediction_proba':predicted_prob})

# same get_prediction_results_laplace but with printing step results
def get_prediction_results_laplace_print(item_string, es):
    regex_pattern = r'[^a-zA-Z0-9.-/"\']'
    regex_pattern = r'^([a-z]|[A-Z]|[0-9]|[\.]|[-]|[/]|["]|[\'])'
    ngrams_ls = m.NGramGenerator_wordwise_interval(item_string, regex_pattern, 1, 1)
    

    print('\nitem_string\n',item_string)
    print('ngrams_ls\n',ngrams_ls)

    # Classes list and probs for each ngram token for each class
    classes_ls, ngram_probs_dict = m.get_ngram_probs_laplace(item_string, regex_pattern, es)
    print('\nclasses_ls\n',classes_ls,'\n')
    print('\nngram_probs_dict\n',ngram_probs_dict,'\n')
    if(len(classes_ls) == 0):
        return 'NA', 0

    docs_share_by_class_dict = m.docs_share_by_class(classes_ls, 'naive_bayes', 'type_1', 'category_path.keyword', 'doc_id', es)
    print('docs_share_by_class_dict\n',docs_share_by_class_dict,'\n')

    # Calculate original predictions
    predictions_dict = {}
    for class_ in classes_ls:
        print(class_)
        prob = 1000000

        for ngram in ngrams_ls:
            if(ngram_probs_dict[ngram][class_]['result'] == 0):
                continue

            prob = prob * ngram_probs_dict[ngram][class_]['result']
            print('ngram',ngram)
            print('prob',prob)
            print('ngram_probs_dict',ngram_probs_dict[ngram][class_]["result"])
        print('point A')
        predictions_dict[class_] = prob * docs_share_by_class_dict[class_]


    #print('\nRaw predictions:')
    #for k,v in predictions_dict.items():
    #    print(k,':',v)
        
    # Normalize predictions by converting them into Liklihood
    probs = {}
    if(len(predictions_dict) == 1):
        #print('\nProba: 1')
        for k,v in predictions_dict.items():
            probs[k] = 1
    else:
        sum_probs = sum(predictions_dict.values())
        #print('\nsum of probs:',sum_probs)
        
        #print('\nLiklihood predictions:')
        for k,v in predictions_dict.items():
            #print(k,':',v)
            probs[k] = v/sum_probs
            
    predicted_class = max(probs, key=probs.get)  # pick key by max value
    predicted_prob = probs[predicted_class]
    #print('\nFinal prediction:')
    #print(predicted_class, predicted_prob)

    return predicted_class, predicted_prob
    #return pd.Series({'Prediction':predicted_class, 'Prediction_proba':predicted_prob})

def get_prediction_results(item_string, es):
    regex_pattern = r'[^a-zA-Z0-9.-/"\']'
    regex_pattern = r'?!([a-z]|[A-Z]|[0-9]|[\.]|[-]|[/]|["]|[\'])*'
    regex_pattern = r' '
    ngrams_ls = m.NGramGenerator_wordwise_interval(item_string, regex_pattern, 1, 1)
    #print('\nitem_string',item_string)
    #print(ngrams_ls)

    
    # Classes list and probs for each ngram token for each class
    classes_ls, ngram_probs_dict = m.get_ngram_probs(item_string, regex_pattern, es)
    #print('\nclasses_ls\n',classes_ls,'\n')
    #print('\nngram_probs_dict\n',ngram_probs_dict,'\n')
    if(len(classes_ls) == 0):
        return 'NA', -1.0

    docs_share_by_class_dict = m.docs_share_by_class(classes_ls, 'naive_bayes', 'type_1', 'category_path.keyword', 'doc_id', es)
    #print('docs_share_by_class_dict\n',docs_share_by_class_dict,'\n')

    # Calculate original predictions
    predictions_dict = {}
    for class_ in classes_ls:
        #print(class_)
        prob = 0

        for ngram in ngrams_ls:
            #print('prob',prob)
            #print('ngram_probs_dict',ngram,ngram_probs_dict[ngram][class_]["result"])
            if(ngram_probs_dict[ngram][class_]['result'] == 0):
                continue
            else:
                prob = prob + ngram_probs_dict[ngram][class_]['result']
            #if(prob == 0.0):
            #    print('Exit on prob == 0.0')
            #    sys.exit()
        predictions_dict[class_] = prob + docs_share_by_class_dict[class_]
    
    #print('\nRaw predictions:')
    #for k,v in predictions_dict.items():
    #    print(k,':',v)
        

    # Normalize predictions by converting them into Liklihood
    probs = {}
    if(len(predictions_dict) == 1):
        #print('\nProba: 1')
        for k,v in predictions_dict.items():
            probs[k] = 1
    else:
        sum_probs = sum(predictions_dict.values())
        #print('\nsum of probs:',sum_probs)
        
        #print('\nLiklihood predictions:')
        for k,v in predictions_dict.items():
            #print(k,':',v)
            probs[k] = v/sum_probs
            
    predicted_class = max(probs, key=probs.get)  # pick key by max value
    predicted_prob = probs[predicted_class]
    #print('\nFinal prediction:')
    #print(predicted_class, predicted_prob)

    return predicted_class, predicted_prob
    #return pd.Series({'Prediction':predicted_class, 'Prediction_proba':predicted_prob})

# same get_prediction_results but with printing step results
def get_prediction_results_print(item_string, es):
    regex_pattern = r'[^a-zA-Z0-9.-/"\']'
    regex_pattern = r'^([a-z]|[A-Z]|[0-9]|[\.]|[-]|[/]|["]|[\'])'
    ngrams_ls = m.NGramGenerator_wordwise_interval(item_string, regex_pattern, 1, 1)
    print('\nitem_string',item_string)
    print(ngrams_ls)

    
    # Classes list and probs for each ngram token for each class
    classes_ls, ngram_probs_dict = m.get_ngram_probs(item_string, regex_pattern, es)
    print('\nclasses_ls\n',classes_ls,'\n')
    print('\nngram_probs_dict\n',ngram_probs_dict,'\n')
    if(len(classes_ls) == 0):
        return 'NA', -1.0

    docs_share_by_class_dict = m.docs_share_by_class(classes_ls, 'naive_bayes', 'type_1', 'category_path.keyword', 'doc_id', es)
    print('docs_share_by_class_dict\n',docs_share_by_class_dict,'\n')

    # Calculate original predictions
    predictions_dict = {}
    for class_ in classes_ls:
        print(class_)
        prob = 0

        for ngram in ngrams_ls:
            print('prob',prob)
            print('ngram_probs_dict',ngram,ngram_probs_dict[ngram][class_]["result"])
            if(ngram_probs_dict[ngram][class_]['result'] == 0):
                continue
            else:
                prob = prob + ngram_probs_dict[ngram][class_]['result']
            if(prob == 0.0):
                print('Exit on prob == 0.0')
                sys.exit()
        predictions_dict[class_] = prob + docs_share_by_class_dict[class_]
    
    print('\nRaw predictions:')
    for k,v in predictions_dict.items():
        print(k,':',v)
        

    # Normalize predictions by converting them into Liklihood
    probs = {}
    if(len(predictions_dict) == 1):
        #print('\nProba: 1')
        for k,v in predictions_dict.items():
            probs[k] = 1
    else:
        sum_probs = sum(predictions_dict.values())
        print('\nsum of probs:',sum_probs)
        
        #print('\nLiklihood predictions:')
        for k,v in predictions_dict.items():
            print(k,':',v)
            probs[k] = v/sum_probs
            
    predicted_class = max(probs, key=probs.get)  # pick key by max value
    predicted_prob = probs[predicted_class]
    #print('\nFinal prediction:')
    #print(predicted_class, predicted_prob)

    return predicted_class, predicted_prob
    #return pd.Series({'Prediction':predicted_class, 'Prediction_proba':predicted_prob})

def apply_predictions(es_client, col_name, input_data_df):
    ## Add 2 columns with predictions and prediction probabilities
    start = time.time()
    print('      Prediction thread ...')
    #predicted_path, predicted_prob = get_prediction_results(row[col_name], es_client)
    #predictions_df['Prediction'] = predicted_path
    #predictions_df['Prediction_proba'] = predicted_prob
    #input_data_df = input_data_df.merge(input_data_df[col_name].apply(lambda x: get_prediction_results(x, es_client)), left_index=True, right_index=True)
    input_data_df['Prediction'] = input_data_df[col_name].apply(lambda x: get_prediction_results(x, es_client))
    print("      Prediction thread took %g s" % (time.time() - start))


    return input_data_df

def get_predictions_parallel(es_client, input_data_df, col_name, num_partitions = 2, num_cores = 1):
    print('\nClassification is in progress ...')
    start = time.time()

    ## Splitting test data into chunks
    #print('   Splitting test data into ' + str(num_partitions) + ' chunks ...')
    input_data_df_split = np.array_split(input_data_df, num_partitions)

    ## Get predictions for each split multithreaded in multiple pools
    print('   Parallelizing predictions ...')
    apply_predictions_partial = partial(apply_predictions, es_client, col_name)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(apply_predictions_partial, input_data_df_split))
    pool.close()
    pool.join()

    ## Get elapsed time
    end = time.time()
    print("Classification took %g s" % (end - start))

    return df


# experimental start
def main_1():
    ## get trace logger and set level
    tracer = logging.getLogger('elasticsearch.trace')
    tracer.setLevel(logging.INFO)
    tracer.addHandler(logging.FileHandler('/tmp/es_trace.log'))

    ## instantiate local ES client, connects to localhost:9200 by default
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

    # Read data
    df = pd.read_csv('/Users/altay.amanbay/Desktop/Power tools csv.csv', encoding = "ISO-8859-1", nrows=100)
    print('Data size:',df.shape)

    # Predictions
    sample_item_string = 'Banquet Brown & Serve Lite Ori - 6.40 oz'
    #sample_item_string = 'ArmorAll AA255 Utility Wet/Dry Vacuum, 2.5 gallon, 2 HP'
    print(get_prediction_results(sample_item_string, es))

    #for idx, row in df.iterrows():
    #    print(get_prediction_results(row['description']))

# single threaded prediction, one description at a time.
def main_2():
    ## get trace logger and set level
    tracer = logging.getLogger('elasticsearch.trace')
    tracer.setLevel(logging.INFO)
    tracer.addHandler(logging.FileHandler('/tmp/es_trace.log'))

    ## instantiate local ES client, connects to localhost:9200 by default
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

    # manual check
    #item_str = 'CWK® New Replacement Laptop Notebook Battery for Samsung NP355E5C NP355E5C-A02US NP355V5C-S01US RV510-S01 RV511-A01 RV511-S03AA-PB9MC6S AA-PB9MC6B NP350E5C NP350E5C-S04AU NP350E5C-A03AU NP350E5C-A02US RF511-S03AU RF511-S04 RF709 RF710-S02US Samsung NP300E5C-A07UK NP365E5C-S04US NP365E5C-S05US NP365E5C-S02UB AA-PB9NC6B NP355E7C NP355V5C NP550P5C NP-RV508I NP-RV509-A02EE NP-RV509-A04 NT300E5Z NT300E7A NT300E7Z 300E4C 300E5C NP300E4C NP300E5C AA-PB9NC6B NP350E7C NP350V5C NP355E5C'
    #item_str = 'Seventh Generation Natural Dish Liquid, Free & Clear Unscented, 25-Ounce Bottles (Pack of 6), Packaging May Vary'
    #item_str = 'Banquet Brown & Serve Lite Ori - 6.40 oz'
    #predicted_path, predicted_prob = get_prediction_results_laplace_print(item_str, es)
    #predicted_path, predicted_prob = get_prediction_results_print(item_str, es)
    #print('\n',predicted_path, predicted_prob)
    #sys.exit()


    ## Get input arguments
    (options, args) = get_args()

    ## Get args from config file
    parser, test_data_arg_section, test_data_arg, model_file_name, model_file_aux_name, prediction_arg_section, prediction_arg, pickle_file_name = check_n_get_config_options(options, args)
    print(test_data_arg_section, test_data_arg, model_file_name, model_file_aux_name, prediction_arg_section, prediction_arg, pickle_file_name)

    # Set searched path
    path_searched = 'Home & Kitchen > Dishwashing > Dishwasher Detergent'
    print('\nPath searched:',path_searched)

    ## Fetch input test data to be classified
    #input_data_df = get_data_df(options, args)
    print('\nReading data ...')
    test_data_df = get_data_df_2(parser, test_data_arg_section, test_data_arg, pickle_file_name)
    test_data_df = test_data_df.loc[test_data_df.category_full_path_mod1 != path_searched,:]
    test_data_df.reset_index(drop=True, inplace=True)
    print('%-22s' % 'Size of dataframe after excluding searched path',':',test_data_df.shape)


    # temp1
    #test_data_df = test_data_df.loc[35000:7784099,:]
    #test_data_df.reset_index(drop=True, inplace=True)
    #print('Test data cut to:',test_data_df.shape)
    #print(test_data_df.head(3))
    #for idx, row in test_data_df.iterrows():
    #    print(row['description_mod1'])
    #    print(get_prediction_results(row['description_mod1'], es))
    #sys.exit()


    
    # Get test-data-portion ranges
    test_df_size = test_data_df.shape[0]
    test_data_ranges = get_ranges_for_df(test_df_size, 1000)


    for r_ in test_data_ranges:
        print('\n'+'='*100)
        print('Test data range:',r_)
        test_data_df_part = test_data_df.loc[int(r_[0]):int(r_[1]),:]
        predictions_df = test_data_df_part.copy(deep=True)

        start = time.time()

        for idx, row in test_data_df_part.iterrows():
            predicted_path, predicted_prob = get_prediction_results_laplace(row['description_mod1'], es)
            predictions_df['Prediction'] = predicted_path
            predictions_df['Prediction_proba'] = predicted_prob
            #print(row['description_mod1'])
            #print(predicted_path, predicted_prob,'\n')


        print('Prediction took %g s' % (time.time() - start))

        ## Drop all predictions with probability < 0.7
        #predictions_df = predictions_df.loc[predictions_df['Prediction_proba'] >= 0.7,:]
        predictions_df = predictions_df.loc[predictions_df['Prediction'] == path_searched,:]
        print('Size of selected predictions : ', predictions_df.shape)

        if(predictions_df.shape[0] == 0):
            continue
        #sys.exit()
        ## Insert data with prediction results into db
        #insert_df_into_db(500, output_data_df, options)
        insert_df_into_db_2(parser, prediction_arg_section, prediction_arg, 500, predictions_df)
        #insert_df_into_db_parallel(parser, prediction_arg_section, prediction_arg, 500, predictions_df, df_partitions = 2, num_cores = None)


# multithreaded prediction (in progress)
def main_3():
    ## get trace logger and set level
    tracer = logging.getLogger('elasticsearch.trace')
    tracer.setLevel(logging.INFO)
    tracer.addHandler(logging.FileHandler('/tmp/es_trace.log'))

    ## instantiate local ES client, connects to localhost:9200 by default
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])


    ## Get input arguments
    (options, args) = get_args()

    ## Get args from config file
    parser, test_data_arg_section, test_data_arg, model_file_name, model_file_aux_name, prediction_arg_section, prediction_arg, pickle_file_name = check_n_get_config_options(options, args)
    print(test_data_arg_section, test_data_arg, model_file_name, model_file_aux_name, prediction_arg_section, prediction_arg, pickle_file_name)

    # Set searched path
    path_searched = 'Home & Kitchen > Dishwashing > Dishwasher Detergent'
    print('\nPath searched:',path_searched)

    ## Fetch input test data to be classified
    #input_data_df = get_data_df(options, args)
    print('\nReading data ...')
    test_data_df = get_data_df_2(parser, test_data_arg_section, test_data_arg, pickle_file_name)
    test_data_df = test_data_df.loc[test_data_df.category_full_path_mod1 != path_searched,:]
    test_data_df.reset_index(drop=True, inplace=True)
    print('%-22s' % 'Size of dataframe after excluding searched path',':',test_data_df.shape)


    # temp1
    #test_data_df = test_data_df.loc[0:100,:]
    #print('Test data cut to:',test_data_df.shape)
    #print(test_data_df.head())

    #test_data_df = test_data_df.merge(test_data_df['description_mod1'].apply(lambda x: get_prediction_results(x, es)), left_index=True, right_index=True)
    #test_data_df['Prediction'] = test_data_df['description_mod1'].apply(lambda x: get_prediction_results(x, es))
    #predictions_df = get_predictions_parallel(es, test_data_df, 'description_mod1', num_partitions = 2, num_cores = 2)
    #print(test_data_df.head())

    #for idx, row in test_data_df.iterrows():
    #    print(row['description_mod1'])
    #    print(get_prediction_results(row['description_mod1'], es))
    #sys.exit()


    # temp2
    # Get test-data-portion ranges
    test_df_size = test_data_df.shape[0]
    test_data_ranges = get_ranges_for_df(test_df_size, 1000)


    for r_ in test_data_ranges:
        print('\n'+'='*100)
        print('Test data range:',r_)
        test_data_df_part = test_data_df.loc[int(r_[0]):int(r_[1]),:]
        predictions_df = test_data_df_part.copy(deep=True)

        start = time.time()

        for idx, row in test_data_df_part.iterrows():
            predicted_path, predicted_prob = get_prediction_results(row['description_mod1'], es)
            predictions_df['Prediction'] = predicted_path
            predictions_df['Prediction_proba'] = predicted_prob

        print('Prediction took %g s' % (time.time() - start))

        ## Drop all predictions with probability < 0.7
        #predictions_df = predictions_df.loc[predictions_df['Prediction_proba'] >= 0.7,:]
        predictions_df = predictions_df.loc[predictions_df['Prediction'] == path_searched,:]
        print('Size of selected predictions : ', predictions_df.shape)

        if(predictions_df.shape[0] == 0):
            continue
        #sys.exit()
        ## Insert data with prediction results into db
        #insert_df_into_db(500, output_data_df, options)
        insert_df_into_db_2(parser, prediction_arg_section, prediction_arg, 500, predictions_df)
        #insert_df_into_db_parallel(parser, prediction_arg_section, prediction_arg, 500, predictions_df, df_partitions = 2, num_cores = None)



## main function
if __name__ == "__main__":

    main_2()



