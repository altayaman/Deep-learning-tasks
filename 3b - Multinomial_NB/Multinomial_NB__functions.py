import re
import time
from copy import deepcopy
from elasticsearch import helpers

## ------------------------------------------------------------------------------------------
## Utility functions
## ------------------------------------------------------------------------------------------


# Removes whitespaces from sentence
def query_string_processor(str):
    str_ = str.replace(" ", "")
    return str_

# Generates all character-wise ngrams for word 
def NGramGenerator_charwise(word):
    all_ngrams = []
    for n in range(1, len(word)+1):
        n_ngram = [word[i:i+n] for i in range(len(word)-n+1)]
        all_ngrams.extend(n_ngram)
    
    return ' '.join(all_ngrams)

# Generates all character-wise ngrams for each word in a sentence
def NGramGenerator_wordCharwise(query_sentence):
    all_ngrams_str = ""

    for s in query_sentence.split():
        n_gram_str = NGramGenerator_charwise(s)
        all_ngrams_str = all_ngrams_str + ' ' + n_gram_str
    
    return all_ngrams_str

# Generates character-wise ngrams for word
# as compared to NGramGenerator_charwise it generates ngrams according to specified min_ngram and max_ngram
def NGramGenerator_charwise_interval(word, min_ngram, max_ngram):
    all_ngrams = []
    for n in range(min_ngram, max_ngram+1):
        n_ngram = [word[i:i+n] for i in range(len(word)-n+1)]
        all_ngrams.extend(n_ngram)
    
    return ' '.join(all_ngrams)

def NGramGenerator_wordwise(phrase):
    all_ngram_lists = []
    #s_split = phrase.split()
    s_split = "".join((char if char.isalnum() else " ") for char in phrase).split()
    
    
    for n in range(len(s_split)+1, 0, -1):
        n_gram = [s_split[i:i+n] for i in range(len(s_split)-n+1)]
        all_ngram_lists.extend(n_gram)
        
    all_ngrams = []
    for n_gram in all_ngram_lists:
        all_ngrams.extend([' '.join(n_gram)])
    
    return all_ngrams

def NGramGenerator_wordwise_interval(phrase, rgx_pattern, min_ngram, max_ngram):
    all_ngram_lists = []

    phrase_ = phrase
    #phrase_ = re.sub(rgx_pattern, " ", phrase_)
    s_split = phrase_.lower().split()
    #s_split = "".join((char if char.isalnum() else " ") for char in phrase_).split()
    
    # remove generic tokens
    #s_split = [token for token in s_split if token not in ['and', '&']]

    for n in range(max_ngram, min_ngram - 1, -1):
        n_gram = [s_split[i:i+n] for i in range(len(s_split)-n+1)]
        all_ngram_lists.extend(n_gram)
        
    all_ngrams = []
    for n_gram in all_ngram_lists:
        all_ngrams.extend([' '.join(n_gram)])
    
    return all_ngrams

def print_search_stats(results):
    print('=' * 80)
    print('Total %d found in %dms' % (results['hits']['total'], results['took']))
    print('-' * 80)


# ### Cheatsheet

# In[4]:

## ==========================================================================================
## CheatSheet
## ==========================================================================================

## ------------------------------------------------------------------------------------------
## print all data in index
# es.search(index="descriptions", doc_type="type_1")

## ------------------------------------------------------------------------------------------
## Delete document
# es.delete(index = "descriptions", doc_type = "type_1", id = item_id)

## ------------------------------------------------------------------------------------------
## Delete index
# es.indices.delete(index = 'descriptions')

## ------------------------------------------------------------------------------------------
def create_index_NaiveBayes(es):
    request_body = {
        "settings" : {
            "number_of_shards": 1,    # set to 1 because if more scores for the same docs may differ in matching results
            "number_of_replicas": 0  # set to 0 as I run ES locally
            ,"analysis":{ 
              "analyzer": {
                "my_analyzer": {
                  "type": "custom",
                  "tokenizer": "standard",
                  "filter": ["lowercase"]
                }
              }
            }
        }
        ,"mappings":{
            "type_1": {
                "properties": {
                    "doc_id": {
                        "type": "integer"
                        #,"fields": {
                        #    "hash": {
                        #      "type": "murmur3"
                        #    }
                        #}
                    },
                    
                    "description": { 
                      "type": "text",
                      "fielddata": True
                        ,"fields": {
                          "keyword": {
                            "type": "keyword",
                            "index": True      # index set as True indicates that keyword field will be indexed,
                          }                    # consequently will be queryable (can be set to False)
                        }
                    },
                    "category_path": {
                      "type": "text"
                      ,"fielddata": True
                        ,"fields": {
                          "keyword": {
                            "type": "keyword",
                            "index": True      # index set as True indicates that keyword field will be indexed,
                          }                    # consequently will be queryable  (can be set to False)
                        }
                    }
                    ,"description_raw": {         # specified type:"string" & "index":"not_analyzed"
                      "type": "string",           # to enable regex search and count by groups
                      "index": "not_analyzed"     # type "string" was used in old ES versions and will be deprecated
                    }
                    ,"category_path_raw": {       # specified type:"string" & "index":"not_analyzed"
                      "type": "string"            # to enable regex search and count by groups
                      ,"index": "not_analyzed"    # type "string" was used in old ES versions and will be deprecated
                    }
                    ,"demo_field": {
                      "type": "text",
                      "fielddata": True,
                      #"store": True,
                      "term_vector": 'yes',
                      #"index_analyzer": "my_analyzer"
                      "analyzer": "my_analyzer"
                    }
                }
            }
        }
    }
    print("Creating customized index for NaiveBayes: naive_bayes ...")
    response = es.indices.create(index = 'naive_bayes', body = request_body)
    print(response) 
    
# create_index_NaiveBayes()


def insert_doc_into_index_asis(item_description, category_path, index_p, type_p, es):
    
    # Generate incremental doc id
    response_4 = get_max_value('naive_bayes','type_1','doc_id', es)
    max_value = response_4['aggregations']['max_value']['value']
    print(type(max_value),max_value)
    
    doc_id = 1
    if(max_value is not None):
        doc_id = max_value + 1

    # Insert item document
    doc = {
        'doc_id': doc_id,
        'description': item_description,
        'category_path': category_path,
        #'description_raw': item_description,
        #'category_path_raw': category_path
        'demo_field': item_description
    }

    res = es.index(index = index_p, doc_type = type_p, body = doc)
    
def insert_doc_into_index_ngrams(item_description_doc, rgx_pattern, category_path, index_p, type_p, es, counter_):
    
    # Generate incremental doc id
    response_4 = get_max_value('naive_bayes','type_1','doc_id', es)
    max_value = response_4['aggregations']['max_value']['value']
    #print(type(max_value),max_value)
    
    #doc_id = 1
    #if(max_value is not None):
    #    doc_id = max_value + 1
    
    # Insert item document by ngrams
    ngrams_ls = NGramGenerator_wordwise_interval(item_description_doc, rgx_pattern, 1, 1)
    for ngram in ngrams_ls:
        doc = {
            'doc_id': counter_, # doc_id,
            'description': ngram,
            'category_path': category_path,
            #'description_raw': item_description,
            #'category_path_raw': category_path
            #'demo_field': item_description
        }

        res = es.index(index = index_p, doc_type = type_p, body = doc)
 
def insert_bulk_doc_into_index_ngrams(data_df, rgx_pattern, index_p, type_p, es, counter_):

    counter = counter_

    # Generate bulk of docs for insertion
    print('   Generating bulk ...')
    start = time.time()
    bulk_ = []
    for idx, row in data_df.iterrows():
        item_description = row['description']
        category_path = row['category_path']

        ngrams_ls = NGramGenerator_wordwise_interval(item_description, rgx_pattern, 1, 1)
        if(len(ngrams_ls) == 0):
            print('item parsed with len 0: ',counter,'-',item_description,)
            #continue

        for ngram in ngrams_ls:
            doc = {
                '_index': index_p,
                '_type': type_p,
                '_source':{
                    'doc_id': counter, # doc_id,
                    'description': ngram,
                    'category_path': category_path
                    }
            }
            bulk_.append(doc)

        counter = counter + 1
    print('   Bulk generation took %g s' % (time.time()-start))

    # Insert bulk
    start = time.time()
    #res = es.bulk_index(index = index_p, doc_type = type_p, body = doc)
    print('\n   Inserting bulk ...')
    res = helpers.bulk(es, bulk_)
    print('   Bulk insertion took %g s' % (time.time()-start))


    return counter

def get_max_value(index_p, type_p, aggs_field, es):
    response_dict = es.search(
    index = index_p,
    doc_type = type_p,
    body={
      "size": 0,
      "aggs" : {
            "max_value" : {
                "max" : {
                    "field" : aggs_field
                }
            }
        }
    }

    ,request_timeout=30
    )

    return response_dict

def docs_share_by_class(classes_ls, index_p, type_p, search_field, aggs_field, es):
    #docs_count_per_class_dict = docs_count_per_class(classes_ls, index_p, type_p,search_field, aggs_field, es)  # deprecated
    #docs_count_ = docs_count(index_p, type_p, aggs_field, es)                                                   # deprecated

    docs_count_per_class_dict = docs_count_per_class_2(classes_ls, index_p, type_p,search_field, aggs_field, es)
    docs_count_ = docs_count_2(index_p, type_p, aggs_field, es)
    
    #print('\nFrom function:')
    #print('docs_count_per_class_dict',docs_count_per_class_dict,'\n')
    #print('docs_count_',docs_count_,'\n')

    docs_share_per_class = {}
    for class_ in docs_count_per_class_dict.keys():
        docs_share_per_class[class_] = docs_count_per_class_dict[class_]/docs_count_
        
    return docs_share_per_class

# deprecated, "cardinality" not accurate as index get large. Try use "terms" in the future as it is accurate when "cardinality" is not.
def docs_count(index_p, type_p, aggs_field, es):
    
    response_dict = es.search(
    index = index_p,
    doc_type = type_p,
    body={
      "size": 0
      ,"aggs" : {
            "doc_count" : {
                "cardinality" : {
                    "field" : aggs_field
                }
            }
        }
    }

    ,request_timeout=30
    )

    doc_count = response_dict['aggregations']['doc_count']['value']

    return doc_count

def docs_count_2(index_p, type_p, aggs_field, es):
    min_value_response_dict = es.search(
        index = index_p,
        doc_type = type_p,
        body={
          "size": 0,
          "aggs" : {
                "min_value" : {
                    "min" : {
                        "field" : aggs_field
                    }
                }
            }
        }

        ,request_timeout=30
    )
    
    min_value = min_value_response_dict['aggregations']['min_value']['value']
    
    max_value_response_dict = es.search(
        index = index_p,
        doc_type = type_p,
        body={
          "size": 0,
          "aggs" : {
                "max_value" : {
                    "max" : {
                        "field" : aggs_field
                    }
                }
            }
        }

        ,request_timeout=30
    )
    
    max_value = max_value_response_dict['aggregations']['max_value']['value']
    
    return ((max_value - min_value) + 1)

# deprecated, "cardinality" not accurate as index get large. Try use "terms" in the future as it is accurate when "cardinality" is not.
def docs_count_per_class(classes_ls, index_p, type_p,search_field, aggs_field, es):
    
    doc_count_by_class = {}
    for class_ in classes_ls:
        response_dict = es.search(
        index = index_p,
        doc_type = type_p,
        body={
          "query": { 
          "match": { 
            search_field: class_
          }},
          "size": 0
          ,"aggs" : {
                "doc_count" : {
                    "cardinality" : {
                        "field" : aggs_field
                    }
                }
            }
        }

        ,request_timeout=30
        )
        
        doc_count = response_dict['aggregations']['doc_count']['value']
        doc_count_by_class[class_] = doc_count

    return doc_count_by_class

def docs_count_per_class_2(classes_ls, index_p, type_p, search_field, aggs_field, es):
    doc_count_by_class = {}
    for class_ in classes_ls:
        min_value_response_dict = es.search(
        index = index_p,
        doc_type = type_p,
        body={
          "query": { 
          "match": { 
            search_field: class_
          }}
          ,"size": 0
          ,"aggs" : {
            "min_value" : {
                "min" : {
                    "field" : aggs_field
                }
            }
          }
        }

        ,request_timeout=30
        )
        
        min_value = min_value_response_dict['aggregations']['min_value']['value']
        
        max_value_response_dict = es.search(
        index = index_p,
        doc_type = type_p,
        body={
          "query": { 
          "match": { 
            search_field: class_
          }}
          ,"size": 0
          ,"aggs" : {
            "max_value" : {
                "max" : {
                    "field" : aggs_field
                }
            }
          }
        }

        ,request_timeout=30
        )
        
        max_value = max_value_response_dict['aggregations']['max_value']['value']
        
        doc_count_by_class[class_] = (max_value - min_value)+1
        
    return doc_count_by_class

def get_ngram_probs_laplace(string_, regex_pattern, es):
    ngram_ls = NGramGenerator_wordwise_interval(string_,regex_pattern,1,1)
    #print('ngram_ls',ngram_ls)
    
    response_3 = Query_unique_nGrams_count('naive_bayes', 'type_1', 'description.keyword', es)
    unique_nGrams_count = response_3['aggregations']['unique_ngrams_count']['value']
    #print('unique_nGrams_count',unique_nGrams_count)

    
    # Create common set of classes from set of classes of each ngram 
    # and create result holder
    laplace_number = 1
    prob_boost = 1
    empty_probs_per_class_dict = {}
    for ngram in ngram_ls:
        #print(ngram)
        response_1 = Query_certain_nGram_count_per_class(es, ngram, 'naive_bayes', 'type_1', 'description.keyword', 'category_path.keyword', output_size=10000)
        for e in response_1['aggregations']['ngram_count']['buckets']:
            class_ = e['key']
            
            response_2 = Query_total_nGrams_count_per_class(es, class_, 'naive_bayes', 'type_1', 'category_path.keyword','category_path.keyword', output_size=10000)
            total_ngrams_count = response_2['aggregations']['class_count']['buckets'][0]['doc_count']
            #print(total_ngrams_count)
            
            formula_ = str(laplace_number)+'/('+str(total_ngrams_count)+'+'+str(unique_nGrams_count)+')='
            result_ = laplace_number/(total_ngrams_count+unique_nGrams_count)
            empty_probs_per_class_dict[class_] = {'formula':formula_, 'result':result_}
    #print(empty_probs_per_class_dict)
            
    # 
    final_results = {}
    for ngram in ngram_ls:
        #print(ngram)
        ngram_probs_per_class_dict = deepcopy(empty_probs_per_class_dict)
        
        response_1 = Query_certain_nGram_count_per_class(es, ngram, 'naive_bayes', 'type_1', 'description.keyword', 'category_path.keyword', output_size=10000)

        for e in response_1['aggregations']['ngram_count']['buckets']:
            class_ = e['key']
            count_ = e['doc_count']
            #print(class_,count_)
            
            response_2 = Query_total_nGrams_count_per_class(es, class_, 'naive_bayes', 'type_1', 'category_path.keyword','category_path.keyword', output_size=10)
            total_ngrams_count = response_2['aggregations']['class_count']['buckets'][0]['doc_count']
            #print(total_ngrams_count)
            
            formula_ = '('+str(count_)+'*'+str(prob_boost)+'+'+str(laplace_number)+')/('+str(total_ngrams_count)+'+'+str(unique_nGrams_count)+')='
            result_ = (laplace_number*prob_boost+count_)/(total_ngrams_count+unique_nGrams_count)
            ngram_probs_per_class_dict[class_]['formula'] = formula_
            ngram_probs_per_class_dict[class_]['result'] = result_
    
        final_results[ngram] = ngram_probs_per_class_dict
    
    classes_ls = list(empty_probs_per_class_dict.keys())
    return classes_ls, final_results

def get_ngram_probs(string_, regex_pattern, es):
    ngram_ls = NGramGenerator_wordwise_interval(string_,regex_pattern,1,1)
    #print('ngram_ls',ngram_ls)
    
    # Create common set of classes from set of classes of each ngram 
    # and create result holder
    empty_probs_per_class_dict = {}
    for ngram in ngram_ls:
        #print(ngram)
        response_1 = Query_certain_nGram_count_per_class(es, ngram, 'naive_bayes', 'type_1', 'description.keyword', 'category_path.keyword', output_size=10000)
        for e in response_1['aggregations']['ngram_count']['buckets']:
            class_ = e['key']
            
            #response_2 = Query_total_nGrams_count_per_class(es, class_, 'naive_bayes', 'type_1', 'category_path.keyword','category_path.keyword', output_size=10000)
            #total_ngrams_count = response_2['aggregations']['class_count']['buckets'][0]['doc_count']
            #print(total_ngrams_count)
            
            #formula_ = '0/('+str(total_ngrams_count)+'+'+str(unique_nGrams_count)+')='
            #result_ = 0/(total_ngrams_count+unique_nGrams_count)
            formula_ = '0='
            result_ = 0
            empty_probs_per_class_dict[class_] = {'formula':formula_, 'result':result_}
            #empty_probs_per_class_dict[class_] = 1

            
    # 
    final_results = {}
    for ngram in ngram_ls:
        #print(ngram)
        ngram_probs_per_class_dict = deepcopy(empty_probs_per_class_dict)
        
        response_1 = Query_certain_nGram_count_per_class(es, ngram, 'naive_bayes', 'type_1', 'description.keyword', 'category_path.keyword', output_size=10000)

        for e in response_1['aggregations']['ngram_count']['buckets']:
            class_ = e['key']
            count_ = e['doc_count']
            #print(class_,count_)
            
            response_2 = Query_total_nGrams_count_per_class(es, class_, 'naive_bayes', 'type_1', 'category_path.keyword','category_path.keyword', output_size=10)
            total_ngrams_count = response_2['aggregations']['class_count']['buckets'][0]['doc_count']
            #print(total_ngrams_count)
            
            formula_ = str(count_)+'/'+str(total_ngrams_count)+'='
            result_ = (count_)/(total_ngrams_count)
            #ngram_probs_per_class_dict[class_] = {'formula':formula_, 'result':result_}
            ngram_probs_per_class_dict[class_]['formula'] = formula_
            ngram_probs_per_class_dict[class_]['result'] = result_
    
        final_results[ngram] = ngram_probs_per_class_dict
    
    classes_ls = list(empty_probs_per_class_dict.keys())
    return classes_ls, final_results

def Query_total_nGrams_count_per_class(es, class_, index_p, type_p, search_field, aggs_field, output_size = 15000):
        
    #ls = []
    #for e in query_response['aggregations']['ngram_count']['buckets']:
    #    ls.append([{ "match" : { search_field : { "query" : e['key'] } } }])
    

    response_dict = es.search(
        index = index_p,
        doc_type = type_p,
        body={
          "query" : { 
            "bool" : { 
              "should" : { "match" : { search_field : { "query" : class_ } } }
            } 
          } 
          ,"size": 0,
          "aggs" : {
                "class_count" : {
                    "terms" : {
                        "field" : aggs_field
                        ,"size" : output_size
                    }
                }
            }
        }

        ,request_timeout=30
    )

    return response_dict

def Query_certain_nGram_count_per_class(es, query_ngram, index_p, type_p, search_field, aggs_field, output_size = 15000):
        
    #ngram_ls = NGramGenerator_wordwise_interval(query_ngram,1,1)
    
    #for ngram in ngram_ls:
    #print(ngram)
    response_dict = es.search(
        index = index_p,
        doc_type = type_p,
        body={
          "query": {
            "match": { search_field: query_ngram}
          }

          ,"aggs" : {
            "ngram_count" : {
                "terms" : {
                    "field" : aggs_field
                    ,"size" : output_size
                }
            }
          }
        ,"size": 0
        #,"_source": ["description", "category_path"]
        }

        ,request_timeout=30
    )

    return response_dict

def Query_unique_nGrams_count(index_p, type_p, aggs_field, es):
    response_dict = es.search(
    index = index_p,
    doc_type = type_p,
    body = {
      "size": 0,
      "aggs" : {
        "unique_ngrams_count" : {
            "cardinality" : {
                "field" : aggs_field
            }
          }
        }
    }

    ,request_timeout=30
    )

    return response_dict




