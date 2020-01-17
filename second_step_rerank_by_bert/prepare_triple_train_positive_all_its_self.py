######################################
#1. triples.train.small.tsv   description_content \t relevant_paper_content \t unrelevant_paper_content
#2. top3.evl.tsv        description's top3 relevant papers.             description_id  \t paper_id \t description_content \t paper_conent
#3. top3.dev.tsv        use BM25 to retrieve results from train file.   description_id  \t paper_id \t description_content \t paper_conent
#4. qrel.dev.tsv        split from TrianFile,  description_content  \t 0    relevant_paper_id \t 1
###################

import  pandas as pd
#from data_processing import pre_process
import numpy as np
import  re
import random
import time


####################  Parameter in local host ############train_release_remove_dev
# train_file_path  = 'D:/backup/wsdm_cup/ms_citation_original_format_desc/dev.csv'
# train_bm25_path = 'D:/backup/wsdm_cup/ms_citation_bm25_result/bm25-dev-top100.csv'
#type = 'dev'

# train_bm25_path = 'D:/backup/wsdm_cup/ms_citation_bm25_result/bm25-train-remove-dev-key-sentence-top100.csv'
# train_file_path  = 'D:/backup/wsdm_cup/ms_citation_original_format_desc/train_release_remove_dev.csv'
# type ='evl'

#train_bm25_path = 'D:/backup/wsdm_cup/ms_citation_original_format_desc4/bm25-train4-k1-0.68-b-1.0-top50.csv'
#train_file_path  = 'D:/backup/wsdm_cup/ms_citation_original_format_desc4/train_release_remove_dev.csv'
validation_path = 'D:/backup/wsdm_cup/ms_citation_test/test-kt.csv'
# type ='dev'

cand_file_name  = 'D:/backup/wsdm_cup/ms_citation_original/candidate_paper_for_wsdm2020.csv'
data_path = 'D:/backup/wsdm_cup/ms_citation_test/'


random_neg = 0
bm25_neg = 0
vali_random_neg = 29

def remove_none(abstract):
    if abstract.strip() == 'NO_CONTENT' or abstract.strip() == '' or abstract is None:
        return ''
    else:
        return abstract

def read_candidate(candidate_paper_path):
    # abstract,journal,keywords,paper_id,title,year
    cand = pd.read_csv(candidate_paper_path)
    cand['journal'] = cand['journal'].fillna('')
    cand['keywords'] = cand['keywords'].fillna('')
    cand['paper_id'] = cand['paper_id'].fillna('')
    cand['abstract'] = cand['abstract'].fillna('')
    cand['abstract'] = cand['abstract'].apply(lambda x: remove_none(x))

    cand['title'] = cand['title'].fillna('')
    print('candidate shape:{}'.format(cand.shape))
    #can_id2abs= cand.set_index('paper_id').to_dict()['abstract']
    #can_id2tilte= cand.set_index('paper_id').to_dict()['title']
    cand['title'] = cand['title'] + ' '+ cand['abstract']
    can_id2doc = cand.set_index('paper_id').to_dict()['title']

    return can_id2doc

def get_positive_document(paper_id, candidate_map_id_doc):
    if paper_id.strip() != '':
        doc = candidate_map_id_doc[paper_id]
        return doc
    else:
        print('skip paper_id: {} not in candidate'.format(paper_id))
        return abs



def get_random_negative_document2(list_cand_map_id_doc):
    ran_paper_id, ran_paper_doc = random.choice(list_cand_map_id_doc)       # ran_paper_doc: contains title and abstract.
    return ran_paper_doc

def gen_triple_train(can_id2doc, bm25list_id2id):

    validation = pd.read_csv(validation_path)
    validation['description_id'] = validation['description_id'].fillna('')
    validation['key_text'] = validation['key_text'].fillna('')
    print('load validation :{}'.format(validation_path))

    triple = 3

    actually_size = 0
    all_size = len(validation['description_id'].values) * (vali_random_neg)

    train_triple_tmp = np.empty((all_size , triple), object)
    list_cand_map_id_doc = list(can_id2doc.items())

    for idx, row in validation.iterrows():
        if idx % 5000 == 0:
            print('validation current iteration number:{}'.format(actually_size))
        if row['description_id'].strip() == '':
            print('description id null.{}'.format(row['description_id']))
            continue
        if row['key_text'].strip() == '':
            print('key_text null.{}'.format(row['description_id']))
            continue
        for idx in range(0, vali_random_neg):
            col = list()
            col.append(row['key_text'].replace('\n', ' ').replace('\r', ' ').replace('\t', ' '))
            col.append(row['key_text'].replace('\n', ' ').replace('\r', ' ').replace('\t', ' '))
            neg_doc = get_random_negative_document2(list_cand_map_id_doc)  # get random paper abstract from candidate.
            col.append(neg_doc.replace('\n', ' ').replace('\r', ' ').replace('\t', ' '))
            train_triple_tmp[actually_size] = col
            actually_size += 1

    train_triple = np.empty((actually_size, triple), object)
    for idx in range(0, actually_size):
        train_triple[idx] = train_triple_tmp[idx]

    print('actually size is:{}'.format(actually_size))
    return train_triple

def read_triple(path):
    print('train triple shape is: {}'.format(pd.read_csv(path, sep='\t').shape))


if __name__ == '__main__':
    # get candidate, with map id2abstract and id2title
    can_id2doc = read_candidate(cand_file_name )
    t1 = time.time()
    train_triple = gen_triple_train(can_id2doc, None)
    print('generate triple train spends:{}'.format(time.time() - t1))
    df = pd.DataFrame(train_triple)
    df.to_csv('{}triples.train-key-sentence-4-p1-r{}-b{}-vali-neg-{}.tsv'.format(data_path, random_neg, bm25_neg, vali_random_neg), header=False, index=False, sep='\t')


