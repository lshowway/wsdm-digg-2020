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

validation_path = 'D:/backup/wsdm_cup/ms_citation_test/test-kt.csv'
train_bm25_path = 'D:/backup/wsdm_cup/ms_citation_test/bm25-test-kt-k1-0.68-b-1.0-top100.csv'

cand_file_name  = 'D:/backup/wsdm_cup/ms_citation_original/candidate_paper_for_wsdm2020.csv'
data_path = 'D:/backup/wsdm_cup/ms_citation_test/'


random_neg = 0
bm25_neg = 0
vali_random_neg = 29


# random_neg = 10
# bm25_neg = 9
# vali_random_neg = 19

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

def read_bm25_first(train_bm25_path):
    train25 = pd.read_csv(train_bm25_path, dtype=str)  # set read_csv as string.
    descid2paper_id= train25.set_index('0').to_dict()['1']
    return descid2paper_id


def read_train_bm25_4_neg (train_bm25_path):
    # bm25 will return top 50 relevant document.
    # every 10 bm25 returns, choose one from candidate sets
    train25 = pd.read_csv(train_bm25_path, dtype=str)      # set read_csv as string.
    ran_int = list()
    list_id2id = list()
    max_neg_cnt = 5
    topk = 50
    for i in range(0,max_neg_cnt):
        ri = random.randint(1 + i * (topk/ max_neg_cnt), (i+1)*(topk/ max_neg_cnt))
        ran_int.append(ri)

    for idx in ran_int:
        desc_id2paper_id = train25.set_index('0').to_dict()[str(idx)]
        list_id2id.append(desc_id2paper_id)
    return list_id2id

def read_train_bm25_9_neg (train_bm25_path):
    # bm25 will return top 50 relevant document.
    # every 10 bm25 returns, choose one from candidate sets
    train25 = pd.read_csv(train_bm25_path, dtype=str)      # set read_csv as string.
    ran_int = list()
    list_id2id = list()
    max_neg_cnt = 10
    topk = 50
    for i in range(0,max_neg_cnt):
        ri = random.randint(1 + i * (topk/ max_neg_cnt), (i+1)*(topk/ max_neg_cnt))
        ran_int.append(ri)

    for idx in ran_int:
        desc_id2paper_id = train25.set_index('0').to_dict()[str(idx)]
        list_id2id.append(desc_id2paper_id)
    return list_id2id

def get_positive_document(paper_id, candidate_map_id_doc):
    if paper_id.strip() != '':
        doc = candidate_map_id_doc[paper_id]
        return doc
    else:
        print('skip paper_id: {} not in candidate'.format(paper_id))
        return abs


def get_random_negative_document(paper_id, list_cand_map_id_doc):
    min_doc_len = 15
    abs = ' '
    if paper_id.strip() != '':
        ran_paper_id = paper_id
        while (ran_paper_id.strip() == paper_id.strip()):
            ran_paper_id, ran_paper_abs = random.choice(list_cand_map_id_doc)
        if len(ran_paper_abs.split(' ')) > min_doc_len:
            return ran_paper_abs
        else:
            return get_random_negative_document(paper_id, list_cand_map_id_doc)

    else:
        return abs

# get negative documents from both bm25 and random result.  bm25 4-5 negative documents, random 5 negative documents
def get_negative_document(paper_id, desc_id, can_id2doc, list_cand_map_id_doc,
                                    bm25list_id2id):
    res_neg = []
    for desc_id_to_paper_id in bm25list_id2id:
        if len(res_neg) == bm25_neg:
            break
        if desc_id_to_paper_id.__contains__(desc_id):
            cur_paperid = desc_id_to_paper_id[desc_id]
            if cur_paperid == paper_id:
                continue
            else:
                if can_id2doc.__contains__(cur_paperid):
                    tmp = get_positive_document(cur_paperid, can_id2doc)      # Note that. some bm25 results may noly contain the title
                    res_neg.append(tmp)

    for idx in range(0, random_neg):
        res_neg.append(get_random_negative_document(paper_id, list_cand_map_id_doc))
    return res_neg

def get_bm25_first(bm25first, desc_id, can_id2doc):
    if bm25first.__contains__(desc_id):
        cur_paperid = bm25first[desc_id]
        tmp = get_positive_document(cur_paperid, can_id2doc)
        return tmp
    else:
        return ''


def get_random_negative_document2(list_cand_map_id_doc):
    ran_paper_id, ran_paper_doc = random.choice(list_cand_map_id_doc)       # ran_paper_doc: contains title and abstract.
    return ran_paper_doc

def gen_triple_train(can_id2doc, bm25fist):
    validation = pd.read_csv(validation_path)
    validation['description_id'] = validation['description_id'].fillna('')
    validation['key_text'] = validation['key_text'].fillna('')
    print('load validation :{}'.format(validation_path))

    triple = 3
    para_size = random_neg + bm25_neg
    actually_size = 0
    not_match = 0
    all_size = len(validation['description_id'].values) * (vali_random_neg)
    vali_size = len(validation['description_id'].values)

    train_triple_tmp = np.empty((all_size , triple), object)
    list_cand_map_id_doc = list(can_id2doc.items())

    '''
    for idx, row in train.iterrows():
        if idx % 5000 == 0:
            print('current iteration number:{}'.format(actually_size))
        if row['description_id'].strip() == '':
            continue

        pos_abstract = get_positive_document(row['paper_id'], can_id2doc)
        if pos_abstract.strip() == '':  ## if positive abstract is null, we will skip this data.
            continue

        cur_desc_text = row['key_text']
        list_neg_doc = get_negative_document(row['paper_id'], row['description_id'], can_id2doc, list_cand_map_id_doc, bm25list_id2id)

        for neg_doc in list_neg_doc:
            if neg_doc.strip() == '':
                continue
            col = list()
            col.append(cur_desc_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' '))
            col.append(pos_abstract.replace('\n', ' ').replace('\r', ' ').replace('\t', ' '))
            col.append(neg_doc.replace('\n', ' ').replace('\r', ' ').replace('\t', ' '))

            train_triple_tmp[actually_size] = col
            actually_size += 1
    '''

    for idx, row in validation.iterrows():
        if (idx+1) % 5000 == 0:
            print('validation current iteration number:{}'.format(actually_size))
        if row['description_id'].strip() == '':
            continue
            # description_id, key_text, query_text, description_text
        if row['key_text'].strip() == '':
            # print('wrong validation content. with description id {}'.format(row['description_id']))
            continue

        if idx < vali_size/2:
            bm25_first_as_positive = get_bm25_first(bm25_first, row['description_id'], can_id2doc)
        else:
            bm25_first_as_positive = row['key_text']

        if(bm25_first_as_positive is None or bm25_first_as_positive.strip() == ''):
            continue

        for idx in range(0, vali_random_neg):
            col = list()
            col.append(row['key_text'].replace('\n', ' ').replace('\r', ' ').replace('\t', ' '))
            col.append(bm25_first_as_positive.replace('\n', ' ').replace('\r', ' ').replace('\t', ' '))
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

    #bm25list_id2id = read_train_bm25(train_bm25_path)
    #bm25list_id2id = read_train_bm25_4_neg(train_bm25_path)
    #bm25list_id2id = read_train_bm25_9_neg(train_bm25_path)

    # get candidate, with map id2abstract and id2title

    bm25_first = read_bm25_first(train_bm25_path)
    can_id2doc = read_candidate(cand_file_name )
    t1 = time.time()

    train_triple = gen_triple_train(can_id2doc, bm25_first)
    print('generate triple train spends:{}'.format(time.time() - t1))
    df = pd.DataFrame(train_triple)
    df.to_csv('{}triples.train-text-positive-half-bm25-first-p1-r{}-b{}-vali-neg-{}.tsv'.format(data_path, random_neg, bm25_neg, vali_random_neg), header=False, index=False, sep='\t')


