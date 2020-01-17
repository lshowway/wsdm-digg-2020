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
# type = 'dev'

#train_bm25_path = 'D:/backup/wsdm_cup/ms_citation_bm25_result/bm25-train-remove-dev-key-sentence-top100.csv'
train_file_path  = 'D:/backup/wsdm_cup/ms_citation_original_format_desc4/train_release_remove_dev.csv'
validation_path = 'D:/backup/wsdm_cup/ms_citation_original_format_desc4/validation.csv'
type ='evl'

cand_file_name  = 'D:/backup/wsdm_cup/ms_citation_original/candidate_paper_for_wsdm2020.csv'
data_path = 'D:/backup/wsdm_cup/ms_citation_original_format_desc4/'


####################  Parameter in GPU############
# train_file_path  = '/home/LAB/zhaoqh/ljf/data_original/train_release_remove_dev.csv'
# candidate_paper_path = '/home/LAB/zhaoqh/ljf/data_original/candidate_paper_for_wsdm2020.csv'
# data_path = '/home/LAB/zhaoqh/ljf/datasets/'


#topk = 10
random_neg = 1
vali_random_neg = 29
#bm25_neg = 1

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

# def read_train_bm25(train_bm25_path):
#     train25 = pd.read_csv(train_bm25_path, dtype=str)      # set read_csv as string.
#     for idx in range (random_neg+1, random_neg + bm25_neg+1):
#         train25[str(idx)] = train25[str(idx)].fillna('')
#     list_id2id =[]
#     for idx in range(random_neg, random_neg + bm25_neg):
#         desc_id2paper_id = train25.set_index('0').to_dict()[str(idx+1)]
#         list_id2id.append(desc_id2paper_id)
#     #train25_descid_paperid0 = train25.set_index('0').to_dict()['1']
#     #print(list_id2id[0]['77bef2'])
#     return list_id2id
#


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

def get_random_negative_document2(list_cand_map_id_doc):
    ran_paper_id, ran_paper_doc = random.choice(list_cand_map_id_doc)       # ran_paper_doc: contains title and abstract.
    return ran_paper_doc


# get negative documents from both bm25 and random result.  bm25 4-5 negative documents, random 5 negative documents
def get_negative_document(paper_id, desc_id, can_id2doc, list_cand_map_id_doc,
                                    bm25list_id2id):
    res_neg = []
    for desc_id_to_paper_id in bm25list_id2id:
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




def gen_triple_train(can_id2doc):

    # train file.  train_release_remove_dev  description_id, paper_id, key_text, query_text, description_text
    train = pd.read_csv(train_file_path)
    train['description_id'] = train['description_id'].fillna('')
    train['paper_id'] = train['paper_id'].fillna('')
    train['key_text'] =train['key_text'].fillna('')
    train['description_text'] = train['description_text'].fillna('')
    print('train shape:{}'.format(train.shape))
    print('load train relase... ')
    # description_id, key_text, query_text, description_text
    validation = pd.read_csv(validation_path)
    validation['description_id'] = validation ['description_id'].fillna('')
    validation['key_text']= validation ['key_text'].fillna('')
    print('load validation :{}'.format(validation_path))

    triple = 3
    para_size = vali_random_neg
    actually_size = 0
    not_match = 0
    train_triple_tmp = np.empty(((len(train['description_id'].values) + len(validation['description_id'].values)) * para_size , triple), object)
    list_cand_map_id_doc = list(can_id2doc.items())

    for idx, row in train.iterrows():
        if idx % 5000 == 0:
            print('current iteration number:{}'.format(actually_size))
        if row['description_id'].strip() =='':
            continue

        pos_abstract = get_positive_document(row['paper_id'], can_id2doc)
        if pos_abstract.strip() == '':  ## if positive abstract is null, we will skip this data.
            continue
        cur_desc_text = row['key_text']
        neg_doc = get_random_negative_document(row['paper_id'], list_cand_map_id_doc)

        col = list()
        col.append(cur_desc_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' '))
        col.append(pos_abstract.replace('\n', ' ').replace('\r', ' ').replace('\t', ' '))
        col.append(neg_doc.replace('\n', ' ').replace('\r', ' ').replace('\t', ' '))

        train_triple_tmp[actually_size] = col
        actually_size += 1

    for idx, row in validation.iterrows():
        if idx % 5000 == 0:
            print('validation current iteration number:{}'.format(actually_size))
        if row['description_id'].strip() == '':
            continue
            # description_id, key_text, query_text, description_text
        if row['key_text'].strip() == '':
            #print('wrong validation content. with description id {}'.format(row['description_id']))
            continue
        for idx in range(0, vali_random_neg):
            col = list()
            col.append(row['key_text'])
            col.append(row['key_text'])
            neg_doc = get_random_negative_document2(list_cand_map_id_doc)  # get random paper abstract from candidate.
            col.append(neg_doc)
            train_triple_tmp[actually_size] = col
            actually_size += 1


    train_triple = np.empty((actually_size, triple), object)
    for idx in range(0, actually_size):
        train_triple[idx] = train_triple_tmp[idx]
    #print('can not find [[**##**]] in description with counts:{}'.format(not_match))
    return train_triple

def read_triple(path):
    print('train triple shape is: {}'.format(pd.read_csv(path, sep='\t').shape))



if __name__ == '__main__':

    #bm25list_id2id = read_train_bm25(train_bm25_path)

    # get candidate, with map id2abstract and id2title
    can_id2doc = read_candidate( cand_file_name )
    t1 = time.time()

    train_triple = gen_triple_train(can_id2doc)
    print('generate triple train spends:{}'.format(time.time() - t1))
    df = pd.DataFrame(train_triple)
    df.to_csv('{}triples.train-random-validation-1-{}.tsv'.format(data_path, vali_random_neg), header=False, index=False, sep='\t')


