######################################
#1. triples.train.small.tsv   description_content \t relevant_paper_content \t unrelevant_paper_content
#2. top3.evl.tsv        description's top3 relevant papers.             description_id  \t paper_id \t description_content \t paper_conent
#3. top3.dev.tsv        use BM25 to retrieve results from train file.   description_id  \t paper_id \t description_content \t paper_conent
#4. qrel.dev.tsv        split from TrianFile,  description_content  \t 0    relevant_paper_id \t 1
###################

import math
import  pandas as pd
#from data_processing import pre_process
import numpy as np
import  re

topk_path = 'D:/backup/wsdm_cup/ms_citation_test/test-wxy2/test_top_50.csv'
validataion_desc_path = 'D:/backup/wsdm_cup/ms_citation_test/test-kt.csv'
candidate_path = 'D:/backup/wsdm_cup/ms_citation_original/candidate_paper_for_wsdm2020.csv'
data_path = 'D:/backup/wsdm_cup/ms_citation_test/test-wxy2/'
type = 'test'
topK = 50


def read_validation_desc(validataion_desc_path):
    validdation_desc = pd.read_csv(validataion_desc_path)
    validdation_desc['description_id'] =validdation_desc['description_id'].fillna('')
    validdation_desc['key_text'] = validdation_desc['key_text'].fillna('')
    validation_map_id_text = validdation_desc.set_index('description_id').to_dict()['key_text']
    gt80, gt160, size = 0, 0, 0

    for idx, row in validdation_desc.iterrows():
        rowarr = row['key_text'].split()
        if len(rowarr) > 80:
            gt80 += 1
        if len(rowarr) > 160:
            gt160 += 1
        size += 1
    print('key_text len more that 80 with {}, more that 160 with{}'.format(gt80*1.0/size*1.0, gt160*1.0/size*1.0 ))
    return validation_map_id_text

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
    cand['title'] = cand['title'] + ' '+ cand['abstract']
    can_id2doc = cand.set_index('paper_id').to_dict()['title']

    return can_id2doc


def read_topK(topk_path, topK):
    top100 = pd.read_csv(topk_path, dtype=str)      # set read_csv as string.
    for idx in range (1, topK + 1):
        top100[str(idx)] = top100[str(idx)].fillna('')
    return top100

def get_desc_by_id(description_id, validation_map_id_text):
    if validation_map_id_text.__contains__(description_id) == False:
        return ' '
    else:
        return  validation_map_id_text[description_id]


def get_content_by_id(paper_id,  can_id2doc):
    abs = ''
    if paper_id.strip() != '':
        doc = can_id2doc[paper_id]
        return doc
    else:
        return abs


def formate_evl_tsv(validation_map_id_text, top100 , can_id2doc):
    # description_id  \t paper_id \t description_content \t paper_conent
    eval_formate_tmp = np.empty((len(top100['0'].values) * topK, 4), object)
    all_eval_cnt  = 0
    for idx, row in top100.iterrows():
        if idx % 3000 == 0:
            print('current formate eval size:{}'.format(all_eval_cnt))
        description_id = row['0']
        desc_content = get_desc_by_id(description_id, validation_map_id_text)           # get description by description_id
        for idx in range(1, topK +1 ):
            col = list()
            paper_id = row[str(idx)]
            paper_content = get_content_by_id(paper_id, can_id2doc)       # get paper abstract
            col.append(description_id)
            col.append(paper_id)
            col.append(desc_content.replace('\n', ' ').replace('\r', ' ').replace('\t', ' '))
            col.append(paper_content.replace('\n', ' ').replace('\r', ' ').replace('\t', ' '))
            eval_formate_tmp[all_eval_cnt] = col
            all_eval_cnt += 1
    eval_formate = np.empty((all_eval_cnt, 4), object)
    for idx in range(0, all_eval_cnt):
        eval_formate[idx] = eval_formate_tmp[idx]

    return eval_formate


if __name__ == '__main__':

    validation_map_id_text =  read_validation_desc(validataion_desc_path)
    print('load validation... ')
    bm25_topk = read_topK(topk_path, topK)
    print('load top topK of desc_id and paper_id pairs...')
    can_id2doc = read_candidate(candidate_path)
    print('load read candidates...')

    evl_bath = 5000
    if type== 'evl' or type == 'test':
        for i in range(0, math.ceil(len(bm25_topk)/evl_bath)):
            baths = bm25_topk[i * evl_bath: (i+1)*evl_bath ]
            formate_eval = formate_evl_tsv(validation_map_id_text, baths, can_id2doc)
            df = pd.DataFrame(formate_eval)
            df.to_csv('{}top{}.{}{}.tsv'.format(data_path, topK, type, (i +1)), header=False, index=False, sep='\t')

    else:
        formate_eval = formate_evl_tsv(validation_map_id_text, bm25_topk , can_id2doc)
        df = pd.DataFrame(formate_eval )
        df.to_csv('{}top{}.{}.tsv'.format(data_path, topK, type), header=False, index=False, sep='\t')







