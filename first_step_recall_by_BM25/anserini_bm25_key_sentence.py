from pyserini.search import pysearch
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import operator
import  tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string( "file_path",   "D:/backup/wsdm_cup/bert_tfrecord/",     "The file path of the input file.  " )
flags.DEFINE_string("query_file", "dev.csv", "The file to run BM25, which contains the column 'query_text' ")
flags.DEFINE_string( "out_file", "bm25-dev.csv", "The result file name." )
flags.DEFINE_string( "out_score_file", "bm25-dev.csv.score", "The result file name." )
flags.DEFINE_string( "default_cite", "55a38b7f2401aa93797cef61", "Default citation when BM25 return none, or less than topk " )
flags.DEFINE_integer("topk", 100, "Retrieve topK relevant documents." )
flags.DEFINE_float("k1", 0.9, "The parameter of K1 in BM25. default 0.9")
flags.DEFINE_float("b", 0.4, "The parameter of b in BM25. default 0.4")
flags.DEFINE_float("default_score", 0.1, "Default citation score when BM25 return none, or less than topk ")



#pkl_path = '/home/liu/workspace/datasets/ms_citation/citation_pkl/'
#json_path = '/home/liu/workspace/datasets/ms_citation/citation_json/'
#tsv_path = '/home/liu/workspace/datasets/ms_citation/citation_tsv/'
#desc_path = '/home/liu/workspace/datasets/ms_citation/description/formate_desc.csv'

#################### arguments in my ubuntu ########################
# index_path = '/home/liu/workspace/datasets/ms_citation/citation_index/citation_candidates_index/'
# results_path = '/home/liu/workspace/datasets/ms_citation/results/'
# desc_path = '/home/liu/workspace/datasets/ms_citation/description/formate_desc.csv'

#################### arguments in GPU ########################
#index_path = '/home/LAB/zhaoqh/ljf/BM25/citation_candidates_index/'
index_path = '/home/LAB/liujf/workspace/wsdm_cup/BM25/citation_candidates_index/'
# results_path = '/home/LAB/zhaoqh/ljf/result/'
# desc_path = '//home/LAB/zhaoqh/ljf/datasets/dev.csv'
#
#default_citation ='55a38b7f2401aa93797cef61'
# # Default (k1=0.9, b=0.4)
# para_k1 = 0.9
# para_b = 0.4


# def read_cadidates():
#     cand = pd.read_csv('../data/candidate_paper.csv')
#     return  cand

def read_des(desc_path):
    valid = pd.read_csv(desc_path)
    valid['query_text'] =valid['query_text'].fillna('')
    #'description_id', 'paper_id', 'key_text', 'query_text', 'description_text'
    print('load description... ')
    return valid


def query_batches_each_item(valid,n):     # default n is 3

    null_size = 0
    query_text = valid['query_text'].values
    ids = valid['description_id'].values
    submit = np.zeros((len(ids), n+1)).astype(np.str)
    submit_score = np.zeros((len(ids), n + 1)).astype(np.str)

    searcher = pysearch.SimpleSearcher(index_path)
    searcher.set_bm25_similarity(FLAGS.k1, FLAGS.b)
    count = len(ids)
    bar = tqdm(range(count))
    for i in bar:
        col = list()
        col_score = list()
        col.append(ids[i])
        col_score.append(ids[i])

        cur_query = query_text[i]
        hits = searcher.search(cur_query.encode('utf-8'), k= n)

        if len(hits) == 0:
            null_size += 1
            for idx in range(0, n):
                col.append(FLAGS.default_cite)
                col_score.append(FLAGS.default_score)
        else:
            min_cnt = min(len(hits), n)
            for idx in range(0, min_cnt):
                col.append(hits[idx].docid)
                col_score.append(hits[idx].score)
            while min_cnt < n:
                col.append(FLAGS.default_cite)
                col_score.append(FLAGS.default_score)
                min_cnt += 1

        submit[i] = col
        submit_score[i] = col_score
    print('nullsize:{}'.format(null_size))
    return submit, submit_score




if __name__ == '__main__':
    topk = FLAGS.topk
    desc_path = FLAGS.file_path + FLAGS.query_file
    valid = read_des(desc_path)
    submit, submit_score = query_batches_each_item(valid, topk)
    df = pd.DataFrame(submit)
    df.to_csv(FLAGS.file_path + FLAGS.out_file , index=False)
    df = pd.DataFrame(submit_score)
    df.to_csv(FLAGS.file_path + FLAGS.out_score_file,  index=False)


    #df.to_csv('{}bm25-key-sentence-k1-{}-b{}-top{}-res.csv'.format(results_path, para_k1, para_b, top_n), header=None, index=False)


'''

python3 anserini_bm25_key_sentence.py \ 
--file_path=/home/LAB/zhaoqh/ljf/data_original/  \
--query_file=dev.csv \
--out_file=bm25-dev-top100.csv \
--kopk=100

python3 anserini_bm25_key_sentence.py --file_path=/home/LAB/zhaoqh/ljf/data_original/  --query_file=dev.csv --out_file=bm25-dev-top100.csv --topk=100

python3 anserini_bm25_key_sentence.py --file_path=/home/LAB/zhaoqh/ljf/data_original/  --query_file=validation.csv --out_file=bm25-vali-key-sentence-top100.csv --topk=100

python3 anserini_bm25_key_sentence.py --file_path=/home/LAB/zhaoqh/ljf/data_original/  --query_file=train_release_remove_dev.csv  --out_file=bm25-train-remove-dev-key-sentence-top100.csv --topk=100

python3 anserini_bm25_key_sentence.py --file_path=/home/LAB/zhaoqh/ljf/data_original/ \  
--query_file=dev.csv --out_file=bm25-dev-top3-k1_0.7-b1_1.0.csv --topk=3 --k1=0.7 --b1=1.0

'''
