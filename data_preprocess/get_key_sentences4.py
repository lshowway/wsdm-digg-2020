import pandas as pd
import pickle
import re
from sklearn.externals import joblib
import os
from data_processing import pre_process_with_number
from tqdm import tqdm
from gensim import corpora,similarities,models
import nltk

split_sentence=nltk.data.load('tokenizers/punkt/english.pickle')

remove_dot = ['\n', '\r', '\t', 'al.', 'et.', 'vs.', 'e.g.', 'fig.', ', ,', ',,', '`']

def get_key_sentence(text):
    for item in remove_dot:
        text = text.replace(item, '')
    text = split_sentence.tokenize(text)

    key_sen_pos = list()  # all reference's position. i.e., key sentence positions.
    max_pos = -1  # minimum position of the ref .

    for i, sentence in enumerate(text):
        if sentence.find('[[**##**]]') != -1 or sentence.find('([**##**])') != -1:
            key_sen_pos.append(i)
    if len(key_sen_pos) == 0:
        # print(text)
        print('wrong description')
    else:
        max_pos = key_sen_pos[len(key_sen_pos) -1]
        if key_sen_pos.__contains__(0) == False:
            key_sen_pos.append(0)
        if max_pos - 1 > 0 and key_sen_pos.__contains__(max_pos - 1) == False:
            key_sen_pos.append(max_pos-1)
    key_sen_pos.sort()
    key_sens = ''

    for idx in key_sen_pos:
        item = re.sub(r'[\[|\(]?[\[|,]+\*\*\#\#\*\*[\]|,]+[\]|\)]?', '', text[idx])
        key_sens += item + ' '

    return key_sens


if __name__ == '__main__':
    original_path = 'D:/backup/wsdm_cup/ms_citation_original/'
    formate_path = 'D:/backup/wsdm_cup/ms_citation_original_format_desc_query/'
    #
    #str = 'he normal and [] asthmatic airways [[**##**]] contain similar  numbers of mast cells. A. Tom an jery. K. et al. 12.32 12L and K-6L mg/ge  increase with asthma severity. The normal and asthmatic airways contain similar numbers of mast cells in the submucosal connective tissues, however there are increased mast cells in the epithelial layer and smooth muscle, as well as the bronchoalveolar lavage (BAL) fluid of patients with asthma ([**##**]). '
    # str = 'this is first. this is second.  this is third . ' \
    #       'The normal and (BAL) fluid of patients with asthma[[**##**]]. '
    # print(get_key_sentence(str))

    # dev_name = 'dev.csv'
    # columns = ['description_id','paper_id','key_text', 'query_text', 'description_text']
    # dev = pd.read_csv(original_path + dev_name).astype(str)
    # dev['key_text'] = dev['description_text'].apply(lambda x: get_key_sentence(x))
    # dev['query_text'] = dev['key_text'].apply(lambda  x : pre_process_with_number(re.sub(' ', ' ', x)))
    # dev.to_csv(formate_path + dev_name, index=False, columns=columns)
    # print('dev done')
    #
    # validation = 'validation.csv'
    # columns = ['description_id', 'key_text', 'query_text', 'description_text']
    # dev = pd.read_csv(original_path + validation).astype(str)
    # print('read {} done. '.format(validation))
    # dev['key_text'] = dev['description_text'].apply(lambda x: get_key_sentence(x))
    # dev['query_text'] = dev['key_text'].apply(lambda x: pre_process_with_number(re.sub(' ', ' ', x)))
    # dev.to_csv(formate_path + validation, index=False, columns=columns)
    # print('format description done， with file: {}'.format(validation))
    #
    # train = 'train_release_remove_dev.csv'
    # columns = ['description_id','paper_id','key_text', 'query_text', 'description_text']
    # dev = pd.read_csv(original_path + train)
    # print('read {} done. '.format(train))
    # dev['key_text'] = dev['description_text'].astype('str').apply(lambda x: get_key_sentence(x))
    # dev['query_text'] = dev['key_text'].apply(lambda x: pre_process_with_number(re.sub(' ', ' ', x)))
    # dev.to_csv(formate_path + train, index=False, columns=columns)
    # print('format description done， with file:{} '.format(train))


    # dev_name = 'dev.csv'
    # columns = ['description_id','paper_id','key_text', 'query_text', 'description_text']
    # dev = pd.read_csv(original_path + dev_name).astype(str)
    # dev['key_text'] = dev['description_text'].apply(lambda x: get_key_sentence(x))
    # dev['query_text'] = dev['description_text'].apply(lambda  x : pre_process_with_number(re.sub(' ', ' ', x)))
    # dev.to_csv(formate_path + dev_name, index=False, columns=columns)
    # print('dev done')

    # validation = 'validation.csv'
    # columns = ['description_id', 'key_text', 'query_text', 'description_text']
    # dev = pd.read_csv(original_path + validation).astype(str)
    # print('read {} done. '.format(validation))
    # dev['key_text'] = dev['description_text'].apply(lambda x: get_key_sentence(x))
    # dev['query_text'] = dev['description_text'].apply(lambda x: pre_process_with_number(re.sub(' ', ' ', x)))
    # dev.to_csv(formate_path + validation, index=False, columns=columns)
    # print('format description done， with file: {}'.format(validation))

    validation = 'test.csv'
    columns = ['description_id', 'key_text', 'query_text', 'description_text']
    dev = pd.read_csv(original_path + validation).astype(str)
    print('read {} done. '.format(validation))
    dev['key_text'] = dev['description_text'].apply(lambda x: get_key_sentence(x))
    dev['query_text'] = dev['description_text'].apply(lambda x: pre_process_with_number(re.sub(' ', ' ', x)))
    dev.to_csv(formate_path + validation, index=False, columns=columns)
    print('format description done， with file: {}'.format(validation))

'''
    first sentence
    current description sentence 
    last sentence.
'''