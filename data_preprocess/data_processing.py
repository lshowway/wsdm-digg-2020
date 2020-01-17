import time
import  numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import re
import codecs
import pandas as pd
from tqdm import tqdm



def tokenize(sentence):
    sentence = re.sub(r'\s+', ' ', sentence)
    token_words = word_tokenize(sentence)  # 输入的是列表
    token_words = pos_tag(token_words)
    return token_words

def stem(token_words):
    wordnet_lematizer = WordNetLemmatizer()  # 单词转换原型
    words_lematizer = []
    spec_words = []
    for word, tag in token_words:
        if tag.startswith('NNP') or tag.startswith('NNPS'): #专有名词单数，专有名词复数
            word_lematizer = wordnet_lematizer.lemmatize(word)
            spec_words.append(word_lematizer)
        elif tag.startswith('NN'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='n')  # n代表名词
        elif tag.startswith('VB'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='v')  # v代表动词
        elif tag.startswith('JJ'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='a')  # a代表形容词
        elif tag.startswith('R'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='r')  # r代表代词
        else:
            word_lematizer = wordnet_lematizer.lemmatize(word)
        words_lematizer.append(word_lematizer)

    return words_lematizer



sr = stopwords.words('english')

def delete_stopwords(token_words):
    cleaned_words = [word for word in token_words if word not in sr]
    return cleaned_words

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


characters = [' ', ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '-', '...', '^', '{', '}', '\'', '\"', '``', '='
              'a', 'an', 'the', 'to', 'of','i', 'no', 'not', 'too', 'very', 'can', 's', 't', 'no-content', '\'\'', '“', '**', ']]', '[[', ',]', '[,']

def delete_characters(token_words):
    words_list = [word for word in token_words if word not in characters and not is_number(word)]
    return words_list

def delete_characters_without_number(token_words):
    words_list = [word for word in token_words if word not in characters]
    return words_list


def to_lower(token_words):
    words_lists = [x.lower() for x in token_words]
    return words_lists


def pre_process(text):
    token_words = tokenize(text)
    token_words = stem(token_words)
    token_words = to_lower(token_words)
    token_words = delete_stopwords(token_words)
    token_words = delete_characters(token_words)
    return token_words

def pre_process_with_number(text):
    token_words = tokenize(text)
    token_words = stem(token_words)
    token_words = to_lower(token_words)
    token_words = delete_stopwords(token_words)
    token_words = delete_characters_without_number(token_words)
    return token_words


def pre_process_with_number_sen(text):
    token_words = tokenize(text)
    token_words = stem(token_words)
    token_words = to_lower(token_words)
    token_words = delete_stopwords(token_words)
    token_words = delete_characters_without_number(token_words)
    return ' '.join(token_words)



# if __name__ == '__main__':
    # papers = pd.read_csv('{}candidate_paper2.csv'.format('../data/'))
    #
    # print(papers.shape)
    # papers = papers[papers['paper_id'].notnull()]
    # print(papers.shape)
    # papers['abstract'] = papers['abstract'].fillna(' ')
    # papers['journal'] = papers['journal'].fillna(' ')
    # papers['keywords'] = papers['keywords'].fillna(' ')
    # papers['title'] = papers['title'].fillna(' ')
    # papers['year'] = papers['year'].fillna(0)
    #
    # train = papers['title'].values + ' ' + papers['abstract'].values + ' ' + \
    #         papers['keywords'].apply(lambda x: x.replace(';', ' ')).values + papers['journal'].values
    #
    # train = list(map(lambda x: pre_process(x), tqdm(train)))

    # text = 'usefulness in the detection of low-bacteraemia carriers.'
    #
    # list_desc = re.split('[\[|\(]?[\[|,]+\*\*\#\#\*\*[\]|,]+[\]|\)]?', text)
    # for item in list_desc:
    #     print(pre_process(item))
