from opencc import OpenCC
import codecs
import os
import re
import string
import json
import jieba
from gensim.models.word2vec import PathLineSentences
from gensim.models.word2vec import Word2Vec


OP = OpenCC('t2s')
sent_cut_pattern = [
    re.compile(r'([。？！?])([^"\'”])'),
    re.compile(r'(\.{6})([^"\'”])'),
    re.compile(r'([。？！?]["\'”])([^\'"”])'),
]
zh_pattern = re.compile(r'^[\u4e00-\u9fa5]+$')
puncs = string.punctuation + '.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'
punc_pattern = re.compile(r'[{}]+'.format(puncs))


def cut_sent(para):
    '''
    句子切分
    :param para:
    :return:
    '''
    for pat in sent_cut_pattern:
        para = pat.sub(r'\1\n\2',para)
    return para.split('\n')


def init_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_json(path,content):
    with codecs.open(path,'w',encoding='utf8') as f:
        json.dump(content,f)


def read_json(path):
    with codecs.open(path,'r',encoding='utf8') as f:
        data = json.load(f)
    return data