import codecs
import os
import re

import jieba
from gensim.models.word2vec import PathLineSentences
from gensim.models.word2vec import Word2Vec
from opencc import OpenCC

from src.common.utils import cut_sent, init_folder

OP = OpenCC('t2s')
zh_pattern = re.compile(r'^[\u4e00-\u9fa5]+$')


def preprocess_wiki(inputpath,outputpath):
    '''
    输入文件经过句子切分、分词后重新写入输出文件
    :param inputpath:
    :param outputpath:
    :return:
    '''
    output = codecs.open(outputpath,'w',encoding='utf8')
    with codecs.open(inputpath,'r',encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if re.search(r'</doc>|<doc [^<>]*?>',line):
                continue
            sentences = cut_sent(line)
            for sent in sentences:
                words = [word for word in jieba.cut(sent) if zh_pattern.search(word)]
                if len(words) == 0:
                    continue
                output.write(" ".join(words) + "\n")
    print('finished processing {}'.format(inputpath))

def fan2jian(inputdir,outputdir):
    '''
    繁体字转换简体字
    :param inputdir:
    :param outputdir:
    :return:
    '''
    init_folder(outputdir)
    for path in os.listdir(inputdir):
        if not path.startswith('wiki'):
            continue
        with codecs.open(os.path.join(inputdir,path),'r',encoding='utf8') as f:
            raw_content = f.read()

        transform_content = OP.convert(raw_content)
        with codecs.open(os.path.join(outputdir,path),'w',encoding='utf8') as f:
            f.write(transform_content)
        print('finish processing {}'.format(path))
    print('processed finished...')


def process_sent(inputdir,outputdir):
    init_folder(outputdir)
    for fname in os.listdir(inputdir):
        inpath = os.path.join(inputdir,fname)
        outpath = os.path.join(outputdir,fname+'_sent')
        preprocess_wiki(inpath,outpath)
    print("finished processing sentence")


def word_vec_model(sent_dir,model_dir):
    sentences = PathLineSentences(sent_dir)
    model = Word2Vec(sentences=sentences,size=100,min_count=5,sg=1,iter=5)
    model.train(total_examples=model.corpus_count,epochs=5)
    model.save(model_dir)
    return model


if __name__ == '__main__':
    inputdir = './wiki_simplified_chn'
    outputdir = './wiki_sent'
    model_path = 'word2vec.model'
    process_sent(inputdir,outputdir)
    word_vec_model(outputdir, model_path)



