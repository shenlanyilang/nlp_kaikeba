# -*- coding: UTF-8 –*-
from sqlalchemy import Column, String, Integer
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from stanfordcorenlp import StanfordCoreNLP
import sys
from common.constants import candidate_words
from common.utils import cut_sent, read_json, write_json
import jieba
from stanfordcorenlp import StanfordCoreNLP
import os


ModelBase = declarative_base()
candidates = set(candidate_words)

class News(ModelBase):
    __tablename__ = "news_chinese"
    id = Column(Integer,primary_key=True,nullable=False)
    source = Column(String,nullable=True)
    content = Column(String,nullable=True)
    feature = Column(String,nullable=True)
    title = Column(String,nullable=True)
    url = Column(String,nullable=True)


def query_db():
    engine = create_engine('mysql+pymysql://root:AI@2019@ai@rm-8vbwj6507z6465505ro.mysql.zhangbei.rds.aliyuncs.com:3306/stu_db?charset=utf8')
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    res = session.query(News).all()
    return res

def text2json(content:str):
    data = []
    paras = content.split('\n')
    for para in paras:
        if len(para.strip()) == 0:
            continue
        sents = cut_sent(para)
        data.append({'para':para, 'sents':sents})
    return data

def format_data(outpath='./data/news_content.json'):
    data = query_db()
    res = []
    for news in data:
        content = news.content
        if content is None:
            continue
        content_json = text2json(content)
        res.append(content_json)
    write_json(outpath,res)

def contain_candidates(words:list):
    contains = set()
    for word in words:
        if word in candidates:
            contains.add(word)
    return contains

def find_nsubj(dependency_list,words,with_modifier=True):
    root = dependency_list[0]
    relations = [[] for i in range(len(words))]
    subj_index = -1
    subj = ''
    for frag in dependency_list[1:]:
        status,parent,index = frag
        parent -= 1
        index -= 1
        relations[parent].append(index)
        if status in ('nsubj') and parent == root[2]-1:
            subj_index = index
            break
    if subj_index != -1:
        subj = words[subj_index]
    return subj

def process_views(views_infos):
    return views_infos

def extract_views(all_sents):
    views_in_sents = []
    with StanfordCoreNLP(r'E:\nlp_kaikeba\stanfordcorenlp',lang='zh') as nlp:
        for i,sents in enumerate(all_sents):
            views_tmp = []
            for sent in sents:
                if sent is None or len(sent.strip()) == 0:
                    continue
                words = nlp.word_tokenize(sent)
                contains = contain_candidates(words)
                if len(contains) == 0:
                    continue
                # ners = nlp.ner(sent)
                dependency_list = nlp.dependency_parse(sent)
                root = dependency_list[0]
                predicate = words[root[2]-1]
                if predicate not in contains:
                    continue
                views_in_sent = ''.join(words[root[2]:])
                nsubj = find_nsubj(dependency_list,words)
                views_tmp.append([nsubj,views_in_sent,i])
            if len(views_tmp) > 0:
                views_in_sents.append({'sents':sents, 'views':views_tmp})
    return views_in_sents


def extract(inpath='./data/news_content.json',outpath='./data/news_views.json'):
    data = read_json(inpath)
    all_sents = []
    for news in data:
        for frag in news:
            para = frag['para']
            sents = frag['sents']
            all_sents.append(sents)
    views_infos = extract_views(all_sents)
    views_infos = process_views(views_infos)
    write_json(outpath,views_infos)
    print(os.path.abspath(os.path.curdir))
    print('successfully extract views from all news...')




if __name__ == '__main__':
    contents = ['《中央日报》称，当前韩国海军陆战队拥有2个师和2个旅，还打算在2021年增设航空团，并从今年开始引进30余架运输直升机和20架攻击直升机。此外，韩军正在研发新型登陆装甲车，比现有AAV-7的速度更快、火力更猛。未来韩国海军陆战队还会配备无人机，“将在东北亚三国中占据优势”。\n 但韩国网友对“韩国海军陆战队世界第二”的说法不以为然。不少网友留言嘲讽称：“这似乎是韩国海军陆战队争取国防预算的软文”。',
                '昨日，雷先生说，交警部门罚了他16次，他只认了一次，交了一次罚款，拿到法院的判决书后，会前往交警队，要求撤销此前的处罚。']
    print(os.path.abspath(os.path.curdir))
    # format_data()
    # extract()
    print(list(jieba.cut(contents[1])))
    # with StanfordCoreNLP(r'E:\nlp_kaikeba\stanfordcorenlp',lang='zh') as nlp:
    #     sent = ' '.join(list(jieba.cut(contents[1])))
    #     words = nlp.word_tokenize(sent)
    #     dependcy = nlp.dependency_parse(sent)
    #     print(words)
    #     print(dependcy)

