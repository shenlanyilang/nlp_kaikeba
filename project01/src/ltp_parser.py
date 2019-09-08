# -*- coding: UTF-8 –*-
from pyltp import Postagger,Parser,Segmentor
from sqlalchemy import Column, String, Integer
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from common.constants import candidate_words
from common.utils import cut_sent, read_json, write_json, puncs
import random
import os
from SentSim import SentEmbedding
import numpy as np


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

class ExtractViews(object):
    def __init__(self):
        self.sents = self.load()
        self.sent_embd = SentEmbedding()
        self.segmentor = Segmentor()
        self.postagger = Postagger()
        self.parser = Parser()

    def load(self,inpath='./data/news_content.json'):
        data = read_json(inpath)
        all_sents = []
        for news in data:
            for frag in news:
                sents = frag['sents']
                all_sents.append(sents)
        print('finished loading all sentences')
        return all_sents

    def prepare_data(self, sampled_sents=1000):
        print('start preparing items for sentences embedding')
        sentences_sampled = random.sample(self.sents,sampled_sents)
        self.sent_embd.prepare(sentences_sampled)
        print('sentence embedding data prepare finished')

    def prepare_nlp_parser(self):
        self.segmentor.load(r'./ltpmodels/cws.model')
        self.postagger.load(r'./ltpmodels/pos.model')
        self.parser.load(r'./ltpmodels/parser.model')

    def extract_news(self,content):
        content = content.strip()
        paras = content.split('\n')
        sentences = []
        for para in paras:
            sentences.append(cut_sent(para))
        views = self._extract_views(sentences)
        return views

    def _extract_views(self,all_sents):
        nums = len(all_sents)
        views_in_sents = []
        print('totally {} paragraphs needing processed'.format(nums))
        for i, sents in enumerate(all_sents):
            views_tmp = []
            if i % 100 == 0:
                print('processing paras : {}/{}'.format(i,nums))
            for j,sent in enumerate(sents):
                sent = sent.replace('\\n', '\n').strip()
                # sentence长度达到1000左右时，ltp会报错
                if len(sent) == 0 or len(sent) > 500:
                    continue
                # words = list(jieba.cut(sent))
                words = list(self.segmentor.segment(sent))
                contains = contain_candidates(words)
                if len(contains) == 0:
                    continue
                tags = list(self.postagger.postag(words))
                arcs = list(self.parser.parse(words, tags))
                sbv, head = get_sbv_head(arcs, words, tags)
                if sbv[0] is None or head[0] is None or head[0] not in contains:
                    continue
                subj = sbv[0]
                view = clean_view(words[head[1] + 1:])
                views_tmp.append((subj, view, j))
            views_final = self._get_final_views(sents, views_tmp)
            if len(views_final) > 0:
                views_in_sents.extend(views_final)
        return views_in_sents

    def extract(self):
        all_sents = self.sents
        views_in_sents = self._extract_views(all_sents)
        return views_in_sents

    def _get_final_views(self, sents, views_tmp):
        def _entire_emb(emb:np.array, sents_no_ind):
            dim = emb.shape[1]
            for ind in sents_no_ind:
                # 不存在embedding的句子补0
                emb = np.insert(emb,ind,np.zeros((1,dim)),axis=0)
            return emb
        embeddings,sents_no_ind = self.sent_embd.sents_embedding(sents)
        # 获得所有句子的embeding
        embeddings = _entire_emb(embeddings, sents_no_ind)
        views_final = []
        for i,view in enumerate(views_tmp):
            start = view[2]
            stop = len(views_tmp)
            if i < len(views_tmp) - 1:
                stop = views_tmp[i+1][2]
            end = self._get_view_end(embeddings,start,stop)
            views_final.append({'subj':view[0],'view':view[1]+''.join(sents[start+1:end])})
        return views_final

    def _get_view_end(self,embeddings,start,stop,sim_threshold=0.8):
        # 判断view是不是在尾句或view不存在embedding
        if start + 1 >= stop or np.sum(np.abs(embeddings[start])) == 0:
            return start
        end = start + 1
        for i in range(start+1,stop):
            sent_emb = embeddings[i]
            curr_emb = np.mean(embeddings[start:i])
            sim = self.sent_embd.cos_similarity(curr_emb,sent_emb)
            if sim < sim_threshold:
                break
            end += 1
        return end

    def release_nlp_parser(self):
        self.segmentor.release()
        self.postagger.release()
        self.parser.release()

    def run(self):
        self.prepare_data()
        self.prepare_nlp_parser()
        views = self.extract()
        self.release_nlp_parser()
        write_json('./data/news_views_final.json', views)
        print('finished extract views')

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
        content_json = text2json(content.replace('\\n', '\n'))
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

def sbv_combine(arcs,sbv,words):
    '''
    位于主语前且连续的修饰词与主语一起返回
    :param arcs:
    :param sbv:
    :return:
    '''
    if sbv[0] is None:
        return (None,None)
    word,index = sbv
    modifiers = []
    for i in range(index)[::-1]:
        arc = arcs[i]
        par_idx = arc.head
        relation = arc.relation
        if par_idx - 1 != index:
            break
        modifiers.append(words[i])
    return (''.join(modifiers[::-1] + [sbv[0]]),sbv[1])

def get_sbv_head(arcs,words,tags):
    tags_cand = set(['j','n','nh','ni','nl','ns','nz','r'])
    head = (None,None)
    sbv_cand = []
    assert len(arcs) == len(words)
    for i in range(len(words)):
        indx,relation = arcs[i].head,arcs[i].relation
        if relation == 'HED':
            head = (words[i],i)
        elif relation == 'SBV' and arcs[indx - 1].relation == 'HED' and tags[i] in tags_cand:
            sbv_cand.append((words[i], i))
    sbv = sbv_cand[-1] if len(sbv_cand) > 0 else (None,None)
    sbv = sbv_combine(arcs,sbv,words)
    return (sbv,head)


def clean_view(words):
    start = 0
    end = 0
    puncs_set = set(puncs)
    for word in words:
        if word not in puncs_set:
            break
        start += 1
    for word in words[::-1]:
        if word not in puncs_set:
            break
        end -= 1
    return ''.join(words[start:end]) if end != 0 else ''.join(words[start:])

def extract_views(all_sents):
    segmentor = Segmentor()
    segmentor.load(r'/home/student/project-01/ltp_data/cws.model')
    postagger = Postagger()
    postagger.load(r'/home/student/project-01/ltp_data/pos.model')
    parser = Parser()
    parser.load(r'/home/student/project-01/ltp_data/parser.model')
    views_in_sents = []
    for i,sents in enumerate(all_sents):
        views_tmp = []
        for sent in sents:
            sent = sent.replace('\\n','\n').strip()
            if len(sent) == 0:
                continue
            # words = list(jieba.cut(sent))
            words = list(segmentor.segment(sent))
            contains = contain_candidates(words)
            if len(contains) == 0:
                continue
            tags = list(postagger.postag(words))
            arcs = list(parser.parse(words,tags))
            sbv,head = get_sbv_head(arcs,words,tags)
            if sbv[0] is None or head[0] is None or head[0] not in contains:
                continue
            subj = sbv[0]
            view = clean_view(words[head[1] + 1:])
            views_tmp.append((subj,view,i))
        if len(views_tmp) > 0:
                views_in_sents.append({'sents': sents, 'views': views_tmp})
    segmentor.release()
    postagger.release()
    parser.release()
    return views_in_sents

def extract(inpath='./data/news_content.json',outpath='./data/news_views.json'):
    data = read_json(inpath)
    all_sents = []
    for news in data:
        for frag in news:
            para = frag['para']
            sents = frag['sents']
            all_sents.append(sents)
    views_infos = extract_views(all_sents[:20])
    views_infos = process_views(views_infos)
    write_json(outpath,views_infos)
    print(os.path.abspath(os.path.curdir))
    print('successfully extract views from all news...')




if __name__ == '__main__':
    contents = ['《中央日报》称，当前韩国海军陆战队拥有2个师和2个旅，还打算在2021年增设航空团，并从今年开始引进30余架运输直升机和20架攻击直升机。此外，韩军正在研发新型登陆装甲车，比现有AAV-7的速度更快、火力更猛。未来韩国海军陆战队还会配备无人机，“将在东北亚三国中占据优势”。\n 但韩国网友对“韩国海军陆战队世界第二”的说法不以为然。不少网友留言嘲讽称：“这似乎是韩国海军陆战队争取国防预算的软文”。',
                '昨日，雷先生说，交警部门罚了他16次，他只认了一次，交了一次罚款，拿到法院的判决书后，会前往交警队，要求撤销此前的处罚。']
    format_data()
    # extract()
    # extract_ins = ExtractViews()
    # views = extract_ins.run()
