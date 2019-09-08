from typing import List
from gensim.models.word2vec import Word2Vec
from collections import defaultdict

def func(distances:List[float])->float:
    pass

params = {
    'topn':10,
    'distance':5,
    'func':func
}


def bfs(model:Word2Vec,word='说',params=params,top_sim=50):
    '''
    广度优先搜索
    :param model:
    :param word:
    :param params:
    :return:
    '''
    topn = params['topn']
    distances = params['distance']
    words_dict = {word:[1.]}
    word_weights = {}
    i = 0
    while i < distances:
        new_word_dict = defaultdict(list)
        for word,weights in words_dict.items():
            similar_w = model.wv.most_similar(word,topn=topn)
            for weight in weights:
                for wd,sim in similar_w:
                    dis = weight * sim
                    word_weights[wd] = word_weights.get(wd,0) + dis
                    new_word_dict[wd].append(dis)
        words_dict = new_word_dict
        i += 1
    res = [(key,value) for key,value in word_weights.items()]
    res.sort(key=lambda x:x[1],reverse=True)
    return res[:top_sim]
