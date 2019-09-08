from gensim.models.word2vec import Word2Vec
import codecs
from SIF import data_io, params, SIF_embedding
from common import utils
import jieba
from scipy import spatial
import gc

class SentEmbedding(object):
    def __init__(self):
        self.wordvec = './model/wordvec.txt'
        self.wordfreq = './model/wordfreq.txt'
        self.params_ = params.params(1)
        self.words = None
        self.We = None
        self.weight4ind = None
        self.pc = None
        self.sent_len_limit = 100

    def generate_wordvecs(self,path='./model/word2vec_update.model'):
        model = Word2Vec.load(path)
        vectors = []
        word_cnt = []
        for word in model.wv.index2word:
            vecs = [str(round(num,4)) for num in model.wv[word]]
            count = model.wv.vocab[word].count
            vectors.append(' '.join([word] + vecs))
            word_cnt.append(' '.join([word,str(count)]))
        with codecs.open(self.wordvec,'w',encoding='utf8') as f:
            f.write('\n'.join(vectors))

        with codecs.open(self.wordfreq,'w',encoding='utf8') as f:
            f.write('\n'.join(word_cnt))
        print('word vectors and word freq write successfully...')


    def prepare(self,all_sents):
        all_sents = [sent for sent_list in all_sents for sent in sent_list]
        self.generate_wordvecs()
        sentences = [' '.join(list(jieba.cut(sent))[:self.sent_len_limit]) for sent in all_sents]
        weightpara = 1e-3
        (words, We) = data_io.getWordmap(self.wordvec)
        # load word weights
        word2weight = data_io.getWordWeight(self.wordfreq, weightpara)  # word2weight['str'] is the weight for the word 'str'
        weight4ind = data_io.getWeight(words, word2weight)  # weight4ind[i] is the weight for the i-th word
        # load sentences
        x, w, sents_no_ind = data_io.sentences2idx2(sentences, words, weight4ind)
        emb = SIF_embedding.get_weighted_average2(We, x, w)
        del x,w
        gc.collect()
        pc = SIF_embedding.get_pc(emb,self.params_.rmpc)
        self.words = words
        self.We = We
        self.weight4ind = weight4ind
        self.pc = pc

    def sents_embedding(self, sentences):
        sentences = [' '.join(list(jieba.cut(st))[:self.sent_len_limit]) for st in sentences]
        x,w,sents_no_ind = data_io.sentences2idx2(sentences, self.words, self.weight4ind)
        embedding_original = SIF_embedding.get_weighted_average2(self.We,x,w)
        embedding = SIF_embedding.embedding_remove_pc(embedding_original,self.pc,self.params_.rmpc)
        return embedding,sents_no_ind

    def cos_similarity(self,sent1,sent2):
        dis = spatial.distance.cosine(sent1,sent2)
        sim = 1. - dis
        return sim

if __name__ == '__main__':
    pass
