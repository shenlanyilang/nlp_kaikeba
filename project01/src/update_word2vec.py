from gensim.models.word2vec import PathLineSentences
from gensim.models.word2vec import Word2Vec



if __name__ == '__main__':
    word2vec_model_path = '../model/word2vec.model'
    news_sent_path = '../data/news/news_sents.txt'
    news_sents = PathLineSentences(news_sent_path)
    model = Word2Vec.load(word2vec_model_path)
    model.build_vocab(news_sents,update=True)
    model.train(news_sents,total_examples=model.corpus_count,epochs=5)
    model.save('../model/word2vec_update.model')

