import json
import codecs
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy import Column,String,Integer
from src.common.utils import cut_sent,punc_pattern
import re
import jieba

ModelBase = declarative_base()

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


def proc_content(content):
    '''
    分句，去标点，分词
    :param content:
    :return:
    '''
    sents = cut_sent(content)
    sents_new = []
    for sent in sents:
        sent = punc_pattern.sub(r' ',sent)
        words = ' '.join([word for word in jieba.cut(sent) if not re.search(r'\s+',word)])
        if len(words.strip()) > 0:
            sents_new.append(words.strip())
    return sents_new


def run(outpath='../data/news/news_sents.txt'):
    data = query_db()
    contents = []
    for d in data:
        content = d.content
        if content == None:
            continue
        sents = proc_content(content.replace('\\n','\n'))
        if len(sents) > 0:
            contents.extend(sents)
    with codecs.open(outpath,'w',encoding='utf8') as f:
        f.write('\n'.join(contents))


if __name__ == '__main__':
    run()
