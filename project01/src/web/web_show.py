# -*- coding:utf-8 -*-
from flask import Flask,render_template,request
import sys
import os
sys.path.append(os.path.abspath('./src'))
from ltp_parser import ExtractViews
import gc



app = Flask(__name__)
extract = ExtractViews()
extract.prepare_data()
extract.sents = []
gc.collect()
extract.prepare_nlp_parser()
print('extract parser finished preparing')


@app.route('/extractViews',methods=['GET','POST'],strict_slashes=False)
def extract_views():
    print('get request')
    news= request.args.get('newsContent')
    if not news:
        views = []
    else:
        # views = [{'subj': '律师骆裕德', 'view': '交警部门在处理交通闻法得程序上存在问题。司机违停了，交警应将处罚单张贴在车上，并告知不服可以行驶申请复议和提起诉讼的权力。这既是交警的告知义务，也是司机的知情权力'}, {'subj': '小明', 'view': '交警部门罚了他16次，他只认了一次，交了一次罚款，拿到法院的判决书后，会前往交警队，要求撤销此前的处罚。'}]
        views = extract.extract_news(news)
    return render_template('index.html',views=views)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=9999)