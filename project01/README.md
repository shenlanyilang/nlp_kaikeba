# 新闻人物观点提取

## 0a. 下载项目,安装依赖库
注意：python版本最好使用python3.6
~~~{.bash}
git clone git@github.com:shenlanyilang/nlp_kaikeba.git
cd nlp_kaikeba/project01
pip install -r requirements.txt
~~~

## 0b. 下载ltpmodel
下载链接http://ltp.ai/download.html。
project01目录下创建目录ltpmodels,将下载好的model解压放置在此路径下，包含cws.model,ner.model,
parser.model,pisrl.model,pos.model

## 1.训练word2vec model
两种方式可供选择
1.自己训练word2vec model并放置在project01/model目录下，命名为word2vec_update.model
2.server 39.100.3.165上有训练好的word2vec model，可以直接登陆此服务器并执行下面命令进入项目目录下，激活虚拟环境，跳到第3步执行。
~~~{.bash}
cd /home/student/project/project-01/nlp_strong/nlp_kaikeba
conda activate nlpstrong
~~~ 

## 2.从mysql数据库下载新闻语料数据,并解析保存成json
~~~{.python}
python src/ltp_parser.py
~~~

## 3.开启web服务器
开启前请查看端口是否被占用，如被占用，则到src/web/web_show.py中修改端口号。
~~~{.python}
python src/web/web_show.py
~~~
注意：开启服务器期间需要进行数据加载和计算，整个过程可能需要6-7分钟时间。

## 4.浏览器访问
服务器开启后即可浏览器中访问，http://ip:port/extractViews,
输入新闻文本内容，点击提交，即可看到人物观点提取之后的结果。
