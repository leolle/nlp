#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os

ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path = [os.path.join(ROOTDIR, "lib")] + sys.path
from pyltp import SentenceSplitter, Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller
LTP_DATA_DIR = '/home/weiwu/projects/sentiment/ltp_data/'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`

segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  # 加载模型
words = segmentor.segment('由于美国总统特朗的女婿库什纳(Jared Kusher)在闭门听证会之前发布的声明中表示其并未与俄罗斯勾结，且周一公布的美国经济数据表现良好，美元当天自逾一年最低水准小幅反弹至94关口上方。眼下美联储即将召开为期两天的货币政策会议，而投资者料密切关注决策者对近期疲弱的零售销售和通胀的评估，以及是否透露缩表的具体时间。另一方面，美元看空情绪已经升至2013年高位，但分析师们认为，有迹象显示美元近期跌势或将迎来终结。')  # 分词
print '\t'.join(words)


ROOTDIR = LTP_DATA_DIR
sys.path = [os.path.join(ROOTDIR, "lib")] + sys.path

# Set your own model path
MODELDIR=os.path.join(ROOTDIR, "ltp_data")


paragraph = '中国进出口银行与中国银行加强合作。中国进出口银行与中国银行加强合作！'

sentence = SentenceSplitter.split(paragraph)[0]

segmentor = Segmentor()
segmentor.load(os.path.join(MODELDIR, "cws.model"))
words = segmentor.segment(sentence)
print "\t".join(words)

postagger = Postagger()
postagger.load(os.path.join(MODELDIR, "pos.model"))
postags = postagger.postag(words)
# list-of-string parameter is support in 0.1.5
# postags = postagger.postag(["中国","进出口","银行","与","中国银行","加强","合作"])
print "\t".join(postags)

parser = Parser()
parser.load(os.path.join(MODELDIR, "parser.model"))
arcs = parser.parse(words, postags)

print "\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs)

recognizer = NamedEntityRecognizer()
recognizer.load(os.path.join(MODELDIR, "ner.model"))
netags = recognizer.recognize(words, postags)
print "\t".join(netags)

labeller = SementicRoleLabeller()
labeller.load(os.path.join(MODELDIR, "srl/"))
roles = labeller.label(words, postags, netags, arcs)

for role in roles:
    print role.index, "".join(
            ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments])

segmentor.release()
postagger.release()
parser.release()
recognizer.release()
labeller.release()
