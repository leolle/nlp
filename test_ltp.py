# -*- coding: utf-8 -*-
from pyltp import SentenceSplitter
sents = SentenceSplitter.split('由于美国总统特朗的女婿库什纳(Jared Kusher)在闭门听证会之前发布的声明中表示其并未与俄罗斯勾结，且周一公布的美国经济数据表现良好，美元当天自逾一年最低水准小幅反弹至94关口上方。眼下美联储即将召开为期两天的货币政策会议，而投资者料密切关注决策者对近期疲弱的零售销售和通胀的评估，以及是否透露缩表的具体时间。另一方面，美元看空情绪已经升至2013年高位，但分析师们认为，有迹象显示美元近期跌势或将迎来终结。')  # 分句
print '\n'.join(sents)