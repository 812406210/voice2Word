# -*-coding:utf-8-*-

import re
import  jieba
import  json
import  numpy as np
from gensim.models import KeyedVectors
import  pandas as pd
def testJieba(data):
    # jieba 分词
    jieba.load_userdict("E:\\python\\Demo\\dict.txt")
    seg_list = jieba.cut(data,cut_all=False)
    print("Full Mode: " + ",".join(seg_list))

def getJsonData(jsonData):
    jsonStr = json.dumps(jsonData)
    print(jsonStr)
    jsonResult = json.loads(jsonStr)
    finalResult = jsonResult['text']
    return  finalResult

def clean_space(originalData):
    match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
    should_replace_list = match_regex.findall(originalData)
    order_replace_list = sorted(should_replace_list, key=lambda i: len(i), reverse=True)
    for i in order_replace_list:
        if i == u' ':
            continue
        new_i = i.strip()
        originalData = originalData.replace(i, new_i)
    return originalData

def to_review_vector(review):
    model = KeyedVectors.load_word2vec_format("E:\\python\\Demo\\5000-small.txt")

    word_vec = np.zeros((1, 300))
    for word in review:
        # word_vec = np.zeros((1,300))
        if word in model:
            word_vec += np.array([model[word]])
    # print (word_vec.mean(axis = 0))
    print(pd.Series(word_vec.mean(axis=0)))
    return pd.Series(word_vec.mean(axis=0))

if __name__ == "__main__":

    testJieba("哇塞，这是武汉长江大桥吗？")

    # json解析
    jsonData = {"result": [{"conf": 0.976326, "end": 1.5, "start": 1.14, "word": "喂"},
                           {"conf": 0.945741, "end": 2.098387, "start": 1.5, "word": "你好"}], "text": "喂 你好"}
    result = getJsonData(jsonData)
    print("解析结果:",result)
    final_result = clean_space(result)
    print("解析最终结果:", final_result)
    print("------------------------------------------","\n")
    model = KeyedVectors.load_word2vec_format("E:\\python\\Demo\\5000-small.txt")
    print(model.similarity('女人', '男人'))
    print(model.most_similar('特朗普',topn=10))

    print(model.doesnt_match("上海 成都 广州 北京".split(" ")))

    print(model.get_vector("哈哈"))

    print(to_review_vector("病"))

