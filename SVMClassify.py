from pyhanlp import *
import numpy as np
import re

'''创建数据集：单词列表postingList, 所属类别classVec'''
def Handle_data(content):
    reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
    CustomDictionary.add("傻缺", "nr 300")
    CustomDictionary.insert("语音识别", "nz 1024")
    CustomDictionary.add("巨大成功", "nz 1024 n 1")
    CustomDictionary.add("深度学习", "nz 1025")
    content=re.sub(reg, '', content)
    Get_value = HanLP.segment(content)
    lists=[]
    for term in Get_value:
        lists.append(term.word)
    return lists
def loadDataSet():
    Train = [
        "你个傻缺",
        "今天天气真的很好，我要开始新的一天去学习",
        "我太不开心了，我日",
        "每一个太阳都很大",
        "你真是个傻缺",
        "你太狗了",
        "加油，我最棒"
    ]
    Train_y = [
        1, 0, 1, 0, 1, 1, 0
    ]
    Train_x = []
    for n in Train:
        a = Handle_data(n)
        Train_x.append(a)
    return Train_x, Train_y

'''获取所有单词的集合:返回不含重复元素的单词列表'''
def createVocabList(dataSet):
    vocabSet = set([])

    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 操作符 | 用于求两个集合的并集
    #print(vocabSet,"PPPPPPP")
    return list(vocabSet)

'''词集模型构建数据矩阵'''
def setOfWords2Vec(vocabList, inputSet):
    # 创建一个和词汇表等长的向量，并将其元素都设置为0
    returnVec = [0] * len(vocabList)
    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("单词: %s 不在词汇表之中!" % word)
    #print(returnVec)
    return returnVec
'''文档词袋模型构建数据矩阵'''
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    #print(returnVec)
    return returnVec

'''朴素贝叶斯分类器训练函数'''
def _trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix) # 文件数
    numWords = len(trainMatrix[0]) # 单词数
    # 侮辱性文件的出现概率，即trainCategory中所有的1的个数，
    # 代表的就是多少个侮辱性文件，与文件的总数相除就得到了侮辱性文件的出现概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)

    # 构造单词出现次数列表
    p0Num = np.zeros(numWords) # [0,0,0,.....]
    p1Num = np.zeros(numWords) # [0,0,0,.....]
    p0Denom = 0.0;p1Denom = 0.0 # 整个数据集单词出现总数
    for i in range(numTrainDocs):
        # 遍历所有的文件，如果是侮辱性文件，就计算此侮辱性文件中出现的侮辱性单词的个数
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i] #[0,1,1,....]->[0,1,1,...]
            p1Denom += sum(trainMatrix[i])
        else:
            # 如果不是侮辱性文件，则计算非侮辱性文件中出现的侮辱性单词的个数
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 类别1，即侮辱性文档的[P(F1|C1),P(F2|C1),P(F3|C1),P(F4|C1),P(F5|C1)....]列表
    # 即 在1类别下，每个单词出现次数的占比
    p1Vect = p1Num / p1Denom# [1,2,3,5]/90->[1/90,...]
    # 类别0，即正常文档的[P(F1|C0),P(F2|C0),P(F3|C0),P(F4|C0),P(F5|C0)....]列表
    # 即 在0类别下，每个单词出现次数的占比
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive
'''训练数据优化版本'''
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix) # 总文件数
    numWords = len(trainMatrix[0]) # 总单词数
    pAbusive = sum(trainCategory) / float(numTrainDocs) # 侮辱性文件的出现概率
    # 构造单词出现次数列表,p0Num 正常的统计,p1Num 侮辱的统计
    # 避免单词列表中的任何一个单词为0，而导致最后的乘积为0，所以将每个单词的出现次数初始化为 1
    p0Num = np.ones(numWords)#[0,0......]->[1,1,1,1,1.....],ones初始化1的矩阵
    p1Num = np.ones(numWords)

    # 整个数据集单词出现总数，2.0根据样本实际调查结果调整分母的值（2主要是避免分母为0，当然值可以调整）
    # p0Denom 正常的统计
    # p1Denom 侮辱的统计
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]  # 累加辱骂词的频次
            p1Denom += sum(trainMatrix[i]) # 对每篇文章的辱骂的频次 进行统计汇总
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表,取对数避免下溢出或浮点舍入出错
    p1Vect = np.log(p1Num / p1Denom)
    # 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

'''分类'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 计算公式  log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    # 使用 NumPy 数组来计算两个向量相乘的结果，这里的相乘是指对应元素相乘，即先将两个向量中的第一个元素相乘，然后将第2个元素相乘，以此类推。这里的 vec2Classify * p1Vec 的意思就是将每个词与其对应的概率相关联起来
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def classifyPrint(i):
    if(i==1):
        return "侮辱性"
    else:
        return "正常"

if __name__ == '__main__':
    #获取数据
    Train_x, Train_y=loadDataSet()
    #获得单词集合
    Value=createVocabList(Train_x)
    #获取每一句话中所有的出现频率
    Translat_x=[]
    for n in Train_x:
        m=setOfWords2Vec(Value,n)
        Translat_x.append(m)

    #训练数据
    p0Vect, p1Vect, pAbusive=trainNB0(Translat_x,Train_y)
    print("在1类别下，每个单词出现次数的占比:\n",p1Vect)
    print("在0类别下，每个单词出现次数的占比:\n", p0Vect)
    print("代表的就是多少个侮辱性文件，与文件的总数相除就得到了侮辱性文件的出现概率:\n", pAbusive)

    #测试
    test="你他妈有病啊"
    testEntry=Handle_data(test)
    thisDoc = np.array(setOfWords2Vec(Value, testEntry))
    i=classifyNB(thisDoc, p0Vect, p1Vect, pAbusive)

    print(testEntry, '分类结果是: ',classifyPrint(i))