# coding: gbk

import jieba
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re  # ʵ��������ʽģ��
import string
from sklearn import metrics
from sklearn.linear_model import SGDClassifier

class RestPython():

    def __init__(self):
        # ��������
        self.stopword_list = self.load_stopword_list()
        self.orginal_corpus, self.orginal_labels = self.get_data()
        self.corpus, self.labels = self.remove_empty_docs(self.orginal_corpus,self.orginal_labels)

        # ���ݴ��� === ��ȡ������ȡ��
        self.train_corpus, self.test_corpus, self.train_labels, self.test_labels = self.prepare_datasets(self.corpus,self.labels,test_data_proportion=0.3)
        self.norm_train_corpus = self.normalize_corpus(self.train_corpus)
        self.norm_test_corpus = self.normalize_corpus(self.test_corpus)
        self.tfidf_vectorizer, self.tfidf_train_features = self.tfidf_extractor(self.norm_train_corpus)

        #��ȡѵ��ģ��
        self.classifier = self.train_model()


    def load_stopword_list(self):
        with open("dict/stop_words.utf8", encoding="utf8") as f:
            stopword_list = f.readlines()
        return stopword_list

    def tokenize_text(self,text):
        jieba.load_userdict("E:\\python\\Demo\\dict.txt")
        tokens = jieba.cut(text)
        tokens = [token.strip() for token in tokens]
        return tokens

    def remove_special_characters(self,text):
        tokens = self.tokenize_text(text)
        # compile ����һ��ƥ����� escape ���ӵ������ַ����壨�൱��ת�壬��ʾ�����壩 string.punctuation ��ʾ���б�����
        pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
        filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def remove_stopwords(self,text):
        tokens = self.tokenize_text(text)
        filtered_tokens = [token for token in tokens if token not in self.stopword_list]
        filtered_text = ''.join(filtered_tokens)
        return filtered_text

    def normalize_corpus(self,corpus: object, tokenize: object = False) -> object:
        normalized_corpus = []
        for text in corpus:

            text = self.remove_special_characters(text)
            text = self.remove_stopwords(text)
            normalized_corpus.append(text)
            if tokenize:
                text = self.tokenize_text(text)
                normalized_corpus.append(text)

        return normalized_corpus


    def get_data(self):
        """
        ��ȡ����
        :return:  �ı����ݣ���Ӧ��labels
        """
        with open("data/ham_data_example.txt", encoding='utf-8') as ham_f, open("data/spam_data_example.txt",
                                                                                encoding='utf-8') as spam_f:
            ham_data = ham_f.readlines()
            spam_data = spam_f.readlines()
            ham_label = np.ones(len(ham_data)).tolist()  # tolist��������������ת��Ϊ�б�����
            spam_label = np.zeros(len(spam_data)).tolist()
            corpus = ham_data + spam_data
            labels = ham_label + spam_label
        return corpus, labels

    def remove_empty_docs(self, corpus, labels):
        """
        ȥ������
        :param corpus:
        :param labels:
        :return:
        """
        filtered_corpus = []
        filtered_labels = []
        for docs, label in zip(corpus, labels):
            if docs.strip():
                filtered_corpus.append(docs)
                filtered_labels.append(label)
        return filtered_corpus, filtered_labels

    def prepare_datasets(self,corpus, labels, test_data_proportion=0.3):
        """
        :param corpus: �ı�����
        :param labels: �ı���ǩ
        :param test_data_proportion:  ���Լ�����ռ��
        :return: ѵ�����ݣ� �������ݣ� ѵ��labels�� ����labels
        """
        x_train, x_test, y_train, y_test = train_test_split(corpus, labels, test_size=test_data_proportion,
                                                            random_state=42)  # �̶�random_state��ÿ�����ɵ�������ͬ����ģ����ͬ��
        return x_train, x_test, y_train, y_test

    def tfidf_extractor(self,corpus, ngram_range=(1, 1)):
        vectorizer = TfidfVectorizer(min_df=1,
                                     norm='l2',
                                     smooth_idf=True,
                                     use_idf=True,
                                     ngram_range=ngram_range)
        features = vectorizer.fit_transform(corpus)
        return vectorizer, features

    def get_metrics(self,true_labels, predicted_labels):
        print('׼ȷ��:', np.round(
            metrics.accuracy_score(true_labels,
                                   predicted_labels),
            2))
        print('����:', np.round(
            metrics.precision_score(true_labels,
                                    predicted_labels,
                                    average='weighted'),
            2))
        print('�ٻ���:', np.round(
            metrics.recall_score(true_labels,
                                 predicted_labels,
                                 average='weighted'),
            2))
        print('F1�÷�:', np.round(
            metrics.f1_score(true_labels,
                             predicted_labels,
                             average='weighted'),
            2))


    def train_model(self):
        """
        ģ��ѵ��
        :return:
        """

        # ������ȡ
        tfidf_test_features = self.tfidf_vectorizer.transform(self.norm_test_corpus)

        svm_classifier = SGDClassifier(loss='hinge', n_iter_no_change=100)

        svm_classifier.fit(self.tfidf_train_features, self.train_labels)
        predictions = svm_classifier.predict(tfidf_test_features)

        self.get_metrics(self.test_labels, predictions)

        return svm_classifier

    def classify(self, input):
        input_text = [input]
        input_counts = self.tfidf_vectorizer.transform(input_text)

        predictions = self.classifier.predict(input_counts)
        print(predictions)
        return predictions

myModel = RestPython()
label_name_map = ['����������', '��������']  # 0 1
text = "�������в���"

result = myModel.classify(text)
print(label_name_map[int(result)])
