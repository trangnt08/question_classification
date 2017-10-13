# -*- encoding: utf8 -*-
import re

import gensim
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from io import open
from operator import itemgetter

import datetime
import pandas as pd
import time
from pyvi.pyvi import ViTokenizer
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter

from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import math
from textblob import TextBlob as tb
WORD_SIZE = 500

class WordInfo:
    # tf_idf = 0
    # w2v = np.zeros(WORD_SIZE)

    def __init__(self, w2v, tf_idf):
        self.w2v = w2v
        self.tf_idf = tf_idf

    def get_w2v(self):
        return self.w2v

    def get_tfidf(self):
        return self.tf_idf


def time_diff_str(t1, t2):
    """
    Calculates time durations.
    """
    diff = t2 - t1
    mins = int(diff / 60)
    secs = round(diff % 60, 2)
    return str(mins) + " mins and " + str(secs) + " seconds"

def load_model(model):
    print('loading model ...')
    if os.path.isfile(model):
        return joblib.load(model)
    else:
        return None
def clean_str_vn(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[~`@#$%^&*-+]", " ", string)
    def sharp(str):
        b = re.sub('\s[A-Za-z]\s\.', ' .', ' '+str)
        while (b.find('. . ')>=0): b = re.sub(r'\.\s\.\s', '. ', b)
        b = re.sub(r'\s\.\s', ' # ', b)
        return b
    string = sharp(string)
    string = re.sub(r" : ", " ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    # return string.strip()
    return string.strip().lower()

def review_to_words(review):
    """
    Function to convert a raw review to a string of words
    :param review
    :return: meaningful_words
    """
    # 1. Convert to lower case, split into individual words
    words = review.lower().split()
    # 2. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    with open('datavn/vietnamese-stopwords-dash.txt', "r") as f3:
        dict_data = f3.read()
        array = dict_data.splitlines()
    # 3. Remove stop words
    meaningful_words = [w for w in words if not w in array]

    # 4. Join the words back into one string separated by space,
    # and return the result.
    return " ".join(meaningful_words)

def print_words_frequency(train_data_features):
    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()
    print "Words in vocabulary:", vocab

    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    print "Words frequency..."
    for tag, count in zip(vocab, dist):
        print count, tag

def inverse_document_frequencies(tokenized_documents):
    # tokenized_documents: list cac list cua nhieu document  [[w1...wn]...[w1...wn]]
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    # sublist: list cua 1 document
    # item: 1 tu trong sublist
    # all_tokens_set: tap cac tu trong tat ca document, cac tu ko lap lai. vd: set(['a','b','c','d'])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        # contains_token: so mang = so phan tu trong all_tokens_set. Moi mang chua n phan tu(n = so document)
        # Phan tu thu i = True neu no nam trong document i, False neu ko nam trong document i
        # contains_token: list
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
        # sum(contains_token): so vb chua "tkn"
    # print idf_values
    return idf_values

def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:  # Neu tu do khong xuat hien trong document
        return 0
    return 1 + math.log(count)

tokenize = lambda doc: doc.lower().split(" ")

def tfidf(documents):
    tokenized_documents = [tokenize(d) for d in documents]  # list cac list cua nhieu document  [[w1...wn]...[w1...wn]]
    print tokenized_documents
    idf = inverse_document_frequencies(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:    # document la 1 list chua cac tu trong 1 doc
        doc_tfidf = []
        for term in idf.keys(): # term la 1 word
            tf = sublinear_term_frequency(term, document)
            print term,tf,idf[term]
            doc_tfidf.append(tf * idf[term])    # list cac tfidf cua cac tu trong document do
        tfidf_documents.append(doc_tfidf)
    # print tfidf_documents
    return tfidf_documents


def load_data(filename):
    res = []
    col1 = []; col2 = []; col3 = []; col4 = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            label1, p , label2, question = line.split(" ", 3)
            # question = review_to_words(question)
            question = clean_str_vn(question)
            question = question.lower()
            # print question
            col1.append(label1)
            col2.append(label2)
            col3.append(question)

            # question = clean_str_vn(question)

            # question = ViTokenizer.tokenize(unicode(question, 'utf8'))
            # print question
            # res.append((label, question))
        d = {"label1":col1, "label2":col2, "question": col3}
        # d = dict(zip(col1,col3))
        train = pd.DataFrame(d)
        # print train
    return train

def get_tfidf_scores(vectorizer, tfidf_result):
    # http://stackoverflow.com/questions/16078015/
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    # scores = zip(vectorizer.get_feature_names(), vectorizer.idf_)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    # for item in sorted_scores:
    #     print('%s : %f' % (item[0], item[1]))
    return scores

def getlist(sentences):
    list_sens = []
    for j in sentences:  # j la 1 cau
        j = clean_str_vn(j)
        words = j.split(' ')  # word la 1 mang cac tu cua 1 cau
        words_clean = []
        for k in words:
            if k != '':
                words_clean.append(k)
        if len(words_clean) > 0:    # bo list chi chua ki tu '\n'
            list_sens.append(words_clean)  # list_sens la 1 list chua cac list tu da tach trong cau
    return list_sens

# def is_exist(word, vocab):
#     try:
#         ind = vocab[word]
#         return  ind
#     except:
#         return -1

def is_exist(word, vocab):
    try:
        _ = vocab[word]
        return  True
    except:
        return False

def vecto_tfidf(train_text,dict_vocab ):
    arr = []
    count = 0
    tfidf = {}
    tokenized_documents = [tokenize(d) for d in train_text]
    # tokenized_documents: list cac list cua nhieu document  [[w1...wn]...[w1...wn]]
    idf = inverse_document_frequencies(tokenized_documents)
    for document in train_text: # document: 1 cau
        sumv = np.zeros(WORD_SIZE)
        document = tokenize(document)
        for term in document:
            tf = sublinear_term_frequency(term, document)
            try:
                tfidf[term] = tf * idf[term]
            except:
                pass
            b = tfidf[term]
            check = is_exist(term,dict_vocab)
            if check == True:
                w = WordInfo(word2vec[term], b)
            else:
                w = WordInfo(np.zeros(WORD_SIZE), 0)

            v = (w.get_tfidf()) * (w.get_w2v())
            sumv = sumv + v
        if sumv.all() == 0:
            count += 1
        arr.append(sumv)
    print "count ", count
    arr = np.asarray(arr)
    return arr

if __name__ == "__main__":
    train = load_data('datavn/train')
    test = load_data('datavn/test')
    # print test

    print "Data dimensions:", train.shape
    print "List features:", train.columns.values
    print "First review:", train["label1"][0], "|", train["question"][0]

    print "Data dimensions:", test.shape
    print "List features:", test.columns.values
    print "First review:", test["label1"][0], "|", test["question"][0]
    # train, test = train_test_split(train, test_size=0.2)

    train_text = train["question"].values
    test_text = test["question"].values
    # list = []
    vectorizer = TfidfVectorizer(ngram_range=(1, 1),norm='l2')

    # X_train = vectorizer.fit_transform(train_text)
    # doc = 0
    # feature_index = X_train[doc, :].nonzero()[1]
    # tfidf_scores = zip(feature_index, [X_train[doc, x] for x in feature_index])

    vectorizer.fit(train_text)
    X_train = vectorizer.transform(train_text)
    # X_train = X_train.toarray()
    sorted_score = get_tfidf_scores(vectorizer, X_train)

    y_train = train["label1"]
    y_train2 = train["label2"]

    X_test = vectorizer.fit_transform(test_text)
    # X_test = X_test.toarray()
    y_test = test["label1"]
    y_test2 = test["label2"]

    """
    Training
    """

    word2vec = load_model('model/w2v.pkl')
    if word2vec == None:
        sentences = []
        for item in train_text:
            item = item.lower()
            # item = item.decode('utf-8').lower()
            # item = item.encode('utf-8')
            sentences.append(item)
        list_sens = getlist(sentences)
        word2vec = gensim.models.Word2Vec(min_count=1, size=WORD_SIZE, window=2, sg=0, iter=10)
        word2vec.build_vocab(list_sens)
        word2vec.train(list_sens, total_examples=word2vec.corpus_count, epochs=word2vec.iter)
        # joblib.dump(word2vec, 'model/w2v.pkl')
    with open('datavn/w2v.txt', 'w', encoding='utf-8') as f:
        for item in word2vec.wv.vocab.keys():
           f.write(item + '\n')
    # kq = word2vec.most_similar(u"hoán_vị", topn=5)
    # for i in range(len(kq)):
    #     print "a ", kq[i][0]


    s = word2vec.wv.vocab.keys()

    sd = sorted(s)

    with open('data/out1.txt', 'w') as f1:
        for i in sorted_score:
            f1.write(i[0] + u"\n")
            # print i[0]
    with open('data/out2.txt', 'w', encoding='utf-8') as f2:
        for i in sd:
            f2.write(i+"\n")
    # for i in sorted_score:
    #     print i[0]
    a = set(x[0] for x in sorted_score) # tap cac word trong vocab chua tfidf
    aa = [x[0] for x in sorted_score]
    b = set(word2vec.wv.vocab.keys())   # tap cac word trong vocab cua w2v
    c = a.intersection(b)

    f = [x for x in sorted_score]   # tap cac word trong vocab chua tfidf gom word va tfidf

    # aab = {w:aa.index(w) for w in c} # a and b
    aab = {w: True for w in c}  # a and b

    f1=[];f2=[]
    for item in f:
        f1.append(item[0])
        f2.append(item[1])
    d=dict(zip(f1,f2))
    # print f
    arr = []; arr2 = []

    clf = load_model('model/svm.pkl')
    if clf == None:
        print "---------------------------"
        print "Training"
        print "---------------------------"
        names = ["RBF SVC"]
        t0 = time.time()
        arr = vecto_tfidf(train_text, aab)
        clf = SVC(kernel='rbf', C=50)
        clf.fit(arr, y_train)
        # clf.fit(X_train, y_train)
        # clf.fit(arr,y_train)
        # joblib.dump(clf, 'model/svm.pkl')
        print " %s - Converting completed %s" % (datetime.datetime.now(), time_diff_str(t0, time.time()))

    # arr2 = get_train(test_text, aab, d, vectorizer,y_test)
    arr2 = vecto_tfidf(test_text, aab)
    y_pred = clf.predict(arr2)
    # print y_pred

    print " accuracy: %0.3f" % accuracy_score(y_test,y_pred)
    print "confuse matrix: \n", confusion_matrix(y_test, y_pred, labels=["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"])


    # get_tfidf_scores(vectorizer,X_train)

    # print "-----------------------"
    # print "fine grained category"
    # print "-----------------------"
    # clf = SVC(kernel='rbf', C=1000)
    # clf.fit(X_train, y_train2)
    # y_pred = clf.predict(X_test)
    # print y_pred
    #
    # print " accuracy: %0.3f" % accuracy_score(y_test2,y_pred)
    #
