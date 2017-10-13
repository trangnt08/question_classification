# -*- encoding: utf8 -*-
import re

import gensim
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from operator import itemgetter

import datetime
import pandas as pd
import time
from pyvi.pyvi import ViTokenizer
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
from sklearn.svm import SVC
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

def load_data(filename):
    res = []
    col1 = []; col2 = []; col3 = []; col4 = []

    with open(filename, 'r') as f:
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
    scores = zip(vectorizer.get_feature_names(), np.asarray(tfidf_result.sum(axis=0)).ravel())
    # print tfidf_result.shape[0]
    # scores = zip(vectorizer.get_feature_names(), vectorizer.idf_)
    # sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
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

def is_exist(word, vocab):
    try:
        ind = vocab[word]
        return  ind
    except:
        return -1

def get_train(train_text, aab, d, vectorizer, y_train):
    # a: set cac word trong vocab cua tfidf
    # ob: cac word trong vocab cua w2v
    # abb: a and b
    arr = []; y1 = []; y2 = []
    for j in y_train:
        y1.append(j)
    count1 = 0
    s = 0
    for i in train_text:  # i la 1 cau

        sum1 = np.zeros(WORD_SIZE)
        words = i.split(' ')  # words la 1 list cac tu
        a = dict((x, words.count(x)) for x in set(words))
        idf = vectorizer.idf_
        for x in a:
            ind = is_exist(x,aab)
            if ind > -1:
                idf1 = idf[ind]
                b = (a.get(x))*idf1/len(i)  # b: tf-idf
                # print "b ",b
                w = WordInfo(word2vec[x], b)
            else:
                w = WordInfo(np.zeros(WORD_SIZE), 0)

            ar = (w.get_tfidf()) * (w.get_w2v())
            sum1 = sum1 + ar

        if sum1.all() == 0.:
            count1 += 1
        else:
            arr.append(sum1)
            y2.append(y1[s])
        s += 1

    arr = np.asarray(arr)
    print type(arr)
    print len(y2)
    return arr

def get_y(train_text, aab, d, vectorizer, y_train):
    # a: set cac word trong vocab cua tfidf
    # ob: cac word trong vocab cua w2v
    # abb: a and b
    arr = []; y1 = []; y2 = []
    for j in y_train:
        y1.append(j)
    count1 = 0
    s = 0
    for i in train_text:  # i la 1 cau

        sum1 = np.zeros(WORD_SIZE)
        words = i.split(' ')  # words la 1 list cac tu
        a = dict((x, words.count(x)) for x in set(words))
        idf = vectorizer.idf_
        for x in a:
            ind = is_exist(x,aab)
            if ind > -1:
                idf1 = idf[ind]
                b = (a.get(x))*idf1/len(i)  # b: tf-idf
                # print "b ",b
                w = WordInfo(word2vec[x], b)
            else:
                w = WordInfo(np.zeros(WORD_SIZE), 0)

            ar = (w.get_tfidf()) * (w.get_w2v())
            sum1 = sum1 + ar

        if sum1.all() == 0.:
            count1 += 1
        else:
            arr.append(sum1)
            y2.append(y1[s])
        s += 1

    arr = np.asarray(arr)
    return y2

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

    X_train = vectorizer.fit_transform(train_text)
    sorted_score = get_tfidf_scores(vectorizer, X_train)
    # doc = 0
    # feature_index = X_train[doc, :].nonzero()[1]
    # tfidf_scores = zip(feature_index, [X_train[doc, x] for x in feature_index])

    # vectorizer.fit(train_text)
    # X_train = vectorizer.transform(train_text)
    # X_train = X_train.toarray()

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
            item = item.decode('utf-8').lower()
            item = item.encode('utf-8')
            sentences.append(item)
        list_sens = getlist(sentences)
        word2vec = gensim.models.Word2Vec(min_count=1, size=WORD_SIZE, window=2, sg=0, iter=10)
        word2vec.build_vocab(list_sens)
        word2vec.train(list_sens, total_examples=word2vec.corpus_count, epochs=word2vec.iter)
        # joblib.dump(word2vec, 'model/w2v.pkl')




    sorted_score2 = get_tfidf_scores(vectorizer, X_test)

    s = word2vec.wv.vocab.keys()
    sd = sorted(s)

    print len(sorted_score2)
    print len(word2vec.wv.vocab)
    with open('data/out1.txt', 'w') as f1:
        for i in sorted_score:
            f1.write(i[0].encode('utf-8')+"\n")
            # print i[0]
    with open('data/out2.txt', 'w') as f2:
        for i in sd:
            f2.write(i+"\n")
    # for i in sorted_score:
    #     print i[0]
    a = set(x[0] for x in sorted_score) # tap cac word trong vocab chua tfidf
    aa = [x[0] for x in sorted_score]
    b = set(word2vec.wv.vocab.keys())   # tap cac word trong vocab cua w2v
    c = a.intersection(b)
    d = a.difference(b)
    e = b.difference(a)
    f = [x for x in sorted_score]   # tap cac word trong vocab chua tfidf gom word va tfidf
    print len(a)
    print len(d)
    print "e ", len(e)
    aab = {w:aa.index(w) for w in c} # a and b

    f1=[];f2=[]
    for item in f:
        f1.append(item[0])
        f2.append(item[1])
    d=dict(zip(f1,f2))
    print d
    # print f
    arr = []; arr2 = []

    # clf = load_model('model/svm.pkl')
    # if clf == None:
    #     print "---------------------------"
    #     print "Training"
    #     print "---------------------------"
    #     names = ["RBF SVC"]
    #     t0 = time.time()
    #     # iterate over classifiers
    #     results = {}
    #     kq = {}
    #     arr = get_train(train_text,aab, d, vectorizer,y_train)
    #     y1 = get_y(train_text,aab, d, vectorizer,y_train)
    #
    #     clf = SVC(kernel='rbf', C=1000)
    #     clf.fit(arr, y1)
    #     # clf.fit(X_train, y_train)
    #     # clf.fit(arr,y_train)
    #     # joblib.dump(clf, 'model/svm.pkl')
    #     print " %s - Converting completed %s" % (datetime.datetime.now(), time_diff_str(t0, time.time()))
    #
    # arr2 = get_train(test_text, aab, d, vectorizer,y_test)
    # y2 = get_y(test_text,aab,d,vectorizer,y_test)
    # y_pred = clf.predict(arr2)
    # # print y_pred
    #
    # print " accuracy: %0.3f" % accuracy_score(y2,y_pred)

