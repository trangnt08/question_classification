# -*- encoding: utf8 -*-
import re
from sklearn.metrics import accuracy_score
from operator import itemgetter

import datetime
import numpy as np
import pandas as pd
import time
from pyvi.pyvi import ViTokenizer
import numpy as np
import os
from sklearn import svm
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

def time_diff_str(t1, t2):
    """
    Calculates time durations.
    """
    diff = t2 - t1
    mins = int(diff / 60)
    secs = round(diff % 60, 2)
    return str(mins) + " mins and " + str(secs) + " seconds"

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
    string = re.sub(r" : ", ":", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
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
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in sorted_scores:
        print('%s : %f' % (item[0], item[1]))

if __name__ == "__main__":

    # load_data("datavn/test")

    # vectorizer = CountVectorizer(analyzer="word",
    #                          tokenizer=None,
    #                          preprocessor=None,
    #                          stop_words=None,
    #                          max_features=1000)
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)
    train = load_data('datavn/train')
    test = load_data('datavn/test')
    print test

    print "Data dimensions:", train.shape
    print "List features:", train.columns.values
    print "First review:", train["label1"][0], "|", train["question"][0]

    print "Data dimensions:", test.shape
    print "List features:", test.columns.values
    print "First review:", test["label1"][0], "|", test["question"][0]
    # train, test = train_test_split(train, test_size=0.2)

    train_text = train["question"].values
    test_text = test["question"].values

    # X_train = vectorizer.fit_transform(train_text)
    vectorizer.fit(train_text)
    X_train = vectorizer.transform(train_text)
    X_train = X_train.toarray()
    y_train = train["label1"]
    y_train2 = train["label2"]

    X_test = vectorizer.transform(test_text)
    X_test = X_test.toarray()
    y_test = test["label1"]
    y_test2 = test["label2"]

    """
    Training
    """

    print "---------------------------"
    print "Training"
    print "---------------------------"
    names = ["RBF SVC"]
    t0 = time.time()
    # iterate over classifiers
    results = {}
    kq = {}
    # clf = SVC(kernel='rbf', C=1000)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # print y_pred
    #
    # print " accuracy: %0.3f" % accuracy_score(y_test,y_pred)
    # print " %s - Converting completed %s" % (datetime.datetime.now(), time_diff_str(t0, time.time()))
    get_tfidf_scores(vectorizer,X_train)

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
