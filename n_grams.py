# -*- encoding: utf8 -*-
import re
from itertools import count

from sklearn.metrics import accuracy_score
import datetime
import pandas as pd
import time
from pyvi.pyvi import ViTokenizer
import numpy as np
import os
from sklearn import svm
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def review_to_words(review, filename):
    """
    Function to convert a raw review to a string of words
    :param review
    :return: meaningful_words
    """
    # 1. Convert to lower case, split into individual words
    words = review.lower().split()
    # 2. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    with open(filename, "r") as f3:
        dict_data = f3.read()
        array = dict_data.splitlines()
    # 3. Remove stop words
    meaningful_words = [w for w in words if not w in array]

    # 4. Join the words back into one string separated by space,
    # and return the result.
    return " ".join(meaningful_words)

def review_to_words2(review, filename,n):
    with open(filename, "r") as f3:
        dict_data = f3.read()
        array = dict_data.splitlines()
    words = [' '.join(x) for x in ngrams(review, n)]
    meaningful_words = [w for w in words if not w in array]
    return build_sentence(meaningful_words)

def word_clean(array, review):
    words = review.lower().split()
    meaningful_words = [w for w in words if w in array]
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

def ngrams(input, n):
  input = input.split(' ')
  output = []
  for i in range(len(input)-n+1):
    output.append(input[i:i+n])
  return output # output dang ['a b','b c','c d']

def ngrams2(input, n):
  input = input.split(' ')
  output = {}
  for i in range(len(input)-n+1):
    g = ' '.join(input[i:i+n])
    output.setdefault(g, 0)
    output[g] += 1
  return output # output la tu dien cac n-gram va tan suat cua no {'a b': 1, 'b a': 1, 'a a': 3}

def ngrams_array(arr,n):
    output = {}
    for x in arr:
        d = ngrams2(x, n)  # moi d la 1 tu dien
        for x in d:
            count = d.get(x)
            output.setdefault(x, 0)
            output[x] += count
    return output

# def build_dict(arr,n,m):
#     d={}
#     ngram = ngrams_array(arr,n)
#     for x in ngram:
#         p = ngram.get(x)
#         if p < m:
#             d.setdefault(x,p)
#     return d
def buid_dict(filename,arr,n,m):
    with open(filename, 'r') as f:
        ngram = ngrams_array(arr, n)
        for x in ngram:
            p = ngram.get(x)
            if p < m:
                f.write(x)

def build_sentence(input_arr):
    d = {}
    for x in range(len(input_arr)):
        d.setdefault(input_arr[x], x)
    chuoi = []
    for i in input_arr:
        x = d.get(i)
        if x == 0:
            chuoi.append(i)
        for j in input_arr:
            y = d.get(j)
            if y == x + 1:
                z = j.split(' ')
                chuoi.append(z[1])
    return " ".join(chuoi)

def load_data(filename, dict):
    res = []
    col1 = []; col2 = []; col3 = []; col4 = []

    with open(filename, 'r') as f,open(dict, "w") as f2:
        for line in f:
            label1, p , label2, question = line.split(" ", 3)
            # question = clean_str_vn(question)
            # question = review_to_words(question,'datavn/vietnamese-stopwords-dash.txt')
            col1.append(label1)
            col2.append(label2)
            col3.append(question)

        ngram = ngrams_array(col3,2)
        dict_arr = []
        for x in ngram:
            p = ngram.get(x)
            if p<1:
                dict_arr.append(x)
                f2.write(x+"\n")
        col4 = []
        for q in col3:
            q = review_to_words2(q,dict,2)
            col4.append(q)
        print "col4 ", col4
        d = {"label1":col1, "label2":col2, "question": col4}
        train = pd.DataFrame(d)
    return train


if __name__ == "__main__":

    # load_data("datavn/test")

    # vectorizer = CountVectorizer(analyzer="word",
    #                          tokenizer=None,
    #                          preprocessor=None,
    #                          stop_words=None,
    #                          max_features=1000)
    vectorizer = TfidfVectorizer(ngram_range=(1,1), max_df=0.7, min_df=2, max_features=1000)
    train = load_data('datavn/train','datavn/dict1')
    test = load_data('datavn/test','datavn/dict2')
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

    vectorizer.fit(train_text)
    X_train = vectorizer.transform(train_text)
    X_train = X_train.toarray()
    y_train = train["label1"]
    y_train2 = train["label2"]
    print X_train

    X_test = vectorizer.transform(test_text)
    X_test = X_test.toarray()
    y_test = test["label1"]
    y_test2 = test["label2"]

    print "---------------------------"
    print "Training"
    print "---------------------------"
    names = ["RBF SVC"]
    t0 = time.time()
    # iterate over classifiers
    results = {}
    kq = {}
    clf = SVC(kernel='rbf', C=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print y_pred

    print " accuracy: %0.3f" % accuracy_score(y_test,y_pred)
    print " %s - Converting completed %s" % (datetime.datetime.now(), time_diff_str(t0, time.time()))

    # print "-----------------------"
    # print "fine grained category"
    # print "-----------------------"
    # clf = SVC(kernel='rbf', C=1000)
    # clf.fit(X_train, y_train2)
    # y_pred = clf.predict(X_test)
    # print y_pred
    #
    # print " accuracy: %0.3f" % accuracy_score(y_test2,y_pred)

