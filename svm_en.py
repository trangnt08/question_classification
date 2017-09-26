# -*- encoding: utf8 -*-
import re
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score

import datetime
import numpy as np
import pandas as pd
import time
from nltk.corpus import stopwords
from pyvi.pyvi import ViTokenizer
import numpy as np
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
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", string)
    string = re.sub(r" : ", ":", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def review_to_words(review, remove_stopwords=False ):
     review_text = BeautifulSoup(review).get_text()
    #
        # 2. Remove non-letters
     review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case and split them
     words = review_text.split()
        #
        # 4. Optionally remove stop words (false by default)
     if remove_stopwords:
         stops = set(stopwords.words("english"))
         words = [w for w in words if not w in stops]
        #
        # 5. Return a list of words
     return " ".join(words)


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
            label1, p = line.split(":", 1)
            label2, question = p.split(" ", 1)
            question = review_to_words(question, False)
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


if __name__ == "__main__":

    # load_data("data/test")

    # vectorizer = CountVectorizer(analyzer="word",
    #                          tokenizer=None,
    #                          preprocessor=None,
    #                          stop_words=None,
    #                          max_features=1000)
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)
    train = load_data('data/train')
    test = load_data('data/test')
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

    X_train = vectorizer.fit_transform(train_text)
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
    clf = SVC(kernel='rbf', C=100)
    clf.fit(X_train, y_train)
    print "BBBBB "
    y_pred = clf.predict(X_test)
    print y_pred

    print " accuracy: %0.3f" % clf.score(X_test,y_test)
    print " %s - Converting completed %s" % (datetime.datetime.now(), time_diff_str(t0, time.time()))

    print "-----------------------"
    print "fine grained category"
    print "-----------------------"
    clf = SVC(kernel='linear', C=100)
    clf.fit(X_train, y_train2)
    y_pred = clf.predict(X_test)
    print y_pred

    print " accuracy: %0.3f" % accuracy_score(y_test2,y_pred)
