# -*- encoding: utf8 -*-
import re
from pyvi.pyvi import ViTokenizer, ViPosTagger
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import datetime
import pandas as pd
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from svm import *
# from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


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
    # words = review.lower().split()
    tup = ViPosTagger.postagging(ViTokenizer.tokenize(unicode(review, encoding='utf-8')))  # gan nhan POS
    words = review.split()
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
def review_add_pos(review, filename):
    words = review.split()
    with open(filename, "r") as f3:
        dict_data = f3.read()
        array = dict_data.splitlines()

    meaningful_words = [w for w in words if not w in array]
    b = " ".join(meaningful_words)  # cau sau khi loai bo stopword
    words_list = b.split()
    tup = ViPosTagger.postagging(ViTokenizer.tokenize(unicode(b,encoding='utf-8')))   # gan nhan POS
    a = tup[1]
    c = words_list + a
    return " ".join(c)

def add_pos_sen(review):
    tup = ViPosTagger.postagging(ViTokenizer.tokenize(unicode(review, encoding='utf-8')))  # gan nhan POS


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
            question = review_to_words(question,'datavn/question_stopwords.txt')
            # question = review_add_pos(question,'datavn/question_stopwords.txt')
            col1.append(label1)
            col2.append(label2)
            col3.append(question)

        ngram = ngrams_array(col3,2)    # tu dien cac tu va so lan xuat hien cua no
        dict_arr = []       # list cac tu co tan suat < 1
        for x in ngram:
            p = ngram.get(x)
            if p<1:
                dict_arr.append(x)
                f2.write(x+"\n")
        col4 = []
        for q in col3:
            r1 = []; r2 = []
            q = review_to_words2(q,dict,2)  # q la 1 cau
            q1 = [' '.join(x) for x in ngrams(q, 1)]  # q1:mang cac 1-grams
            s1 = ViPosTagger.postagging(ViTokenizer.tokenize(unicode(q, encoding='utf-8')))  # gan nhan POS
            for i1,i2 in zip(s1[0],s1[1]):
                t1 = i1 +"_" +i2
                t1 = t1.encode('utf-8')
                r1.append(t1)
            s2 = ' '.join(i for i in s1[1])     # Nhan tu loai cua cau dang str. vd: "N V E N"
            q2 = [' '.join(x) for x in ngrams(q, 2)]  # q2: mang cac phan tu 2-grams la word
            s22 = [' '.join(x) for x in ngrams(s2, 2)]  # s22: mang cac phan tu 2-grams la tag
            q3 = (' '.join(x.replace(' ', '_') for x in q2)).split()
            s3 = (' '.join(x.replace(' ', '_') for x in s22)).split()
            for i1,i2 in zip(q3,s3):
                t2 = i1 +"_" +i2
                r2.append(t2)
            y = r1 + r2
            # z1 = [' '.join(x) for x in y]
            z = ' '.join(y)
            col4.append(z)
            # col4.append(q)
        d = {"label1":col1, "label2":col2, "question": col4}
        train = pd.DataFrame(d)
    return train


if __name__ == "__main__":
    vectorizer = load_model('model/vectorizer2.pkl')
    if vectorizer == None:
        vectorizer = TfidfVectorizer(ngram_range=(1,1), max_df=0.7, min_df=2, max_features=1000)
    train = load_data('datavn/train','datavn/dict1')
    test = load_data('datavn/test','datavn/dict2')

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

    X_test = vectorizer.transform(test_text)
    X_test = X_test.toarray()
    y_test = test["label1"]
    y_test2 = test["label2"]
    # joblib.dump(vectorizer, 'model/vectorizer2.pkl')
    print "---------------------------"
    print "Training"
    print "---------------------------"
    names = ["RBF SVC"]
    t0 = time.time()
    # iterate over classifiers
    clf = load_model('model/uni_big2.pkl')
    clf = SVC(kernel='rbf', C=1000)
    clf.fit(X_train, y_train)
    # joblib.dump(clf, 'model/uni_big2.pkl')
    y_pred = clf.predict(X_test)
    print y_pred

    print " accuracy: %0.3f" % accuracy_score(y_test,y_pred)
    print " %s - Converting completed %s" % (datetime.datetime.now(), time_diff_str(t0, time.time()))
    print "confuse matrix: \n", confusion_matrix(y_test,y_pred,labels=["ABBR", "DESC", "ENTY","HUM","LOC","NUM"])

    # print "-----------------------"
    # print "fine grained category"
    # print "-----------------------"
    # clf = SVC(kernel='rbf', C=1000)
    # clf.fit(X_train, y_train2)
    # y_pred = clf.predict(X_test)
    # # print y_pred
    #
    # print " accuracy: %0.3f" % accuracy_score(y_test2,y_pred)

