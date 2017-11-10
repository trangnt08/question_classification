# -*- coding: utf-8 -*-
from underthesea import pos_tag
from pyvi.pyvi import ViTokenizer,ViPosTagger
from scipy.sparse import coo_matrix, np
from sklearn.utils import resample

# def ngrams(input, n):
#   input = input.split(' ')
#   output = []
#   for i in range(len(input)-n+1):
#     output.append(input[i:i+n])
#   return output
#
#
# def ngrams2(input, n):
#   input = input.split(' ')
#   output = {}
#   for i in range(len(input)-n+1):
#     g = ' '.join(input[i:i+n])
#     output.setdefault(g, 0)
#     output[g] += 1
#   return output
#
# def ngrams_array(arr,n):
#     output = {}
#     for x in arr:
#         d = ngrams2(x, n)  # moi d la 1 tu dien
#         for x in d:
#             count = d.get(x)
#             output.setdefault(x, 0)
#             output[x] += count
#     return output
#
# def build_dict(arr, n):
#   ngram = ngrams_array(arr, n)
#   d = {}
#   for x in ngram:
#     p = ngram.get(x)
#     if p >=2:
#       d.setdefault(x,p)
#       # print x + ":" + str(p)
#   return d
# def is_exist(word, vocab):
#     try:
#         word in vocab
#         return True
#     except:
#         return False
#
# print ngrams('a b c d', 2) # [['a', 'b'], ['b', 'c'], ['c', 'd']]
# a = [' '.join(x) for x in ngrams('a b c d', 2)] #['a b', 'b c', 'c d']
# b=""
# for x in a:
#   b += x +" "
#
# print b
#
# c = ngrams2('a b a a a a', 2) # {'a b': 1, 'b a': 1, 'a a': 3}
# print c
# for x in c:
#   a = c.get(x)
#   if a>=2:
#     print x + ":" + str(a)
#
# d = ngrams2('Xin chao xin chao xin chao, xin chao',3)
# print d
# arr =[]
# # arr.append(unicode("xin chao cac ban", encoding='utf-8'))
# # arr.append(unicode("rat vui gap cac ban",encoding='utf-8'))
# # arr.append(unicode("cac ban rat vui phai khong",encoding='utf-8'))
#
# arr.append("xin chao cac ban")
# arr.append("rat vui gap cac ban cac ban")
# arr.append("cac ban rat vui phai khong cac ban cac ban cac ban")
# arr.append("cac ban rat vui phai khong cac ban")
# print "------------- "
# n = ngrams_array(arr, 2 )
# # print "n", n
#
# for x in arr:
#   print ngrams2(x, 3)
# print "NNNNNNNNNNNNNN"
# out = {}
# for x in arr:
#   a = ngrams2(x,3) # moi a la 1 tu dien
#   for x in a:
#     out.setdefault(x,0)
#     out[x] +=1
# print "out", out
#
#
# d = build_dict(arr,2)
# print "aaaaaa ", d
#
# a=['i go','go to','a a']
# d={}
# for x in range(len(a)):
#     d.setdefault(a[x],x)
# print d
# chuoi=[]
# for i in a:
#     x = d.get(i)
#     if x==0:
#         chuoi.append(i)
#         # print " ".join(chuoi)
#     for j in a:
#         y = d.get(j)
#         if y==x+1:
#             z = j.split(' ')
#             print z
#             chuoi.append(z[1])
#
# print "chuoi ",chuoi
# print " ".join(chuoi)
#
# d_arr = ngrams_array(arr,2)
# print "ddd ",d_arr
# col4=[]
# for q in arr:
#     q1 = [' '.join(x) for x in ngrams(q, 1)]# q1:mang cac 1-grams
#     q2 = [' '.join(x) for x in ngrams(q, 2)]  # q2: mang cac phan tu 2-grams
#     print "q2 ",q2
#     q3 = [' '.join(x.replace(' ','_') for x in q2)]
#     print "q3  ",q3
#     y=q1+q3
#     z = " ".join(y)
#     print "yyyy ",z
#     col4.append(z)
# print "col4 ",col4
#
# a =['a b','c d']
# b = ['a b','c d','e f']

row  = np.array([0, 3, 1, 0])
# print row.shape
# X2 = np.array([[1., 0.], [2., 1.], [0., 0.]])
X = np.array([[1., 0.], [2., 1.], [0., 0.],[3., 5.], [0, 0]])
# print X.shape
y = np.array([0, 1, 2, 3, 4])
# y2 = np.array([0, 1, 2])
# X_sparse = coo_matrix(X)
# print X_sparse
X, y = resample(X, y, n_samples=7)

print X

print y