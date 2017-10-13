# -*- encoding: utf8 -*-
import math


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
    print idf_values
    return idf_values   # list idf cua tat ca cac tu trong tat ca cac van ban

def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:  # Neu tu do khong xuat hien trong document
        return 0
    return 1 + math.log(count)

def tfidf(documents):
    tokenized_documents = [tokenize(d) for d in documents]  # list cac list cua nhieu document  [[w1...wn]...[w1...wn]]
    print tokenized_documents
    idf = inverse_document_frequencies(tokenized_documents)
    print len(idf)
    tfidf_documents = []
    for document in tokenized_documents:    # document la 1 list chua cac tu trong 1 doc
        doc_tfidf = []
        for term in idf.keys(): # term la 1 word
            # term = term.decode('utf-8')
            # term = unicode(term, encoding='utf-8')
            tf = sublinear_term_frequency(term, document)
            print term,tf,idf[term]
            doc_tfidf.append(tf * idf[term])    # list cac tfidf cua tat ca cac tu trong tat ca cac vb doi voi 1 vb
        print len(doc_tfidf)
        tfidf_documents.append(doc_tfidf)
    # print tfidf_documents
    return tfidf_documents

# def tfidf(documents):
#     tokenized_documents = [tokenize(d) for d in documents]  # list cac list cua nhieu document  [[w1...wn]...[w1...wn]]
#     idf = inverse_document_frequencies(tokenized_documents)
#     tfidf_documents = []
#     for document in tokenized_documents:
#         doc_tfidf = []
#         for term in idf.keys():
#             tf = sublinear_term_frequency(term, document)
#             doc_tfidf.append(tf * idf[term])
#         tfidf_documents.append(doc_tfidf)
#     return tfidf_documents

document_0 = "Việt Nam has a strong economy that is growing at a rapid pace."
document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_6 = "Vladimir Putin is riding a horse while hunting deer. Is he crazy?"
tokenize = lambda doc: doc.lower().split(" ")
# print tokenize
all_documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]

tokenized_documents = [tokenize(d) for d in all_documents]  # list cac list cua nhieu document  [[w1...wn]...[w1...wn]]
# print tokenized_documents
# idf = inverse_document_frequencies(tokenized_documents)
tfidfs = tfidf(all_documents)
print tfidfs