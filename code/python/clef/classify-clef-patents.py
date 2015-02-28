
# coding: utf-8

## classify patents documents stored in sqlite DB

# In[122]:

import sklearn
print sklearn.__version__


# In[2]:

def get_cur_time():
    import time
    return time.strftime("%d/%m/%y %H:%m:%s")


# In[3]:

import sqlite3 as sqlitedb
from sklearn.feature_extraction.text import TfidfTransformer
from clef_globals import *

print get_cur_time()

# load patents from sqlite DB
corpus = []
labels = []
patents_query = 'select title, tags from patents group by lower(title) having id like \'%A1\' or id like \'%A2\''
con = sqlitedb.connect(db_path)
with con:
    cur = con.execute(patents_query)
    while True:
        patent = cur.fetchone()
        if patent==None:
            break
        corpus.append(patent[0])
        tags = patent[1].split(' ')
        if len(tags[len(tags)-1])==0:
            labels.append(tags[0:len(tags)-1])            
        else:
            labels.append(tags)        
#print corpus[0]
#print labels


# map labels into unique class names
labels_dic = {}
labels_arr = []
for i in range(len(labels)):
    for j in range(len(labels[i])):
        #x = labels[i][j]
        if labels[i][j] not in labels_dic:
            labels_dic[labels[i][j]] = len(labels_arr)
            labels_arr.append(labels[i][j])
            labels[i][j] = len(labels_arr)-1
        else:
            labels[i][j] = labels_dic[labels[i][j]]
        #print x,labels[i][j]

print len(labels_arr)
print len(corpus)

print get_cur_time()


# In[4]:

from nltk import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
class LemmaTokenizer(object):
    pattern = u'(?u)\b[a-z]+\-*[a-z]+|\b(?u)\b[a-z]\b'
    def __init__(self):
        self.tokenizer = RegexpTokenizer(pattern)
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in self.tokenizer.tokenize(doc)]


# In[5]:

# tokenize text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer

print get_cur_time()

min_term_freq = 3
                                           
vectorizer = CountVectorizer(min_df=2,tokenizer=LemmaTokenizer(),ngram_range=(1,1),stop_words={})
corpus_vectors = vectorizer.fit_transform(corpus)
print len(vectorizer.vocabulary_)
#print vectorizer.vocabulary_
#print vectorizer.stop_words_
#print corpus_vectors.todense()
corpus_vectors = corpus_vectors.tolil()
term_freq = corpus_vectors.sum(axis=0) # sum on culomns to obtain term frequencies
terms_to_remove = []
for k,v in vectorizer.vocabulary_.iteritems():
    #print k,v,term_freq[0,vectorizer.vocabulary_[k]]
    if(term_freq[0,vectorizer.vocabulary_[k]]<min_term_freq):
        corpus_vectors[:,v] = 0
        terms_to_remove.append(k)
  
#print corpus_vectors.todense()
print len(terms_to_remove)
for k in terms_to_remove:
    del vectorizer.vocabulary_[k]
    
#print vectorizer.vocabulary_
#print sum(corpus_vectors.getcol(vectorizer.vocabulary_[i])).todense()[0,0]>0
#print corpus_vectors[:,vectorizer.vocabulary_['cairo']]
#print corpus_vectors
print len(vectorizer.vocabulary_)

corpus_vectors = corpus_vectors.tocsr()
#print corpus_vectors.todense()
#corpus_vectors?
    
print len(vectorizer.vocabulary_)

print get_cur_time()


# In[6]:

#del corpus
#del vectorizer


# In[ ]:

# save vocabulary in DB for future use
import sqlite3 as sqlitedb
from clef_globals import *

print get_cur_time()

l = []
l.extend([i] for i in vectorizer.vocabulary_.keys())
con = sqlitedb.connect(db_path)
with con:
    con.execute('drop table if exists vocabulary_lemmatized_uni_df2_tf3')
    con.execute('create table vocabulary_lemmatized_uni_df2_tf3(term text)')
    con.executemany('insert into vocabulary_lemmatized_uni_df2_tf3(term) values(?)',l)
    
print get_cur_time()


# In[7]:

print get_cur_time()

# tokenize text
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer()#(min_df=0.000001,max_df=0.95)
#corpus = ['hi this is cairo','hi here in usa']
corpus_tfidf_vectors = transformer.fit_transform(corpus_vectors)

#print vectorizer
print corpus_tfidf_vectors.shape
#print corpus_tfidf_vectors

print get_cur_time()


# In[20]:

# given classifier predictions probabilities, return predictions with top n probabilities for each instance
import heapq
import numpy

def get_max_n_pred(pred_proba, n_pred):
    max_n_pred = numpy.ndarray(shape=pred_proba.shape)
    for i in range(len(pred_proba)):
        largest_n_proba = heapq.nlargest(n_pred,pred_proba[i])
        max_n_pred[i] = numpy.array(((pred_proba[i]>0.5) & (pred_proba[i]>=largest_n_proba[len(largest_n_proba)-1]) & 1))
        if max_n_pred[i].sum(axis=0)==0: # at least one label should be returned
            max_n_pred[i] = numpy.array(((pred_proba[i]>=max(pred_proba[i])) & 1))
    return max_n_pred


# In[21]:

# binarize the labels
from sklearn.preprocessing import MultiLabelBinarizer

print get_cur_time()

mlb = MultiLabelBinarizer()
labels_binarized = mlb.fit_transform(labels)
labels_binarized.shape

# classify
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(corpus_tfidf_vectors, labels_binarized, test_size=0.33, random_state=1)
cls = OneVsRestClassifier(LogisticRegression())
#cls = OneVsRestClassifier(MultinomialNB(alpha=0.01))
#cls = OneVsRestClassifier(SVC(kernel='linear',probability=True))
cls.fit(x_train, y_train)

# evaluate
#pred = cls.predict(x_test)
pred_proba = cls.predict_proba(x_test)
print len(pred_proba[0]) # make sure it is 121

# use only top 4 probabilities labels as maximum labels per patent is 4
max_labels = 4
pred_labels = mlb.inverse_transform(get_max_n_pred(pred_proba, max_labels))
actual_labels = mlb.inverse_transform(y_test)
# http://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification
print metrics.precision_score(actual_labels, pred_labels, average='micro')
print metrics.recall_score(actual_labels, pred_labels, average='micro')
print metrics.f1_score(actual_labels, pred_labels, average='micro')
#pred_probs = cls.predict_proba(x_test)

print get_cur_time()


# In[ ]:



