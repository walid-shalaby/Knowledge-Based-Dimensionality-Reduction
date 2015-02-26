
# coding: utf-8

## Classify 20ng docs

####### Classifiy 20ng docs using unigrams and bigrams features with different classifiers and report classification results

# In[ ]:

def vectorize_corpus(corpus,tokenizer,vocabulary,max_ngram_size):
    # tokenize text
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from ng20_globals import *
    
    # generate corpus vectors
    vectorizer = CountVectorizer(min_df=min_df,tokenizer=tokenizer,ngram_range=(1,max_ngram_size),vocabulary=vocabulary,stop_words={})
    corpus_vectors = vectorizer.fit_transform(corpus)
    
    print 'vectorizing done, {0} terms vocabulary tokenized'.format(len(vectorizer.vocabulary_))
    
    # generate tfidf vectors
    transformer = TfidfTransformer()
    corpus_tfidf_vectors = transformer.fit_transform(corpus_vectors)

    return corpus_tfidf_vectors


# In[ ]:

# given classifier predictions probabilities, return predictions with top n probabilities > 0.5 for each instance or greatest one if all are <=0.5
def get_max_n_pred(pred_proba, n_pred, threshold):
    import heapq
    import numpy
    max_n_pred = numpy.ndarray(shape=pred_proba.shape)
    for i in range(len(pred_proba)):
        largest_n_proba = heapq.nlargest(n_pred,pred_proba[i])
        max_n_pred[i] = numpy.array(((pred_proba[i]>threshold) & (pred_proba[i]>=largest_n_proba[len(largest_n_proba)-1]) & 1))
        if max_n_pred[i].sum(axis=0)==0: # at least one label should be returned
            max_n_pred[i] = numpy.array(((pred_proba[i]>=max(pred_proba[i])) & 1))
    return max_n_pred


# In[ ]:

# reference: http://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification
def classify(x_train,y_train,x_test,y_test,max_labels):
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.preprocessing import binarize
    from sklearn.cross_validation import train_test_split
    from sklearn import metrics
    import numpy
        
    # binarize the labels
    #mlb = MultiLabelBinarizer()
    #y_train_binarized = mlb.fit_transform(y_train)
    
    # train/test split
    #corpus_tfidf_vectors, labels_binarized = shuffle(corpus_tfidf_vectors, labels_binarized)
    #x_train, x_test, y_train, y_test = train_test_split(corpus_tfidf_vectors, labels_binarized, test_size=test_size, random_state=1)
    
    # classify
    #cls = OneVsRestClassifier(LogisticRegression(class_weight='auto'))
    #cls = OneVsRestClassifier(LogisticRegression())
    cls = MultinomialNB(alpha=0.01)
    #cls = OneVsRestClassifier(BernoulliNB()need binarize(x_train and x_test))
    #cls = OneVsRestClassifier(SVC(kernel='linear',probability=True,max_iter=1000))
    #cls = LinearSVC()
    cls.fit(x_train, y_train)
    y_train_pred = cls.predict(x_train)
    print metrics.f1_score(y_train, y_train_pred, average='micro')
    #pred_proba = 1/(1+numpy.exp(-1*cls.decision_function(x_test)))
    #threshold = 0.45#1/(1+numpy.exp(-1))
    #y_pred = mlb.inverse_transform(get_max_n_pred(pred_proba, max_labels,threshold))
    #y_pred = mlb.inverse_transform(cls.predict(x_test))
    y_pred = cls.predict(x_test)
    # evaluate
    #pred_proba = cls.predict_proba(x_test)
    #print len(pred_proba[0]) # make sure it is 121
    #actual_labels = mlb.inverse_transform(y_test)
    #precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred, pos_label=0)
    #print precision
    #print recall
    #print thresholds
    return {'precision':metrics.precision_score(y_test, y_pred, average='micro'),
            'recall':metrics.recall_score(y_test, y_pred, average='micro'),
            'f1':metrics.f1_score(y_test, y_pred, average='micro')}


# In[ ]:

import sklearn
from sklearn.svm import LinearSVC
c = LinearSVC()
print sklearn.__version__


# In[ ]:

def test_lemmatized_unigrams(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import *
    from ng20_vocabulary_loader import load_vocabulary
    
    max_ngram_size = 1
    
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = LemmaTokenizer()
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawLemmaTokenizer()
    
    # load vocabulary
    vocabulary_tbl_name = 'ng20{0}_lemmas{1}_unigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary = load_vocabulary(vocabulary_tbl_name)

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print vocabulary_tbl_name,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_lemmatized_bigrams(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import *
    from ng20_vocabulary_loader import load_vocabulary
    
    max_ngram_size = 2
    
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = LemmaTokenizer()
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawLemmaTokenizer()
    
    # load vocabulary
    vocabulary_tbl_name = 'ng20{0}_lemmas{1}_bigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary = load_vocabulary(vocabulary_tbl_name)

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print vocabulary_tbl_name,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_lemmatized_bigrams_with_LSA(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens,num_components):
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import *
    from ng20_vocabulary_loader import load_vocabulary
    from sklearn.decomposition import TruncatedSVD
    from scipy import sparse
    import numpy
    
    max_ngram_size = 2
    
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = LemmaTokenizer()
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawLemmaTokenizer()
    
    # load vocabulary
    vocabulary_tbl_name = 'ng20{0}_lemmas{1}_bigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary = load_vocabulary(vocabulary_tbl_name)

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # apply LSA
    #print numpy.max(corpus_train_tfidf_vectors)
    #print numpy.min(corpus_train_tfidf_vectors)
    lsa = TruncatedSVD(n_components=num_components)
    lsa.fit(corpus_train_tfidf_vectors)
    #corpus_train_tfidf_vectors = numpy.dot(corpus_train_tfidf_vectors,pca.components_.transpose())
    corpus_train_tfidf_vectors = lsa.transform(corpus_train_tfidf_vectors)
    corpus_test_tfidf_vectors = lsa.transform(corpus_test_tfidf_vectors)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print 'LSA ^' , vocabulary_tbl_name,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_stemmed_unigrams(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from stemming_tokenizer import StemmingTokenizer
    from stemming_tokenizer import RawStemmingTokenizer    
    from ng20_globals import *
    from ng20_vocabulary_loader import load_vocabulary
    
    max_ngram_size = 1
    
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = StemmingTokenizer()    
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawStemmingTokenizer()    
    
    # load vocabulary
    vocabulary_tbl_name = 'ng20{0}_stems{1}_unigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)    
    vocabulary = load_vocabulary(vocabulary_tbl_name)

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print vocabulary_tbl_name,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_stemmed_bigrams(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from stemming_tokenizer import StemmingTokenizer
    from stemming_tokenizer import RawStemmingTokenizer    
    from ng20_globals import *
    from ng20_vocabulary_loader import load_vocabulary
    
    max_ngram_size = 2
        
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = StemmingTokenizer()
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawStemmingTokenizer()
    
    # load vocabulary
    vocabulary_tbl_name = 'ng20{0}_stems{1}_bigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary = load_vocabulary(vocabulary_tbl_name)

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print vocabulary_tbl_name,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_lemmatized_wiki_bigrams(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import *
    from ng20_vocabulary_loader import load_common_vocabulary
    
    max_ngram_size = 2
        
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = LemmaTokenizer()
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawLemmaTokenizer()
    
    # load vocabulary
    vocabulary_tbl_name1 = 'ng20{0}_lemmas{1}_unigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary_tbl_name2 = 'ng20{0}_lemmas{1}_bigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    
    vocabulary_tbl_intersect = 'wiki_bigrams'
    vocabulary = load_common_vocabulary(vocabulary_tbl_name1,vocabulary_tbl_name2,vocabulary_tbl_intersect,'lemma')
    print 'done loading vocabulary'
    
    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_lemmatized_wiki_bigrams_unigrams(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import *
    from ng20_vocabulary_loader import load_common_vocabulary_extend_unigrams
    
    max_ngram_size = 2
        
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = LemmaTokenizer()
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawLemmaTokenizer()
    
    # load vocabulary
    vocabulary_tbl_name = 'ng20{0}_lemmas{1}_bigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    
    vocabulary_tbl_intersect = 'wiki_bigrams'
    vocabulary = load_common_vocabulary_extend_unigrams(vocabulary_tbl_name,vocabulary_tbl_intersect,'lemma')
    print 'done loading vocabulary'
    
    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print vocabulary_tbl_name,'^',vocabulary_tbl_intersect,'(extended unigrams) --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_lemmatized_wiktionary_bigrams(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import *
    from ng20_vocabulary_loader import load_common_vocabulary
    
    max_ngram_size = 2
    
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = LemmaTokenizer()    
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawLemmaTokenizer()    
    
    # load vocabulary
    vocabulary_tbl_name1 = 'ng20{0}_lemmas{1}_unigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary_tbl_name2 = 'ng20{0}_lemmas{1}_bigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    
    vocabulary_tbl_intersect = 'wiktionary_bigrams'
    vocabulary = load_common_vocabulary(vocabulary_tbl_name1,vocabulary_tbl_name2,vocabulary_tbl_intersect,'lemma')
    print 'done loading vocabulary'

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_lemmatized_google_bigrams(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import *
    from ng20_vocabulary_loader import load_common_vocabulary
    
    max_ngram_size = 2
        
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = LemmaTokenizer()
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawLemmaTokenizer()
    
    # load vocabulary
    vocabulary_tbl_name1 = 'ng20{0}_lemmas{1}_unigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary_tbl_name2 = 'ng20{0}_lemmas{1}_bigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    
    vocabulary_tbl_intersect = 'google_bigrams'
    vocabulary = load_common_vocabulary(vocabulary_tbl_name1,vocabulary_tbl_name2,vocabulary_tbl_intersect,'lemma')    

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_lemmatized_wiki_wiktionary_bigrams(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import *
    from ng20_vocabulary_loader import load_common_vocabulary
    
    max_ngram_size = 2
        
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = LemmaTokenizer()
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawLemmaTokenizer()
    
    # load vocabulary
    vocabulary_tbl_name1 = 'ng20{0}_lemmas{1}_unigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary_tbl_name2 = 'ng20{0}_lemmas{1}_bigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    
    vocabulary_tbl_intersect = 'wiki_wiktionary_bigrams_vw'
    vocabulary = load_common_vocabulary(vocabulary_tbl_name1,vocabulary_tbl_name2,vocabulary_tbl_intersect,'lemma')

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_lemmatized_wiki_google_bigrams(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import *
    from ng20_vocabulary_loader import load_common_vocabulary
    
    max_ngram_size = 2
    
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = LemmaTokenizer()
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawLemmaTokenizer()
    
    # load vocabulary
    vocabulary_tbl_name1 = 'ng20{0}_lemmas{1}_unigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary_tbl_name2 = 'ng20{0}_lemmas{1}_bigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    
    vocabulary_tbl_intersect = 'wiki_google_bigrams_vw'
    vocabulary = load_common_vocabulary(vocabulary_tbl_name1,vocabulary_tbl_name2,vocabulary_tbl_intersect,'lemma')

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_lemmatized_wiktionary_google_bigrams(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import *
    from ng20_vocabulary_loader import load_common_vocabulary
    
    max_ngram_size = 2
        
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = LemmaTokenizer()
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawLemmaTokenizer()
    
    # load vocabulary
    vocabulary_tbl_name1 = 'ng20{0}_lemmas{1}_unigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary_tbl_name2 = 'ng20{0}_lemmas{1}_bigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    
    vocabulary_tbl_intersect = 'wiktionary_google_bigrams_vw'
    vocabulary = load_common_vocabulary(vocabulary_tbl_name1,vocabulary_tbl_name2,vocabulary_tbl_intersect,'lemma')

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_stemmed_wiki_wiktionary_bigrams(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from stemming_tokenizer import StemmingTokenizer
    from stemming_tokenizer import RawStemmingTokenizer
    from ng20_globals import *
    from ng20_vocabulary_loader import load_common_vocabulary
    
    max_ngram_size = 2
        
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = StemmingTokenizer()
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawStemmingTokenizer()
    
    # load vocabulary
    vocabulary_tbl_name1 = 'ng20{0}_stems{1}_unigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary_tbl_name2 = 'ng20{0}_stems{1}_bigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    
    vocabulary_tbl_intersect = 'wiki_wiktionary_bigrams_vw'
    vocabulary = load_common_vocabulary(vocabulary_tbl_name1,vocabulary_tbl_name2,vocabulary_tbl_intersect,'stem')

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_stemmed_wiki_google_bigrams(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from stemming_tokenizer import StemmingTokenizer
    from stemming_tokenizer import RawStemmingTokenizer
    from ng20_globals import *
    from ng20_vocabulary_loader import load_common_vocabulary
    
    max_ngram_size = 2
        
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = StemmingTokenizer()
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawStemmingTokenizer()
    
    # load vocabulary
    vocabulary_tbl_name1 = 'ng20{0}_stems{1}_unigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary_tbl_name2 = 'ng20{0}_stems{1}_bigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    
    vocabulary_tbl_intersect = 'wiki_google_bigrams_vw'
    vocabulary = load_common_vocabulary(vocabulary_tbl_name1,vocabulary_tbl_name2,vocabulary_tbl_intersect,'stem')

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_stemmed_wiktionary_google_bigrams(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from stemming_tokenizer import StemmingTokenizer
    from stemming_tokenizer import RawStemmingTokenizer
    from ng20_globals import *
    from ng20_vocabulary_loader import load_common_vocabulary
    
    max_ngram_size = 2
        
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = StemmingTokenizer()
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawStemmingTokenizer()
    
    # load vocabulary
    vocabulary_tbl_name1 = 'ng20{0}_stems{1}_unigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary_tbl_name2 = 'ng20{0}_stems{1}_bigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    
    vocabulary_tbl_intersect = 'wiktionary_google_bigrams_vw'
    vocabulary = load_common_vocabulary(vocabulary_tbl_name1,vocabulary_tbl_name2,vocabulary_tbl_intersect,'stem')

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_lemmatized_all_bigrams(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import *
    from ng20_vocabulary_loader import load_common_vocabulary
    
    max_ngram_size = 2
        
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = LemmaTokenizer()
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawLemmaTokenizer()
    
    # load vocabulary
    vocabulary_tbl_name1 = 'ng20{0}_lemmas{1}_unigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary_tbl_name2 = 'ng20{0}_lemmas{1}_bigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    
    vocabulary_tbl_intersect = 'wiki_wiktionary_google_bigrams_vw'
    vocabulary = load_common_vocabulary(vocabulary_tbl_name1,vocabulary_tbl_name2,vocabulary_tbl_intersect,'lemma')

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_stemmed_wiki_bigrams(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from stemming_tokenizer import StemmingTokenizer
    from stemming_tokenizer import RawStemmingTokenizer
    from ng20_globals import *
    from ng20_vocabulary_loader import load_common_vocabulary
    
    max_ngram_size = 2
        
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = StemmingTokenizer()    
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawStemmingTokenizer()    
    
    # load vocabulary
    vocabulary_tbl_name1 = 'ng20{0}_stems{1}_unigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary_tbl_name2 = 'ng20{0}_stems{1}_bigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    
    vocabulary_tbl_intersect = 'wiki_bigrams'
    vocabulary = load_common_vocabulary(vocabulary_tbl_name1,vocabulary_tbl_name2,vocabulary_tbl_intersect,'stem')    

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_stemmed_wiktionary_bigrams(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from stemming_tokenizer import StemmingTokenizer
    from stemming_tokenizer import RawStemmingTokenizer
    from ng20_globals import *
    from ng20_vocabulary_loader import load_common_vocabulary
    
    max_ngram_size = 2
        
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = StemmingTokenizer()
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawStemmingTokenizer()
    
    # load vocabulary
    vocabulary_tbl_name1 = 'ng20{0}_stems{1}_unigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary_tbl_name2 = 'ng20{0}_stems{1}_bigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    
    vocabulary_tbl_intersect = 'wiktionary_bigrams'
    vocabulary = load_common_vocabulary(vocabulary_tbl_name1,vocabulary_tbl_name2,vocabulary_tbl_intersect,'stem')

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_stemmed_google_bigrams(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from stemming_tokenizer import StemmingTokenizer
    from stemming_tokenizer import RawStemmingTokenizer
    from ng20_globals import *
    from ng20_vocabulary_loader import load_common_vocabulary
    
    max_ngram_size = 2
        
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = StemmingTokenizer()
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawStemmingTokenizer()
    
    # load vocabulary
    vocabulary_tbl_name1 = 'ng20{0}_stems{1}_unigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary_tbl_name2 = 'ng20{0}_stems{1}_bigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    
    vocabulary_tbl_intersect = 'google_bigrams'
    vocabulary = load_common_vocabulary(vocabulary_tbl_name1,vocabulary_tbl_name2,vocabulary_tbl_intersect,'stem')

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_stemmed_all_bigrams(corpus_train_data,corpus_test_data,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from stemming_tokenizer import StemmingTokenizer
    from stemming_tokenizer import RawStemmingTokenizer
    from ng20_globals import *
    from ng20_vocabulary_loader import load_common_vocabulary
    
    max_ngram_size = 2
    
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    if use_raw_tokens==False:
        raw_tokens_pattern = ''
        tokenizer = StemmingTokenizer()
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawStemmingTokenizer()
    
    # load vocabulary
    vocabulary_tbl_name1 = 'ng20{0}_stems{1}_unigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary_tbl_name2 = 'ng20{0}_stems{1}_bigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    
    vocabulary_tbl_intersect = 'wiki_wiktionary_google_bigrams_vw'
    vocabulary = load_common_vocabulary(vocabulary_tbl_name1,vocabulary_tbl_name2,vocabulary_tbl_intersect,'stem')

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test():
    from ng20_corpus_loader import load_corpus_and_labels
    from ng20_corpus_loader import load_corpus_with_labels_mappings
    
    # load 20ng docs with class lables from DB
    corpus_train_data = load_corpus_and_labels('train')
    print 'done loading {0} train records and {1} labels.'.format(len(corpus_train_data['corpus']),len(corpus_train_data['labels_dic']))

    corpus_test_data = load_corpus_with_labels_mappings('test',corpus_train_data['labels_dic'])
    print 'done loading {0} test records.'.format(len(corpus_test_data['corpus']))
    
    stopwords_removal_mask = 1
    chi_features_mask = 2
    raw_tokens_mask = 4
    for i in range(4,6): # test w/o stopword removal and w/o chi square features and w/o raw tokens
        stopwords_removal = i&stopwords_removal_mask==stopwords_removal_mask
        use_chi_features = i&chi_features_mask==chi_features_mask
        use_raw_tokens = i&raw_tokens_mask==raw_tokens_mask
        
        test_lemmatized_unigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 stopwords_removal,use_chi_features,use_raw_tokens)
        
        #test_lemmatized_bigrams_with_LSA({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
        #                         {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
        #                         stopwords_removal,use_chi_features,use_raw_tokens,113240)
        
        test_lemmatized_wiki_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_wiktionary_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_google_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_wiki_wiktionary_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_wiki_google_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_wiktionary_google_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_all_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_unigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_wiki_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_wiktionary_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_google_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_wiki_wiktionary_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_wiki_google_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_wiktionary_google_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_all_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 stopwords_removal,use_chi_features,use_raw_tokens)


# In[ ]:

# test using all vocabulary
test()

print 'done!'

