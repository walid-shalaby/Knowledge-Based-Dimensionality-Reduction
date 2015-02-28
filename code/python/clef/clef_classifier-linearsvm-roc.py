
# coding: utf-8

## ROC for CLEF-IP2010 patents

####### ROC for clef-ip2010 patents

# In[ ]:

def vectorize_corpus(corpus,tokenizer,vocabulary,max_ngram_size):
    # tokenize text
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from clef_globals import *
    
    # generate corpus vectors
    vectorizer = CountVectorizer(min_df=min_df,tokenizer=tokenizer,ngram_range=(1,max_ngram_size),vocabulary=vocabulary,stop_words={})
    corpus_vectors = vectorizer.fit_transform(corpus)
    
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
def classify(labels,corpus_tfidf_vectors,test_size,max_labels,threshold):
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.cross_validation import train_test_split
    from sklearn import metrics
    import numpy
        
    # binarize the labels
    mlb = MultiLabelBinarizer()
    labels_binarized = mlb.fit_transform(labels)
    
    # train/test split
    #corpus_tfidf_vectors, labels_binarized = shuffle(corpus_tfidf_vectors, labels_binarized)
    print 'test size=0.2'
    x_train, x_test, y_train, y_test = train_test_split(corpus_tfidf_vectors, labels_binarized, test_size=0.2, random_state=1)
    
    # classify
    #cls = OneVsRestClassifier(LogisticRegression(class_weight='auto'))
    #cls = OneVsRestClassifier(LogisticRegression())
    #cls = OneVsRestClassifier(MultinomialNB(alpha=0.01))
    cls = OneVsRestClassifier(SVC(kernel='linear',probability=True,max_iter=1000))
    #cls = OneVsRestClassifier(SGDClassifier(loss='hinge',penalty='l2',max_iter=1000))
    #cls = OneVsRestClassifier(LinearSVC())
    cls.fit(x_train, y_train)
    pred_proba = 1/(1+numpy.exp(-1*cls.decision_function(x_test)))
    # evaluate
    #pred_proba = cls.predict_proba(x_test)
    #print len(pred_proba[0]) # make sure it is 121
    pred_labels = mlb.inverse_transform(get_max_n_pred(pred_proba, max_labels,threshold))
    actual_labels = mlb.inverse_transform(y_test)
    result = 'threshold: {0}, precision: {1}, recall: {2}, f1: {3}'.format(threshold,metrics.precision_score(actual_labels, pred_labels, average='micro'),metrics.recall_score(actual_labels, pred_labels, average='micro'),metrics.f1_score(actual_labels, pred_labels, average='micro'))    
    print result


# In[ ]:

def test_lemmatized_unigrams(corpus,labels,vocabulary_src,with_stopwords_removal,use_chi_features):
    import thread
    from lemmatizing_tokenizer import LemmaTokenizer
    from clef_globals import *
    from clef_vocabulary_loader import load_vocabulary
    
    max_ngram_size = 1
    tokenizer = LemmaTokenizer()
    
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'
    
    # load vocabulary
    vocabulary_tbl_name = 'clef_2010_{0}_lemmas{1}_unigrams{2}_df{3}_tf{4}'.format(vocabulary_src,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary = load_vocabulary(vocabulary_tbl_name)

    # generate tfidf vectors
    corpus_tfidf_vectors = vectorize_corpus(corpus,tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate
    classify(labels,corpus_tfidf_vectors,test_set_size,max_labels,0.5)
    for i in range(0,101):
        #thread.start_new_thread(classify, (labels,corpus_tfidf_vectors,test_set_size,max_labels,i/100.0))
        classify(labels,corpus_tfidf_vectors,test_set_size,max_labels,i/100.0)


# In[ ]:

def test(vocabulary_src):
    from clef_corpus_loader import load_corpus_with_labels
    
    # load clef patents with class lables from DB using only vocabulary_src as main field for vocabulary (e.g., abstract, description, claims...)
    corpus_data = load_corpus_with_labels(vocabulary_src)
    corpus = corpus_data['corpus']
    labels = corpus_data['labels']
    labels_dic = corpus_data['labels_dic']
    labels_arr = corpus_data['labels_arr']
    print 'done loading {0} records and {1} labels.'.format(len(corpus),len(labels_dic))

    test_lemmatized_unigrams(corpus,labels,vocabulary_src,True,True)


# In[ ]:

# test using abstracts vocabulary
test('abstract')    

