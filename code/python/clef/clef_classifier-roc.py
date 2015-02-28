
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
def classify(x_train,y_train,x_test,y_test,test_size,max_labels,threshold):
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.cross_validation import train_test_split
    from sklearn import metrics
    import numpy
        
    # binarize the labels
    mlb = MultiLabelBinarizer()
    y_train_binarized = mlb.fit_transform(y_train)
    
    # train/test split
    #corpus_tfidf_vectors, labels_binarized = shuffle(corpus_tfidf_vectors, labels_binarized)
    #x_train, x_test, y_train, y_test = train_test_split(corpus_tfidf_vectors, labels_binarized, test_size=test_size, random_state=1)
    
    # classify
    #cls = OneVsRestClassifier(LogisticRegression(class_weight='auto'))
    #cls = OneVsRestClassifier(LogisticRegression())
    #cls = OneVsRestClassifier(MultinomialNB(alpha=0.01))
    #cls = OneVsRestClassifier(SVC(kernel='linear',probability=True,max_iter=1000))
    cls = OneVsRestClassifier(LinearSVC())
    cls.fit(x_train, y_train_binarized)
    pred_proba = 1/(1+numpy.exp(-1*cls.decision_function(x_train)))
    # evaluate
    y_pred = mlb.inverse_transform(get_max_n_pred(pred_proba, max_labels,threshold))
    result = 'threshold: {0}, precision: {1}, recall: {2}, f1: {3}'.format(threshold,metrics.precision_score(y_train, y_pred, average='micro'),metrics.recall_score(y_train, y_pred, average='micro'),metrics.f1_score(y_train, y_pred, average='micro'))    
    print result


# In[ ]:

def test_lemmatized_unigrams(corpus_train_data,corpus_test_data,vocabulary_src,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from clef_globals import *
    from clef_vocabulary_loader import load_vocabulary
    
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
    vocabulary_tbl_name = 'clef_2010_{0}{1}_lemmas{2}_unigrams{3}_df{4}_tf{5}'.format(vocabulary_src,raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary = load_vocabulary(vocabulary_tbl_name)

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    # classify & evaluate
    for i in range(40,56):
        classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       test_set_size,max_labels,
                       i/100.0)


# In[ ]:

def test(vocabulary_src):
    from clef_corpus_loader import load_corpus_and_labels
    from clef_corpus_loader import load_corpus_with_labels_mappings
    
    # load clef patents with class lables from DB using only vocabulary_src as main field for vocabulary (e.g., abstract, description, claims...)
    corpus_train_data = load_corpus_and_labels(vocabulary_src,'train')
    print 'done loading {0} train records and {1} labels.'.format(len(corpus_train_data['corpus']),len(corpus_train_data['labels_dic']))

    corpus_test_data = load_corpus_with_labels_mappings(vocabulary_src,'test',corpus_train_data['labels_dic'])
    print 'done loading {0} test records.'.format(len(corpus_test_data['corpus']))
    
    stopwords_removal = False
    use_chi_features = False
    use_raw_tokens = True
    test_lemmatized_unigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                             {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                             vocabulary_src,stopwords_removal,use_chi_features,use_raw_tokens)    


# In[ ]:

# test using abstracts vocabulary
test('abstract')    


# In[ ]:



