## Classify CLEF-IP2010 patents

# Classifiy clef-ip2010 patents using unigrams and bigrams features with different classifiers and report classification results

def vectorize_corpus(corpus,tokenizer,vocabulary,max_ngram_size):
    # tokenize text
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from clef_globals import min_df
    
    # generate corpus vectors
    vectorizer = CountVectorizer(min_df=min_df,tokenizer=tokenizer,ngram_range=(1,max_ngram_size),vocabulary=vocabulary,stop_words={})
    corpus_vectors = vectorizer.fit_transform(corpus)
    
    print 'vectorizing done, {0} terms vocabulary tokenized'.format(len(vectorizer.vocabulary_))
    
    # generate tfidf vectors
    transformer = TfidfTransformer()
    corpus_tfidf_vectors = transformer.fit_transform(corpus_vectors)

    return vectorizer, corpus_tfidf_vectors


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

def print_top_feature_names(features_weights, feature_names, file_prefix, label_names, max_features=1000):
    import os
    import numpy as np
    from sklearn.utils.extmath import density
    if features_weights!=None:
        print 'dimensionality: {0}'.format(features_weights.shape[1])
        print 'density: {0}'.format(density(features_weights))
        for i, category in enumerate(label_names):
            f = open(os.path.dirname(os.path.realpath(__file__))+'/results/linearsvc-tf1df1maxdf1.0-preprocessed-subject-shuffle-nopreprocess-4cv-featurename/'+str(file_prefix)+'_'+str(category),'w')
            f.write('number of features {0}\n'.format(features_weights.shape[1]))
            f.write('density: {0}\n\n'.format(density(features_weights)))
            top = np.argsort(features_weights[i])[max_features*-1:]
            for j in reversed(top):
                f.write(feature_names[j]+','+''.join('{0:0.2f}\n'.format(features_weights[i][j])))
                
                #print category,",".join(feature_names[top])

# reference: http://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification
def classify(x_train,y_train,x_test,y_test,max_labels):
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
    pred_proba = 1/(1+numpy.exp(-1*cls.decision_function(x_test)))
    # evaluate
    #pred_proba = cls.predict_proba(x_test)
    #print len(pred_proba[0]) # make sure it is 121
    threshold = 0.45#1/(1+numpy.exp(-1))
    y_pred = mlb.inverse_transform(get_max_n_pred(pred_proba, max_labels,threshold))
    #actual_labels = mlb.inverse_transform(y_test)
    return {'precision':metrics.precision_score(y_test, y_pred, average='micro'),
            'recall':metrics.recall_score(y_test, y_pred, average='micro'),
            'f1':metrics.f1_score(y_test, y_pred, average='micro')}


def classify_cv(x_train,y_train,x_test,y_test,max_labels):
    from sklearn.svm import LinearSVC
    from sklearn.cross_validation import StratifiedKFold
    from sklearn import metrics
    import numpy as np
    from numpy import mean
    import scipy
    
    # combine train and test vectors
    if x_test.shape[0]>0:
        x = scipy.sparse.vstack((x_train,x_test))
        y = np.array(y_train + y_test)
    else:
        x = x_train
        y = np.array(y_train)

    # split samples
    num_folds = 4
    folds = StratifiedKFold(y, num_folds)
    
    cls = LinearSVC()
    avg_features_weights = None
    acc_scores = []
    for train, test in folds:
        cls.fit(x[train], y[train])
        y_predicted = cls.predict(x[test])
        
        # add this fold's accueacy score
        acc_scores.append(metrics.accuracy_score(y[test],y_predicted))
        
        # add this fold's features weights
        if hasattr(cls,'coef_'):
            if avg_features_weights==None:
                avg_features_weights = cls.coef_
            else:
                avg_features_weights += cls.coef_
        
    avg_features_weights = avg_features_weights/num_folds

    print 'accuracy: ',acc_scores    
    return {'features_weights':avg_features_weights, 'accuracy':round(mean(acc_scores)*100,1)}


def test_lemmatized_unigrams(corpus_train_data,corpus_test_data,vocabulary_src,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from commons.lemmatizing_tokenizer import LemmaTokenizer
    from commons.lemmatizing_tokenizer import RawLemmaTokenizer
    from clef_globals import min_df, min_tf, test_set_size, max_labels
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
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       test_set_size,max_labels)
    
    print vocabulary_tbl_name,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


def run_experiment(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal=False,
            use_chi_features=False,preprocess_tokens='raw',ngram_size=1,knowledge_set=None):
    import numpy as np
    from commons.lemmatizing_tokenizer import RawTokenizer
    from commons.lemmatizing_tokenizer import RawLemmaTokenizer
    from commons.stemming_tokenizer import RawStemmingTokenizer
    from ng20_globals import max_labels,min_df,min_tf
    from ng20_vocabulary_loader import load_vocabulary
    from ng20_vocabulary_loader import load_common_vocabulary
    from commons.globals import knowledge_dic
        
    if with_stopwords_removal==False:
        stopwords_pattern = ''
    else:
        stopwords_pattern = '_stopwords'
    if use_chi_features==False:
        chi_features_pattern = ''
    else:
        chi_features_pattern = '_chi'

    if preprocess_tokens=='lemmatize':
        raw_tokens_pattern = '_raw_lemmas'
        tokenizer = RawLemmaTokenizer()
    elif preprocess_tokens=='stem':
        raw_tokens_pattern = '_raw_stems'
        tokenizer = RawStemmingTokenizer()
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawTokenizer()
    
    if ngram_size==1:
        ngram_pattern = '_unigrams'
    elif ngram_size==2:
        ngram_pattern = '_bigrams'

    # load vocabulary
    if knowledge_set!=None:
        vocabulary_tbl_name1 = 'ng20{0}{1}_unigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
        vocabulary_tbl_name2 = 'ng20{0}{1}_bigrams{2}_df{3}_tf{4}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    
        vocabulary_tbl_intersect = knowledge_dic[knowledge_set.lower()]
        vocabulary = load_common_vocabulary(vocabulary_tbl_name1,vocabulary_tbl_name2,vocabulary_tbl_intersect,'stem')    
        vocabulary_tbl_name = vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect

    else:
        vocabulary_tbl_name = 'ng20{0}{1}{2}{3}_df{4}_tf{5}'.format(raw_tokens_pattern,chi_features_pattern,ngram_pattern,stopwords_pattern,min_df,min_tf)
        vocabulary = load_vocabulary(vocabulary_tbl_name)

    # generate tfidf vectors
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,ngram_size)
    
    # classify & evaluate    
    results = classify_cv(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)

    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name, label_names)
    
    print vocabulary_tbl_name,' --> ','accuracy',results['accuracy']#print vocabulary_tbl_name,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


def test_lemmatized_bigrams(corpus_train_data,corpus_test_data,vocabulary_src,with_stopwords_removal,use_chi_features,use_raw_tokens):
    from commons.lemmatizing_tokenizer import LemmaTokenizer
    from commons.lemmatizing_tokenizer import RawLemmaTokenizer
    from clef_globals import min_df, min_tf, test_set_size, max_labels
    from clef_vocabulary_loader import load_vocabulary
    
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
    vocabulary_tbl_name = 'clef_2010_{0}{1}_lemmas{2}_bigrams{3}_df{4}_tf{5}'.format(vocabulary_src,raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
    vocabulary = load_vocabulary(vocabulary_tbl_name)

    # generate tfidf vectors
    corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       test_set_size,max_labels)
    
    print vocabulary_tbl_name,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[ ]:

def test_lemmatized_bigrams_with_LSA(corpus_train_data,corpus_test_data,vocabulary_src,with_stopwords_removal,use_chi_features,use_raw_tokens,num_components):
    from commons.lemmatizing_tokenizer import LemmaTokenizer
    from commons.lemmatizing_tokenizer import RawLemmaTokenizer
    from clef_globals import min_df, min_tf, test_set_size, max_labels
    from clef_vocabulary_loader import load_vocabulary
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
    vocabulary_tbl_name = 'clef_2010_{0}{1}_lemmas{2}_bigrams{3}_df{4}_tf{5}'.format(vocabulary_src,raw_tokens_pattern,chi_features_pattern,stopwords_pattern,min_df,min_tf)
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
                       test_set_size,max_labels)
    
    print 'LSA ^' , vocabulary_tbl_name,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


def test_lemmatized_bigrams_unigrams(bigrams_src,corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from commons.lemmatizing_tokenizer import LemmaTokenizer
    from commons.lemmatizing_tokenizer import RawLemmaTokenizer
    from clef_globals import max_labels,min_df,min_tf
    from clef_vocabulary_loader import load_common_vocabulary_extend_unigrams
    
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
    
    if len(bigrams_src)==1:
        vocabulary_tbl_intersect = '{0}_bigrams'.format(bigrams_src[0])
    else:
        vocabulary_tbl_intersect = '{0}_'.format(bigrams_src[0])
        for i in range(len(bigrams_src)-1):
            vocabulary_tbl_intersect = '{0}{1}_'.format(vocabulary_tbl_intersect,bigrams_src[i+1])
        vocabulary_tbl_intersect = '{0}bigrams_vw'.format(vocabulary_tbl_intersect)
        
    vocabulary = load_common_vocabulary_extend_unigrams(vocabulary_tbl_name,vocabulary_tbl_intersect,'lemma')
    print 'done loading vocabulary'
    
    # generate tfidf vectors
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name+'_'+vocabulary_tbl_intersect, label_names)
    
    print vocabulary_tbl_name,'^',vocabulary_tbl_intersect,'(extended unigrams) --> ','accuracy ',results['accuracy']#print vocabulary_tbl_name,'^',vocabulary_tbl_intersect,'(extended unigrams) --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


def test_stemmed_bigrams_unigrams(bigrams_src,corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from commons.stemming_tokenizer import StemmingTokenizer
    from commons.stemming_tokenizer import RawStemmingTokenizer
    from clef_globals import max_labels,min_df,min_tf
    from clef_vocabulary_loader import load_common_vocabulary_extend_unigrams
    
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
    
    if len(bigrams_src)==1:
        vocabulary_tbl_intersect = '{0}_bigrams'.format(bigrams_src[0])
    else:
        vocabulary_tbl_intersect = '{0}_'.format(bigrams_src[0])
        for i in range(len(bigrams_src)-1):
            vocabulary_tbl_intersect = '{0}{1}_'.format(vocabulary_tbl_intersect,bigrams_src[i+1])
        vocabulary_tbl_intersect = '{0}bigrams_vw'.format(vocabulary_tbl_intersect)
        
    vocabulary = load_common_vocabulary_extend_unigrams(vocabulary_tbl_name,vocabulary_tbl_intersect,'stem')
    print 'done loading vocabulary'
    
    # generate tfidf vectors
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name+'_'+vocabulary_tbl_intersect, label_names)
    
    print vocabulary_tbl_name,'^',vocabulary_tbl_intersect,'(extended unigrams) --> ','accuracy ',results['accuracy']#print vocabulary_tbl_name,'^',vocabulary_tbl_intersect,'(extended unigrams) --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


def run(vocabulary_src):
    from clef_corpus_loader import load_corpus_and_labels
    from clef_corpus_loader import load_corpus_with_labels_mappings
    # from clef_vocabulary_builder import build
    
    # build()
    
    # load clef patents with class lables from DB using only vocabulary_src as main field for vocabulary (e.g., abstract, description, claims...)
    corpus_train_data = load_corpus_and_labels(vocabulary_src,'train')
    print 'done loading {0} train records and {1} labels.'.format(len(corpus_train_data['corpus']),len(corpus_train_data['labels_dic']))

    corpus_test_data = load_corpus_with_labels_mappings(vocabulary_src,'test',corpus_train_data['labels_dic'])
    print 'done loading {0} test records.'.format(len(corpus_test_data['corpus']))
    
    label_names = corpus_train_data['labels_arr']
    
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'raw',1,None)
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'raw',2,None)
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'raw',2,'w')
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'raw',2,'t')
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'raw',2,'g')
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'raw',2,'wtg')
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'raw',1,None)
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'raw',2,None)
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'raw',2,'w')
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'raw',2,'t')
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'raw',2,'g')
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'raw',2,'wtg')
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'lemmatize',1,None)
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'lemmatize',2,None)
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'lemmatize',2,'w')
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'lemmatize',2,'t')
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'lemmatize',2,'g')
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'lemmatize',2,'wtg')

    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'lemmatize',1,None)
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'lemmatize',2,None)
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'lemmatize',2,'w')
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'lemmatize',2,'t')
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'lemmatize',2,'g')
            
    run_experiment({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'lemmatize',2,'wtg')

if __name__ == "__main__" and __package__ is None:
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

vocabulary_src = ("abstract", "claims", "description")
for src in vocabulary_src:
	print "running experiment on {0}...".format(src)
	run(src)

print 'done!'


