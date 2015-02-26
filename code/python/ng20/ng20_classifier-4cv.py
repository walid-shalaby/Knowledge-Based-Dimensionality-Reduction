
# coding: utf-8

## Classify 20ng docs

####### Classifiy 20ng docs using unigrams and bigrams features with different classifiers and report classification results

# In[1]:

def vectorize_corpus(corpus,tokenizer,vocabulary,max_ngram_size):
    # tokenize text
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from ng20_globals import min_df
    
    # generate corpus vectors
    vectorizer = CountVectorizer(min_df=min_df,tokenizer=tokenizer,ngram_range=(1,max_ngram_size),vocabulary=vocabulary,stop_words={})
    corpus_vectors = vectorizer.fit_transform(corpus)
    
    print 'vectorizing done, {0} terms vocabulary tokenized'.format(len(vectorizer.vocabulary_))
    
    # generate tfidf vectors
    transformer = TfidfTransformer()
    corpus_tfidf_vectors = transformer.fit_transform(corpus_vectors)

    return vectorizer, corpus_tfidf_vectors


# In[2]:

def vectorize_corpus_new(corpus,tokenizer,vocabulary,max_ngram_size):
    # tokenize text
    from sklearn.feature_extraction.text import TfidfVectorizer
    from ng20_globals import min_df
    
    # generate corpus vectors
    vectorizer = TfidfVectorizer(min_df=min_df,tokenizer=tokenizer,ngram_range=(1,max_ngram_size),vocabulary=vocabulary,stop_words={})
    corpus_tfidf_vectors = vectorizer.fit_transform(corpus)
    
    print 'vectorizing done, {0} terms vocabulary tokenized'.format(len(vectorizer.vocabulary_))
    
    return corpus_tfidf_vectors


# In[3]:

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


# In[4]:

def print_top_feature_names(features_weights, feature_names, file_prefix, label_names, max_features=1000):
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


# In[5]:

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
    import numpy as np
    from numpy import mean
    import scipy
    from sklearn.cross_validation import cross_val_score
    from sklearn.metrics import make_scorer
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.cross_validation import StratifiedKFold
    
    # combine train and test vectors
    #print 'merging training and testing samples x({0}),x({1})'.format(x_train.shape,x_test.shape)
    #print 'merging training and testing samples y({0}),y({1})'.format(len(y_train),len(y_test))
    if x_test.shape[0]>0:
        x = scipy.sparse.vstack((x_train,x_test))
        y = y_train + y_test
    else:
        x = x_train
        y = y_train

    #print 'merged into x({0})'.format(x.shape)
    #print 'merged into y({0})'.format(len(y))
    # binarize the labels
    #mlb = MultiLabelBinarizer()
    #y_train_binarized = mlb.fit_transform(y_train)
    
    # train/test split
    #corpus_tfidf_vectors, labels_binarized = shuffle(corpus_tfidf_vectors, labels_binarized)
    #x_train, x_test, y_train, y_test = train_test_split(corpus_tfidf_vectors, labels_binarized, test_size=test_size, random_state=1)
    
    # classify
    #cls = OneVsRestClassifier(LogisticRegression(class_weight='auto'))
    #cls = OneVsRestClassifier(LogisticRegression())
    #cls = MultinomialNB(alpha=0.01)
    #cls = OneVsRestClassifier(BernoulliNB()need binarize(x_train and x_test))
    #cls = OneVsRestClassifier(SVC(kernel='linear',probability=True,max_iter=1000))
    #cls = LinearSVC(dual=False,penalty='l1')
    cls = LinearSVC()
    acc_scores = cross_val_score(cls,x,y,scoring='accuracy',cv=4,n_jobs=-1)
    print 'accuracy scores = {0},{1:0.1f}'.format(acc_scores,mean(acc_scores)*100)
    #macro_p_scores = cross_val_score(cls,x,y,scoring=make_scorer(precision_score,average='macro'),cv=4,n_jobs=-1)
    #print 'macro precision scores = {0},{1:0.1f}'.format(macro_p_scores,mean(macro_p_scores)*100)
    #macro_r_scores = cross_val_score(cls,x,y,scoring=make_scorer(recall_score,average='macro'),cv=4,n_jobs=-1)
    #print 'macro recall scores = {0},{1:0.1f}'.format(macro_r_scores,mean(macro_r_scores)*100)
    #macro_f1_scores = cross_val_score(cls,x,y,scoring=make_scorer(f1_score,average='macro'),cv=4,n_jobs=-1)
    #print 'macro f1 scores = {0},{1:0.1f}'.format(macro_f1_scores,mean(macro_f1_scores)*100)
    #p_scores = cross_val_score(cls,x,y,scoring=make_scorer(precision_score,average='weighted'),cv=4,n_jobs=-1)
    #print 'weighted average precision scores = {0},{1:0.1f}'.format(p_scores,mean(p_scores)*100)
    #r_scores = cross_val_score(cls,x,y,scoring=make_scorer(precision_score,average='weighted'),cv=4,n_jobs=-1)
    #print 'weighted average recall scores = {0},{1:0.1f}'.format(r_scores,mean(r_scores)*100)
    #f1_scores = cross_val_score(cls,x,y,scoring=make_scorer(f1_score,average='weighted'),cv=4,n_jobs=-1)
    #print 'weighted f1 scores = {0},{1:0.1f}'.format(f1_scores,mean(f1_scores)*100)
   
    cls.fit(x,y)
    
    #return {'classifier':cls,
    #        'accuracy':round(mean(acc_scores)*100,1),
    #        'precision':round(mean(macro_p_scores)*100,1),
    #        'recall':round(mean(macro_r_scores)*100,1),
    #        'f1':round(mean(macro_f1_scores)*100,1)}
            
    return {'classifier':cls,
            'accuracy':round(mean(acc_scores)*100,1)}

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


# In[6]:

import sklearn
from sklearn.svm import LinearSVC
c = LinearSVC()
print sklearn.__version__


# In[7]:

knowledge_data = {'w':'wiki_bigrams','t':'wiktionary_bigrams','g':'google_bigrams',
                  'wt':'wiki_wiktionary_bigrams_vw','wg':'wiki_google_bigrams_vw',
                  'tg':'wiktionary_google_bigrams_vw','wtg':'wiki_wiktionary_google_bigrams_vw'}

def test_me(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal=False,
            use_chi_features=False,preprocess_tokens='raw',ngram_size=1,knowledge_set=None):
    import numpy as np
    from lemmatizing_tokenizer import RawTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from stemming_tokenizer import RawStemmingTokenizer
    from ng20_globals import max_labels,min_df,min_tf
    from ng20_vocabulary_loader import load_vocabulary
    from ng20_vocabulary_loader import load_common_vocabulary
        
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
    
        vocabulary_tbl_intersect = knowledge_data[knowledge_set.lower()]
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

'''def test_all_unigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from lemmatizing_tokenizer import RawTokenizer
    from ng20_globals import max_labels
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
        tokenizer = None
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawTokenizer()
    
    # load vocabulary
    vocabulary_tbl_name = 'ng20{0}{1}_unigrams{2}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern)
    print vocabulary_tbl_name
    vocabulary = load_vocabulary(vocabulary_tbl_name)

    # generate tfidf vectors
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name, label_names)
    
    print vocabulary_tbl_name,' --> ','accuracy',results['accuracy']#print vocabulary_tbl_name,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']

'''
# In[8]:

'''def test_lemmatized_unigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import max_labels,min_tf,min_df
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
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name, label_names)
    
    print vocabulary_tbl_name,' --> ','accuracy',results['accuracy']#print vocabulary_tbl_name,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']

'''
# In[9]:

'''def test_all_bigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from lemmatizing_tokenizer import RawTokenizer
    from ng20_globals import max_labels,min_df,min_tf
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
        tokenizer = None
    else:
        raw_tokens_pattern = '_raw'
        tokenizer = RawTokenizer()
    
    # load vocabulary
    vocabulary_tbl_name = 'ng20{0}{1}_bigrams{2}'.format(raw_tokens_pattern,chi_features_pattern,stopwords_pattern)
    vocabulary = load_vocabulary(vocabulary_tbl_name)

    # generate tfidf vectors
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)

    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name, label_names)
    
    print vocabulary_tbl_name,' --> ','accuracy',results['accuracy']#print vocabulary_tbl_name,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[10]:
'''
'''def test_lemmatized_bigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import max_labels,min_df,min_tf
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
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)

    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name, label_names)
    
    print vocabulary_tbl_name,' --> ','accuracy',results['accuracy']#print vocabulary_tbl_name,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[11]:
'''
def test_lemmatized_bigrams_with_LSA(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens,num_components):
    import numpy as np
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import max_labels,min_df,min_tf
    from ng20_vocabulary_loader import load_vocabulary
    from sklearn.decomposition import TruncatedSVD
    
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
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
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

    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name, 'lsa_'+label_names)
    
    print 'LSA ^' , vocabulary_tbl_name,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[12]:

'''def test_stemmed_unigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from stemming_tokenizer import StemmingTokenizer
    from stemming_tokenizer import RawStemmingTokenizer    
    from ng20_globals import max_labels,min_df,min_tf
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
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)

    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name, label_names)
    
    print vocabulary_tbl_name,' --> ','accuracy',results['accuracy']#print vocabulary_tbl_name,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[13]:
'''
'''def test_stemmed_bigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from stemming_tokenizer import StemmingTokenizer
    from stemming_tokenizer import RawStemmingTokenizer    
    from ng20_globals import max_labels,min_df,min_tf
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
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)

    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name, label_names)
        
    print vocabulary_tbl_name,' --> ','accuracy',results['accuracy']#print vocabulary_tbl_name,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']


# In[14]:
'''
'''def test_lemmatized_wiki_bigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import max_labels,min_df,min_tf
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
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)

    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name1+'_'+vocabulary_tbl_name2, label_names)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','accuracy ',results['accuracy']#print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']
'''

# In[15]:

def test_lemmatized_bigrams_unigrams(bigrams_src,corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import max_labels,min_df,min_tf
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


# In[16]:

'''def test_lemmatized_wiktionary_bigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import max_labels,min_df,min_tf
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
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name1+'_'+vocabulary_tbl_name2, label_names)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','accuracy ',results['accuracy']#print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']

'''
# In[17]:

'''def test_lemmatized_google_bigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import max_labels,min_df,min_tf
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
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name1+'_'+vocabulary_tbl_name2, label_names)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','accuracy ',results['accuracy']#print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']

'''
# In[18]:

'''def test_lemmatized_wiki_wiktionary_bigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import max_labels,min_df,min_tf
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
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name1+'_'+vocabulary_tbl_name1, label_names)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','accuracy ',results['accuracy']#print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']

'''
# In[19]:

'''def test_lemmatized_wiki_google_bigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import max_labels,min_df,min_tf
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
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name1+'_'+vocabulary_tbl_name1, label_names)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','accuracy ',results['accuracy']#print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']

'''
# In[20]:

'''def test_lemmatized_wiktionary_google_bigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import max_labels,min_df,min_tf
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
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name1+'_'+vocabulary_tbl_name1, label_names)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','accuracy ',results['accuracy']#print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']

'''
# In[21]:

'''def test_stemmed_wiki_wiktionary_bigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from stemming_tokenizer import StemmingTokenizer
    from stemming_tokenizer import RawStemmingTokenizer
    from ng20_globals import max_labels,min_df,min_tf
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
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name1+'_'+vocabulary_tbl_name2, label_names)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','accuracy ',results['accuracy']#print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']

'''
# In[22]:

'''def test_stemmed_wiki_google_bigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from stemming_tokenizer import StemmingTokenizer
    from stemming_tokenizer import RawStemmingTokenizer
    from ng20_globals import max_labels,min_df,min_tf
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
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name1+'_'+vocabulary_tbl_name2, label_names)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','accuracy ',results['accuracy']#print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']

'''
# In[23]:

'''def test_stemmed_wiktionary_google_bigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from stemming_tokenizer import StemmingTokenizer
    from stemming_tokenizer import RawStemmingTokenizer
    from ng20_globals import max_labels,min_df,min_tf
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
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name1+'_'+vocabulary_tbl_name2, label_names)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','accuracy ',results['accuracy']#print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']

'''
# In[24]:

'''def test_lemmatized_all_bigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from lemmatizing_tokenizer import LemmaTokenizer
    from lemmatizing_tokenizer import RawLemmaTokenizer
    from ng20_globals import max_labels,min_df,min_tf
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
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name1+'_'+vocabulary_tbl_name2, label_names)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','accuracy ',results['accuracy']#print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']

'''
# In[25]:

'''def test_stemmed_wiki_bigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from stemming_tokenizer import StemmingTokenizer
    from stemming_tokenizer import RawStemmingTokenizer
    from ng20_globals import max_labels,min_df,min_tf
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
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name1+'_'+vocabulary_tbl_name2, label_names)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','accuracy ',results['accuracy']#print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']

'''
# In[26]:

def test_stemmed_bigrams_unigrams(bigrams_src,corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from stemming_tokenizer import StemmingTokenizer
    from stemming_tokenizer import RawStemmingTokenizer
    from ng20_globals import max_labels,min_df,min_tf
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


# In[27]:

'''def test_stemmed_wiktionary_bigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from stemming_tokenizer import StemmingTokenizer
    from stemming_tokenizer import RawStemmingTokenizer
    from ng20_globals import max_labels,min_df,min_tf
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
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name1+'_'+vocabulary_tbl_name2, label_names)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','accuracy ',results['accuracy']#print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']

'''
# In[28]:

'''def test_stemmed_google_bigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from stemming_tokenizer import StemmingTokenizer
    from stemming_tokenizer import RawStemmingTokenizer
    from ng20_globals import max_labels,min_df,min_tf
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
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name1+'_'+vocabulary_tbl_name2, label_names)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','accuracy ',results['accuracy']#print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']
'''

# In[29]:

'''def test_stemmed_all_bigrams(corpus_train_data,corpus_test_data,label_names,with_stopwords_removal,use_chi_features,use_raw_tokens):
    import numpy as np
    from stemming_tokenizer import StemmingTokenizer
    from stemming_tokenizer import RawStemmingTokenizer
    from ng20_globals import max_labels,min_df,min_tf
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
    vectorizer, corpus_train_tfidf_vectors = vectorize_corpus(corpus_train_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    _, corpus_test_tfidf_vectors = vectorize_corpus(corpus_test_data['corpus'],tokenizer,vocabulary,max_ngram_size)
    
    # classify & evaluate    
    results = classify(corpus_train_tfidf_vectors,corpus_train_data['labels'],
                       corpus_test_tfidf_vectors,corpus_test_data['labels'],
                       max_labels)
    
    print_top_feature_names(results['features_weights'], np.asarray(vectorizer.get_feature_names()), vocabulary_tbl_name, label_names)
    
    print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','accuracy ',results['accuracy']#print vocabulary_tbl_name1,'^',vocabulary_tbl_name2,'^',vocabulary_tbl_intersect,' --> ','precision ',results['precision'],'recall ',results['recall'],'f1 ',results['f1']

'''
# In[30]:

def test():
    from ng20_corpus_loader import load_corpus_and_labels
    from ng20_corpus_loader import load_corpus_with_labels_mappings
    from ng20_corpus_loader import load_label_names
    # from ng20_vocabulary_builder import build
    
    # build()
    
    # load 20ng docs with class lables from DB
    corpus_train_data = load_corpus_and_labels('train')
    print 'done loading {0} train records and {1} labels.'.format(len(corpus_train_data['corpus']),len(corpus_train_data['labels_dic']))

    corpus_test_data = load_corpus_with_labels_mappings('test',corpus_train_data['labels_dic'])
    print 'done loading {0} test records.'.format(len(corpus_test_data['corpus']))
    
    label_names = load_label_names()
    
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'raw',1,None)
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'raw',2,None)
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'raw',2,'w')
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'raw',2,'t')
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'raw',2,'g')
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'raw',2,'wtg')
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'raw',1,None)
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'raw',2,None)
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'raw',2,'w')
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'raw',2,'t')
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'raw',2,'g')
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'raw',2,'wtg')
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'lemmatize',1,None)
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'lemmatize',2,None)
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'lemmatize',2,'w')
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'lemmatize',2,'t')
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'lemmatize',2,'g')
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,False,False,'lemmatize',2,'wtg')

    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'lemmatize',1,None)
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'lemmatize',2,None)
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'lemmatize',2,'w')
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'lemmatize',2,'t')
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'lemmatize',2,'g')
            
    test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
            {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
            label_names,True,False,'lemmatize',2,'wtg')
            
    #test_me({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
    #        {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
    #        label_names,False,False,'stem',2,'wtg')
'''def test():
    from ng20_corpus_loader import load_corpus_and_labels
    from ng20_corpus_loader import load_corpus_with_labels_mappings
    from ng20_corpus_loader import load_label_names
    
    # load 20ng docs with class lables from DB
    corpus_train_data = load_corpus_and_labels('train')
    print 'done loading {0} train records and {1} labels.'.format(len(corpus_train_data['corpus']),len(corpus_train_data['labels_dic']))

    corpus_test_data = load_corpus_with_labels_mappings('test',corpus_train_data['labels_dic'])
    print 'done loading {0} test records.'.format(len(corpus_test_data['corpus']))
    
    label_names = load_label_names()
    
    stopwords_removal_mask = 1
    chi_features_mask = 2
    raw_tokens_mask = 4
    for i in range(4,6): # test w/o stopword removal and w/o chi square features and w/o raw tokens
        stopwords_removal = i&stopwords_removal_mask==stopwords_removal_mask
        use_chi_features = i&chi_features_mask==chi_features_mask
        use_raw_tokens = i&raw_tokens_mask==raw_tokens_mask
        
        test_all_unigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_all_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_unigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        #test_lemmatized_bigrams_with_LSA({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
        #                         {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
        #                         label_names,stopwords_removal,use_chi_features,use_raw_tokens,113240)
        
        test_lemmatized_wiki_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_bigrams_unigrams(['wiki'],{'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_wiktionary_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_bigrams_unigrams(['wiktionary'],{'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_google_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_bigrams_unigrams(['google'],{'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_wiki_wiktionary_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_bigrams_unigrams(['wiki','wiktionary'],{'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_wiki_google_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_bigrams_unigrams(['wiki','google'],{'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_wiktionary_google_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_bigrams_unigrams(['wiktionary','google'],{'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_all_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_lemmatized_bigrams_unigrams(['wiki','wiktionary','google'],{'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_unigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_wiki_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_bigrams_unigrams(['wiki'],{'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_wiktionary_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_bigrams_unigrams(['wiktionary'],{'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_google_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_bigrams_unigrams(['google'],{'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_wiki_wiktionary_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_bigrams_unigrams(['wiki','wiktionary'],{'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_wiki_google_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_bigrams_unigrams(['wiki','google'],{'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_wiktionary_google_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_bigrams_unigrams(['wiktionary','google'],{'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_all_bigrams({'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)
        
        test_stemmed_bigrams_unigrams(['wiki','wiktionary','google'],{'corpus':corpus_train_data['corpus'],'labels':corpus_train_data['labels']},
                                 {'corpus':corpus_test_data['corpus'],'labels':corpus_test_data['labels']},
                                 label_names,stopwords_removal,use_chi_features,use_raw_tokens)

'''
# In[31]:

# test using all vocabulary
test()

print 'done!'

