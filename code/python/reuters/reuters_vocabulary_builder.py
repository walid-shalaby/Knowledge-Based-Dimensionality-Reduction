
# coding: utf-8

## REUTERS 21578 Vocabulary Builder

####### Build stemmed and lemmatized vocabulary (unigrams + bigrams) from reuters 21578 corpus and store into DB

# In[ ]:

def build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf):    
    from sklearn.feature_extraction.text import CountVectorizer

    # tokenize text
    vectorizer = CountVectorizer(min_df=min_df,tokenizer=tokenizer,ngram_range=(1,max_ngram_size),stop_words=stop_words)
    corpus_vectors = vectorizer.fit_transform(corpus)

    # apply minimum term frequency threshold
    term_freq = corpus_vectors.sum(axis=0) # sum on culomns to obtain term frequencies
    terms_to_remove = []
    for k,v in vectorizer.vocabulary_.iteritems():
        if(term_freq[0,vectorizer.vocabulary_[k]]<min_tf):
            terms_to_remove.append(k)

    print 'removing ({0}) terms under tf threshold'.format(len(terms_to_remove))
    for k in terms_to_remove:
        del vectorizer.vocabulary_[k]

    return vectorizer.vocabulary_


# In[ ]:

def save_vocabulary(vocabulary,tbl_name):
    # save vocabulary in DB for future use
    import sqlite3 as sqlitedb
    from reuters_globals import *

    l = []
    l.extend([i] for i in vocabulary)
    con = sqlitedb.connect(db_path)
    with con:
        con.execute('drop table if exists {0}'.format(tbl_name))
        con.execute('create table {0}(term text)'.format(tbl_name))
        con.executemany('insert into {0}(term) values(?)'.format(tbl_name),l)


# In[ ]:

# build raw unigrams vocabulary
def build_raw_unigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    
    tokenizer = None
    stop_words = {}
    max_ngram_size = 1
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,1,1)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_unigrams'.format(vocabulary_src)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build unigrams vocabulary
def build_unigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    
    tokenizer = None
    stop_words = {}
    max_ngram_size = 1
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_unigrams_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build lemmatized unigrams vocabulary
# uses alphanumeric tokenizer
def build_raw_lemmatized_unigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    from lemmatizing_tokenizer import RawLemmaTokenizer
    
    tokenizer = RawLemmaTokenizer()
    stop_words = {}
    max_ngram_size = 1
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_lemmas_unigrams_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build lemmatized test unigrams vocabulary
# uses alphanumeric tokenizer
def build_raw_lemmatized_test_unigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    from lemmatizing_tokenizer import RawLemmaTokenizer
    
    tokenizer = RawLemmaTokenizer()
    stop_words = {}
    max_ngram_size = 1
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_lemmas_test_unigrams_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build stemmed test unigrams vocabulary
# uses alphanumeric tokenizer
def build_raw_stemmed_test_unigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    from stemming_tokenizer import RawStemmingTokenizer
    
    tokenizer = RawStemmingTokenizer()
    stop_words = {}
    max_ngram_size = 1
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_stems_test_unigrams_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build lemmatized all unigrams vocabulary
# uses alphanumeric tokenizer
def build_raw_lemmatized_all_unigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    from lemmatizing_tokenizer import RawLemmaTokenizer
    
    tokenizer = RawLemmaTokenizer()
    stop_words = {}
    max_ngram_size = 1
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_lemmas_all_unigrams_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build stemmed all unigrams vocabulary
# uses alphanumeric tokenizer
def build_raw_stemmed_all_unigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    from stemming_tokenizer import RawStemmingTokenizer
    
    tokenizer = RawStemmingTokenizer()
    stop_words = {}
    max_ngram_size = 1
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_stems_all_unigrams_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build lemmatized unigrams vocabulary
# uses alphabetic tokenizer
def build_lemmatized_unigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    from lemmatizing_tokenizer import LemmaTokenizer
    
    tokenizer = LemmaTokenizer()
    stop_words = {}
    max_ngram_size = 1
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_lemmas_unigrams_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build unigrams stopwords vocabulary
def build_unigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src):
    from reuters_globals import *
    
    tokenizer = None
    max_ngram_size = 1
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_unigrams_stopwords_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build bigrams stopwords vocabulary
def build_bigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src):
    from reuters_globals import *
    
    tokenizer = None
    max_ngram_size = 2
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_bigrams_stopwords_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build raw unigrams stopwords vocabulary
def build_raw_unigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src):
    from reuters_globals import *
    
    tokenizer = None
    max_ngram_size = 1
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,1,1)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_unigrams_stopwords'.format(vocabulary_src)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build raw bigrams stopwords vocabulary
def build_raw_bigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src):
    from reuters_globals import *
    
    tokenizer = None
    max_ngram_size = 2
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,1,1)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_bigrams_stopwords'.format(vocabulary_src)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build lemmatized unigrams stopwords vocabulary
# uses alphanumeric tokenizer
def build_raw_lemmatized_unigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src):
    from reuters_globals import *
    from lemmatizing_tokenizer import RawLemmaTokenizer
    
    tokenizer = RawLemmaTokenizer()
    max_ngram_size = 1
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_lemmas_unigrams_stopwords_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build lemmatized unigrams stopwords vocabulary
# uses alphabetic tokenizer
def build_lemmatized_unigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src):
    from reuters_globals import *
    from lemmatizing_tokenizer import LemmaTokenizer
    
    tokenizer = LemmaTokenizer()
    max_ngram_size = 1
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_lemmas_unigrams_stopwords_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build raw bigrams vocabulary
def build_raw_bigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    
    tokenizer = None
    stop_words = {}
    max_ngram_size = 2
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,1,1)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_bigrams'.format(vocabulary_src)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name    


# In[ ]:

# build lemmatized bigrams vocabulary
# uses alphanumeric tokenizer
def build_raw_lemmatized_bigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    from lemmatizing_tokenizer import RawLemmaTokenizer
    
    tokenizer = RawLemmaTokenizer()
    stop_words = {}
    max_ngram_size = 2
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_lemmas_bigrams_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name    


# In[ ]:

# build lemmatized all bigrams vocabulary
# uses alphanumeric tokenizer
def build_raw_lemmatized_all_bigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    from lemmatizing_tokenizer import RawLemmaTokenizer
    
    tokenizer = RawLemmaTokenizer()
    stop_words = {}
    max_ngram_size = 2
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_lemmas_all_bigrams_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name    


# In[ ]:

# build stemmed all bigrams vocabulary
# uses alphanumeric tokenizer
def build_raw_stemmed_all_bigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    from stemming_tokenizer import RawStemmingTokenizer
    
    tokenizer = RawStemmingTokenizer()
    stop_words = {}
    max_ngram_size = 2
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_stems_all_bigrams_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name    


# In[ ]:

# build lemmatized test bigrams vocabulary
# uses alphanumeric tokenizer
def build_raw_lemmatized_test_bigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    from lemmatizing_tokenizer import RawLemmaTokenizer
    
    tokenizer = RawLemmaTokenizer()
    stop_words = {}
    max_ngram_size = 2
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_lemmas_test_bigrams_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name    


# In[ ]:

# build stemmed test bigrams vocabulary
# uses alphanumeric tokenizer
def build_raw_stemmed_test_bigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    from stemming_tokenizer import RawStemmingTokenizer
    
    tokenizer = RawStemmingTokenizer()
    stop_words = {}
    max_ngram_size = 2
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_stems_test_bigrams_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name    


# In[ ]:

# build bigrams vocabulary
def build_bigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    
    tokenizer = None
    stop_words = {}
    max_ngram_size = 2
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_bigrams_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name    


# In[ ]:

# build lemmatized bigrams vocabulary
# uses alphabetic tokenizer
def build_lemmatized_bigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    from lemmatizing_tokenizer import LemmaTokenizer
    
    tokenizer = LemmaTokenizer()
    stop_words = {}
    max_ngram_size = 2
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_lemmas_bigrams_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name    


# In[ ]:

# build lemmatized bigrams stopwords vocabulary
# uses alphanumeric tokenizer
def build_raw_lemmatized_bigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src):
    from reuters_globals import *
    from lemmatizing_tokenizer import RawLemmaTokenizer
    
    tokenizer = RawLemmaTokenizer()
    max_ngram_size = 2
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_lemmas_bigrams_stopwords_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name    


# In[ ]:

# build lemmatized bigrams stopwords vocabulary
# uses alphabetic tokenizer
def build_lemmatized_bigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src):
    from reuters_globals import *
    from lemmatizing_tokenizer import LemmaTokenizer
    
    tokenizer = LemmaTokenizer()
    max_ngram_size = 2
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_lemmas_bigrams_stopwords_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name    


# In[ ]:

# build stemmed unigrams vocabulary
# uses alphanumeric tokenizer
def build_raw_stemmed_unigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    from stemming_tokenizer import RawStemmingTokenizer
    
    tokenizer = RawStemmingTokenizer()    
    stop_words = {}
    max_ngram_size = 1
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_stems_unigrams_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build stemmed unigrams vocabulary
# uses alphabetic tokenizer
def build_stemmed_unigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    from stemming_tokenizer import StemmingTokenizer
    
    tokenizer = StemmingTokenizer()    
    stop_words = {}
    max_ngram_size = 1
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_stems_unigrams_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build stemmed unigrams stopwords vocabulary
# uses alphanumeric tokenizer
def build_raw_stemmed_unigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src):
    from reuters_globals import *
    from stemming_tokenizer import RawStemmingTokenizer
    
    tokenizer = RawStemmingTokenizer()    
    max_ngram_size = 1
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_stems_unigrams_stopwords_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build stemmed unigrams stopwords vocabulary
# uses alphabetic tokenizer
def build_stemmed_unigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src):
    from reuters_globals import *
    from stemming_tokenizer import StemmingTokenizer
    
    tokenizer = StemmingTokenizer()    
    max_ngram_size = 1
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_stems_unigrams_stopwords_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build stemmed bigrams vocabulary
# uses alphanumeric tokenizer
def build_raw_stemmed_bigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    from stemming_tokenizer import RawStemmingTokenizer
    
    tokenizer = RawStemmingTokenizer() 
    stop_words = {}
    max_ngram_size = 2    
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_stems_bigrams_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build stemmed bigrams vocabulary
# uses alphabetic tokenizer
def build_stemmed_bigrams_vocabulary(corpus,vocabulary_src):
    from reuters_globals import *
    from stemming_tokenizer import StemmingTokenizer
    
    tokenizer = StemmingTokenizer() 
    stop_words = {}
    max_ngram_size = 2    
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_stems_bigrams_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build stemmed bigrams stopwords vocabulary
# uses alphanumeric tokenizer
def build_raw_stemmed_bigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src):
    from reuters_globals import *
    from stemming_tokenizer import RawStemmingTokenizer
    
    tokenizer = RawStemmingTokenizer() 
    max_ngram_size = 2    
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_raw_stems_bigrams_stopwords_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

# build stemmed bigrams stopwords vocabulary
# uses alphabetic tokenizer
def build_stemmed_bigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src):
    from reuters_globals import *
    from stemming_tokenizer import StemmingTokenizer
    
    tokenizer = StemmingTokenizer() 
    max_ngram_size = 2    
    vocabulary = build_vocabulary(corpus,tokenizer,stop_words,max_ngram_size,min_df,min_tf)
    # save to DB
    tbl_name = 'reuters21578_{0}_stems_bigrams_stopwords_df{1}_tf{2}'.format(vocabulary_src,min_df,min_tf)
    save_vocabulary(vocabulary,tbl_name)
    print 'done '+tbl_name


# In[ ]:

def build(vocabulary_src):
    from reuters_corpus_loader import load_corpus_and_labels
    from stopwords_loader import load_inquiry_stopwords
    
    # load reuters docs from DB using only vocabulary_src as main field for vocabulary (e.g., abstract, description, claims...)
    corpus_train = load_corpus_and_labels(vocabulary_src,'train')
    corpus = corpus_train['corpus']
    
    # build vocabulary without stopwords removal
    build_raw_unigrams_vocabulary(corpus,vocabulary_src)
    #build_unigrams_vocabulary(corpus,vocabulary_src)
    #build_lemmatized_unigrams_vocabulary(corpus,vocabulary_src)
    build_raw_lemmatized_unigrams_vocabulary(corpus,vocabulary_src)    
    #build_lemmatized_bigrams_vocabulary(corpus,vocabulary_src)
    build_raw_lemmatized_bigrams_vocabulary(corpus,vocabulary_src)
    build_raw_bigrams_vocabulary(corpus,vocabulary_src)
    #build_bigrams_vocabulary(corpus,vocabulary_src)
    #build_stemmed_unigrams_vocabulary(corpus,vocabulary_src)
    build_raw_stemmed_unigrams_vocabulary(corpus,vocabulary_src)
    #build_stemmed_bigrams_vocabulary(corpus,vocabulary_src)
    build_raw_stemmed_bigrams_vocabulary(corpus,vocabulary_src)

    # load inquiry stopwords list
    stop_words = load_inquiry_stopwords()

    # build vocabulary with stopwords removal
    #build_unigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src)
    build_raw_unigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src)
    #build_bigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src)
    build_raw_bigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src)
    #build_lemmatized_unigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src)
    build_raw_lemmatized_unigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src)
    #build_lemmatized_bigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src)
    build_raw_lemmatized_bigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src)
    #build_stemmed_unigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src)
    build_raw_stemmed_unigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src)
    #build_stemmed_bigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src)
    build_raw_stemmed_bigrams_stopwords_vocabulary(corpus,stop_words,vocabulary_src)
        
    #corpus_test = load_corpus(vocabulary_src,'test')
    #build_raw_lemmatized_test_unigrams_vocabulary(corpus_test['corpus'],vocabulary_src)    
    #build_raw_stemmed_test_unigrams_vocabulary(corpus_test['corpus'],vocabulary_src)    
    #build_raw_lemmatized_test_bigrams_vocabulary(corpus_test['corpus'],vocabulary_src)            
    #build_raw_stemmed_test_bigrams_vocabulary(corpus_test['corpus'],vocabulary_src)            
    
    #corpus = load_corpus(vocabulary_src,'both')
    #build_raw_lemmatized_all_unigrams_vocabulary(corpus['corpus'],vocabulary_src)    
    #build_raw_stemmed_all_unigrams_vocabulary(corpus['corpus'],vocabulary_src)    
    #build_raw_lemmatized_all_bigrams_vocabulary(corpus['corpus'],vocabulary_src)
    #build_raw_stemmed_all_bigrams_vocabulary(corpus['corpus'],vocabulary_src)


# In[ ]:

# build reuters docs vocabulary
vocabulary_src = 'all'
#target_topic = 'earn'
build(vocabulary_src)

# build reuters titles vocabulary
vocabulary_src = 'title'
#build(vocabulary_src)

# build reuters body vocabulary
vocabulary_src = 'body'
#build(vocabulary_src)

print 'done!'


# In[ ]:



