
# coding: utf-8

## CLEF-IP2010 Corpus Splitter

####### Split clef-ip2010 patents corpus into train/test sets

# In[ ]:

def split_corpus(test_size):
    from sklearn.cross_validation import train_test_split
    import sqlite3 as sqlitedb
    from clef_corpus_loader import load_corpus
    
    ids = []
    
    # load patents from sqlite DB
    corpus = load_corpus('abstract','both')
    ids = corpus['ids']    
        
    ids_train, ids_test = train_test_split(ids, test_size=test_size, random_state=100)
    
    con = sqlitedb.connect(db_path)
    with con:
        for i in ids_train:
            con.execute('insert or replace into patent_train_test values(\'{0}\',{1})'.format(i,1))
        for i in ids_test:
            con.execute('insert or replace into patent_train_test values(\'{0}\',{1})'.format(i,0))            


# In[ ]:

from clef_globals import *
split_corpus(test_set_size)
print 'done'


# In[ ]:



