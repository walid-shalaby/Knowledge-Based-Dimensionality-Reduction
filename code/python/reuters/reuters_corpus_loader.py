
# coding: utf-8

## REUTERS 21578 Corpus Loader

####### Load reuters 21578 news corpus from DB table

# In[ ]:

def load_corpus(vocabulary_src,train_or_test):
    import sqlite3 as sqlitedb
    from reuters_globals import *

    if train_or_test not in train_or_test_values:
        raise ValueError('\'{0}\' is an invalid value. use {1}'.format(train_or_test,train_or_test_values))
    if vocabulary_src not in vocabulary_src_values:
        raise ValueError('\'{0}\' is an invalid value. use {1}'.format(vocabulary_src,vocabulary_src_values))
        
    if train_or_test=='both':
        if vocabulary_src=='all':
            query = 'select id,lower(title)||" "||lower(body) from reuters21578'
        else:
            query = 'select id,lower({0}) from reuters21578'.format(vocabulary_src)
    else:
        if train_or_test=='train':
            is_train = 1
        elif train_or_test=='test':
            is_train = 0
        if vocabulary_src=='all':
            query = 'select id,lower(title)||" "||lower(body) from reuters21578 where is_train={0}'.format(is_train)
        else:
            query = 'select id,lower({0}) from reuters21578 where is_train={1}'.format(vocabulary_src,is_train)
   
    # load reuters text from sqlite DB using only vocabulary_src as main field for vocabulary (e.g., abstract, description, claims...)
    ids = []
    corpus = []
    
    con = sqlitedb.connect(db_path)
    with con:
        cur = con.execute(query)
        while True:
            record = cur.fetchone()
            if record==None or record[0]==None or record[1]==None:
                break
            ids.append(record[0])
            corpus.append(record[1])
        print 'loaded {0} records.'.format(len(ids))
    return {'ids':ids,'corpus':corpus}


# In[ ]:

def load_corpus_and_labels(vocabulary_src,train_or_test):
    import sqlite3 as sqlitedb
    from reuters_globals import *
    from sets import Set

    if train_or_test not in train_or_test_values:
        raise ValueError('\'{0}\' is an invalid value. use {1}'.format(train_or_test,train_or_test_values))
    if vocabulary_src not in vocabulary_src_values:
        raise ValueError('\'{0}\' is an invalid value. use {1}'.format(vocabulary_src,vocabulary_src_values))
        
    if train_or_test=='both':
        if vocabulary_src=='all':
            query = 'select id,lower(title)||" "||lower(body) from reuters21578'
        else:
            query = 'select id,lower({0}) from reuters21578'.format(vocabulary_src)
    else:
        if train_or_test=='train':
            is_train = 1
        elif train_or_test=='test':
            is_train = 0
        if vocabulary_src=='all':
            query = 'select id,lower(title)||" "||lower(body) from reuters21578 where is_train={0}'.format(is_train)
        else:
            query = 'select id,lower({0}) from reuters21578 where is_train={1}'.format(vocabulary_src,is_train)
    
    # load docs text from sqlite DB using only vocabulary_src as main field for vocabulary (e.g., abstract, description, claims...)
    ids = []
    corpus = []
    labels = []
    con = sqlitedb.connect(db_path)
    with con:
        # load valid topics (topics that have at least 1 instance in training set and in all-topics-strings.lc.txt)
        valid_topics = Set()
        topics_cur = con.execute('select distinct(topic) from valid_topics')
        while True:
            topic = topics_cur.fetchone()
            if topic==None or topic[0]==None:
                break
            valid_topics.add(topic[0])
        
        cur = con.execute(query)
        while True:
            record = cur.fetchone()
            if record==None or record[0]==None or record[1]==None:
                break
            # retrieve topics (more that one topic separated by space)
            topics = ''
            topics_cur = con.execute('select distinct(topic) from reuters21578_topics rt,reuters21578_topics_join rtj where rt.id=rtj.topic_id and rtj.doc_id={0} and rt.id=rtj.topic_id'.format(record[0]))            
            while True:
                topic = topics_cur.fetchone()
                if topic==None or topic[0]==None:
                    break
                if (topic[0] in valid_topics)==True:
                    if topics=='':
                        topics = topic[0]
                    else:
                        topics = topics + ',' + topic[0]                

            if len(topics)>0:
                # retrieve record text
                ids.append(record[0])
                corpus.append(record[1])            
                record = topics.split(',')
                if len(record[len(record)-1])==0:
                    labels.append(record[0:len(record)-1])            
                else:
                    labels.append(record)        
        print 'loaded {0} records.'.format(len(ids))
    # map labels into unique class names
    labels_dic = {}
    labels_arr = []
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] not in labels_dic:
                labels_dic[labels[i][j]] = len(labels_arr)
                labels_arr.append(labels[i][j])
                labels[i][j] = len(labels_arr)-1
            else:
                labels[i][j] = labels_dic[labels[i][j]]
    return {'ids':ids,'corpus':corpus,'labels':labels,'labels_dic':labels_dic,'labels_arr':labels_arr}


# In[ ]:

def load_corpus_with_labels_mappings(vocabulary_src,train_or_test,labels_dic):
    import sqlite3 as sqlitedb
    from reuters_globals import *
    from sets import Set

    if train_or_test not in train_or_test_values:
        raise ValueError('\'{0}\' is an invalid value. use {1}'.format(train_or_test,train_or_test_values))
    if vocabulary_src not in vocabulary_src_values:
        raise ValueError('\'{0}\' is an invalid value. use {1}'.format(vocabulary_src,vocabulary_src_values))
    
    if train_or_test=='both':
        if vocabulary_src=='all':
            query = 'select id,lower(title)||" "||lower(body) from reuters21578'
        else:
            query = 'select id,lower({0}) from reuters21578'.format(vocabulary_src)
    else:
        if train_or_test=='train':
            is_train = 1
        elif train_or_test=='test':
            is_train = 0
        if vocabulary_src=='all':
            query = 'select id,lower(title)||" "||lower(body) from reuters21578 where is_train={0}'.format(is_train)
        else:
            query = 'select id,lower({0}) from reuters21578 where is_train={1}'.format(vocabulary_src,is_train)
    
    # load docs text from sqlite DB using only vocabulary_src as main field for vocabulary (e.g., abstract, description, claims...)
    ids = []
    corpus = []
    labels = []
    con = sqlitedb.connect(db_path)
    with con:
        # load valid topics (topics that have at least 1 instance in training set and in all-topics-strings.lc.txt)
        valid_topics = Set()
        topics_cur = con.execute('select distinct(topic) from valid_topics')
        while True:
            topic = topics_cur.fetchone()
            if topic==None or topic[0]==None:
                break
            valid_topics.add(topic[0])

        cur = con.execute(query)
        while True:
            record = cur.fetchone()
            if record==None or record[0]==None or record[1]==None:
                break
            # retrieve topics (more that one topic separated by space)
            topics = ''
            topics_cur = con.execute('select topic from reuters21578_topics rt,reuters21578_topics_join rtj where rt.id=rtj.topic_id and rtj.doc_id={0} and rt.id=rtj.topic_id'.format(record[0]))            
            while True:
                topic = topics_cur.fetchone()
                if topic==None or topic[0]==None:
                    break
                if (topic[0] in valid_topics)==True:
                    if topics=='':
                        topics = topic[0]
                    else:
                        topics = topics + ',' + topic[0]
            if len(topics)>0:
                # retrieve record text
                ids.append(record[0])
                corpus.append(record[1])
                record = topics.split(',')
                if len(record[len(record)-1])==0:
                    labels.append(record[0:len(record)-1])            
                else:
                    labels.append(record)        
        print 'loaded {0} records.'.format(len(ids))
    
    # map labels into unique class names
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            labels[i][j] = labels_dic[labels[i][j]]
    return {'ids':ids,'corpus':corpus,'labels':labels}


# In[ ]:

#corpus_train_data = load_corpus_and_labels('all','train')
#print 'done loading {0} train records and {1} labels.'.format(len(corpus_train_data['corpus']),len(corpus_train_data['labels_dic']))


# In[ ]:



