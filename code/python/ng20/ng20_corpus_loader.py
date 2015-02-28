
# coding: utf-8

## 20ng Corpus Loader

####### Load 20ng corpus from DB table

# In[ ]:

def load_corpus(train_or_test):
    import sqlite3 as sqlitedb
    from commons.globals import train_or_test_values
    from ng20_globals import db_path

    if train_or_test not in train_or_test_values:
        raise ValueError('\'{0}\' is an invalid value. use {1}'.format(train_or_test,train_or_test_values))
        
    if train_or_test=='both':
        query = 'select id,lower(data) from ng20'
    else:
        if train_or_test=='train':
            is_train = 1
        elif train_or_test=='test':
            is_train = 0
        
        query = 'select id,lower(data) from ng20 where is_train={0}'.format(is_train)
   
    # load 20ng text from sqlite DB
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

def load_corpus_and_labels(train_or_test):
    import sqlite3 as sqlitedb
    from commons.globals import train_or_test_values
    from ng20_globals import db_path

    if train_or_test not in train_or_test_values:
        raise ValueError('\'{0}\' is an invalid value. use {1}'.format(train_or_test,train_or_test_values))
        
    if train_or_test=='both':
        query = 'select id,lower(data) from ng20'
    else:
        if train_or_test=='train':
            is_train = 1
        elif train_or_test=='test':
            is_train = 0
        
        query = 'select id,lower(data),topic_id from ng20 where is_train={0}'.format(is_train)
    
    # load docs text from sqlite DB
    ids = []
    corpus = []
    labels = []
    con = sqlitedb.connect(db_path)
    with con:
        labels_dic = {}
        topics_cur = con.execute('select id,topic from ng20_topics')
        while True:
            topic = topics_cur.fetchone()
            if topic==None or topic[0]==None or topic[1]==None:
                break
            labels_dic[topic[1]] = topic[0]
        
        #for k, v in labels_dic.iteritems():
        #    print k, v
                
        cur = con.execute(query)
        while True:
            record = cur.fetchone()
            if record==None or record[0]==None or record[1]==None or record[2]==None:
                break
            
            ids.append(record[0])
            corpus.append(record[1])            
            labels.append(record[2])
        print 'loaded {0} records.'.format(len(ids))
    
    return {'ids':ids,'corpus':corpus,'labels':labels,'labels_dic':labels_dic,'labels_arr':labels}


# In[ ]:

def load_corpus_with_labels_mappings(train_or_test,labels_dic):
    import sqlite3 as sqlitedb
    from commons.globals import train_or_test_values
    from ng20_globals import db_path

    if train_or_test not in train_or_test_values:
        raise ValueError('\'{0}\' is an invalid value. use {1}'.format(train_or_test,train_or_test_values))
    
    if train_or_test=='both':
        query = 'select id,lower(data) from ng20'
    else:
        if train_or_test=='train':
            is_train = 1
        elif train_or_test=='test':
            is_train = 0
        query = 'select id,lower(data),topic_id from ng20 where is_train={0}'.format(is_train)
    
    # load docs text from sqlite DB
    ids = []
    corpus = []
    labels = []
    con = sqlitedb.connect(db_path)
    with con:
        cur = con.execute(query)
        while True:
            record = cur.fetchone()
            if record==None or record[0]==None or record[1]==None or record[2]==None:
                break
            ids.append(record[0])
            corpus.append(record[1])
            labels.append(record[2])
        print 'loaded {0} records.'.format(len(ids))
    
    # make sure each labels has an entry in labels dic
    for i in range(len(labels)):
        if labels[i] not in labels_dic.values():
            raise ValueError('\'{0}\' is an invalid value.'.format(labels[i]))
    return {'ids':ids,'corpus':corpus,'labels':labels}


# In[ ]:

def load_label_names():
    import sqlite3 as sqlitedb
    from ng20_globals import db_path

    query = 'select topic from ng20_topics order by id'
   
    label_names = []
    
    con = sqlitedb.connect(db_path)
    with con:
        cur = con.execute(query)
        while True:
            record = cur.fetchone()
            if record==None or record[0]==None:
                break
            label_names.append(record[0])
        print 'loaded {0} labels.'.format(len(label_names))
    return label_names


# In[ ]:

#corpus_train_data = load_corpus_and_labels('train')
#print 'done loading {0} train records and {1} labels.'.format(len(corpus_train_data['corpus']),len(corpus_train_data['labels_dic']))

#corpus_test_data = load_corpus_and_labels('test')
#print 'done loading {0} test records and {1} labels.'.format(len(corpus_test_data['corpus']),len(corpus_test_data['labels_dic']))

#corpus_test_data = load_corpus_with_labels_mappings('test',corpus_train_data['labels_dic'])
#print corpus_test_data['labels']

