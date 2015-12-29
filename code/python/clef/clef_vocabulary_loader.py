## CLEF-IP2010 Vocabulary Loader

# Load clef-ip2010 patents vocabulary from DB table
def load_vocabulary(tbl_name):
    import sqlite3 as sqlitedb
    from clef_globals import db_path

    # in case using min_df as fractions
    tbl_name = tbl_name.replace('.','_')
    
    # load vocabulary from sqlite DB
    vocabulary = []
    stmt = 'select term from {0}'.format(tbl_name)
    con = sqlitedb.connect(db_path)
    with con:
        cur = con.execute(stmt)
        while True:
            term = cur.fetchone()
            if term==None or term[0]==None:
                break
            # retrieve patent text
            vocabulary.append(term[0])

    print 'loaded ({0}) terms'.format(len(vocabulary))
    return vocabulary


# load all unigrams from tbl_name_full1 and only bigrams existing in both tbl_name_full2&tbl_name_intersect
def load_common_vocabulary(tbl_name_full1,tbl_name_full2,tbl_name_intersect,stem_or_lemma):
    import sqlite3 as sqlitedb
    from clef_globals import db_path

    # in case using min_df as fractions
    tbl_name_full1 = tbl_name_full1.replace('.','_')
    tbl_name_full2 = tbl_name_full2.replace('.','_')
    
    # load vocabulary from sqlite DB
    vocabulary = []
    stmt = 'select term from {0} union select {1} from {2},{3} where {1}=term union select bigram from {2},{3} where bigram=term'.format(tbl_name_full1,stem_or_lemma,tbl_name_full2,tbl_name_intersect)
    con = sqlitedb.connect(db_path)
    with con:
        cur = con.execute(stmt)
        while True:
            term = cur.fetchone()
            if term==None or term[0]==None:
                break
            # retrieve patent text
            vocabulary.append(term[0])

    print 'loaded ({0}) terms'.format(len(vocabulary))
    return vocabulary


# load only bigrams existing in both tbl_name_full1&tbl_name_intersect and extend the vocabulary with unigrams of common bigrams
def load_common_vocabulary_extend_unigrams(tbl_name_full,tbl_name_intersect,stem_or_lemma):
    import sqlite3 as sqlitedb
    from clef_globals import db_path

    # in case using min_df as fractions
    tbl_name_full = tbl_name_full.replace('.','_')
    
    # load vocabulary from sqlite DB
    vocabulary = []
    stmt = 'select {0} from {1},{2} where {0}=term union select bigram from {1},{2} where bigram=term'.format(stem_or_lemma,tbl_name_full,tbl_name_intersect)
    con = sqlitedb.connect(db_path)
    with con:
        cur = con.execute(stmt)
        while True:
            term = cur.fetchone()
            if term==None or term[0]==None:
                break
            # retrieve patent text
            vocabulary.append(term[0])

    print 'loaded ({0}) terms'.format(len(vocabulary))
    dic = {}
    count = len(vocabulary)    
    for i in range(count):
        u1,u2 = vocabulary[i].split(' ')
        if dic.has_key(u1)==False:
            vocabulary.append(u1)
            dic[u1] = ''
        if dic.has_key(u2)==False:
            vocabulary.append(u2)
            dic[u2] = ''
    print 'extended to ({0}) terms'.format(len(vocabulary))
    return vocabulary

