
# coding: utf-8

## CLEF-IP2010 Corpus Language Detector

####### Detect clef-ip2010 patents corpus language and save language as well as confidence scores to DB

# In[ ]:

def detect_corpus_lang(vocabulary_src):
    import sys
    sys.path.append('langid.py-master/langid/')
    import sqlite3 as sqlitedb
    from clef_globals import *
    import langid

    # load patents text from sqlite DB using only vocabulary_src as main field for vocabulary (e.g., abstract, description, claims...)
    patents_query = 'select id,lower({0}) from patents'.format(vocabulary_src)
    con = sqlitedb.connect(db_path)
    with con:
        count = 0
        tbl_name = 'patent_{0}_language'.format(vocabulary_src)
        con.execute('drop table if exists {0}'.format(tbl_name))
        con.execute('create table {0}(id text,language text,confidence real)'.format(tbl_name))
        cur = con.execute(patents_query)
        while True:
            patent = cur.fetchone()
            if patent==None or patent[0]==None or patent[1]==None:
                break
            patent_id = patent[0]
            text = patent[1]
            lang,confidence = langid.classify(text)
            lang = (patent_id,lang,confidence)
            con.execute('insert into {0} values(?,?,?)'.format(tbl_name),lang)
            count = count + 1
            if count%10000==0:
                print 'processsed {0} records'.format(count)
        
        print 'processsed {0} records'.format(count)


# In[ ]:

detect_corpus_lang('abstract')
#detect_corpus_lang('description')
#detect_corpus_lang('claims')
print 'done'

