
# coding: utf-8

## REUTERS 21578 Corpus Preprocessing

####### preprocess REUTERS 21578 corpus by removing numbers and punctuation characters

# In[5]:

def preprocess_corpus():
    import sqlite3 as sqlitedb    
    from reuters_globals import *
    import re
    import string

    # load docs text from sqlite DB using only vocabulary_src as main field for vocabulary (e.g., title, body)
    corpus = []
    labels = []
    #regexp1 = re.compile('(\([ ]*[0-9][0-9a-z,.; ]*\))')
    #regexp2 = re.compile('(\([ ]*[A-Za-z]\))')
    #regexp3 = re.compile(';gt&/>')
    regexp4 = re.compile('[0-9]')
    regexp5 = re.compile('[\'"_\-*/\^&><\[\]]')
    docs_query = 'select id, title, body from reuters21578'
    con = sqlitedb.connect(db_path)
    with con:        
        cur = con.execute(docs_query)
        record = 1
        while True:
            doc = cur.fetchone()
            if doc==None or doc[0]==None or doc[1]==None or doc[2]==None:
                break
            new_doc = []
            new_doc.append(doc[0])
            for i in range(1,3):
                tmp = doc[i]
                tmp = string.lower(tmp)
                tmp = tmp.replace('\n',' ')
                tmp = tmp.replace('\x0a','')
                tmp = tmp.replace('\x1d','')
                tmp = tmp.replace('\x1e','')
                tmp = tmp.replace('\x1f','')
                #tmp = regexp1.sub('',tmp)
                #tmp = regexp2.sub('',tmp)
                #tmp = regexp3.sub('',tmp)
                tmp = regexp4.sub('',tmp)
                tmp = regexp5.sub(' ',tmp)
                new_doc.append(tmp)
            con.execute(u'update reuters21578 set title=\'{0}\',body=\'{1}\' where id=\'{2}\''.format(new_doc[1],new_doc[2],new_doc[0]))    
            if record%10000==0:
                print 'processed ',record
            record = record + 1


# In[6]:

preprocess_corpus()
print 'done!'

