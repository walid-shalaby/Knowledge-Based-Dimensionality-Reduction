
# coding: utf-8

## CLEF-IP2010 Corpus Preprocessing

####### preprocess clef-ip2010 patents corpus by removing numbers and punctuation characters

# In[ ]:

def preprocess_corpus():
    import sqlite3 as sqlitedb    
    from clef_globals import *
    import re

    # load patents text from sqlite DB using only vocabulary_src as main field for vocabulary (e.g., abstract, description, claims...)
    corpus = []
    labels = []
    regexp1 = re.compile('(\([ ]*[0-9][0-9a-z,.; ]*\))')
    regexp2 = re.compile('(\([ ]*[A-Za-z]\))')
    regexp3 = re.compile(';gt&/>')
    regexp4 = re.compile('[0-9]')
    regexp5 = re.compile('[\'"_\-*/\^&]')
    patents_query = 'select id, title, abstract, description, claims from patents'
    con = sqlitedb.connect(db_path)
    with con:        
        cur = con.execute(patents_query)
        record = 1
        while True:
            patent = cur.fetchone()
            if patent==None or patent[0]==None or patent[1]==None or patent[2]==None or patent[3]==None or patent[4]==None:
                break
            new_patent = []
            new_patent.append(patent[0])
            for i in range(1,5):
                tmp = patent[i]
                tmp = lower(tmp)
                tmp = tmp.replace('\n',' ')
                tmp = tmp.replace('\x0a','')
                tmp = tmp.replace('\x1d','')
                tmp = tmp.replace('\x1e','')
                tmp = tmp.replace('\x1f','')
                tmp = regexp1.sub('',tmp)
                tmp = regexp2.sub('',tmp)
                tmp = regexp3.sub('',tmp)
                tmp = regexp4.sub('',tmp)
                tmp = regexp5.sub(' ',tmp)
                new_patent.append(tmp)
            con.execute(u'update patents set title=\'{0}\',abstract=\'{1}\',description=\'{2}\',claims=\'{3}\' where id=\'{4}\''.format(new_patent[1],new_patent[2],new_patent[3],new_patent[4],new_patent[0]))    
            if record%10000==0:
                print 'processed ',record
            record = record + 1


# In[ ]:

preprocess_corpus()
print 'done!'


# In[118]:

import re
reg = re.compile('[0-9\'"\n]')
str = '3.hey\'12-l\n"yo_u'
print str
print reg.sub('',str)
print str.replace('\n','')


# In[112]:

import re
regexp1 = re.compile('(\([ ]*[0-9][0-9a-z,.; ]*\))')
regexp2 = re.compile('(\([ ]*[A-Za-z]\))')
regexp3 = re.compile(';gt&/>')
regexp4 = re.compile('[0-9]')
regexp5 = re.compile('[\'"_\-*/\^&]')
str = 'this i^&s a \'/(t) remove'
print regexp5.sub(' ',str)
str = str.replace('this','')
print str


# In[101]:

a = 100
print a%10


# In[102]:

a = 'A'
print a.lower()


# In[ ]:



