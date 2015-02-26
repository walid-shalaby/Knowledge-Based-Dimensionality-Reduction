
# coding: utf-8

## Crawl and Save 20ng data into sqlite DB

# In[80]:

import os.path
def stream_20ng_dataset(data_path=None):
    from sklearn.datasets import fetch_20newsgroups
    
    train = fetch_20newsgroups(data_home=data_path, subset='train')
    test = fetch_20newsgroups(data_home=data_path, subset='test')
    return train, test


# In[81]:

# Iterator over parsed Reuters SGML files and save in DB.
from ng20_globals import *
import sqlite3 as sqlite
train, test = stream_20ng_dataset('../../../data/ng20/data')
con = sqlite.connect(db_path)
with con:
    con.execute('drop table if exists ng20')
    con.execute('create table ng20 (id integer primary key, data text, topic_id integer, is_train integer)')
    con.execute('drop table if exists ng20_topics')
    con.execute('create table ng20_topics (id integer primary key, topic text)')
    
    # insert topics
    for i in range(len(train.target_names)):
        con.execute(u'insert into ng20_topics values({0},\'{1}\')'.format(i,train.target_names[i]))
    
    # insert train text
    for i in range(len(train.data)):
        con.execute(u'insert into ng20 values({0},lower(\'{1}\'),{2},{3})'.format(train.filenames[i][train.filenames[i].rfind('/')+1:]+str(train.target[i]),train.data[i].replace('\'','\'\''),train.target[i],1))        

    print 'done with ',len(train.data),'training records'
    
    # insert test text
    for i in range(len(test.data)):
        con.execute(u'insert into ng20 values({0},lower(\'{1}\'),{2},{3})'.format(test.filenames[i][test.filenames[i].rfind('/')+1:]+str(test.target[i]),test.data[i].replace('\'','\'\''),test.target[i],0))
        
    print 'done with ',len(test.data),'testing records'


# In[82]:

#from ng20_globals import *
#import sqlite3 as sqlite
#import string
#train, test = stream_20ng_dataset('../../../data/ng20/data')
#print len(train.data)
#print train.filenames[0]
#print train.filenames[1]
#print train.filenames[1][train.filenames[1].rfind('/')+1:]
#print len(train.target)
#print train.target[0]
#print train.target_names[train.target[0]]
#print list(train.target_names)
#print train.filenames[0][train.filenames[0].rfind('/')+1:]
#print train.filenames[0][train.filenames[0].rfind('/')+1:]+str(train.target[0])
#for i in range(0,10):
#    print train.filenames[i][train.filenames[i].rfind('/')+1:]+str(train.target[i])
#    print train.data[i]
#    print train.target[i]
#    print train.target_names[train.target[i]]


# In[82]:



