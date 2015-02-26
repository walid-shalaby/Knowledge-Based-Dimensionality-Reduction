
# coding: utf-8

## Crawl and Save different versions of 20ng data into sqlite DB

# In[1]:

def stream_cleaned__shuffled_20ng_dataset(dataset, data_path=None, remove=()):
    from twenty_newsgroups import fetch_20newsgroups
    return fetch_20newsgroups(dataset=dataset, data_home=data_path, shuffle=True, remove=remove)


# In[2]:

# Iterator over parsed Reuters SGML files and save in DB.
from ng20_globals import db_path
import sqlite3 as sqlite
data = stream_cleaned__shuffled_20ng_dataset(dataset='preprocessed',data_path='../../../data/ng20/data',remove='from')
con = sqlite.connect(db_path)
with con:
    con.execute('drop table if exists ng20')
    con.execute('create table ng20 (id integer primary key, data text, topic_id integer, is_train integer)')
    con.execute('drop table if exists ng20_topics')
    con.execute('create table ng20_topics (id integer primary key, topic text)')
    
    # insert topics
    for i in range(len(data.target_names)):
        con.execute(u'insert into ng20_topics values({0},\'{1}\')'.format(i,data.target_names[i]))
    
    # insert text
    for i in range(len(data.data)):
        con.execute(u'insert into ng20 values({0},lower(\'{1}\'),{2},{3})'.format(data.filenames[i][data.filenames[i].rfind('/')+1:]+str(data.target[i]),data.data[i].replace('\'','\'\''),data.target[i],1)) 

    print 'done with ',len(data.data),' records'

