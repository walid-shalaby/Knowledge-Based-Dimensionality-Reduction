
# coding: utf-8

## CLEF-IP2010 statistics

####### collect some statistics about clef-ip2010 corpus

# In[43]:

from clef_corpus_loader import load_corpus_and_labels
from clef_corpus_loader import load_corpus_with_labels_mappings
corpus_data = load_corpus_and_labels('abstract','both')
print 'done loading {0} records and {1} labels.'.format(len(corpus_data['corpus']),len(corpus_data['labels_dic']))
corpus_train_data = load_corpus_with_labels_mappings('abstract','train',corpus_data['labels_dic'])
print 'done loading {0} train records.'.format(len(corpus_train_data['corpus']))
corpus_test_data = load_corpus_with_labels_mappings('abstract','test',corpus_data['labels_dic'])
print 'done loading {0} test records.'.format(len(corpus_test_data['corpus']))


# In[45]:

# number of labels per # of samples & average # labels per sample
all_labels_count = [0,0,0,0,0,0,0,0,0,0,0,0]
train_labels_count = [0,0,0,0,0,0,0,0,0,0,0,0]
test_labels_count = [0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(len(corpus_data['corpus'])):
    all_labels_count[len(corpus_data['labels'][i])-1] = all_labels_count[len(corpus_data['labels'][i])-1] + 1
    
for i in range(len(corpus_train_data['corpus'])):
    train_labels_count[len(corpus_train_data['labels'][i])-1] = train_labels_count[len(corpus_train_data['labels'][i])-1] + 1
    
for i in range(len(corpus_test_data['corpus'])):
    test_labels_count[len(corpus_test_data['labels'][i])-1] = test_labels_count[len(corpus_test_data['labels'][i])-1] + 1
    
print 'total number of samples: ',sum(all_labels_count)
print 'total number of train samples: ',sum(train_labels_count)
print 'total number of test samples: ',sum(test_labels_count)
print 'number of labels,total number of samples,total number of train samples,total number of test samples'
total = 0
total_train = 0
total_test = 0
for i in range(len(all_labels_count)):
    print i+1,',',all_labels_count[i],',',train_labels_count[i],',',test_labels_count[i]
    total = total + (i+1.0)*all_labels_count[i]
    total_train = total_train + (i+1.0)*train_labels_count[i]
    total_test = total_test + (i+1.0)*test_labels_count[i]
    
print 'average labels per sample: ', total/sum(all_labels_count)
print 'average labels per train sample: ', total_train/sum(train_labels_count)
print 'average labels per test sample: ', total_test/sum(test_labels_count)

all_labels_dist = [0]*(len(corpus_data['labels_dic']))
train_labels_dist = [0]*(len(corpus_data['labels_dic']))
test_labels_dist = [0]*(len(corpus_data['labels_dic']))

for i in range(len(corpus_data['corpus'])):
    for j in range(len(corpus_data['labels'][i])):        
        all_labels_dist[corpus_data['labels'][i][j]] = all_labels_dist[corpus_data['labels'][i][j]] + 1

for i in range(len(corpus_train_data['corpus'])):
    for j in range(len(corpus_train_data['labels'][i])):        
        train_labels_dist[corpus_train_data['labels'][i][j]] = train_labels_dist[corpus_train_data['labels'][i][j]] + 1

for i in range(len(corpus_test_data['corpus'])):
    for j in range(len(corpus_test_data['labels'][i])):        
        test_labels_dist[corpus_test_data['labels'][i][j]] = test_labels_dist[corpus_test_data['labels'][i][j]] + 1

print 'label,number of samples,number of train samples,number of tedst samples'
for k,v in corpus_data['labels_dic'].items():
    print k.upper(),',',all_labels_dist[v],',',train_labels_dist[v],',',test_labels_dist[v]


# In[48]:

import sqlite3 as sqldb
from clef_globals import *
con = sqldb.connect(db_path)
with con:
    cur = con.execute('select count(*) from clef_2010_abstract_raw_lemmas_all_unigrams_df2_tf3')
    print 'clef_2010_abstract_raw_lemmas_all_unigrams_df2_tf3: ',cur.fetchone()[0]
    cur = con.execute('select count(*) from clef_2010_abstract_raw_lemmas_all_bigrams_df2_tf3 where term like \'% %\'')
    print 'clef_2010_abstract_raw_lemmas_all_bigrams_df2_tf3: ',cur.fetchone()[0]
    cur = con.execute('select count(*) from clef_2010_abstract_raw_lemmas_unigrams_df2_tf3')
    print 'clef_2010_abstract_raw_lemmas_unigrams_df2_tf3: ',cur.fetchone()[0]
    cur = con.execute('select count(*) from clef_2010_abstract_raw_lemmas_bigrams_df2_tf3 where term like \'% %\'')
    print 'clef_2010_abstract_raw_lemmas_bigrams_df2_tf3: ',cur.fetchone()[0]
    cur = con.execute('select count(*) from clef_2010_abstract_raw_lemmas_test_unigrams_df2_tf3')
    print 'clef_2010_abstract_raw_lemmas_test_unigrams_df2_tf3: ',cur.fetchone()[0]
    cur = con.execute('select count(*) from clef_2010_abstract_raw_lemmas_test_bigrams_df2_tf3 where term like \'% %\'')
    print 'clef_2010_abstract_raw_lemmas_test_bigrams_df2_tf3: ',cur.fetchone()[0]


# In[ ]:



