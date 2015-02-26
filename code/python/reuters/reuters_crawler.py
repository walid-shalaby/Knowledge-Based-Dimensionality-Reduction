
# coding: utf-8

## Crawl and Save reuters 21578 data into sqlite DB, Using the ModApte split

# In[25]:

#http://www.daviddlewis.com/resources/testcollections/reuters21578/readme.txt
from __future__ import print_function

from glob import glob
import itertools
import os.path
import re
import tarfile
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.externals import six
from sklearn.externals.six.moves import html_parser
from sklearn.externals.six.moves import urllib
from sklearn.datasets import get_data_home

import string 

def _not_in_sphinx():
    # Hack to detect whether we are running by the sphinx builder
    return '__file__' in globals()


###############################################################################
# Reuters Dataset related routines
###############################################################################


class ReutersParser(html_parser.HTMLParser):
    """Utility class to parse a SGML file and yield documents one at a time."""

    def __init__(self, encoding='latin-1'):
        html_parser.HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def handle_starttag(self, tag, attrs):
        method = 'start_' + tag
        getattr(self, method, lambda x: None)(attrs)

    def handle_endtag(self, tag):
        method = 'end_' + tag
        getattr(self, method, lambda: None)()

    def _reset(self):
        self.in_title = 0
        self.in_body = 0
        self.in_topics = 0
        self.in_topic_d = 0
        self.is_train = 2
        self.title = ""
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_title:
            self.title += data
        elif self.in_topic_d:
            self.topic_d += data

    def start_reuters(self, attributes):
        lewissplit = ''
        topics = ''
        for i in range(len(attributes)):
            if string.lower(attributes[i][0])=='lewissplit':
               lewissplit = string.lower(attributes[i][1])
            elif string.lower(attributes[i][0])=='topics':
               topics = string.lower(attributes[i][1])
        if lewissplit=='train' and topics=='yes':
            self.is_train = 1
        if lewissplit=='test' and topics=='yes':
            self.is_train = 0        
        
    def end_reuters(self):
        if self.is_train==0 or self.is_train==1:
            self.body = re.sub(r'\s+', r' ', self.body)
            self.docs.append({'title': self.title,
                              'body': self.body,
                              'topics': self.topics,
                              'is_train': self.is_train})
        self._reset()

    def start_title(self, attributes):
        self.in_title = 1

    def end_title(self):
        self.in_title = 0

    def start_body(self, attributes):
        self.in_body = 1

    def end_body(self):
        self.in_body = 0

    def start_topics(self, attributes):
        self.in_topics = 1

    def end_topics(self):
        self.in_topics = 0

    def start_d(self, attributes):
        self.in_topic_d = 1

    def end_d(self):
        self.in_topic_d = 0
        self.topics.append(self.topic_d)
        self.topic_d = ""


def stream_reuters_documents(data_path=None):
    """Iterate over documents of the Reuters dataset.

    The Reuters archive will automatically be downloaded and uncompressed if
    the `data_path` directory does not exist.

    Documents are represented as dictionaries with 'body' (str),
    'title' (str), 'topics' (list(str)) keys.

    """

    DOWNLOAD_URL = ('http://archive.ics.uci.edu/ml/machine-learning-databases/'
                    'reuters21578-mld/reuters21578.tar.gz')
    ARCHIVE_FILENAME = 'reuters21578.tar.gz'

    if data_path is None:
        data_path = os.path.join(get_data_home(), "reuters")
    if not os.path.exists(data_path):
        """Download the dataset."""
        print("downloading dataset (once and for all) into %s" %
              data_path)
        os.mkdir(data_path)

        def progress(blocknum, bs, size):
            total_sz_mb = '%.2f MB' % (size / 1e6)
            current_sz_mb = '%.2f MB' % ((blocknum * bs) / 1e6)
            if _not_in_sphinx():
                print('\rdownloaded %s / %s' % (current_sz_mb, total_sz_mb),
                      end='')

        archive_path = os.path.join(data_path, ARCHIVE_FILENAME)
        urllib.request.urlretrieve(DOWNLOAD_URL, filename=archive_path,
                                   reporthook=progress)
        if _not_in_sphinx():
            print('\r', end='')
        print("untarring Reuters dataset...")
        tarfile.open(archive_path, 'r:gz').extractall(data_path)
        print("done.")

    parser = ReutersParser()
    for filename in glob(os.path.join(data_path, "*.sgm")):
        for doc in parser.parse(open(filename, 'rb')):
            yield doc


# In[26]:

# Iterator over parsed Reuters SGML files and save in DB.
from reuters_globals import *
import sqlite3 as sqlite
data_stream = stream_reuters_documents('../../../data/reuters21578/data')

con = sqlite.connect(db_path)
with con:
    con.execute('drop table if exists reuters21578')
    con.execute('create table reuters21578 (id integer, title text, body text, is_train integer)')
    con.execute('drop table if exists reuters21578_topics')
    con.execute('create table reuters21578_topics (id integer unique, topic text)')
    con.execute('drop table if exists reuters21578_topics_join')
    con.execute('create table reuters21578_topics_join (doc_id integer, topic_id integer)')    
    record_no = 0
    topic_no = 0
    topic_dic = dict()
    for record in data_stream:
        record_no = record_no + 1        
        con.execute(u'insert into reuters21578 values("{0}",lower("{1}"),lower("{2}"),{3})'.format(record_no,record['title'].replace('"','\''),record['body'].replace('"','\''),record['is_train']))
        if len(record['topics'])==0:
            topics = ''
        else:
            for i in range(len(record['topics'])):
                if topic_dic.has_key(record['topics'][i])==False:
                    topic_no = topic_no + 1
                    con.execute('insert into reuters21578_topics values("{0}",lower("{1}"))'.format(topic_no,record['topics'][i]))
                    topic_dic[record['topics'][i]] = topic_no
                    topic_id = topic_no
                else:
                    topic_id = topic_dic[record['topics'][i]]
                con.execute('insert into reuters21578_topics_join values("{0}","{1}")'.format(record_no,topic_id))
print('done with ',record_no,'records')


# In[26]:




# In[26]:



