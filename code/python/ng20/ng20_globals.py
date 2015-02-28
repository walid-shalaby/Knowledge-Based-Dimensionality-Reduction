## 20ng global definitions

from sets import Set

max_labels = 1 # use only top 1 probabilities labels as maximum labels per doc is 1

min_df = 1
max_df = 1.0
min_tf = 1

db_path = '../../../data/ng20/ng20.db'

vocabulary_src_values = Set(['all','title','body'])