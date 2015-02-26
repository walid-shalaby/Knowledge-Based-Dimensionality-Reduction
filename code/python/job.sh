#!/bin/sh

# ===== PBS OPTIONS =====
### Set the job name
#PBS -N clef_classifier

### Run in the queue named "urc"
#PBS -q viper

# ==== Main ======
python /users/wshalaby/patents/patent-similarity/code/python/clef_classifier.py 1>/users/wshalaby/clef_classifier_o.txt 2>/users/wshalaby/clef_classifier_e.txt

