#!/bin/sh

# ===== PBS OPTIONS =====
### Set the job name
#PBS -N clef_stats

### Run in the queue named "urc"
#PBS -q viper

# ==== Main ======
python /users/wshalaby/patents/patent-similarity/code/python/clef_stats.py 1>/users/wshalaby/clef_stats_o.txt 2>/users/wshalaby/clef_stats_e.txt

