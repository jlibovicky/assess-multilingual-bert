#!/bin/bash

for FILE in data/50k_per_lng/*; do
    LNG=$(echo $FILE | sed -e 's/\.txt//;s#.*/##')
    qsub -o logs/entropies -q cpu-troja.q -pe smp 4 -l mem_free=48G,act_mem_free=48G,h_vmem=48G -V -cwd -b y -j y -N entropies_$LNG source env/bin/activate  \; ./att_entropies_per_lng.py bert-base-multilingual-cased ${FILE}
done
