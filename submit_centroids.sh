#!/bin/bash

QSUB_ARGS="-q gpu-* -l hostname=*dll*,gpu=1 -V -cwd -b y -j y -o logs/centroids"
ENV="source env/bin/activate"
DATA="bert_languages.tsv data/50k_per_lng"

for TYPE in cased uncased; do
    for LAYER in {0..11}; do
        #qsub $QSUB_ARGS -N centroids_${TYPE}_${LAYER}_cls $ENV \; \
        #    ./save_centroids.py bert-base-multilingual-$TYPE $LAYER $DATA \
        #    centroids/${TYPE}_${LAYER}_cls.npz --batch-size 16

        qsub $QSUB_ARGS -N centroids_${TYPE}_${LAYER}_meanpool $ENV \; \
            ./save_centroids.py bert-base-multilingual-$TYPE $LAYER $DATA \
            centroids/${TYPE}_${LAYER}_meanpool.npz --mean-pool --batch-size 16
    done
done
