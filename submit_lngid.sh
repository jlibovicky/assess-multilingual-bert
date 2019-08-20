#!/bin/bash

QSUB_ARGS="-q gpu-* -l hostname=*dll*,gpu=1 -V -cwd -b y -j y -o logs/lngid"
ENV="source env/bin/activate"
DATA="data/lng_id/train.tsv data/lng_id/val.tsv data/lng_id/test.tsv"

for TYPE in cased uncased; do
    # MEAN POOL
    for LAYER in {0..11}; do
        if grep -q 'Mean test acc' logs/lngid/lngid_${TYPE}_${LAYER}_meanpool.o* ; then
            echo Done
        else
            qsub $QSUB_ARGS -N lngid_${TYPE}_${LAYER}_meanpool $ENV \; \
                ./lang_id.py bert-base-multilingual-$TYPE $LAYER $DATA \
                --hidden 1024 --mean-pool --test-output logs/lngid/test_${TYPE}_${LAYER}_meanpool.txt
        fi

        if grep -q 'Mean test acc' logs/lngid/lngid_${TYPE}_${LAYER}_meanpool_center.o* ; then
            echo Done
        else
            qsub $QSUB_ARGS -N lngid_${TYPE}_${LAYER}_meanpool_center $ENV \; \
                ./lang_id.py bert-base-multilingual-$TYPE $LAYER $DATA \
                --hidden 1024 --mean-pool --test-output logs/lngid/test_${TYPE}_${LAYER}_meanpool_center.txt --center-lng
        fi
    done

    # CLS VECTOR
    for LAYER in {-1..11}; do
        if grep -q 'Mean test acc' logs/lngid/lngid_${TYPE}_${LAYER}_cls.o* ; then
            echo Done
        else
            qsub $QSUB_ARGS -N lngid_${TYPE}_${LAYER}_cls $ENV \; \
                ./lang_id.py bert-base-multilingual-$TYPE $LAYER $DATA \
                --hidden 1024 --test-output logs/lngid/test_${TYPE}_${LAYER}_cls.txt
        fi

        if grep -q 'Mean test acc' logs/lngid/lngid_${TYPE}_${LAYER}_cls.o* ; then
            echo Done
        else
            qsub $QSUB_ARGS -N lngid_${TYPE}_${LAYER}_cls_center $ENV \; \
                ./lang_id.py bert-base-multilingual-$TYPE $LAYER $DATA \
                --hidden 1024 --test-output logs/lngid/test_${TYPE}_${LAYER}_cls_center.txt --center-lng
        fi

    done
done
