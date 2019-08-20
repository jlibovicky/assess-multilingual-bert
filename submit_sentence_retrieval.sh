#!/bin/bash

QSUB_ARGS="-o logs/retrieval -q cpu-troja.q -pe smp 4 -l mem_free=32G,act_mem_free=32G,h_vmem=32G -V -cwd -b y -j y"
ENV="source env/bin/activate"
SCRIPT="./sentence_retrieval.py bert-base-multilingual-cased"

for LAYER in {-1..11}; do
    echo $LAYER
    qsub ${QSUB_ARGS} -N retrieval_${LAYER}_cls         ${ENV} \; ${SCRIPT} ${LAYER} data/wmt/*.txt
    qsub ${QSUB_ARGS} -N retrieval_${LAYER}_cls_center  ${ENV} \; ${SCRIPT} ${LAYER} data/wmt/*.txt --center-lng
    qsub ${QSUB_ARGS} -N retrieval_${LAYER}_mean        ${ENV} \; ${SCRIPT} ${LAYER} data/wmt/*.txt --mean-pool
    qsub ${QSUB_ARGS} -N retrieval_${LAYER}_mean_center ${ENV} \; ${SCRIPT} ${LAYER} data/wmt/*.txt --mean-pool --center-lng
done
