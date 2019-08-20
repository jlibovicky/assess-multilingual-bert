#!/bin/bash

QSUB_ARGS="-o logs/qe -e logs/qe -q cpu-troja.q -pe smp 4 -l mem_free=20G,act_mem_free=20G,h_vmem=20G -V -cwd -b y"
ENV="source env/bin/activate"
SCRIPT="./qe_by_cosine.py bert-base-multilingual-cased"

for PAIR in en-de en-ru; do
    for LAYER in {0..11}; do
        echo $LAYER
        qsub ${QSUB_ARGS} -N qe_${PAIR}_${LAYER}_cls         ${ENV} \; \
            ${SCRIPT} ${LAYER} data/quality_estimation/${PAIR}/test.{src,mt}
        qsub ${QSUB_ARGS} -N qe_${PAIR}_${LAYER}_cls_center  ${ENV} \; \
            ${SCRIPT} ${LAYER} data/quality_estimation/${PAIR}/test.{src,mt} --center-lng
        qsub ${QSUB_ARGS} -N qe_${PAIR}_${LAYER}_mean        ${ENV} \; \
            ${SCRIPT} ${LAYER} data/quality_estimation/${PAIR}/test.{src,mt} --mean-pool
        qsub ${QSUB_ARGS} -N qe_${PAIR}_${LAYER}_mean_center ${ENV} \; \
            ${SCRIPT} ${LAYER} data/quality_estimation/${PAIR}/test.{src,mt} --mean-pool --center-lng
    done
done
