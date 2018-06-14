#!/bin/sh

TRAIN_STRIDE=(0 1)
CONCAT_LABELS=(0 1)
HIDDEN_STATE_SIZE=(128 256 512)
ATTENTION=(0 1)
SHARE_WEIGHTS=(1 0)
LEARNING_RATE=('exponential' 'fixed')

echo ${LEARNING_RATE[0]}

for TS in ${TRAIN_STRIDE[@]}
do
    for CL in ${CONCAT_LABELS[@]}
    do
        for HS in ${HIDDEN_STATE_SIZE[@]}
        do
            for ATT in ${ATTENTION[@]}
            do
                for SW in ${SHARE_WEIGHTS[@]}
                do
                    for LR in ${LEARNING_RATE[@]}
                    do
                        python train.py $TS $CL $HS $ATT $SW $LR
                    done
                done
            done
        done
    done
done