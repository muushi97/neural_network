#!/bin/bash

LRS=(0.0001 0.001 0.01 0.1)
BSS=(64 128 256 512)
WS=(1.0 0.5 0.1)
HLNS=(10 30 50)

ES=500
H=20

N=10
NUM_PROCESS=5


for hln in ${HLNS[@]}; do
    for w in ${WS[@]}; do
        nw=`awk "BEGIN { print -1*$w }"`
        for bs in ${BSS[@]}; do
            for lr in ${LRS[@]}; do
                pf=out/param_${hln}_${w}_${bs}_${lr}
                echo "-lr" ${lr} "-bs" ${bs} "-w uni" ${nw} ${w} "-es" ${ES} "-m 3 784 sigm" ${hln} "sigm 10 -or 0.1 0.9 -ep" ${H} "-o out/ttt" 0 > ${pf}
                seq 1 ${N} | xargs -P ${NUM_PROCESS} -I{} ./neural_network.out expe1 -lr ${lr} -bs ${bs} -w  uni ${nw} ${w} -es ${ES} -m 3 784 sigm ${hln} sigm 10 -or 0.1 0.9 -ep ${H} -o ${pf} {}
            done
        done
    done
done

#./neural_network.out expe1 -lr 0.01 -bs 128 -w  uni -0.4 0.4 -es 200 -m 2 784 sigm 10 -or 0.1 0.9 -ep 20 -o out/ttt 1

