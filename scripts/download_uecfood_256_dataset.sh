#!/usr/bin/env bash

readonly CURRENT_DIR=$(dirname $(realpath $0))
readonly DATA_PATH_BASE=$(realpath ${CURRENT_DIR}/../data)
readonly DATA_PATH=${DATA_PATH_BASE}/dataset256
if [ ! -d ${DATA_PATH} ]; then
    mkdir -p ${DATA_PATH}
fi

if [ ! -f ${DATA_PATH}.zip ] || [ ! -d ${DATA_PATH} ]; then
    mkdir -p ${DATA_PATH}
    wget http://foodcam.mobi/dataset256.zip -P ${DATA_PATH_BASE}
    unzip ${DATA_PATH}.zip -d ${DATA_PATH}
    python3 ${CURRENT_DIR}/split_train_val_uecfood_256_dataset.py
fi
