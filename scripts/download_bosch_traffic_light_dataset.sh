#!/usr/bin/env bash

if [ "$#" -ne 1 ] || ! [ -f "$1" ]; then
  echo "Usage: $0 [path/to/cookies.txt]" >&2
  exit 1
fi

readonly CURRENT_DIR=$(dirname $(realpath $0))
readonly DATA_PATH_BASE=$(realpath ${CURRENT_DIR}/../data)
readonly DATA_PATH=${DATA_PATH_BASE}/bosch_traffic_light
if [ ! -d ${DATA_PATH} ]; then
    mkdir -p ${DATA_PATH}
fi

DOWNLOAD_FILES_RGB='
https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/470246335/dataset_additional_rgb.zip

https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/1916835690/dataset_train_rgb.zip.001
https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/2004912193/dataset_train_rgb.zip.002
https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/384278926/dataset_train_rgb.zip.003
https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/759073144/dataset_train_rgb.zip.004

https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/1861771760/dataset_test_rgb.zip.001
https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/1886457574/dataset_test_rgb.zip.002
https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/1581800089/dataset_test_rgb.zip.003
https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/379276232/dataset_test_rgb.zip.004
https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/865572993/dataset_test_rgb.zip.005
https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/1883532915/dataset_test_rgb.zip.006
https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/1549636406/dataset_test_rgb.zip.007
'

DOWNLOAD_FILES_PGM='
https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/283804895/dataset_additional_riib.zip

https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/1430601510/dataset_train_riib.zip.001
https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/1447506613/dataset_train_riib.zip.002
https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/2125697332/dataset_train_riib.zip.003
https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/194292996/dataset_train_riib.zip.004

https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/1681105773/dataset_test_riib.zip.001
https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/1944557291/dataset_test_riib.zip.002
https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/1581131605/dataset_test_riib.zip.003
https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/1444693807/dataset_test_riib.zip.004
https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/1672418119/dataset_test_riib.zip.005
https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/1441613972/dataset_test_riib.zip.006
'

for download_file in ${DOWNLOAD_FILES_RGB}; do
    file_name=$(basename $download_file)
    if [ ! -f $DATA_PATH/$file_name ]; then
        wget --cookies=on --load-cookies $1 --keep-session-cookies --no-check-certificate ${download_file} -P $DATA_PATH
        extension="${file_name##*.}"
        if [[ ${extension} == "zip" ]]; then
            unzip $DATA_PATH/${file_name} -d ${DATA_PATH}
        fi
    fi
done

function package_exists {
    dpkg -s "$1" &> /dev/null ;
}

if ! package_exists p7zip; then
    echo "install file archivers..."
    sudo apt-get install p7zip-full
fi

for entry in ${DATA_PATH}/*.001; do
    file_name=$(basename ${entry})
    mkdir -p ${DATA_PATH}/${file_name%.*.*}
    7za x ${entry} -o${DATA_PATH}/${file_name%.*.*}
done
