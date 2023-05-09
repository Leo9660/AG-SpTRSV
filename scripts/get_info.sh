#!/bin/bash

if [ $# -ne 2 ]; then
    echo "[Usage]: get_info.sh {input matrix} {output file}"
    exit 1
fi

matrix=$1
output=$2

./matrix/info -i ${matrix} -o ${output}
