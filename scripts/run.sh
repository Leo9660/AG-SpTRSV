#!/bin/bash

if [ $# -ne 1 ]; then
    echo "[Usage]: run.sh {matrix_file_name}"
    exit 1
fi

matrix=$1

./test/test -i ${matrix}
