#!/bin/bash

if [ ! -d "build" ]; then
    mkdir build
fi

cd build
cmake ..
make
cd ..

cd matrix
make clean
make
cd ..

cd test
make clean
make all
cd ..
