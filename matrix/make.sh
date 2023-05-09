UTILS_DIR=../utils
LIB_DIR=../build

g++ transfer_to_tri.cpp -o transfer -fpermissive -I${UTILS_DIR} -L${LIB_DIR} -lutils