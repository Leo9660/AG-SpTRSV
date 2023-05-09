CC=nvcc

LFLAGS=-lSpTRSV -lutils -lcusparse

INCLUDE_DIR=../include
UTILS_DIR=../utils
LIB_DIR=../build

all: test test_search

test: test.cu test_baseline.cu
	$(CC) $^ -o $@ -I$(INCLUDE_DIR) -I$(UTILS_DIR) -L$(LIB_DIR) $(LFLAGS)

test_search: test_search.cu test_baseline.cu
	$(CC) $^ -o $@ -I$(INCLUDE_DIR) -I$(UTILS_DIR) -L$(LIB_DIR) $(LFLAGS)

.PHONY:

clean:
	rm test test_search
