#ifndef TEST_H__
#define TEST_H__

#include "AG-SpTRSV.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <unistd.h>

#define YY_TEST true
#define CU_TEST true

#define VALUE_TYPE double
#define VALUE_SIZE 8

#define ERROR_THRESH 1e-4

#define REPEAT_TIME 10

#define duration(a, b) (1.0 * (b.tv_usec - a.tv_usec + (b.tv_sec - a.tv_sec) * 1.0e6))

#endif