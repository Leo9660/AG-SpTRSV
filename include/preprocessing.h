#ifndef PREPROCESSING__
#define PREPROCESSING__

#include "common.h"
#include "strategy.h"
#include <cuda.h>

void write_graph(const char* file_name, ptr_handler handler, unsigned int max_depthm,
int layer, float parallelism);

void show_graph_layer(ptr_handler handler);

void get_matrix_info(const int    m,
                const int         nnz,
                const int        *csrRowPtr,
                const int        *csrColIdx,
                float            *avg_rnnz,
                float            *cov_rnnz,
                float            *avg_lnnz,
                float            *cov_lnnz);

void write_matrix_info(const char* file_name,
                const char*       matrix_name,
                const int         m,
                const int         nnz,
                const int        *csrRowPtr,
                const int        *csrColIdx);

#endif