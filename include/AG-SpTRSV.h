#ifndef AG_SPTRSV__
#define AG_SPTRSV__

#include "GPU_setup.h"
#include "preprocessing.h"
#include "schedule.h"
#include "common.h"
#include "strategy.h"
#include "specialization.h"
#include "finalize.h"
#include "format_def.h"
#include "transformation.h"
#include "elimination.h"

// Matrix inspection and graph generation in Analyse stage
ptr_handler SpTRSV_preprocessing(const int m,
                const int nnz,
                const int *csrRowPtr,
                const int *csrColIdx,
                PREPROCESSING_STRATEGY strategy,
                int row_block);

// Graph reordering
void graph_reorder_with_level(ptr_handler handler);

// Matrix storage transformation
template <typename T>
void matrix_reorder(ptr_handler handler, int *permutation,
            int *csrRowPtr, int *csrColIdx, T* csrValue);

// Schedule nodes onto GPU warps in Schedule stage
void sptrsv_schedule(ptr_handler handler, SCHEDULE_STRATEGY strategy);

// Kernel Execution in Execute stage
template <typename T>
void SpTRSV_executor(ptr_handler handler, 
            const int *csrRowPtr, const int *csrColIdx, const T* csrValue,
            const T* b, T* x);

// In development
// template <typename T>
// void SpTRSV_executor_hybrid(ptr_handler handler, 
//             const int *RowPtr_d, const int *ValPtr_d,
//             const int *idx_d, const T* value_d,
//             const T* b_d, T* x_d);

#endif