#ifndef TRANSFORMATION__
#define TRANSFORMATION__

#include "common.h"
#include "strategy.h"
#include "format_def.h"

template <typename T>
void local_format(ptr_handler handler, int *csrRowPtr, int *csrColIdx, T* csrValue,
int **hybrid_RowPtr, int **hybrid_ValPtr,
int &RowPtr_len, int **hybrid_idx,
int &ValPtr_len, T** hybrid_Value);

#endif