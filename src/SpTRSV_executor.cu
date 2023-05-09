#include "../include/AG-SpTRSV.h"
#include <cuda_runtime.h>

// In this implementation, we assume that
// diagonal elements of the matrix are explicitly stored, 
// together with off-diagonal elements in CSR format
template<typename T>
__global__ void SpTRSV_simple(int *level, node_info **info, 
            const int *csrRowPtr, const int *csrColIdx, const T* csrValue,
            const T* b, T* x, int *get_value)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid / WARP_SIZE;
    int global_wid = bid * WARP_NUM_PER_BLOCK + wid;
    int lane_id = tid % WARP_SIZE;

    for (int schedule_i = 0; schedule_i < level[global_wid]; schedule_i++)
    {
        node_info current_info = info[global_wid][schedule_i];

        int row_st = current_info.start_row;
        int row_ed = current_info.end_row;

        SYNC_ELIM elim = current_info.elim;

        // One warp for one row
        if (elim == NO_ELIM)
        {
            #define WRITE_FENCE __threadfence()
            #define READ_FENCE __threadfence()

            #include "SpTRSV_executor_code.cu"

            #undef WRITE_FENCE
            #undef READ_FENCE
        }
        else if (elim == NO_WRITE_FENCE)
        {
            #define WRITE_FENCE
            #define READ_FENCE __threadfence()

            #include "SpTRSV_executor_code.cu"

            #undef WRITE_FENCE
            #undef READ_FENCE
        }
        else if (elim == WRITE_FENCE_BLOCK)
        {
            #define WRITE_FENCE __threadfence_block()
            #define READ_FENCE __threadfence()

            #include "SpTRSV_executor_code.cu"

            #undef WRITE_FENCE
            #undef READ_FENCE
        }
    }
}

template<typename T>
__global__ void SpTRSV_simple_no_schedule(node_info *info, int total_node,
            const int *csrRowPtr, const int *csrColIdx, const T* csrValue,
            const T* b, T* x, int *get_value)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid / WARP_SIZE;
    int global_wid = bid * WARP_NUM_PER_BLOCK + wid;
    int lane_id = tid % WARP_SIZE;

    if (global_wid > total_node) return;

    node_info current_info = info[global_wid];

    int row_st = current_info.start_row;
    int row_ed = current_info.end_row;

    SYNC_ELIM elim = current_info.elim;

    if (elim == NO_ELIM)
    {
        #define WRITE_FENCE __threadfence()
        #define READ_FENCE __threadfence()

        #include "SpTRSV_executor_code.cu"

        #undef WRITE_FENCE
        #undef READ_FENCE
    }
    else if (elim == NO_WRITE_FENCE)
    {
        #define WRITE_FENCE
        #define READ_FENCE __threadfence()

        #include "SpTRSV_executor_code.cu"

        #undef WRITE_FENCE
        #undef READ_FENCE
    }
    else if (elim == WRITE_FENCE_BLOCK)
    {
        #define WRITE_FENCE __threadfence_block()
        #define READ_FENCE __threadfence()

        #include "SpTRSV_executor_code.cu"

        #undef WRITE_FENCE
        #undef READ_FENCE
    }
}

template <typename T>
void SpTRSV_executor(ptr_handler handler, 
            const int *csrRowPtr_d, const int *csrColIdx_d, const T* csrValue_d,
            const T* b_d, T* x_d)
{
    if (handler->sched_s != SEQUENTIAL2)
        SpTRSV_simple<T><<<BLOCK_NUM, THREAD_NUM_PER_BLOCK>>>(handler->schedule_level_d, 
        handler->schedule_info_d, csrRowPtr_d, csrColIdx_d, csrValue_d, 
        b_d, x_d, handler->get_value);
    else
    {
        int total_node = handler->graph->global_node;
        int total_warp_num = total_node;
        int total_thread_num = total_node * WARP_SIZE;
        int block_num = (total_thread_num - 1) / THREAD_NUM_PER_BLOCK + 1;

        SpTRSV_simple_no_schedule<T><<<block_num, THREAD_NUM_PER_BLOCK>>>
        (handler->no_schedule_info_d, total_node, csrRowPtr_d, csrColIdx_d, csrValue_d,
        b_d, x_d, handler->get_value);
    }
}

// In development
template<typename T>
__global__ void SpTRSV_hybrid(int *level, node_info **info, 
            const int *RowPtr, const int *ValPtr, const int *idx, const T* value,
            const T* b, T* x, int *get_value)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid / WARP_SIZE;
    int global_wid = bid * WARP_NUM_PER_BLOCK + wid;
    int lane_id = tid % WARP_SIZE;

    for (int schedule_i = 0; schedule_i < level[global_wid]; schedule_i++)
    {
        node_info current_info = info[global_wid][schedule_i];

        int row_st = current_info.start_row;
        int row_ed = current_info.end_row;

        if (current_info.format == TELL)
        {
            int row_num = row_ed - row_st;
            int idx_st = RowPtr[row_st];
            int idx_ed = RowPtr[row_ed];
            int val_st = ValPtr[row_st];
            int val_ed = ValPtr[row_ed];

            for (int row_iter = row_st + lane_id; row_iter < row_ed; row_iter += WARP_SIZE)
            {
                T leftsum = 0;
                T rh_value = b[row_iter];

                __threadfence();

                int row_offset = row_iter - row_st;
                int val_i = val_st + row_offset;
                for (int idx_i = idx_st + row_offset; idx_i < idx_ed; idx_i += row_num)
                {
                    int dep_row = idx[idx_i];
                    while (!get_value[dep_row])
                    {
                        __threadfence();
                    }
                    leftsum += value[val_i] * x[dep_row];
                    val_i += row_num;
                }
                x[row_iter] = (rh_value - leftsum) / value[val_i];
                __threadfence();
                get_value[row_iter] = 1;
            }
        }
        else if (current_info.format == CSR)
        {
            printf("Not implemented!\n");
        }
    }
}

template <typename T>
void SpTRSV_executor_hybrid(ptr_handler handler, 
            const int *RowPtr_d, const int *ValPtr_d,
            const int *idx_d, const T* value_d,
            const T* b_d, T* x_d)
{
    SpTRSV_hybrid<T><<<BLOCK_NUM, THREAD_NUM_PER_BLOCK>>>(handler->schedule_level_d, 
    handler->schedule_info_d, RowPtr_d, ValPtr_d, idx_d, value_d, 
    b_d, x_d, handler->get_value);
}

// instance
template void SpTRSV_executor<float>(ptr_handler handler, 
            const int *csrRowPtr, const int *csrColIdx, const float* csrValue,
            const float* b, float* x);

template void SpTRSV_executor<double>(ptr_handler handler, 
            const int *csrRowPtr, const int *csrColIdx, const double* csrValue,
            const double* b, double* x);

template void SpTRSV_executor_hybrid<float>(ptr_handler handler, 
            const int *RowPtr_d, const int *ValPtr_d,
            const int *idx_d, const float* value_d,
            const float* b_d, float* x_d);

template void SpTRSV_executor_hybrid<double>(ptr_handler handler, 
            const int *RowPtr_d, const int *ValPtr_d,
            const int *idx_d, const double* value_d,
            const double* b_d, double* x_d);