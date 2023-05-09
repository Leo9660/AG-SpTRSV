// As for kernel implementation for a single computation task, we leverage warp-level and thread-level parallelism
// use the implementation in YYSpTRSV (more details in ../include/YYSpTRSV) for reference.
// We further improve the performance of the kernels with optimizations of memory access and synchronizations.

if (row_st + 1 == row_ed)
{
    int idx_st = csrRowPtr[row_st];
    int idx_ed = csrRowPtr[row_st + 1];

    T rh_value = b[row_st];
    T diag_value = csrValue[idx_ed - 1];

    T leftsum = 0;
    for (int idx = idx_st + lane_id; idx < idx_ed - 1; idx += WARP_SIZE)
    {
        int dep_row = csrColIdx[idx];

        T dep_value = csrValue[idx];

        while (!get_value[dep_row])
        {
            READ_FENCE;
        }
        
        leftsum += dep_value * x[dep_row];
    }

    // warp-level reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        leftsum += __shfl_down_sync(0xffffffff, leftsum, offset);
    
    if (!lane_id)
    {
        x[row_st] = (rh_value - leftsum) / diag_value;
        WRITE_FENCE;
        get_value[row_st] = 1;
    }
}
else
{
    for (int row_iter = row_st + lane_id; row_iter < row_ed; row_iter += WARP_SIZE)
    {
        int idx = csrRowPtr[row_iter];
        T rh_value = b[row_iter];

        T leftsum = 0;

        while (idx < csrRowPtr[row_iter + 1])
        {
            int dep_row = csrColIdx[idx];

            if (dep_row == row_iter)
            {
                //x[row_iter] = (b[row_iter] - leftsum) / csrValue[idx];
                x[row_iter] = (rh_value - leftsum) / csrValue[idx];
                WRITE_FENCE;
                get_value[row_iter] = 1;
                idx++;
            }

            if (get_value[dep_row] == 1)
            {
                leftsum += csrValue[idx] * x[dep_row];
                idx++;
                dep_row = csrColIdx[idx];
            }
            else
            {
                READ_FENCE;
            }
        }
    }
}