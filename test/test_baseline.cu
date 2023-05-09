#include "test.h"
#include "YYSpTRSV.h"

struct timeval tv_begin, tv_end;

float test_cusparse(int m, int nnzL, int *csrRowPtr_d, int *csrColIdx_d, 
VALUE_TYPE *csrValue_d, VALUE_TYPE *b_d, VALUE_TYPE *x_d)
{
    // cuSparse
    cusparseHandle_t cusparse_handler;
    cusparseStatus_t ErrorStatus;
    ErrorStatus = cusparseCreate(&cusparse_handler);

    cusparseMatDescr_t desc;
    ErrorStatus = cusparseCreateMatDescr(&desc);
    cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(desc, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(desc, CUSPARSE_DIAG_TYPE_NON_UNIT);

    bsrsv2Info_t cusparse_info;
    cusparseCreateBsrsv2Info(&cusparse_info);

    int buffer_size;
#if (VALUE_SIZE == 4)
    ErrorStatus = cusparseSbsrsv2_bufferSize(cusparse_handler,
    CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnzL,
    desc, csrValue_d, csrRowPtr_d, csrColIdx_d, 1, cusparse_info, &buffer_size);
#else
    ErrorStatus = cusparseDbsrsv2_bufferSize(cusparse_handler,
    CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnzL,
    desc, csrValue_d, csrRowPtr_d, csrColIdx_d, 1, cusparse_info, &buffer_size);
#endif

    if (ErrorStatus != CUSPARSE_STATUS_SUCCESS)
    {
        printf("Error in buffersize stage!\n");
        exit(1);
    }

    void *cusparse_buffer;
    cudaMalloc((void **)&cusparse_buffer, buffer_size);

#if (VALUE_SIZE == 4)
    ErrorStatus = cusparseSbsrsv2_analysis(cusparse_handler,
    CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnzL,
    desc, csrValue_d, csrRowPtr_d, csrColIdx_d, 1, cusparse_info, 
    CUSPARSE_SOLVE_POLICY_USE_LEVEL, cusparse_buffer);
#else
    ErrorStatus = cusparseDbsrsv2_analysis(cusparse_handler,
    CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnzL,
    desc, csrValue_d, csrRowPtr_d, csrColIdx_d, 1, cusparse_info, 
    CUSPARSE_SOLVE_POLICY_USE_LEVEL, cusparse_buffer);
#endif
    if (ErrorStatus != CUSPARSE_STATUS_SUCCESS)
    {
        printf("Error in analysis stage!\n");
        printf("%s\n", cusparseGetErrorString(ErrorStatus));
        exit(1);
    }
    int structural_zero;
    ErrorStatus = cusparseXbsrsv2_zeroPivot(cusparse_handler, cusparse_info, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == ErrorStatus)
        printf("L(%d,%d) is missing\n", structural_zero, structural_zero);

    VALUE_TYPE alpha = 1.0;

    float cusparse_time = 0;

    for (int i = 0; i < REPEAT_TIME; i++)
    {
        cudaMemset(x_d, 0, m * sizeof(VALUE_TYPE));

        gettimeofday(&tv_begin, NULL);

#if (VALUE_SIZE == 4)
        ErrorStatus = cusparseSbsrsv2_solve(cusparse_handler,
        CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnzL,
        &alpha, desc, csrValue_d, csrRowPtr_d, csrColIdx_d, 1, cusparse_info,
        b_d, x_d, CUSPARSE_SOLVE_POLICY_USE_LEVEL, cusparse_buffer);
#else
        ErrorStatus = cusparseDbsrsv2_solve(cusparse_handler,
        CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnzL,
        &alpha, desc, csrValue_d, csrRowPtr_d, csrColIdx_d, 1, cusparse_info,
        b_d, x_d, CUSPARSE_SOLVE_POLICY_USE_LEVEL, cusparse_buffer);
#endif
        cudaDeviceSynchronize();

        gettimeofday(&tv_end, NULL);
        
        cusparse_time += duration(tv_begin, tv_end);

        if (ErrorStatus != CUSPARSE_STATUS_SUCCESS)
        {
            printf("Error in solve stage!\n");
            exit(1);
        }
    }

    // L has unit diagonal, so no numerical zero is reported.
    int numerical_zero;
    ErrorStatus = cusparseXbsrsv2_zeroPivot(cusparse_handler, cusparse_info, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == ErrorStatus){
        printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }

    cusparse_time /= REPEAT_TIME;

    cudaFree(cusparse_buffer);

    return cusparse_time;
}

float test_yy(int m, int nnzL, int *csrRowPtr, int *csrColIdx, VALUE_TYPE *csrValue,
int *csrRowPtr_d, int *csrColIdx_d, VALUE_TYPE *csrValue_d, 
VALUE_TYPE *b_d, VALUE_TYPE *x_d, int avg_thresh)
{

    // YYSpTRSV
    int Len;
    int *warp_num=(int *)malloc((m+1)*sizeof(int));
    if (warp_num==NULL)
        printf("warp_num error\n");
    memset(warp_num, 0, sizeof(int)*(m+1));
    
    double warp_occupy=0,element_occupy=0;
    matrix_warp(m, m, nnzL, csrRowPtr, csrColIdx,
    avg_thresh, &Len, warp_num, &warp_occupy, &element_occupy);

    int *d_warp_num;
    cudaMalloc((void **)&d_warp_num, Len  * sizeof(int));
    cudaMemcpy(d_warp_num, warp_num, Len * sizeof(int), cudaMemcpyHostToDevice);

    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int num_blocks = ceil ((double)((Len-1)*WARP_SIZE) / (double)(num_threads));
    
    int *d_id_extractor;
    cudaMalloc((void **)&d_id_extractor, sizeof(int));

    int *yy_get_value;
    cudaMalloc(&yy_get_value, m * sizeof(int));
    cudaMemset(yy_get_value, 0, sizeof(int) * m);
    
    float yy_time = 0;

    for (int i = 0; i < REPEAT_TIME; i++)
    {
        cudaMemset(yy_get_value, 0, sizeof(int) * m);
        cudaMemset(d_id_extractor, 0, sizeof(int));
        cudaMemset(x_d, 0, m * sizeof(VALUE_TYPE));

        cudaDeviceSynchronize();
        
        gettimeofday(&tv_begin, NULL);
        
        yySpTRSV_csr_kernel<<< num_blocks, num_threads >>> (csrRowPtr_d, csrColIdx_d, 
        csrValue_d, yy_get_value, m, nnzL, b_d, x_d, 0, d_warp_num, Len, d_id_extractor);
        cudaDeviceSynchronize();
        
        gettimeofday(&tv_end, NULL);
        
        yy_time += duration(tv_begin, tv_end);
        
    }

    yy_time /= REPEAT_TIME;

    cudaFree(d_warp_num);
    cudaFree(d_id_extractor);
    cudaFree(yy_get_value);

    return yy_time;
}

int cmp_vector(int m, const char* name1, const char* name2, VALUE_TYPE *x1, VALUE_TYPE *x2)
{
    for (int i = 0; i < m; i++)
    {
        if (fabs(x1[i] - x2[i]) > ERROR_THRESH)
        {
            printf("%s vs %s error at index %d, x1 = %.5f, x2 = %.5f!\n", 
            name1, name2, i, x1[i], x2[i]);
            return 0;
        }
    }
    printf("%s vs %s correct!\n", name1, name2);
    return 1;
}