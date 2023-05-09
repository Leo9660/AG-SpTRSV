#include "test.h"

extern float test_cusparse(int m, int nnzL, int *csrRowPtr_d, int *csrColIdx_d, 
VALUE_TYPE *csrValue_d, VALUE_TYPE *b_d, VALUE_TYPE *x_d);
extern float test_yy(int m, int nnzL, int *csrRowPtr, int *csrColIdx, VALUE_TYPE *csrValue,
int *csrRowPtr_d, int *csrColIdx_d, VALUE_TYPE *csrValue_d, 
VALUE_TYPE *b_d, VALUE_TYPE *x_d, int avg_thresh);
int cmp_vector(int m, const char* name1, const char* name2, VALUE_TYPE *x1, VALUE_TYPE *x2);

int main(int argc, char* argv[])
{
    struct timeval tv_begin, tv_end;

    int ch;
    int input_flag = 0;
    char *input_name;

    while ((ch = getopt(argc, argv, "i:")) != -1)
    {
        switch (ch)
        {
            case 'i':
                input_flag = 1;
                input_name = optarg;
                break;
        }
    }

    if (input_flag == 0)
    {
        printf("[Usage]: ./main_batch -i {input_filename}\n");
        exit(1);
    }

    // Triangular matrix L;
    int m;
    int nnzL;
    int *csrRowPtrL;
    int *csrColIdxL;
    VALUE_TYPE *csrValL;

    read_tri<VALUE_TYPE>(input_name, &m, &nnzL, &csrRowPtrL, &csrColIdxL, &csrValL);

    int layer;
    double parallelism;
    matrix_layer(m, m, nnzL, csrRowPtrL, csrColIdxL, &layer, &parallelism);

    // print matrix information
    printf("matrix information: location %s\n"
            "m %d nnz %d layer %d parallelism %.2f\n", 
            input_name, m, nnzL, layer, parallelism);

    // x & randomized b
    VALUE_TYPE *x, *b;
    x = (VALUE_TYPE*)malloc(m * sizeof(VALUE_TYPE));
    b = (VALUE_TYPE*)malloc(m * sizeof(VALUE_TYPE));
    srand(0);
    for (int i = 0; i < m; i++)
    {
        b[i] = rand() * 1.0 / RAND_MAX;
    }

    // copy matrix and vector from CPU to GPU memory
    int *csrRowPtr_d, *csrColIdx_d;
    VALUE_TYPE *csrValL_d, *b_d, *x_d;
    cudaMalloc(&csrRowPtr_d, (m + 1) * sizeof(int));
    cudaMemcpy(csrRowPtr_d, csrRowPtrL, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&csrColIdx_d, nnzL * sizeof(int));
    cudaMemcpy(csrColIdx_d, csrColIdxL, nnzL * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&csrValL_d, nnzL * sizeof(VALUE_TYPE));
    cudaMemcpy(csrValL_d, csrValL, nnzL * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
    cudaMalloc(&b_d, m * sizeof(VALUE_TYPE));
    cudaMemcpy(b_d, b, m * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
    cudaMalloc(&x_d, m * sizeof(VALUE_TYPE));
    cudaMemset(x_d, 0, sizeof(VALUE_TYPE) * m);

//--------------------AG-SpTRSV-RUN--------------------//

    // strategies
    PREPROCESSING_STRATEGY ps = ROW_BLOCK;
    SCHEDULE_STRATEGY ss = SIMPLE;
    int rb = 1;

    // preprocessing and get handler
    ptr_handler handler = SpTRSV_preprocessing(m, nnzL, csrRowPtrL, csrColIdxL, ps, rb);

    sptrsv_schedule(handler, ss);

    float sptrsv_time;
    for (int i = 0; i < REPEAT_TIME; i++)
    {
        cudaMemset(handler->get_value, 0, sizeof(int) * m);

        gettimeofday(&tv_begin, NULL);
        
        SpTRSV_executor<VALUE_TYPE>(handler, csrRowPtr_d, csrColIdx_d, csrValL_d, b_d, x_d);
        cudaDeviceSynchronize();

        gettimeofday(&tv_end, NULL);

        sptrsv_time += duration(tv_begin, tv_end);
    }

    sptrsv_time /= REPEAT_TIME;

    cudaMemcpy(x, x_d, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    SpTRSV_finalize(handler);

    delete handler;

    printf("AG solve time: %.2f us\n", sptrsv_time);

//--------------------AG-SpTRSV-END--------------------//

    // solution vector of on host memory for correctness test
    VALUE_TYPE *x_base;
    x_base = (VALUE_TYPE*)malloc(m * sizeof(VALUE_TYPE));

#if (CU_TEST == true)

    float cusparse_time = test_cusparse(m, nnzL, csrRowPtr_d, csrColIdx_d, csrValL_d, b_d, x_d);

    memset(x_base, 0, sizeof(VALUE_TYPE)*m);
    cudaMemcpy(x_base, x_d, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    cmp_vector(m, "AG", "cuSPARSE", x, x_base);

    printf("cuSPARSE solve time: %.2f us\n", cusparse_time);
    printf("AG Speedup over cuSPARSE: %.2f\n", cusparse_time / sptrsv_time);

#endif

#if (YY_TEST == true)

    float yy_time = test_yy(m, nnzL, csrRowPtrL, csrColIdxL, csrValL,
    csrRowPtr_d, csrColIdx_d, csrValL_d, b_d, x_d, 10);

    memset(x_base, 0, sizeof(VALUE_TYPE)*m);
    cudaMemcpy(x_base, x_d, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);

    cmp_vector(m, "AG", "YY", x, x_base);

    printf("YYSpTRSV solve time: %.2f us\n", yy_time);
    printf("AG Speedup over YYSpTRSV: %.2f\n", yy_time / sptrsv_time);

#endif

    // Finalize
    cudaFree(csrRowPtr_d);
    cudaFree(csrColIdx_d);
    cudaFree(csrValL_d);
    cudaFree(x_d);
    cudaFree(b_d);

}
