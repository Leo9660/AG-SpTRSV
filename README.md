# AG-SpTRSV
An auto-tuning framework to accelerate Sparse Triangular Solve on GPU

## Environmental setup
### Configurations to run AG-SpTRSV
GPU:   NVIDIA Telsa A100, RTX3080 Ti  
CUDA:  11.4  
GCC:   >= 7.5.0  
CMAKE: >= 3.5  
Linux: Ubuntu 18.04
### Extra configurations for the performance model
Python: 3.9  
Python packages: Pytorch 3.1.12

## Build up & Execution
### Compile
```
cd scripts
sh build.sh
```
### Transform the matrix
1. Download the matrix from SuiteSparse Matrix Collection (<https://sparse.tamu.edu/>). Sample matrices are in ``matrix/matrix_sample``.
2. Transform the general matrix into a triangular matrix. Run ``./matrix/transfer {.mtx file} {output file}``. For example: 
```
./matrix/transfer matrix/matrix_sample/delaunay_n13.mtx matrix/matrix_sample_csr/delaunay_n13.csr
```
### Run AG-SpTRSV with manually specified strategies
1. Modify Line 80-82 in code file ``test/test.cu``. Strategies are provided in ``include/strategy.h``. Supported strategies are listed as follows:  

| Analyse Strategy | Description | parameter | 
| -----------      | ----------- | --------- |
| ROW_BLOCK | Merge nodes with every $rb$ rows | $rb$ |
| ROW_BLOCK_THRESH | Merge nodes until one row has more than $rb$ nnz or there are 32 nodes | $rb$ |
| ROW_BLOCK_AVG | Merge nodes if the average nnz of 32 rows is more that $rb$ | $rb$ |

| Schedule Strategy | Description |
| ----------------- | ----------- |
| SIMPLE | Schedule nodes on warps in a round-robin way |
| WORKLOAD_BALANCE | Schedule nodes on the warp with minimum workload |
| WARP_LOCALITY | Schedule nodes on the near warp of their parents |
| BALANCE_AND_LOCALITY | Estimated cost model, described in the paper |
| SEQUENTIAL |  |
| SEQUENTIAL2 | Leave scheduling to GPU hardware |

2. Run ``sh scripts/run.sh {input matrix}`` to evaluate the performance of AG-SpTRSV with a single matrix. For example:
```
cd scripts
sh scripts/run.sh matrix/matrix_sample_csr/delaunay_n13.csr
```
### Run AG-SpTRSV with exaustive search

