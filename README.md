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
Python packages: Pytorch 3.

## Build up & Execution
### Compile
```
cd scripts
sh build.sh
```
### Transform a general matrix into a triangular matrix
1. Download the matrix from SuiteSparse Matrix Collection (<https://sparse.tamu.edu/>)
