#ifndef STRATEGY__
#define STRATEGY__

enum PREPROCESSING_STRATEGY
{
    ROW_BLOCK,
    ROW_BLOCK_THRESH,
    ROW_BLOCK_AVG,
    // The following strategies are in development, do not recommend to use
    SUPERNODE_BLOCK
};

enum SCHEDULE_STRATEGY
{
    SIMPLE,
    WORKLOAD_BALANCE,
    WARP_LOCALITY,
    BALANCE_AND_LOCALITY,
    SEQUENTIAL,
    SEQUENTIAL2,
    // The following strategies are in development, do not recommend to use
    STRUCTURED,
    WINDOW
};

#endif