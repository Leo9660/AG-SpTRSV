#ifndef COMMON__
#define COMMON__

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "GPU_setup.h"
#include "strategy.h"
#include <string>
using namespace std;

enum LOCAL_FORMAT
{
    CSR,
    TELL
};

enum SYNC_ELIM
{
    NO_ELIM,
    //READ_WRITE_BLOCK,
    //NO_READ_FENCE,
    NO_WRITE_FENCE,
    WRITE_FENCE_BLOCK
    //NO_FENCE,
    //NO_WRITE_FLAG
};

typedef struct node_info_
{
    // node row ID
    int start_row;
    int end_row;

    // node format
    LOCAL_FORMAT format;
    SYNC_ELIM elim;

    node_info_(int start_row_input, int end_row_input);
    node_info_();

} node_info;

typedef struct node* ptr_node;
struct node
{
    node_info info;

    // node id
    int id;

    // in & out degree, used in topological sort
    int in_degree;
    int out_degree;
    int in_degree_tmp;

    // number of nnz in the row_block
    int num_nnz;

    // row start position in original matrix (in case of reordering)
    int ori_start;

    // topological level
    int topo_level;

    // ID of the warp the node scheduled to, and schedule level
    int warp_id;

    // schedule level of current warp
    int warp_sche_level;

    // child nodes of data dependency edges
    vector<ptr_node> child;

    // parent nodes of data dependency edges
    vector<ptr_node> parent;

    // locality edge
    ptr_node locality_node;

    node(int nid, int start_row_input, int end_row_input, int nnz_input);
};

typedef struct graph* ptr_graph;
struct graph
{
    // total number of nodes
    int global_node;

    // total number of edges
    int global_edge;

    // List all nodes with no parent node
    vector<ptr_node> start_nodes;

    graph() 
    {
        global_node = global_edge = 0;
    }
};

typedef struct SpTRSV_handler* ptr_handler;
struct SpTRSV_handler
{
    // Number of rows
    int m;
    int nnz;
    int row_block;

    // Dependency graph of the matrix
    ptr_graph graph;

    // Schedule vector
    vector<ptr_node> warp_schedule[WARP_NUM];

    // Schedule info for each warp
    int *schedule_level;
    node_info **schedule_info;

    // Schedule info on device memory
    // Currently, preprocessing is implemented on CPU, 
    // considering transferring this stage to GPU in the future
    int *schedule_level_d;
    node_info **schedule_info_d;

    // When no schedule strategy is enabled, using the hardware scheduler
    node_info *no_schedule_info;
    node_info *no_schedule_info_d;

    // Array for data dependency
    int *get_value;
    int *warp_runtime;

    // Schedule strategy
    SCHEDULE_STRATEGY sched_s;

    ~SpTRSV_handler();

};

#endif