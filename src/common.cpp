#include "../include/common.h"

node_info_ :: node_info_(int start_row_input, int end_row_input):
            start_row(start_row_input), end_row(end_row_input)
{
    format = CSR;
    elim = NO_ELIM;
    //start_idx = end_idx = -1;
}

node_info_ :: node_info_()
{}

// node_info_ :: node_info_(int start_row_input, int end_row_input,
//             int start_idx_input, int end_idx_input):
//             start_row(start_row_input), end_row(end_row_input),
//             start_idx(start_idx_input), end_idx(end_idx_input) {}


node :: node(int nid, int start_row_input, int end_row_input, int nnz_input):
            info(start_row_input, end_row_input)
{
    id = nid;
    topo_level = 0;
    num_nnz = nnz_input;
    in_degree = out_degree = in_degree_tmp = 0;
    warp_sche_level = 0;
    ori_start = start_row_input;
    locality_node = NULL;
};

SpTRSV_handler :: ~SpTRSV_handler()
{
    // Actual destructor is implemented in finalize.cpp
}
