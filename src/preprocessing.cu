#include <stdlib.h>
#include <string>
#include <queue>
#include <algorithm>
#include <math.h>
#include <unistd.h>
#include "../include/preprocessing.h"
#include "../include/common.h"

void show_level(ptr_graph ret, int topo_max)
{
    //int topo_max = 2;
    queue<ptr_node> tmp_queue;
    for (int i = 0; i < ret->start_nodes.size(); i++)
    {
        tmp_queue.push(ret->start_nodes[i]);
    }
    while (!tmp_queue.empty())
    {
        ptr_node tmp_node = tmp_queue.front();
        if (tmp_node->topo_level < topo_max)
        {
            for (auto i = tmp_node->child.begin(); i != tmp_node->child.end(); i++)
            {
                (*i)->in_degree_tmp++;
                if ((*i)->in_degree_tmp == (*i)->in_degree)
                {
                    tmp_queue.push(*i);
                    (*i)->in_degree_tmp = 0;
                }
            }
        }
        printf("node id %d row start %d topo_level %d\n", tmp_node->id, tmp_node->info.start_row, tmp_node->topo_level);
        tmp_queue.pop();
    }
    printf("\n");
}

// Generate dependency graph every (row_block) rows
ptr_graph generate_graph_row_block(const int m, const int nnz,
                const int *csrRowPtr, const int *csrColIdx,
                const int row_block)
{
    ptr_graph ret = new graph();

    ptr_node tmp_nodes[m / row_block];

    ptr_node previous_node = NULL;

    for (int row = 0; row < m; row += row_block)
    {
        int row_ed = min(row + row_block, m);
        int cur_row_block = row / row_block;

        tmp_nodes[cur_row_block] = new node(ret->global_node,
        row, row_ed, csrRowPtr[row_ed] - csrRowPtr[row]);

        ret->global_node++;

        vector<int> dep_list;

        int only_diag = 1;
        for (int idx = csrRowPtr[row]; idx < csrRowPtr[row_ed]; idx++)
        {
            int col_idx = csrColIdx[idx];
            int dep_row_block = col_idx / row_block;
            if (dep_row_block < cur_row_block)
            {
                dep_list.push_back(dep_row_block);
                only_diag = 0;
            }
        }

        sort(dep_list.begin(), dep_list.end());
        auto end_pos = unique(dep_list.begin(), dep_list.end());

        for (auto dep_i = dep_list.begin(); dep_i != end_pos; dep_i++)
        {
            int dep_row_block = *(dep_i);

            tmp_nodes[dep_row_block]->child.push_back(tmp_nodes[cur_row_block]);
            tmp_nodes[dep_row_block]->out_degree++;
            tmp_nodes[cur_row_block]->parent.push_back(tmp_nodes[dep_row_block]);
            tmp_nodes[cur_row_block]->in_degree++;
            if (tmp_nodes[dep_row_block]->topo_level + 1 > tmp_nodes[cur_row_block]->topo_level)
                tmp_nodes[cur_row_block]->topo_level = tmp_nodes[dep_row_block]->topo_level + 1;
            ret->global_edge++;
        }

        if (previous_node)
        {
            previous_node->locality_node = tmp_nodes[cur_row_block];
        }
        previous_node = tmp_nodes[cur_row_block];

        // Identify row blocks with only diagonal elements
        if (only_diag)
        {
            ret->start_nodes.push_back(tmp_nodes[cur_row_block]);
        }
    }

    return ret;
}

ptr_graph generate_graph_row_block_thresh(const int m, const int nnz,
                const int *csrRowPtr, const int *csrColIdx, int thresh)
{
    ptr_graph ret = new graph();
    
    ptr_node tmp_nodes[m];

    int hash_idx[m];

    for (int i = 0; i < m; i++) hash_idx[i] = -1;

    int row = 0;
    int node_count = 0;

    ptr_node previous_node = NULL;

    while (row < m)
    {
        int row_st = row;
        hash_idx[row] = node_count;
        if (csrRowPtr[row + 1] - csrRowPtr[row] > thresh)
        {
            hash_idx[row] = node_count;
            row++;
        }
        else
        {
            //while (row < m && csrRowPtr[row + 1] - csrRowPtr[row] <= thresh)
            while (row < m && row - row_st < WARP_SIZE && csrRowPtr[row + 1] - csrRowPtr[row] <= thresh)
            {
                hash_idx[row] = node_count;
                row++;
            }
        }
        tmp_nodes[node_count] = new node(ret->global_node,
        row_st, row, csrRowPtr[row] - csrRowPtr[row_st]);
        ret->global_node++;

        int cur_row_block = hash_idx[row_st];

        vector<int> dep_list;

        int only_diag = 1;
        for (int idx = csrRowPtr[row_st]; idx < csrRowPtr[row]; idx++)
        {
            int col_idx = csrColIdx[idx];
            int dep_row_block = hash_idx[col_idx];
            if (dep_row_block < cur_row_block)
            {
                dep_list.push_back(dep_row_block);
                only_diag = 0;
            }
        }

        sort(dep_list.begin(), dep_list.end());
        auto end_pos = unique(dep_list.begin(), dep_list.end());

        for (auto dep_i = dep_list.begin(); dep_i != end_pos; dep_i++)
        {
            int dep_row_block = *(dep_i);

            tmp_nodes[dep_row_block]->child.push_back(tmp_nodes[cur_row_block]);
            tmp_nodes[dep_row_block]->out_degree++;
            tmp_nodes[cur_row_block]->parent.push_back(tmp_nodes[dep_row_block]);
            tmp_nodes[cur_row_block]->in_degree++;
            if (tmp_nodes[dep_row_block]->topo_level + 1 > tmp_nodes[cur_row_block]->topo_level)
                tmp_nodes[cur_row_block]->topo_level = tmp_nodes[dep_row_block]->topo_level + 1;
            ret->global_edge++;
        }

        if (previous_node)
        {
            previous_node->locality_node = tmp_nodes[cur_row_block];
        }
        previous_node = tmp_nodes[cur_row_block];

        // Identify row blocks with only diagonal elements
        if (only_diag)
        {
            ret->start_nodes.push_back(tmp_nodes[node_count]);
        }

        node_count++;

    }

    // for (auto i = tmp_nodes[0]->child.begin(); i != tmp_nodes[0]->child.end(); i++)
    //     printf("child id %d\n", (*i)->id);

    return ret;
}

ptr_graph generate_graph_row_block_avg(const int m, const int nnz,
                const int *csrRowPtr, const int *csrColIdx, int thresh)
{
    ptr_graph ret = new graph();
    
    ptr_node tmp_nodes[m];

    int hash_idx[m];

    for (int i = 0; i < m; i++) hash_idx[i] = -1;

    ptr_node previous_node = NULL;

    for (int row_st = 0; row_st < m; row_st += WARP_SIZE)
    {
        int row_ed = min(row_st + WARP_SIZE, m);

        float avg_nnz = csrRowPtr[row_ed] - csrRowPtr[row_st];
        if (row_ed - row_st) avg_nnz /= (row_ed - row_st);

        if (avg_nnz < thresh)
        {
            //printf("thread_level\n");
            for (int i = row_st; i < row_ed; i++)
                hash_idx[i] = ret->global_node;
            
            int cur_row_block = ret->global_node;

            tmp_nodes[cur_row_block] = new node(ret->global_node,
            row_st, row_ed, csrRowPtr[row_ed] - csrRowPtr[row_st]);
            ret->global_node++;

            vector<int> dep_list;

            int only_diag = 1;
            for (int idx = csrRowPtr[row_st]; idx < csrRowPtr[row_ed]; idx++)
            {
                int col_idx = csrColIdx[idx];
                int dep_row_block = hash_idx[col_idx];
                if (dep_row_block < cur_row_block)
                {
                    dep_list.push_back(dep_row_block);
                    only_diag = 0;
                }
            }

            sort(dep_list.begin(), dep_list.end());
            auto end_pos = unique(dep_list.begin(), dep_list.end());

            for (auto dep_i = dep_list.begin(); dep_i != end_pos; dep_i++)
            {
                int dep_row_block = *(dep_i);

                tmp_nodes[dep_row_block]->child.push_back(tmp_nodes[cur_row_block]);
                tmp_nodes[dep_row_block]->out_degree++;
                tmp_nodes[cur_row_block]->parent.push_back(tmp_nodes[dep_row_block]);
                tmp_nodes[cur_row_block]->in_degree++;
                if (tmp_nodes[dep_row_block]->topo_level + 1 > tmp_nodes[cur_row_block]->topo_level)
                    tmp_nodes[cur_row_block]->topo_level = tmp_nodes[dep_row_block]->topo_level + 1;
                ret->global_edge++;
            }

            if (previous_node)
            {
                previous_node->locality_node = tmp_nodes[cur_row_block];
            }
            previous_node = tmp_nodes[cur_row_block];

            // Identify row blocks with only diagonal elements
            if (only_diag)
            {
                ret->start_nodes.push_back(tmp_nodes[cur_row_block]);
            }
        }
        else
        {
            //printf("warp_level\n");
            int cur_row_block;
            for (int i = row_st; i < row_ed; i++)
            {
                hash_idx[i] = ret->global_node;

                cur_row_block = ret->global_node;

                tmp_nodes[cur_row_block] = new node(ret->global_node,
                i, i + 1, csrRowPtr[i + 1] - csrRowPtr[i]);
                ret->global_node++;

                vector<int> dep_list;

                int only_diag = 1;
                for (int idx = csrRowPtr[i]; idx < csrRowPtr[i+1]; idx++)
                {
                    int col_idx = csrColIdx[idx];
                    int dep_row_block = hash_idx[col_idx];
                    if (dep_row_block < cur_row_block)
                    {
                        dep_list.push_back(dep_row_block);
                        only_diag = 0;
                    }
                }

                sort(dep_list.begin(), dep_list.end());
                auto end_pos = unique(dep_list.begin(), dep_list.end());

                for (auto dep_i = dep_list.begin(); dep_i != end_pos; dep_i++)
                {
                    int dep_row_block = *(dep_i);

                    tmp_nodes[dep_row_block]->child.push_back(tmp_nodes[cur_row_block]);
                    tmp_nodes[dep_row_block]->out_degree++;
                    tmp_nodes[cur_row_block]->parent.push_back(tmp_nodes[dep_row_block]);
                    tmp_nodes[cur_row_block]->in_degree++;
                    if (tmp_nodes[dep_row_block]->topo_level + 1 > tmp_nodes[cur_row_block]->topo_level)
                        tmp_nodes[cur_row_block]->topo_level = tmp_nodes[dep_row_block]->topo_level + 1;
                    ret->global_edge++;
                }

                if (previous_node)
                {
                    previous_node->locality_node = tmp_nodes[cur_row_block];
                }
                previous_node = tmp_nodes[cur_row_block];

                // Identify row blocks with only diagonal elements
                if (only_diag)
                {
                    //printf("only diag node %d\n", cur_row_block);
                    ret->start_nodes.push_back(tmp_nodes[cur_row_block]);
                }

            }
        }
    }

    return ret;
}

ptr_graph generate_graph_row_block_supernode(const int m, const int nnz,
                const int *csrRowPtr, const int *csrColIdx, int thresh)
{
    ptr_graph ret = new graph();

    ptr_node tmp_nodes[m];

    int hash_idx[m];

    for (int i = 0; i < m; i++) hash_idx[i] = -1;

    ptr_node previous_node = NULL;

    for (int row_st = 0; row_st < m;)
    {
        int row_ed = row_st + 1;
        int flag = 1;
        while (flag && row_ed < m && row_ed < row_st + WARP_SIZE)
        {
            // diagonal element do not count
            int rp1 = csrRowPtr[row_st];
            int rp1_ed = csrRowPtr[row_st + 1] - 1;
            int rp2 = csrRowPtr[row_ed];
            int rp2_ed = csrRowPtr[row_ed + 1] - 1;

            int correct = 1, total = 1;
            while (rp1 < rp1_ed || (rp2 < rp2_ed && csrColIdx[rp2] < row_st))
            {
                total++;
                if (csrColIdx[rp1] == csrColIdx[rp2])
                {
                    rp1++;
                    rp2++;
                    correct++;
                }
                else if (csrColIdx[rp1] > csrColIdx[rp2]) rp1++;
                else rp2++;
            }
            float correct_rate = correct / total;

            if (correct_rate == 1.0) row_ed++; else flag = 0;
        }

        //if (row_ed > row_st + 1) printf("supernode %d %d\n", row_st, row_ed);

        for (int i = row_st; i < row_ed; i++)
            hash_idx[i] = ret->global_node;

        int cur_row_block = ret->global_node;

        tmp_nodes[cur_row_block] = new node(ret->global_node,
        row_st, row_ed, csrRowPtr[row_ed] - csrRowPtr[row_st]);
        
        ret->global_node++;

        vector<int> dep_list;

        int only_diag = 1;
        for (int idx = csrRowPtr[row_st]; idx < csrRowPtr[row_ed]; idx++)
        {
            int col_idx = csrColIdx[idx];
            int dep_row_block = hash_idx[col_idx];
            if (dep_row_block < cur_row_block)
            {
                dep_list.push_back(dep_row_block);
                only_diag = 0;
            }
        }

        sort(dep_list.begin(), dep_list.end());
        auto end_pos = unique(dep_list.begin(), dep_list.end());

        for (auto dep_i = dep_list.begin(); dep_i != end_pos; dep_i++)
        {
            int dep_row_block = *(dep_i);

            tmp_nodes[dep_row_block]->child.push_back(tmp_nodes[cur_row_block]);
            tmp_nodes[dep_row_block]->out_degree++;
            tmp_nodes[cur_row_block]->parent.push_back(tmp_nodes[dep_row_block]);
            tmp_nodes[cur_row_block]->in_degree++;
            if (tmp_nodes[dep_row_block]->topo_level + 1 > tmp_nodes[cur_row_block]->topo_level)
                tmp_nodes[cur_row_block]->topo_level = tmp_nodes[dep_row_block]->topo_level + 1;
            ret->global_edge++;
        }

        if (previous_node)
        {
            previous_node->locality_node = tmp_nodes[cur_row_block];
        }
        previous_node = tmp_nodes[cur_row_block];

        // Identify row blocks with only diagonal elements
        if (only_diag)
        {
            ret->start_nodes.push_back(tmp_nodes[cur_row_block]);
        }

        row_st = row_ed;
    }

    return ret;

}

void graph_reorder_with_level(ptr_handler handler)
{
    ptr_graph g = handler->graph;

    queue<ptr_node> topo_queue;
    for (auto i = g->start_nodes.begin(); i != g->start_nodes.end(); i++)
    {
        topo_queue.push((*i));
    }

    int row_pos = 0;
    ptr_node last_node = NULL;

    while(!topo_queue.empty())
    {
        ptr_node current_node = topo_queue.front();
        for (auto iter = current_node->child.begin(); iter != current_node->child.end(); iter++)
        {
            (*iter)->in_degree_tmp++;
            if ((*iter)->in_degree_tmp == (*iter)->in_degree)
            {
                topo_queue.push(*iter);
                (*iter)->in_degree_tmp = 0;
            }
        }
        if (last_node) last_node->locality_node = current_node;
        int node_len = current_node->info.end_row - current_node->info.start_row;
        current_node->info.start_row = row_pos;
        current_node->info.end_row = row_pos + node_len;
        row_pos += node_len;
        last_node = current_node;

        topo_queue.pop();
    }
    last_node->locality_node = NULL;
}

void merge_with_size(ptr_handler handler, const int size)
{
    ptr_graph g = handler->graph;

    for (ptr_node i = g->start_nodes[0]; i != NULL;)
    {
        int t_size = 0;
        while (i->locality_node != NULL && t_size < size)
            t_size++;
    }
}

ptr_handler SpTRSV_preprocessing(const int m, const int nnz,
                const int *csrRowPtr, const int *csrColIdx,
                PREPROCESSING_STRATEGY strategy, int row_block)
{
    ptr_handler ret = new SpTRSV_handler();

    ret->m = m;
    ret->nnz = nnz;
    ret->row_block = row_block;

    if (strategy == ROW_BLOCK)
    {
        ret->graph = generate_graph_row_block(m, nnz, csrRowPtr, csrColIdx, row_block);
    }
    else if (strategy == ROW_BLOCK_THRESH)
    {
        ret->graph = generate_graph_row_block_thresh(m, nnz, csrRowPtr, csrColIdx, row_block);
    }
    else if (strategy == ROW_BLOCK_AVG)
    {
        ret->graph = generate_graph_row_block_avg(m, nnz, csrRowPtr, csrColIdx, row_block);
    }
    else if (strategy == SUPERNODE_BLOCK)
    {
        ret->graph = generate_graph_row_block_supernode(m, nnz, csrRowPtr, csrColIdx, row_block);
    }
    else
    {
        // Not implemented
        printf("Error: scheduling strategy not implemented!\n");
    }

    cudaMalloc(&ret->get_value, ret->m * sizeof(int));
    cudaMemset(ret->get_value, 0, ret->m * sizeof(int));

    cudaMalloc(&ret->warp_runtime, ret->m * sizeof(int));
    cudaMemset(ret->warp_runtime, 0, ret->m * sizeof(int));

    ret->schedule_level = NULL;
    ret->schedule_info = NULL;

    // for (auto i = ret->graph->start_nodes.begin(); i != ret->graph->start_nodes.end(); i++)
    // {
    //     printf("id %d row %d\n", (*i)->id, (*i)->info.start_row);
    // }

    return ret;
}

void show_graph_layer(ptr_handler handler)
{
    ptr_graph graph = handler->graph;

    int max_layer = -1;
    for (ptr_node i = graph->start_nodes[0]; i != NULL; i = i->locality_node)
    {
        if (i->topo_level > max_layer) max_layer = i->topo_level;
    }

    int layer[max_layer + 1];
    for (int i = 0; i < max_layer; i++)
        layer[i] = 0;

    printf("Total node number      : %d\n", graph->global_node);
    printf("Total topological layer: %d\n", max_layer);
    int count = 0;
    for (ptr_node i = graph->start_nodes[0]; i != NULL; i = i->locality_node)
    {
        layer[i->topo_level]++;

        //printf("node %d layer %d\n", count, i->topo_level);
        count++;
    }
    printf("layer distribution: ");
    for (int i = 0; i < max_layer; i++)
        printf("%d ", layer[i]);
    printf("\n");

}

void write_graph(const char* file_name, ptr_handler handler, unsigned int max_depth,
int layer, float parallelism)
{
    string f1 = file_name;
    f1 += ".global";
    string f2 = file_name;
    f2 += ".node";
    string f3 = file_name;
    f3 += ".edge";

    FILE* fp1 = fopen(f1.c_str(), "w");
    FILE* fp2 = fopen(f2.c_str(), "w");
    FILE* fp3 = fopen(f3.c_str(), "w");

    ptr_graph g = handler->graph;

    int node_flag[g->global_node];
    memset(node_flag, 0, g->global_node * sizeof(int));

    queue<ptr_node> node_queue;
    ptr_node depth_ptr = NULL;

    for (auto i = g->start_nodes.begin(); i != g->start_nodes.end(); i++)
    {
        node_queue.push(*i);
        if (i + 1 == g->start_nodes.end())
            depth_ptr = *i;
        node_flag[(*i)->id] = 1;
    }
    
    int num_node = 0;
    int num_edge = 0;
    // Get number of nodes and edges
    int current_depth = 0;
    while (!node_queue.empty() && current_depth <= max_depth)
    {
        ptr_node current_node = node_queue.front();

        if (current_depth < max_depth)
        {
            for (auto i = current_node->child.begin(); i != current_node->child.end(); i++)
            {
                if (current_node->topo_level <= max_depth)
                {
                    if (!node_flag[(*i)->id])
                    {
                        node_queue.push(*i);
                        node_flag[(*i)->id] = 1;
                    }
                    num_edge++;
                    fprintf(fp3, "%d %d\n", current_node->id, (*i)->id);
                }
            }
        }
        if (current_node == depth_ptr)
        {
            current_depth++;
            depth_ptr = node_queue.back();
        }

        // output node information
        // feature: number of nnzs
        fprintf(fp2, "%d %d %d\n", current_node->id,
        current_node->child.size(), current_node->parent.size());
        num_node++;

        // start row id / number of rows,
        // start nnz id / number of nnzs,
        // end nnz id / number of nnzs, 
        // number of row nnzs / number of nnzs

        node_queue.pop();
    }

    // output global information
    fprintf(fp1, "%d %d %d %f\n", num_node, num_edge, layer, parallelism);
}

// Features to output
// 1. avg_nnz        : Average number of non-zero elements per row
// 2. csrCoefficient : Coefficient of Variance of number of non-zero elements per row
// 3. Parallelism    : Average number of non-zero elements per level
// 4. parCoefficient : Coefficient of Variance of number of non-zero elements per level

void get_matrix_info(const int    m,
                const int         nnz,
                const int        *csrRowPtr,
                const int        *csrColIdx,
                float            *avg_rnnz,
                float            *cov_rnnz,
                float            *avg_lnnz,
                float            *cov_lnnz)
{
    int total_nnz = 0;

    int *layer=(int*)malloc(m * sizeof(int));
    if (layer == NULL)
        printf("layer error\n");
    memset(layer, 0, sizeof(int) * m);

    int *layer_num = (int*)malloc((m + 1) * sizeof(int));
    if (layer_num == NULL)
        printf("layer_num error\n");
    memset (layer_num, 0, sizeof(int) * (m + 1));

    int max_layer;
    int max_layer2 = 0;
    int max = 0;
    unsigned int min = -1;

    // count layer
    int row, j;
    for (row = 0; row < m; row++)
    {
        max_layer = 0;

        total_nnz += csrRowPtr[row+1] - csrRowPtr[row];

        for (j = csrRowPtr[row]; j < csrRowPtr[row+1]; j++)
        {
            int col = csrColIdx[j];

            if((layer[col] + 1) > max_layer)
                max_layer = layer[col] + 1;

        }
        layer[row] = max_layer;
        layer_num[max_layer]++;
        if (max_layer > max_layer2)
            max_layer2 = max_layer;
    }

    *avg_rnnz = 1.0 * total_nnz / m;

    *cov_rnnz = 0;
    for (int i = 0; i < m; i++)
    {
        int row_nnz = csrRowPtr[i+1] - csrRowPtr[i];
        *cov_rnnz += (row_nnz - *avg_rnnz) * (row_nnz - *avg_rnnz);
    }
    *cov_rnnz = sqrt(*cov_rnnz / m) / *avg_rnnz;

    int total_layer_num = 0;
    for(j = 1; j <= max_layer2; j++)
    {
        if(max < layer_num[j])
            max = layer_num[j];
        if(min > layer_num[j])
            min = layer_num[j];
        total_layer_num += layer_num[j];
    }
    *avg_lnnz = 1.0 * total_layer_num / max_layer2;

    *cov_lnnz = 0;
    for(j = 1; j <= max_layer2; j++)
    {
        *cov_lnnz += (layer_num[j] - *avg_lnnz) * (layer_num[j] - *avg_lnnz);
    }
    *cov_lnnz = sqrt(*cov_lnnz / max_layer2) / *avg_lnnz;

    free(layer);
    free(layer_num);
}

void write_matrix_info(const char* file_name,
                const char*       matrix_name,
                const int         m,
                const int         nnz,
                const int        *csrRowPtr,
                const int        *csrColIdx)
{
    float avg_rnnz, cov_rnnz, avg_lnnz, cov_lnnz;
    get_matrix_info(m, nnz, csrRowPtr, csrColIdx,
    &avg_rnnz, &cov_rnnz, &avg_lnnz, &cov_lnnz);

    FILE *fp = fopen(file_name, "a");
    fprintf(fp, "%s,%d,%d,%.2f,%.2f,%.2f,%.2f\n", matrix_name,
    m, nnz, avg_rnnz, cov_rnnz, avg_lnnz, cov_lnnz);
    fclose(fp);
    
}