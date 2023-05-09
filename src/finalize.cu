#include "../include/finalize.h"
#include <queue>
#include <cuda.h>

void graph_finalize(ptr_handler handler)
{
    ptr_graph graph = handler->graph;

    queue<ptr_node> topo_queue;

    // inital
    for (auto iter = graph->start_nodes.begin(); iter != graph->start_nodes.end(); iter++)
    {
        topo_queue.push(*iter);
    }

    while(!topo_queue.empty())
    {
        ptr_node current_node = topo_queue.front();
        for (auto iter = current_node->child.begin(); iter != current_node->child.end(); iter++)
        {
            (*iter)->in_degree--;
            if ((*iter)->in_degree == 0)
            {
                topo_queue.push(*iter);
            }
        }
        topo_queue.pop();
        delete current_node;
    }

}

void schedule_finalize(ptr_handler handler)
{
    if (handler->sched_s != SEQUENTIAL2)
    {
        if (handler->schedule_level != NULL)
        {
            free(handler->schedule_level);
        }
        if (handler->schedule_info != NULL)
        {
            for (int i = 0; i < WARP_NUM; i++)
            {
                free(handler->schedule_info[i]);
            }
            free(handler->schedule_info);
        }

        node_info* tmp_schedule_info[WARP_NUM];
        cudaMemcpy(tmp_schedule_info, handler->schedule_info_d, WARP_NUM * sizeof(node_info*), cudaMemcpyDeviceToHost);

        for (int i = 0; i < WARP_NUM; i++)
        {
            cudaFree(tmp_schedule_info[i]);
            handler->warp_schedule[i].clear();
        }
        cudaFree(handler->schedule_level_d);
        cudaFree(handler->schedule_info_d);
    }
    else
    {
        if (handler->no_schedule_info != NULL)
            free(handler->no_schedule_info);
        cudaFree(handler->no_schedule_info_d);
    }

    ptr_graph graph = handler->graph;

    queue<ptr_node> topo_queue;

    // inital
    for (auto iter = graph->start_nodes.begin(); iter != graph->start_nodes.end(); iter++)
    {
        topo_queue.push(*iter);
    }

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
        topo_queue.pop();

        current_node->in_degree_tmp = 0;
        current_node->topo_level = 0;
    }

}

void SpTRSV_finalize(ptr_handler handler)
{
    schedule_finalize(handler);

    //free the graph, together with the nodes
    if (handler->graph != NULL)
        graph_finalize(handler);

}