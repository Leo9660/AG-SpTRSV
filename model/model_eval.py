import csv
import sys

import torch
import torch.nn.functional as F

import pyltr

from sklearn.model_selection import GroupShuffleSplit

import pandas as pds

import numpy as np

import csv
import pickle

if (len(sys.argv) != 4 and len(sys.argv) != 5):
    print("Usage: python model_eval.py {matrix info} {perf result} {training-set split proportion} {output name|optional}")
    exit(-1)

matrix_info_file = sys.argv[1]
perf_file = sys.argv[2]
test_size = 1 - float(sys.argv[3])
if (len(sys.argv) == 5):
    output_name = sys.argv[4]
else:
    output_name = None

matrix_data = list(csv.reader(open(matrix_info_file)))
perf_data = list(csv.reader(open(perf_file)))

matrix_table_head = ['matrix_name', 'm', 'nnz', 'avg_nnz', \
'csrCoefficient', 'avg_layer_num', 'lnumCoefficient']
perf_table_head = ['matrix_name', 'ps', 'ss', 'rb', 'time']

mt_matrix_name_idx = matrix_table_head.index('matrix_name')
mt_m_idx = matrix_table_head.index('m')
mt_nnz_idx = matrix_table_head.index('nnz')
mt_avg_nnz_idx = matrix_table_head.index('avg_nnz')
mt_csr_coe_idx = matrix_table_head.index('csrCoefficient')
mt_avg_layer_idx = matrix_table_head.index('avg_layer_num')
mt_lnum_coe_idx = matrix_table_head.index('lnumCoefficient')

pf_matrix_name_idx = perf_table_head.index('matrix_name')
pf_ps_idx = perf_table_head.index('ps')
pf_ss_idx = perf_table_head.index('ss')
pf_rb_idx = perf_table_head.index('rb')
pf_time_idx = perf_table_head.index('time')

matrix_data = matrix_data[0:]
perf_data = perf_data[0:]

x_count = []
x_data = []
y_label = []
y_time = []

# Data loading and preprocessing
perf_index = 0
count = 0
while (perf_index < len(perf_data)):
    perf_item = perf_data[perf_index]
    matrix_name = perf_item[pf_matrix_name_idx]

    perf_set = list(filter(lambda x: x[pf_matrix_name_idx] == matrix_name, perf_data))
    perf_index += len(perf_set)

    st = perf_item[pf_matrix_name_idx].rfind('/')
    matrix_name = perf_item[pf_matrix_name_idx][st:]
    
    matrix_info = list(filter(lambda x: matrix_name in x[mt_matrix_name_idx], matrix_data))

    if (len(matrix_info) == 0):
        continue
    matrix_info = matrix_info[0]

    if (float(matrix_info[mt_nnz_idx]) < 100000):
        continue

    count += 1

    y_current = []

    con_flag = 0
    for perf_item in perf_set:
        time = float(perf_item[pf_time_idx])
        if time < 10:
            con_flag = 1
    if con_flag:
        continue

    for perf_item in perf_set:

        ps = int(perf_item[pf_ps_idx])
        ss = int(perf_item[pf_ss_idx])
        rb = int(perf_item[pf_rb_idx])
        time = float(perf_item[pf_time_idx])

        info_item = [float(matrix_info[mt_m_idx]), \
        float(matrix_info[mt_nnz_idx]), \
        float(matrix_info[mt_avg_nnz_idx]), \
        float(matrix_info[mt_csr_coe_idx]), \
        float(matrix_info[mt_avg_layer_idx]), \
        float(matrix_info[mt_lnum_coe_idx])]

        ps_item = F.one_hot(torch.tensor(ps), num_classes=5).tolist()
        ss_item = F.one_hot(torch.tensor(ss), num_classes=8).tolist()

        info_item += list(ps_item)
        info_item += list(ss_item)
        info_item += [rb]

        x_count.append(count)
        x_data.append(info_item)
        y_current.append(time)

    y_time.extend(y_current)
    if (np.max(y_current) > np.min(y_current)):
       y_current = 1.0 - (y_current - np.min(y_current)) / (np.max(y_current) - np.min(y_current))

    y_label.extend(y_current)

x_df = {"query_id": x_count, "data": x_data}
x_df = pds.DataFrame(x_df)
y_df = {"label": y_label, "time": y_time}
y_df = pds.DataFrame(y_df)

# Model definition
gss = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state = 7).split(x_df, groups=x_df["query_id"])
train_ids, test_ids = next(gss)

train_x = x_df.iloc[train_ids]
train_y = y_df.iloc[train_ids]
test_x = x_df.iloc[test_ids]
test_y = y_df.iloc[test_ids]

metric = pyltr.metrics.NDCG(k=10)
model = pyltr.models.LambdaMART( \
        metric=metric, \
        max_depth=9, \
        n_estimators=1000, \
        learning_rate=0.01, \
        max_features=0.75, \
        verbose=1)

model.fit(train_x['data'].tolist(), train_y['label'].tolist(), train_x['query_id'].tolist())

test_data = test_x['data'].tolist()
qid_data = test_x['query_id'].tolist()
test_data_y = test_y['label'].tolist()
test_data_time = test_y['time'].tolist()
i = 0

# Evaluation
mape = 0.0
count = 0
good_count = 0.0
while (i < len(test_data)):
    j = i
    while (j < len(test_data) and qid_data[j] == qid_data[i]):
        j += 1
    pred = model.predict(test_data[i: j])
    grnd = test_data_y[i: j]

    best_idx = np.where(pred == np.max(pred))[0]

    act_perf = []
    for idx in best_idx:
        act_perf.append(grnd[idx])
    act_perf = np.max(act_perf)

    bst_perf = np.max(grnd)
    current_mape = abs(bst_perf - act_perf) / bst_perf

    mape += current_mape
    if (current_mape < 0.05):
        good_count += 1

    i = j
    count += 1

mape = mape / count
good_count = good_count / count

print("Test split size", test_size)
print("MAPE: ", mape)
print("Accuracy (95% perf): ", good_count)

if (output_name != None):
    with open(output_name, 'wb') as f:
        pickle.dump(model, f)

