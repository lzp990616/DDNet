# import os

# # 设置文件夹路径
# folder_path = "./log_kfold"

# # 遍历文件夹，找到所有的log文件
# for file_name in os.listdir(folder_path):
#     if file_name.endswith(".log"):
#         # 读取所有医学指标及其对应的数值
#         with open(os.path.join(folder_path, file_name), "r") as f:
#             lines = f.readlines()
#             metrics = {}
#             for line in lines:
#                 if ":" in line:
#                     metric_name, metric_value = line.split(":")
#                     metric_name = metric_name.strip()
#                     metric_value = float(metric_value.strip())
#                     metrics[metric_name] = metric_value

#             # 找到testloss指标对应的数值最低的那一行医学指标
#             min_testloss = float('inf')
#             min_testloss_metrics = ""
#             for metric_name, metric_value in metrics.items():
#                 if metric_name == "testloss" and metric_value < min_testloss:
#                     min_testloss = metric_value
#                     min_testloss_metrics = f"{metric_name}: {metric_value}\n"
#                 elif metric_name != "testloss":
#                     min_testloss_metrics += f"{metric_name}: {metric_value}\n"

#             # 输出结果
#             print(f"文件名: {file_name}")
#             print(min_testloss_metrics)




# import re
# import numpy as np

# # 读取log文件
# with open('./log_kfold/bus_bce_0dot1dice_lr1e4.log', 'r') as f:
#     log_data = f.readlines()

# # 使用正则表达式找到每个交叉实验的test_loss
# pattern = r'Epoch \d+: train_loss=[\d.]+,.*test_loss=([\d.]+),'
# test_loss = []
# for line in log_data:
#     result = re.findall(pattern, line)
#     if len(result) > 0:
#         test_loss.append(float(result[0]))

# # 将test_loss按交叉实验分组，并取每组的最小值所在的行号
# min_loss_rows = []
# for i in range(4):
#     fold_test_loss = test_loss[i::4]
#     min_loss_index = i * 6 + fold_test_loss.index(min(fold_test_loss))
#     min_loss_rows.append(min_loss_index)

# # 使用正则表达式找到每个交叉实验中test_loss最低的一行，并提取其中的医学指标
# pattern2 = r'train_loss=([\d.]+), pre=([\d.]+), recall=([\d.]+), dice=([\d.]+), jaccard=([\d.]+),spe=([\d.]+),acc=([\d.]+),f1=([\d.]+),test_loss=([\d.]+),'
# med_indicators = []
# for row in min_loss_rows:
#     result = re.findall(pattern2, log_data[row])
#     med_indicators.append([float(x) for x in result[0]])

# # 计算各个医学指标的均值
# med_indicators = np.array(med_indicators)
# mean_med_indicators = np.mean(med_indicators, axis=0)
# print("bus_bce_0dot1dice_lr1e4.log")
# print("train_loss, precision, recall, dice, jaccard, spe")
# # 输出结果
# print(f"4次交叉实验中test_loss最低的行的各个医学指标的均值为{mean_med_indicators}")

import pdb
import pandas as pd
import os
import statistics
import numpy as np
def parse_log_file1(log_file_path):

    with open(log_file_path, "r") as f:
        lines = f.readlines()
    
    
    
    min_test_loss = float('inf')
    best_metrics = {}
    for line in lines:
        if "test_loss=" in line:
            test_loss = float(line.split("test_loss=")[-1].strip(",\n"))
            if test_loss < min_test_loss:
                min_test_loss = test_loss
                best_metrics = {}
                for metric in ["pre", "recall", "dice", "jaccard", "spe", "acc", "f1"]:
                    metric_val = float(line.split(f"{metric}=")[-1].split(",")[0])
                    best_metrics[metric] = metric_val
    return best_metrics


def parse_log_file(log_file_path, experiment_lines=101):
    with open(log_file_path, "r") as f:
        lines = f.readlines()
    # min_test_loss = float('inf')
    best_metrics = {}
    all_metrics = []
    for i, line in enumerate(lines):
        if i % experiment_lines == 0:
            best_metrics = {}
            min_test_loss = float('inf')
        else:
            if "test_loss=" in line:
                test_loss = float(line.split("test_loss=")[-1].strip(",\n"))
                if test_loss < min_test_loss:
                    min_test_loss = test_loss
                    best_metrics = {}
                    for metric in ["pre", "recall", "dice", "jaccard", "spe", "acc", "f1"]:
                        metric_val = float(line.split(f"{metric}=")[-1].split(",")[0])
                        best_metrics[metric] = metric_val
        if "Epoch 100" in line:
            all_metrics.append(best_metrics)    
    return all_metrics
# log/kfolder
# log/kfolder_compare
# log_Polyp/dnm
log_folder = "./log_Polyp/dnm"
# log_folder = "./log_kfold/busi"
all_metrics = []
all_metrics_df = pd.DataFrame()
all_avg_metrics = []
for i, log_file_name in enumerate(os.listdir(log_folder)):
    if log_file_name.endswith(".log"):
        log_file_path = os.path.join(log_folder, log_file_name)
        best_metrics = parse_log_file(log_file_path)
        # best_metrics['log_file_name'] = log_file_name
        all_metrics.append(best_metrics)
        
    avg_metrics = {}
    std_metrics = {}
    all_avg_metrics = []
   #  pdb.set_trace()
    for metric in ["pre", "recall", "dice", "jaccard", "spe", "acc", "f1"]:
        metric_vals = [m[metric] for m in best_metrics if metric in m]
        # avg_metrics[metric] = sum(metric_vals) / len(metric_vals)
        
        
        mean_metric = statistics.mean(metric_vals)
        std_metric = statistics.stdev(metric_vals)
        mean_metric = round(mean_metric, 3)
        std_metric = round(std_metric, 3)
        all_avg_metrics.append((metric, mean_metric, std_metric))
        
        
    
    print(log_file_name)
    print(all_avg_metrics)
    print()

all_metrics_df.to_csv('metrics.csv', index=False) 
