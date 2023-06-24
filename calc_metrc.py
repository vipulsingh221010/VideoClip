import numpy as np
import csv
import torch
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
import pandas as pd
import os

test_set_dir="/home/intern/interndata/vipul/kinetic_dataset"
df = pd.read_csv('kinetics_scores.csv',index_col=0)
#labels=pd.read_csv('kinetics400_labels.csv')
#labels=labels[0]
ground_truth_labels={}
precision=[]
recall=[]
f1=[]
accuracy=[]
for class_folder in os.listdir(test_set_dir):
    class_folder_path=os.path.join(test_set_dir,class_folder)
    for video_file in os.listdir(class_folder_path):
        video_name=os.path.splitext(video_file)[0]
        ground_truth_labels[video_name]=class_folder
        
        if video_name in df.index:
            row=df.loc[video_name]
            truth_list=[]
            predicted_list=[]
            for key,value in row.items():
                if key==class_folder:
                    truth_list.append(1)
                else:
                    truth_list.append(0)
               # print(value)
                value_str = value.replace("tensor([", "").replace("], device='cuda:0')", "")
                value = float(value_str)
                predicted_list.append(value>4)
                
            precision.append(precision_score(truth_list, predicted_list))
            recall.append(recall_score(truth_list, predicted_list))
            f1.append(f1_score(truth_list, predicted_list))
            accuracy.append(accuracy_score(truth_list, predicted_list))


file_path = 'results.csv'
data = list(zip(precision,recall,f1,accuracy))
with open(file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['precision', 'recall', 'f1-score','accuracy'])
    writer.writerows(data)
precision = sum(precision) / len(precision)
f1= sum(f1) / len(f1)
recall = sum(recall) / len(recall)
accuracy= sum(accuracy) / len(accuracy)
print(precision)
print(accuracy)
print(f1)
print(recall)
        
