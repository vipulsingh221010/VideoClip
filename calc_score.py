import torch
torch.set_num_threads(32)
import pandas as pd
import av
import os
import numpy as np
import csv
import cv2
import time
from mmpt.models import MMPTModel
import traceback
ground_truth_labels={}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return torch.from_numpy(np.stack([x.to_ndarray(format="rgb24") for x in frames]))



def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

print(device)
model, tokenizer, aligner = MMPTModel.from_pretrained("projects/retri/videoclip/how2.yaml")
x=0
model.to(device)
#tokenizer.to(device)
#aligner.to(device)
model.eval()
test_set_dir="/home/intern/interndata/vipul/kinetic_dataset"
score=[]
with open('kinetics400_labels.csv', 'r') as file:
     csv_reader = csv.reader(file)

    # Convert each row into a list and store them in a main list
     labels = [row for row in csv_reader]
labels=labels[0]
#print(labels)
video_scores={}

cap_dict={}
for label in labels:
    caps, cmasks = aligner._build_text_seq(tokenizer(label, add_special_tokens=False)["input_ids"])
    caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1
    cap_dict[label]=[caps,cmasks]

for class_folder in os.listdir(test_set_dir):

    class_folder_path=os.path.join(test_set_dir,class_folder)
    num=0
    for video_file in os.listdir(class_folder_path):

        x+=1
        print(x)
        dict={}
        num+=1
        if num==3:
            break
        t1=time.time()
        video_name=os.path.splitext(video_file)[0]
        ground_truth_labels[video_name]=class_folder
        video_path = os.path.join(class_folder_path,video_file)
        try :
            container = av.open(video_path)
            indices = sample_frame_indices(clip_len=10, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
            video = read_video_pyav(container, indices)
            video = torch.unsqueeze(video, 0)  # Add dimension at position 0
            video = torch.unsqueeze(video, 0)  # Add dimension at position 0 again
            caps=cap_dict[class_folder][0]
            cmasks=cap_dict[class_folder][1]
            with torch.no_grad():
                output = model(video.float().to(device), caps.to(device), cmasks.to(device), return_score=True)            
            dict[class_folder]=output['score']
            
            for label in labels:

                caps=cap_dict[label][0]
                cmasks=cap_dict[label][1]
                with torch.no_grad():
    
                    output = model(video.float().to(device), caps.to(device), cmasks.to(device), return_score=True)
                   # print(t1-time.time())                                                                
               # print(label)
                #print(output)
                #print(num)
        
                dict[label]=output['score']
                print(output['score'])


            video_scores[video_name]=dict


        except Exception as e:
           # print("An error occurred while downloading the video:", str(e))
            traceback.print_exc()
        print(time.time()-t1)    
# Open the CSV file in write mode
file_path = 'kinetics_scores_new.csv'


# Get all unique labels from the data
labels = set(label for video_scores in video_scores.values() for label in video_scores.keys())

# Open the CSV file in write mode
with open(file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

            # Write the header row with labels
    header = ['Video'] + list(labels)
    writer.writerow(header)

                        # Write the data rows
    for video, scores in video_scores.items():
        row = [video] + [scores.get(label, '') for label in labels]
        writer.writerow(row)
