import os
import csv

test_set_dir="/home/intern/interndata/vipul/kinetic_dataset"
classes=[]
for class_folder in os.listdir(test_set_dir):
    classes.append(class_folder)


file_path = 'kinetics400_labels.csv'

with open(file_path, 'w', newline='') as csvfile:
                # Create a CSV writer object
    writer = csv.writer(csvfile)

                        # Write the list to the CSV file
    writer.writerow(classes)
