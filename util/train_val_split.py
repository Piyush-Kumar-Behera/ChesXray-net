#Python script to generate training and validation dataframe to prevent any form of data-leakage

import os
import random
import pandas as pd

current_dir = os.path.split(os.getcwd())[1]
parent_directory_path = ''

if current_dir == 'util':
    parent_directory_path = os.path.split(os.getcwd())[0]
else:
    parent_directory_path = os.getcwd()

print("Reading train_val_list.txt...")
file_path = os.path.join(parent_directory_path,'train_val_list.txt')

with open(file_path,'r') as f:
    train_val_list = [x.rstrip() for x in f.readlines()]

labels = pd.read_csv(os.path.join(parent_directory_path,'labels.csv'))
labels = labels.set_index('Image', drop = False)

train_val_patient_list = list(set([x[:8] for x in train_val_list]))
train_val_patient_list_r = random.sample(train_val_patient_list, len(train_val_patient_list))

print("Creating train and val dataset...")
d = int(0.8 * len(train_val_patient_list))
train_patient = train_val_patient_list_r[:d]
val_patient = train_val_patient_list_r[d:]

train_list = [x for x in train_val_list if x[:8] in train_patient]
val_list = [x for x in train_val_list if x[:8] in val_patient]

traindf = labels.loc[train_list]
valdf = labels.loc[val_list]

train_path = os.path.join(parent_directory_path,'train.csv')
val_path = os.path.join(parent_directory_path,'val.csv')

print("Saving Result...")
traindf.to_csv(train_path, index = False)
valdf.to_csv(val_path, index = False)

# print(len(train_val_list))
# print(len(train_val_patient_list))
# print(train_list)
# print(val_list)
# print(len(train_list))
# print(len(val_list))