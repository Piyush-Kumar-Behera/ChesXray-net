#Python script to create the test label dataframe

import os
import pandas as pd

current_dir = os.path.split(os.getcwd())[1]
parent_directory_path = ''

if current_dir == 'util':
    parent_directory_path = os.path.split(os.getcwd())[0]
else:
    parent_directory_path = os.getcwd()

print("Reading test_list.txt...")
file_path = os.path.join(parent_directory_path,'test_list.txt')

with open(file_path,'r') as f:
    test_list = [x.rstrip() for x in f.readlines()]

labels = pd.read_csv(os.path.join(parent_directory_path,'labels.csv'))
labels = labels.set_index('Image', drop = False)


testdf = labels.loc[test_list]

test_path = os.path.join(parent_directory_path,'test.csv')

print("Saving Result...")
testdf.to_csv(test_path, index = False)