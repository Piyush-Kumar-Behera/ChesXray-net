#Program to generate a csv of the disease labels in one-hot encoding format with only patient-id and diseases

import pandas as pd
import os

current_dir = os.path.split(os.getcwd())[1]
parent_directory_path = ''

if current_dir == 'util':
    parent_directory_path = os.path.split(os.getcwd())[0]
else:
    parent_directory_path = os.getcwd()

file_path = os.path.join(parent_directory_path,'Data_Entry_2017.csv')

print("Importing Original Labels...")
base_data = pd.read_csv(file_path)
base_data_columns = list(base_data.columns)
data_columns = base_data_columns[:4]
data_columns.remove(data_columns[2])


update_data = base_data[data_columns]

diseases = set()
print("Creating Disease List...")

for i in range(len(update_data['Finding Labels'])):
    condition = update_data['Finding Labels'][i].split('|')
    for c in condition:
        diseases.add(c)

disease_list = list(diseases)

disease_columns = [0] * len(update_data['Finding Labels'])
print("Creating Dataframe...")
for d in disease_list:
    update_data[d] = disease_columns

for i in range(len(update_data['Finding Labels'])):
    condition = update_data['Finding Labels'][i].split('|')
    for c in condition:
        update_data.loc[i,c] = 1
    
update_data.drop(["Finding Labels", "No Finding"], axis = 1,inplace = True)
update_data.rename(columns = {'Image Index' : 'Image'}, inplace = True)

print("Exporting CSV...")
path_to_labels = os.path.join(parent_directory_path,'labels.csv')
update_data.to_csv(path_to_labels, index = False)

# print(update_data)
# print(file_path)
# print(current_dir)
# print(parent_directory_path)