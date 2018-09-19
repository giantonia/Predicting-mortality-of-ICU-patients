import os
import numpy as np
import pandas as pd

set_name = 'c'

# # Rename files
os.chdir('set-' + set_name)
# for i, filename in enumerate(os.listdir(".")):
#     os.rename(filename, str(i) + ".txt")

# Get all the features
feature_list = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT',
                'AST', 'Bilirubin', 'BUN', 'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
                'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP',
                'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate', 'SaO2', 'SysABP', 'Temp', 'TropI',
                'TropT', 'Urine', 'WBC']
new_feature_list = feature_list[:5]
for i in range(5, 42):
    f = feature_list[i]
    new_feature_list.extend([f + 'Min', f + 'Max', f + 'Mean', f + 'Variance', f + 'Count'])

# Standardised format
df = pd.DataFrame(columns=new_feature_list[:])

# Read data

for i in range(4000):
    row = dict(zip(feature_list, [[] for k in range(42)]))
    new_row = dict(zip(new_feature_list, [[] for k in range(190)]))
    text = open(str(i) + '.txt', 'r').read().split('\n')
    count = [0]*42
    for line in text[1:]:
        line = line.split(',')
        if len(line) == 3:
            try:
                f = line[1]
                val = line[2]
                row[str(f)].append([float(val)])
            except KeyError:
                pass
    for k in range(5):
        new_row[feature_list[k]] = row[feature_list[k]]
    for k in range(5, 42):
        f = feature_list[k]
        if len(row[f]) > 0:
            new_row[f + 'Min'] = min(row[f])[0]
            new_row[f + 'Max'] = max(row[f])[0]
            new_row[f + 'Mean'] = np.mean(row[f])
            new_row[f + 'Variance'] = np.var(row[f])
            new_row[f + 'Count'] = len(row[f])
        else:
            new_row[f + 'Min'] = 0
            new_row[f + 'Max'] = 0
            new_row[f + 'Mean'] = 0
            new_row[f + 'Variance'] = 0
            new_row[f + 'Count'] = 0
    new_row = list(new_row.values())
    for i in range(5):
        new_row[i] = new_row[i][0][0]
    df.loc[len(df)] = new_row


# Outcome
os.chdir("..")

mortality = open('Outcomes-' + set_name + '.txt', 'r').read().split('\n')
outcome = []
for line in mortality[1:]:
    if line != '':
        if line[-1] == '1':
            outcome.append(1)
        else:
            outcome.append(0)
df['Outcome'] = pd.Series(outcome)


# Write to csv
os.chdir("..")
df.to_csv('set-' + set_name + '.csv')
