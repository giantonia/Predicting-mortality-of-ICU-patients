import pandas as pd
import numpy as np

set_name = 'c'

df = pd.read_csv('set-' + set_name + '.csv')
feature_list = list(df.columns.values)

# Remove variance features and those with same values
variance_list = []
for feature in feature_list:
    if 'Variance' in feature:
        variance_list.append(feature)
df.drop(variance_list, axis=1, inplace=True)

other_drop = ['ALPMax', 'ALTMin', 'ALTMax', 'ALTMean', 'ASTMin', 'ASTMax', 'ASTMean', 'BilirubinMin', 'BilirubinMax',
              'BilirubinMean', 'CholesterolMin', 'CholesterolMax', 'CholesterolMean',
              'CholesterolCount', 'pHMax', 'RespRateMin', 'RespRateMax', 'RespRateMean', 'RespRateCount']
df.drop(other_drop, axis=1, inplace=True)


# Features with different distributions for two classes
def createNew(col, type, up=None, down=None):
    new_col = []
    if type == 'up':
        for item in col:
            if item > up:
                new_col.append(1)
            else:
                new_col.append(0)
    elif type == 'down':
        for item in col:
            if item < down:
                new_col.append(1)
            else:
                new_col.append(0)
    else:
        for item in col:
            if item < up and item > down:
                new_col.append(1)
            else:
                new_col.append(0)
    return new_col
def replaceCol(col_name, type, up=None, down=None):
    col = df[col_name].tolist()
    new_col = createNew(col, type, up, down)
    df[col_name + 'Dis'] = pd.Series(new_col)
    df.drop(col_name, axis=1, inplace=True)
replaceCol('ALPCount', 'up', 2)
replaceCol('ALTCount', 'up', 2)
replaceCol('ASTCount', 'up', 2)
replaceCol('BilirubinCount', 'up', 2)
replaceCol('FiO2Min', 'both', 0.8, 0.2)
replaceCol('FiO2Mean', 'down', None, 0.3)
replaceCol('GCSMax', 'down', None, 10)
replaceCol('PaCO2Min', 'both', 60, 10)
replaceCol('PaCO2Max', 'both', 80, 20)
replaceCol('PaCO2Mean', 'both', 70, 20)

# Remove TropI and TropT
for var in ["TropI", "TropT"]:
    for type in ["Min", "Max", "Mean", "Count"]:
        df.drop(var+type, 1, inplace=True)

# Imputation
def replaceByMean(col_name, stan):
    col = np.array(df[col_name])
    colMean = np.mean(col[col > stan])
    for i in range(len(col)):
        if col[i] <= stan:
            col[i] = colMean
    df[col_name] = col[:]

def replaceByMode(col_name, stan):
    col = np.array(df[col_name])
    unique, count = np.unique(np.array(col[col > stan]), return_counts=True)
    colMode = unique[np.argmax(count)]
    for i in range(len(col)):
        if col[i] <= stan:
            col[i] = colMode
    df[col_name] = col[:]
mean_replace = ['Height 100', 'WeightMin 20', 'WeightMax 20', 'WeightMean 20', 'AlbuminMin 0', 'AlbuminMax 0',
                'AlbuminMean 0', 'ALPMean 0', 'ALPMean 0', 'BUNMean 0', 'DiasABPMin 0', 'DiasABPMax 0', 'DiasABPMean 0',
                'GCSMin 0', 'GCSMean 0', 'GlucoseMin 0', 'GlucoseMean 0',
                'GlucoseMax 0', 'HCO3Min 0', 'HCO3Max 0', 'HCO3Mean 0', 'HCTMin 0', 'HCTMean 0', 'HCTMax 0', 'HRMin 0',
                'HRMax 0', 'HRMean 0', 'KMean 0', 'KMin 0', 'KMax 0', 'LactateMin 0', 'LactateMean 0', 'LactateMax 0',
                'MgMin 0', 'MgMax 0', 'MgMean 0', 'MAPMin 0', 'MAPMean 0', 'MAPMax 0', 'NaMin 0', 'NaMean 0', 'NaMax 0',
                'NIDiasABPMin 0', 'NIDiasABPMean 0', 'NIDiasABPMax 0', 'NIMAPMin 0', 'NIMAPMean 0', 'NIMAPMax 0',
                'NISysABPMin 0', 'NISysABPMean 0', 'NISysABPMax 0',
                'PaO2Min 0', 'PaO2Mean 0', 'PaO2Max 0', 'pHMin 0', 'pHMean 0', 'PlateletsMin 0', 'PlateletsMean 0',
                'PlateletsMax 0', 'SaO2Min 0', 'SaO2Mean 0', 'SaO2Max 0', 'SysABPMin 0', 'SysABPMean 0', 'SysABPMax 0',
                'TempMin 24', 'TempMax 0', 'TempMean 0', 'UrineMin 0', 'UrineMax 0', 'UrineMean 0', 'WBCMin 0', 'WBCMax 0',
                'WBCMean 0']
for item in mean_replace:
    col_name, stan = item.split()
    replaceByMean(col_name, float(stan))

mode_replace = ['BUNMin 0', 'BUNMax 0']
for item in mode_replace:
    col_name, stan = item.split()
    replaceByMode(col_name, float(stan))

# Normalise
# for i in range(1, len(df.columns)):
#     col = df.ix[:,i]
#     min_col = min(col)
#     max_col = max(col)
#     col = (col - min_col)/(max_col-min_col)
#     df.ix[:,i] = col[:]

# Write files
if set_name == 'a':
    df.to_csv('train.csv')
else:
    df.to_csv('validation.csv')
