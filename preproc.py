import os
import numpy as np
import pandas as pd

class Preprocessing():
    def __init__(self, set_name, cwd):
        self.set_name = set_name
        self.cwd = cwd # current working directory
        self.df = None

    def change_file_name(self):
        # Rename files
        os.chdir(cwd + '\\set-' + self.set_name)
        for i, filename in enumerate(os.listdir(".")):
            os.rename(filename, str(i) + ".txt")

    def read_data(self):
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
        self.df = pd.DataFrame(columns=new_feature_list[:])
        # Read data from text files
        self.change_file_name()
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
            self.df.loc[len(self.df)] = new_row
        # Read outcome
        os.chdir("..")
        mortality = open('Outcomes-' + self.set_name + '.txt', 'r').read().split('\n')
        outcome = []
        for line in mortality[1:]:
            if line != '':
                if line[-1] == '1':
                    outcome.append(1)
                else:
                    outcome.append(0)
        self.df['Outcome'] = pd.Series(outcome)

    def createNew(self, col, type, up=None, down=None):
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

    def replaceCol(self, col_name, type, up=None, down=None):
        col = self.df[col_name].tolist()
        new_col = self.createNew(col, type, up, down)
        self.df[col_name + 'Dis'] = pd.Series(new_col)
        self.df.drop(col_name, axis=1, inplace=True)

    def replaceByMean(self, col_name, stan):
        col = np.array(self.df[col_name])
        colMean = np.mean(col[col > stan])
        for i in range(len(col)):
            if col[i] <= stan:
                col[i] = colMean
        self.df[col_name] = col[:]

    def replaceByMode(self, col_name, stan):
        col = np.array(self.df[col_name])
        unique, count = np.unique(np.array(col[col > stan]), return_counts=True)
        colMode = unique[np.argmax(count)]
        for i in range(len(col)):
            if col[i] <= stan:
                col[i] = colMode
        self.df[col_name] = col[:]

    def process_data(self):
        feature_list = list(self.df.columns.values)
        # Remove variance features and those with same values
        variance_list = []
        for feature in feature_list:
            if 'Variance' in feature:
                variance_list.append(feature)
        self.df.drop(variance_list, axis=1, inplace=True)

        other_drop = ['ALPMax', 'ALTMin', 'ALTMax', 'ALTMean', 'ASTMin', 'ASTMax', 'ASTMean', 'BilirubinMin', 'BilirubinMax',
                      'BilirubinMean', 'CholesterolMin', 'CholesterolMax', 'CholesterolMean',
                      'CholesterolCount', 'pHMax', 'RespRateMin', 'RespRateMax', 'RespRateMean', 'RespRateCount']
        self.df.drop(other_drop, axis=1, inplace=True)
        # Features with different distributions for two classes
        self.replaceCol('ALPCount', 'up', 2)
        self.replaceCol('ALTCount', 'up', 2)
        self.replaceCol('ASTCount', 'up', 2)
        self.replaceCol('BilirubinCount', 'up', 2)
        self.replaceCol('FiO2Min', 'both', 0.8, 0.2)
        self.replaceCol('FiO2Mean', 'down', None, 0.3)
        self.replaceCol('GCSMax', 'down', None, 10)
        self.replaceCol('PaCO2Min', 'both', 60, 10)
        self.replaceCol('PaCO2Max', 'both', 80, 20)
        self.replaceCol('PaCO2Mean', 'both', 70, 20)
        # Remove TropI and TropT
        for var in ["TropI", "TropT"]:
            for type in ["Min", "Max", "Mean", "Count"]:
                self.df.drop(var+type, 1, inplace=True)
        # Imputation
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
            self.replaceByMean(col_name, float(stan))
        mode_replace = ['BUNMin 0', 'BUNMax 0']
        for item in mode_replace:
            col_name, stan = item.split()
            self.replaceByMode(col_name, float(stan))
        # Normalise
        # for i in range(1, len(df.columns)):
        #     col = df.ix[:,i]
        #     min_col = min(col)
        #     max_col = max(col)
        #     col = (col - min_col)/(max_col-min_col)
        #     df.ix[:,i] = col[:]

    def write_file(self):
        self.read_data()
        self.process_data()
        # Write files
        if self.set_name == 'a':
            self.df.to_csv('train.csv')
        else:
            self.df.to_csv('validation.csv')

cwd = os.getcwd()

prep = Preprocessing('a', cwd)
prep.write_file()

prep = Preprocessing('c', cwd)
prep.write_file()
