import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report, accuracy_score, roc_curve, confusion_matrix, average_precision_score, precision_recall_curve, r2_score
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore")

def classify_size(shape, min_size, max_size, df):
    df['{}_s'.format(shape)] = ((df['{}_avg'.format(shape)] < min_size) & (df['{}'.format(shape)] == 1)).astype(int)
    df['{}_m'.format(shape)] = ((df['{}_avg'.format(shape)] <= max_size) & (df['{}_avg'.format(shape)] >= min_size) & (df['{}'.format(shape)] == 1)).astype(int)
    df['{}_l'.format(shape)] = ((df['{}_avg'.format(shape)] > max_size) & (df['{}'.format(shape)] == 1)).astype(int)

def create_dataset(drop_nonimportant_columns = False, standard_scaler = False):
    df = pd.read_excel('../Datasets/dataset_labeled.xlsx')
    df['Cube_avg'] = df.loc[:,['Cube_min', 'Cube_max']].mean(axis=1)
    df['Stick_avg'] = df.loc[:,['Stick_min', 'Stick_max']].mean(axis=1)
    df['Sphere_avg'] = df.loc[:,['Sphere_min', 'Sphere_max']].mean(axis=1)
    df['Flat_avg'] = df.loc[:,['Flat_min', 'Flat_max']].mean(axis=1)
    df['Amorphous_avg'] = df.loc[:,['Amorphous_min', 'Amorphous_max']].mean(axis=1)
    df = df.drop(columns=['Image_id', 'Cube_min', 'Cube_max',
       'Stick_min', 'Stick_max', 'Sphere_min', 'Sphere_max', 'Flat_min',
       'Flat_max', 'Amorphous_min', 'Amorphous_max',])
    if drop_nonimportant_columns:
        df = df.drop(columns=['Stirring, rpm', 'Ca ion, mM', 'CO3 ion, mM', 'Hexadecyltrimethylammonium bromide', 'Triton X-100', '1-Hexanol', 'Methyl alcohol'])

    classify_size("Cube", 15, 20, df)
    classify_size("Sphere", 10, 14, df)
    classify_size("Stick", 35, 45, df)
    classify_size("Flat", 23, 30, df)
    classify_size("Amorphous", 12, 20, df)
    df = df.drop(columns=['Cube_avg', 'Stick_avg', 'Sphere_avg',
       'Flat_avg', 'Amorphous_avg'])
    X = df.iloc[:,0:29]
    num_cols = X.iloc[:,0:10].columns
    if standard_scaler:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    return df, X
